# Reproducibility and Operations Runbook

## 1. Local Setup

```powershell
git clone https://github.com/ABHISHEK1139/mri-inference-stack.git
cd mri-inference-stack
git lfs install
git lfs pull

# Pinned stack -- reproduces the environment the tests were validated against
python -m pip install -r requirements.lock

# Or the floating ranges, if you need a different resolution
python -m pip install -r requirements.txt

# Optional extras
python -m pip install -e ".[segmentation]"   # BraTS NIfTI volumes
python -m pip install -e ".[dev]"            # pytest + ruff
```

> `requirements.lock` is generated from the validated environment, not resolved
> on every build. Regenerate it deliberately with `pip-compile` and re-run the
> test suite before committing the result.

## 2. Determinism and Seeds

Every source of randomness is driven from a single seed, applied in
`training/reproducibility.py`:

| Source | Controlled by |
| --- | --- |
| Weight initialisation | `tf.keras.utils.set_random_seed` |
| Dropout masks | same |
| Data augmentation | same (augmentation draws from the TF graph RNG) |
| `tf.data` shuffling | explicit `seed=` on every builder |
| Split shuffling | `random_state=42` defaults in `data/dataset.py` |
| Synthetic data mixing | `np.random.default_rng(seed)` |

Select a seed per run:

```powershell
python train.py --track detection --seed 1234
```

Or set it for the whole session:

```powershell
$env:SEED = 1234
python train.py --track all
```

For stricter reproducibility (slower; some ops have no deterministic kernel):

```powershell
python train.py --track detection --deterministic
```

The seed, TensorFlow/NumPy/Python versions, and the determinism flag are written
into every `checkpoints/**/training_state.json` and `checkpoints/gan/*_state.json`
under a `reproducibility` block, so a run can be traced back to its configuration.

**What seeding does and does not guarantee.** Two runs with the same seed on the
same machine, same TensorFlow build, and same data produce the same weights. It
does not make results bit-identical across different hardware or TensorFlow
versions, because those change kernel selection and floating-point reduction
order. Use `requirements.lock` and record the seed to get comparable runs.

## 3. Readiness Preflight

```powershell
python scripts/preflight.py                  # structural checks
python scripts/preflight.py --require-weights
python scripts/preflight.py --require-datasets
python scripts/preflight.py --json           # machine-readable output
```

Preflight validates the Python version, required files and training modules, the
presence of the new k8s manifests, that `weights/*.keras` are real archives
rather than Git LFS pointer stubs, and that
`weights/detection_inference_config.json` contains a usable threshold.

## 4. Quality Gate

```powershell
python -m pip install -r requirements-dev.txt
ruff check .
python -m compileall app.py train.py config.py preprocessing.py data models training evaluation scripts tests
python -m pytest tests/ -m "not slow"        # unit + regression
python -m pytest tests/ --cov=. --cov-report=term-missing
```

Trainer integration tests run a real single-epoch training run per track against
a generated fixture. They are marked `slow` and split out so the fast feedback
loop stays quick:

```powershell
python -m pytest tests/ -m slow
```

## 5. Application Runtime

```powershell
streamlit run app.py
docker compose up -d --build
ansible-playbook -i ansible/inventory.ini ansible/site.yml
```

### Container variants

```powershell
# GPU-capable (default, ~4.4 GB)
docker build -t mri .

# CPU-only (~3.3 GB)
docker build --build-arg TENSORFLOW_DIST=tensorflow-cpu -t mri .
```

Both variants pin the same TensorFlow version from `requirements.lock`; only the
CUDA payload differs. Verify either with:

```powershell
docker compose ps                                     # expect "(healthy)"
docker compose exec mri-app id                        # uid=10001(app)
docker compose exec mri-app python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health').status)"
```

The container runs as a non-root user against a read-only root filesystem, with
`/tmp` provided as scratch space. That matches the Kubernetes manifests, so the
same image is validated by the compose run.

Kubernetes, in order:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/networkpolicy.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/pdb.yaml
```

The image tag in `k8s/deployment.yaml` is versioned rather than `:latest`, so a
rollout is an explicit, reversible act. The container runs as UID 10001 with a
read-only root filesystem; Streamlit writes to the `/tmp` emptyDir.

The Service is `ClusterIP`. To expose the app, front it with an Ingress that
terminates TLS and authentication -- a commented example is in
`k8s/service.yaml`. Do not put the inference endpoint back on a `NodePort`.

## 6. Training Paths

```powershell
python train.py --track all
python train.py --track detection --patient_level
python train.py --track classifier --epochs 40 --seed 7
python train.py --track gan --gan_type conditional
python train.py --track gan_v2
python train.py --track segmentation
python train.py --only_download
python train.py --no_resume --track detection      # ignore checkpoints
```

Useful environment variables:

| Variable | Purpose |
| --- | --- |
| `SEED` | Default seed for all runs |
| `DETERMINISTIC_OPS` | Request deterministic kernels |
| `LOW_VRAM_MODE` / `GPU_MEMORY_GB` | Tune batch sizes and resolutions for small GPUs |
| `MIXED_PRECISION` | Toggle float16 training on GPU |
| `TF_MEMORY_GROWTH` | Toggle TensorFlow GPU memory growth |
| `PATIENT_LEVEL_SPLIT` | Force patient-level splitting without the flag |
| `GAN_LAMBDA_GP` / `GAN_D_STEPS` / `GAN_G_STEPS` | WGAN-GP tuning |
| `LOG_LEVEL` / `--log_level` | Console verbosity |

## 7. Training Engine Layout

`train.py` is a thin CLI. The engine lives in `training/`:

```
training/
  runtime.py          device policy, seeding, shared helpers
  state.py            atomic checkpoint state for resume
  reproducibility.py  seeding and determinism
  data_sources.py     dataset acquisition
  callbacks.py        checkpointing, logging, collapse detection
  tracks/
    detection.py  classifier.py  segmentation.py
    gan.py        gan_v2.py      gan_augmented.py
```

Each track module is independently importable, which is what makes the
integration tests in `tests/test_trainers_integration.py` possible.

## 8. CI Behavior

`.github/workflows/quality.yml` runs on every push and pull request:

- Ruff lint
- Python compile check
- Preflight structural verification (`--ci-mode`)
- Model construction smoke tests (detection, classifier, U-Net, GAN v2)
- `tf.data` pipeline smoke test
- Unit and regression tests (`-m "not slow"`)
- Trainer integration tests (`-m slow`)
- Coverage report uploaded as an artifact

The workflow installs from `requirements.lock`, so CI runs the pinned stack.
