# MRI Inference Stack

A deployment-ready ML system focused on **resource-constrained brain MRI screening and subtype classification**, with optional research extensions for segmentation and synthesis.

The repository is structured to demonstrate an operational machine learning lifecycle:

- reproducible packaging with Docker
- infrastructure automation with Ansible
- cluster deployment path with Kubernetes
- Git LFS artifact handling for runnable model weights

## Why this is structured this way

This project aligns with rigorous engineering expectations where software is expected to be:

- reproducible across heterogeneous systems
- infrastructure-aware (container and orchestration friendly)
- practical to hand over and operate

Instead of adding a huge platform, the stack here stays compact and transparent.

## Results Snapshot

| Component | Split / context | Key metrics |
| --- | --- | --- |
| Detection | Calibrated validation operating point | threshold `0.225`, accuracy `0.9813`, precision `0.9939`, recall `0.9839`, specificity `0.9667`, F1 `0.9889` |
| Classification | Held-out evaluation from training logs | accuracy `0.9893`, precision `0.9894`, recall `0.9893`, F1 `0.9893`, AUC `0.9998` |
| Segmentation | Experimental validation track | validation Dice `0.4808`, validation IoU `0.4216` |
| GAN | Experimental research track | qualitative samples and training curves |

Core visual outputs are stored in `outputs/`.

## Core Workflow

1. Upload MRI image
2. Run calibrated tumour detection
3. If positive, run subtype classification
4. (Optional) inspect segmentation and GAN research tabs

## Quality and Readiness Checks

Run the preflight checker before demos or interviews:

```powershell
python scripts/preflight.py
```

Require core inference artifacts during validation:

```powershell
python scripts/preflight.py --require-weights
```

Run the static quality gate:

```powershell
python -m pip install -r requirements-dev.txt
ruff check .
python -m compileall app.py train.py config.py data models training evaluation scripts
```

Automated checks are defined in `.github/workflows/quality.yml`.

## Local Run

```powershell
git clone <your-repo-url>
cd "Abhishek Kumar"
git lfs install
git lfs pull
python -m pip install -r requirements.txt
streamlit run app.py
```

## Docker Run

```powershell
docker compose up -d --build
```

App URL: `http://localhost:8501`

## Ansible Run

```powershell
ansible-playbook -i ansible/inventory.ini ansible/site.yml
```

This playbook performs dependency checks, pulls LFS artifacts, and starts the Docker stack.

## Kubernetes Run

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

The default setup uses `hostPath` volumes for portability. You can replace them with PVCs in managed clusters.

## Weights and Outputs Policy

This repo intentionally keeps runtime-ready artifacts:

- tracked model weights in `weights/` (LFS-managed)
- curated outputs in `outputs/`

It intentionally excludes heavy transient artifacts:

- training checkpoints
- raw logs
- raw datasets

## Project Structure

```text
.
|-- .github/workflows/       # CI quality checks
|-- app.py                   # Streamlit application
|-- train.py                 # Multi-track training entrypoint
|-- config.py                # Runtime and profile configuration
|-- docker-compose.yml
|-- Dockerfile
|-- ansible/                 # Automation playbooks
|-- k8s/                     # Kubernetes manifests
|-- data/                    # Dataset loaders and local data roots
|-- models/                  # Model architectures
|-- training/                # Callback and training utilities
|-- evaluation/              # Metrics and evaluation logic
|-- scripts/                 # Readiness and helper scripts
|-- tests/                   # Test scaffolding and future unit tests
|-- docs/                    # Architecture, runbooks, and alignment docs
|-- weights/                 # LFS-managed runtime model artifacts
|-- outputs/                 # Curated evaluation artifacts
```

## Documentation

- [docs/README.md](docs/README.md)
- [docs/architecture_diagrams.md](docs/architecture_diagrams.md)
- [docs/dataset.md](docs/dataset.md)
- [docs/innovation.md](docs/innovation.md)
- [docs/research_extensions.md](docs/research_extensions.md)
- [docs/system-design.md](docs/system-design.md)
- [docs/reproducibility_runbook.md](docs/reproducibility_runbook.md)
- [ansible/README.md](ansible/README.md)
- [k8s/README.md](k8s/README.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)

## Limitations

- Raw medical datasets are not included.
- Exact runtime behavior depends on host TensorFlow/CUDA compatibility.
- Segmentation and GAN tracks are still positioned as research modules.

## License

This repository uses the [MIT License](LICENSE).
