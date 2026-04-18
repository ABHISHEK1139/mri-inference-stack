# Reproducibility and Operations Runbook

## 1. Local Setup

```powershell
git clone <your-repo-url>
cd "Abhishek Kumar"
git lfs install
git lfs pull
python -m pip install -r requirements.txt
```

## 2. Readiness Preflight

Run structural checks before launching demos:

```powershell
python scripts/preflight.py
```

Require core model artifacts:

```powershell
python scripts/preflight.py --require-weights
```

## 3. Quality Gate

Install dev tooling and lint the codebase:

```powershell
python -m pip install -r requirements-dev.txt
ruff check .
python -m compileall app.py train.py config.py data models training evaluation scripts
```

## 4. Application Runtime Paths

Local Streamlit:

```powershell
streamlit run app.py
```

Docker Compose:

```powershell
docker compose up -d --build
```

Ansible automation:

```powershell
ansible-playbook -i ansible/inventory.ini ansible/site.yml
```

Kubernetes:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## 5. Training Paths

Run all tracks:

```powershell
python train.py --track all
```

Run a specific track:

```powershell
python train.py --track detection
python train.py --track classifier
python train.py --track gan --gan_type conditional
python train.py --track segmentation
```

Download datasets only:

```powershell
python train.py --only_download
```

## 6. CI Behavior

The GitHub Actions quality workflow performs:

- Ruff lint checks
- Python compile smoke checks
- preflight structural verification in CI mode

The workflow file is in `.github/workflows/quality.yml`.
