# MRI Inference Stack

[![Quality Gate](https://github.com/ABHISHEK1139/mri-inference-stack/actions/workflows/quality.yml/badge.svg)](https://github.com/ABHISHEK1139/mri-inference-stack/actions/workflows/quality.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Multi--stage-blue.svg)](Dockerfile)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red.svg)](https://streamlit.io/)

A deployment-ready ML system focused on **resource-constrained 2D brain MRI screening and tumour type classification**, with optional research extensions for segmentation and synthesis.

> [!IMPORTANT]
> **Scope & Clinical Disclaimer**: This pipeline processes **2D grayscale MRI slices** (PNG/JPG/BMP/TIFF). It does **not** handle volumetric formats (NIfTI, DICOM) and is **not a clinical diagnostic tool**. Model predictions represent screening probabilities and must not be used for clinical decision-making.

The repository is structured to demonstrate an operational machine learning lifecycle:

- reproducible packaging with multi-stage Docker builds
- infrastructure automation with Ansible
- container orchestration with Kubernetes manifests
- Git LFS artifact handling for runnable model weights
- leak-free dataset partitioning and centralized preprocessing contracts

## Architectural Highlights

- **Calibrated Detection**: Binary screening model (Normal vs. Tumour) utilizing residual blocks and Squeeze-and-Excitation channel attention, evaluated at an empirically calibrated threshold (`0.225`) optimizing sensitivity and F1.
- **Tumour Type Classification**: Multi-class classifier with an EfficientNetB0 backbone and dual attention pooling, categorizing detected lesions into `glioma`, `meningioma`, `pituitary`, or `normal`.
- **Dynamic Attention U-Net**: Medical image segmentation featuring skip connections modulated by dynamically resized attention gates and residual blocks, trained with combined Dice + BCE loss.
- **Conditional GAN / WGAN-GP**: Synthesis pipeline utilizing spectral normalization, self-attention, projection discrimination, and Exponential Moving Average (EMA) weight tracking for conditional MRI generation.
- **Centralized Preprocessing**: [`preprocessing.py`](preprocessing.py) serves as the single source of truth for input resizing, channel adaptation, and range scaling (`[0, 1]` for classification/detection/segmentation, `[-1, 1]` for GANs).
- **Leakage-Free Splitting**: Native support for patient-level grouping (`GroupShuffleSplit`) to prevent multi-slice data leakage between training and evaluation partitions.

## Results Snapshot

| Component | Split / Context | Key Metrics |
| --- | --- | --- |
| **Detection** | Calibrated validation operating point | Threshold `0.225`, Accuracy `0.9813`, Precision `0.9939`, Recall `0.9839`, Specificity `0.9667`, F1 `0.9889` |
| **Classification** | Held-out evaluation from training logs | Accuracy `0.9893`, Precision `0.9894`, Recall `0.9893`, F1 `0.9893`, AUC `0.9998` |
| **Segmentation** | Experimental validation track | Validation Dice `0.4808`, Validation IoU `0.4216` |
| **GAN** | Experimental research track | Qualitative samples, WGAN-GP loss curves, FID/FS tracking |

> [!NOTE]
> **Metrics Caveat**: Baseline metrics were computed using image-level splitting. To eliminate potential inter-slice data leakage across the same patient, use the `--patient_level` flag during training and evaluation.

Core visual outputs are stored in `outputs/`.

## Core Workflow

1. Upload 2D grayscale MRI slice via the Streamlit web app
2. Run calibrated screening detection (decision threshold: `0.225`)
3. If positive, automatically route to tumour type classification
4. (Optional) inspect interactive Attention U-Net segmentation and conditional GAN research tabs

## Quality and Readiness Checks

Run the preflight checker before demos or interviews:

```powershell
python scripts/preflight.py
```

Require core inference artifacts during validation:

```powershell
python scripts/preflight.py --require-weights
```

Run static analysis and compile check:

```powershell
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m compileall app.py train.py config.py preprocessing.py data models training evaluation scripts tests -q
```

Run the automated test suite:

```powershell
python -m pytest tests/ -v
```

Automated checks are defined in [`.github/workflows/quality.yml`](.github/workflows/quality.yml).

## Local Run

```powershell
git clone https://github.com/ABHISHEK1139/mri-inference-stack.git
cd mri-inference-stack
git lfs install
git lfs pull
python -m pip install -r requirements.txt
streamlit run app.py
```

App URL: `http://localhost:8501`

## Docker Run

The multi-stage Docker build includes build-time Git LFS validation to ensure complete model artifacts:

```powershell
docker compose up -d --build
```

App URL: `http://localhost:8501`

## Ansible Automation

```powershell
ansible-playbook -i ansible/inventory.ini ansible/site.yml
```

This playbook performs dependency verification, pulls Git LFS artifacts, and starts the Docker container.

## Kubernetes Deployment

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

> **Storage Configuration**: By default, the pod consumes model weights baked directly into the container image (`/app/weights`). The `outputs-volume` mount preserves runtime exports. To supply weights externally via cluster storage, uncomment the `weights-volume` PVC in [`k8s/deployment.yaml`](k8s/deployment.yaml).

## Training Pipelines

Train individual tracks or the entire stack with GPU memory management:

```powershell
# Train detection with patient-level grouping
python train.py --track detection --patient_level

# Train classification with patient-level grouping
python train.py --track classifier --patient_level

# Train Attention U-Net segmentation
python train.py --track segmentation --patient_level

# Train WGAN-GP v2 synthesis model
python train.py --track gan_v2 --epochs 100

# Train all tracks sequentially
python train.py --track all --patient_level
```

Key CLI flags:
- `--track`: `all`, `detection`, `segmentation`, `classifier`, `gan`, `gan_v2`, `gan_augmented`
- `--patient_level`: Groups slices by patient ID using `GroupShuffleSplit` to prevent multi-slice data leakage
- `--download_figshare`: Automatically download and extract Figshare dataset
- `--download_brats`: Automatically download BraTS dataset
- `--no_resume`: Start fresh training run instead of resuming from checkpoints

## Weights and Outputs Policy

This repository tracks runtime-ready inference artifacts:

- tracked model weights in `weights/` (Git LFS-managed binary models)
- calibrated detection configuration in `weights/detection_inference_config.json`
- curated visual outputs in `outputs/`

Heavy transient artifacts are excluded from version control:

- training checkpoints (`checkpoints/`)
- raw logs (`logs/`)
- raw datasets (`data/raw/`, `data/processed/`)

## Project Structure

```text
.
|-- .github/workflows/       # CI quality checks and model smoke tests
|-- app.py                   # Streamlit web application & UI
|-- preprocessing.py         # Centralized preprocessing contracts
|-- train.py                 # Multi-track training engine
|-- config.py                # System profiles, paths, and hyperparameters
|-- docker-compose.yml       # Container orchestration spec
|-- Dockerfile               # Multi-stage container definition with LFS check
|-- ansible/                 # Provisioning and deployment playbooks
|-- k8s/                     # Kubernetes production manifests
|-- data/                    # Dataset loaders, splits, and augmentation
|-- models/                  # Detection, Classifier, U-Net, and GAN architectures
|-- training/                # Callbacks, state managers, and training utilities
|-- evaluation/              # Metrics, threshold calibration, and confusion matrices
|-- scripts/                 # Preflight readiness checker and utilities
|-- tests/                   # Test suite (dataset, models, metrics, preprocessing)
|-- docs/                    # Architecture diagrams, system design, and runbooks
|-- weights/                 # LFS-managed runnable model weights
|-- outputs/                 # Curated evaluation plots and figures
```

## Documentation

- [System Design Document](docs/system-design.md)
- [Architecture Diagrams](docs/architecture_diagrams.md)
- [Dataset Specifications](docs/dataset.md)
- [Reproducibility Runbook](docs/reproducibility_runbook.md)
- [Research Extensions](docs/research_extensions.md)
- [Innovation Overview](docs/innovation.md)
- [Kubernetes Operations](k8s/README.md)
- [Ansible Automation](ansible/README.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## Limitations

- **2D Slices Only**: The pipeline accepts 2D grayscale images (PNG/JPG/BMP/TIFF). It does not process volumetric MRI formats (NIfTI `.nii/.nii.gz`, DICOM `.dcm`).
- **Not a Diagnostic Tool**: Model predictions are screening-level likelihoods and must not be used as clinical diagnoses.
- **No Domain-Level Input Filtering**: The system does not verify that an uploaded image is a brain MRI scan; non-medical images will produce unvalidated predictions.
- **Image-Level Benchmark Baseline**: Legacy benchmark metrics were computed using image-level train/test splits. Future training passes should employ `--patient_level`.
- **Pretrained Initialization**: The saved classifier checkpoint was originally trained from scratch; recent code updates enable ImageNet transfer learning for future training runs.
- **Hardware Requirements**: Segmentation and GAN tracks are experimental modules and require sufficient GPU VRAM for extended runs.

## License

This repository uses the [MIT License](LICENSE).
