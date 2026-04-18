"""Project configuration for the Brain MRI intelligence system."""
import os
from dataclasses import dataclass, field
from typing import Tuple


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


GPU_MEMORY_GB = float(os.getenv("GPU_MEMORY_GB", "0") or 0)
LOW_VRAM_MODE = _env_flag("LOW_VRAM_MODE", default=bool(GPU_MEMORY_GB and GPU_MEMORY_GB <= 4))
RUNTIME_PROFILE = "low_vram" if LOW_VRAM_MODE else "default"
GAN_IMAGE_SIZE = _env_int("GAN_IMAGE_SIZE", 64 if LOW_VRAM_MODE else 128)

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

# Dataset sources
DATASET_CONFIG = {
    "figshare": {
        "url": "https://figshare.com/ndownloader/files/41354957",
        "description": "Figshare Brain Tumour MRI (glioma, meningioma, pituitary, normal)",
        "classes": ["glioma", "meningioma", "pituitary", "normal"],
    },
    "brats": {
        "url": "https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation",
        "description": "BraTS 2020 - Segmentation + Classification",
        "classes": ["glioma", "meningioma", "pituitary", "normal"],
    },
}


@dataclass
class ImageConfig:
    detection_size: Tuple[int, int] = (224, 224)
    segmentation_size: Tuple[int, int] = (128, 128) if LOW_VRAM_MODE else (256, 256)
    classifier_size: Tuple[int, int] = (224, 224)
    gan_size: Tuple[int, int] = field(default_factory=lambda: (GAN_IMAGE_SIZE, GAN_IMAGE_SIZE))
    channels: int = 1
    normalize_range: Tuple[float, float] = (-1.0, 1.0)


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    early_stopping_patience: int = 7
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7


if LOW_VRAM_MODE:
    TRACK_CONFIGS = {
        "detection": TrainConfig(epochs=30, batch_size=_env_int("DETECTION_BATCH_SIZE", 16), learning_rate=1e-3),
        "segmentation": TrainConfig(epochs=50, batch_size=_env_int("SEG_BATCH_SIZE", 2), learning_rate=1e-4),
        "classifier": TrainConfig(epochs=40, batch_size=_env_int("CLASSIFIER_BATCH_SIZE", 8), learning_rate=1e-4),
        "gan": TrainConfig(epochs=100, batch_size=_env_int("GAN_BATCH_SIZE", 8), learning_rate=2e-4),
    }
else:
    TRACK_CONFIGS = {
        "detection": TrainConfig(epochs=30, batch_size=_env_int("DETECTION_BATCH_SIZE", 32), learning_rate=1e-3),
        "segmentation": TrainConfig(epochs=50, batch_size=_env_int("SEG_BATCH_SIZE", 8), learning_rate=1e-4),
        "classifier": TrainConfig(epochs=40, batch_size=_env_int("CLASSIFIER_BATCH_SIZE", 16), learning_rate=1e-4),
        "gan": TrainConfig(epochs=100, batch_size=_env_int("GAN_BATCH_SIZE", 64), learning_rate=2e-4),
    }


LATENT_DIM = 100
GAN_LABEL_SMOOTHING = 0.9
GAN_NOISE_STD = 0.1

FID_BATCH_SIZE = 64
FS_BATCH_SIZE = 64

PROJECT_NAME = "MRI Inference Stack"
FLAGSHIP_TRACKS = ("detection", "classifier")
EXPERIMENTAL_TRACKS = ("segmentation", "gan")

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "normal"]
NUM_CLASSES = len(CLASS_NAMES)

for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, CHECKPOINT_DIR, LOG_DIR, OUTPUT_DIR, WEIGHTS_DIR]:
    os.makedirs(d, exist_ok=True)
