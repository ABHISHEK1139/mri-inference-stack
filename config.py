"""Project configuration for the Brain MRI intelligence system."""
import os
from dataclasses import dataclass, field


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
# Source datasets, keyed by name. The inner shape varies (BraTS carries a
# "task" key, figshare does not), so it is typed as a mapping rather than a
# fixed dataclass: every consumer reads the keys it needs.
DATASET_CONFIG: dict[str, dict[str, object]] = {
    "figshare": {
        "url": "https://figshare.com/ndownloader/files/41354957",
        "description": "Figshare Brain Tumour MRI (glioma, meningioma, pituitary, normal)",
        "classes": ["glioma", "meningioma", "pituitary", "normal"],
    },
    "brats": {
        "url": "https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation",
        "description": "BraTS 2020 - Glioma segmentation benchmark (binary mask task)",
        "task": "segmentation",
        "classes": ["tumour"],  # BraTS is a glioma segmentation dataset, not 4-class
    },
}


def dataset_url(name: str) -> str:
    """Return the download URL for a configured dataset.

    Typed accessor so callers do not have to narrow the heterogeneous
    ``DATASET_CONFIG`` mapping themselves.

    Raises:
        KeyError: The dataset is not configured.
        TypeError: The configured entry has no usable ``url``.
    """
    if name not in DATASET_CONFIG:
        raise KeyError(
            f"Unknown dataset {name!r}. Configured: {sorted(DATASET_CONFIG)}"
        )
    url = DATASET_CONFIG[name].get("url")
    if not isinstance(url, str) or not url:
        raise TypeError(f"Dataset {name!r} has no usable 'url' entry.")
    return url


@dataclass
class ImageConfig:
    detection_size: tuple[int, int] = (224, 224)
    segmentation_size: tuple[int, int] = (128, 128) if LOW_VRAM_MODE else (256, 256)
    classifier_size: tuple[int, int] = (224, 224)
    gan_size: tuple[int, int] = field(default_factory=lambda: (GAN_IMAGE_SIZE, GAN_IMAGE_SIZE))
    channels: int = 1
    normalize_range: tuple[float, float] = (-1.0, 1.0)


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
        "detection": TrainConfig(epochs=30, batch_size=_env_int("DETECTION_BATCH_SIZE", 16),
            learning_rate=1e-3),
        "segmentation": TrainConfig(epochs=50, batch_size=_env_int("SEG_BATCH_SIZE", 2),
            learning_rate=1e-4),
        "classifier": TrainConfig(epochs=40, batch_size=_env_int("CLASSIFIER_BATCH_SIZE", 8),
            learning_rate=1e-4),
        "gan": TrainConfig(epochs=100, batch_size=_env_int("GAN_BATCH_SIZE", 8),
            learning_rate=2e-4),
    }
else:
    TRACK_CONFIGS = {
        "detection": TrainConfig(epochs=30, batch_size=_env_int("DETECTION_BATCH_SIZE", 32),
            learning_rate=1e-3),
        "segmentation": TrainConfig(epochs=50, batch_size=_env_int("SEG_BATCH_SIZE", 8),
            learning_rate=1e-4),
        "classifier": TrainConfig(epochs=40, batch_size=_env_int("CLASSIFIER_BATCH_SIZE", 16),
            learning_rate=1e-4),
        "gan": TrainConfig(epochs=100, batch_size=_env_int("GAN_BATCH_SIZE", 64),
            learning_rate=2e-4),
    }


LATENT_DIM = 100
GAN_LABEL_SMOOTHING = 0.9
GAN_NOISE_DROPOUT_STD = 0.1

FID_BATCH_SIZE = 64
FS_BATCH_SIZE = 64

# ── Reproducibility ──────────────────────────────────────────────────────
# Every source of randomness is driven from this seed: Python's `random`,
# NumPy's legacy global generator, and TensorFlow/Keras (weight init, dropout,
# augmentation, tf.data shuffling).
#
# Seeding makes a run repeatable on the *same* machine and TensorFlow build. It
# does not make results bit-identical across different hardware, TF versions, or
# `nondeterministic` kernels, which is why the seed is recorded in every
# checkpoint state file next to the epoch it belongs to.
DEFAULT_SEED = _env_int("SEED", 42)
DETERMINISTIC_OPS = _env_flag("DETERMINISTIC_OPS", default=False)

PROJECT_NAME = "MRI Inference Stack"
FLAGSHIP_TRACKS = ("detection", "classifier")
EXPERIMENTAL_TRACKS = ("segmentation", "gan")

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "normal"]
NUM_CLASSES = len(CLASS_NAMES)

MANAGED_DIRS = (
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
    OUTPUT_DIR,
    WEIGHTS_DIR,
)


def ensure_directories() -> None:
    """Create the project's managed output directories.

    Called explicitly by entry points instead of at import time: merely
    ``import config`` (which every test and every module does) previously created
    seven directories as a side effect, which fails on read-only filesystems and
    pollutes the working tree during test collection.
    """
    for directory in MANAGED_DIRS:
        os.makedirs(directory, exist_ok=True)
