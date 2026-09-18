"""Centralized preprocessing contracts for all model tracks.

Each preprocessor documents its normalization range, target size, and channel
handling. Both the Streamlit app and the training pipeline should import from
this module to ensure training↔inference consistency.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

from config import ImageConfig

_IMG_CFG = ImageConfig()


def _to_grayscale_array(
    image: Image.Image,
    target_size: Tuple[int, int],
    resample: int = Image.BILINEAR,
) -> np.ndarray:
    """Convert a PIL image to a grayscale float32 array with channel dim."""
    gray = image.convert("L")
    resized = gray.resize((target_size[1], target_size[0]), resample=resample)
    return np.asarray(resized, dtype=np.float32)


# ── Detection ────────────────────────────────────────────────────────────

def preprocess_detection(
    image: Image.Image,
    target_size: Tuple[int, int] | None = None,
) -> np.ndarray:
    """Preprocess a single image for the binary tumour detection model.

    Normalization: [0.0, 1.0]
    Output shape:  (1, H, W, 1)
    """
    target_size = target_size or _IMG_CFG.detection_size
    array = _to_grayscale_array(image, target_size) / 255.0
    return np.expand_dims(array, axis=(0, -1))


# ── Classifier ───────────────────────────────────────────────────────────

def preprocess_classifier(
    image: Image.Image,
    target_size: Tuple[int, int] | None = None,
) -> np.ndarray:
    """Preprocess a single image for the multi-class tumour classifier.

    Normalization: [0.0, 1.0]
    Output shape:  (1, H, W, 1)
    """
    target_size = target_size or _IMG_CFG.classifier_size
    array = _to_grayscale_array(image, target_size) / 255.0
    return np.expand_dims(array, axis=(0, -1))


# ── Segmentation ─────────────────────────────────────────────────────────

def preprocess_segmentation(
    image: Image.Image,
    target_size: Tuple[int, int] | None = None,
) -> np.ndarray:
    """Preprocess a single image for the segmentation U-Net.

    Normalization: [0.0, 1.0]
    Output shape:  (1, H, W, 1)
    """
    target_size = target_size or _IMG_CFG.segmentation_size
    array = _to_grayscale_array(image, target_size) / 255.0
    return np.expand_dims(array, axis=(0, -1))


def preprocess_segmentation_mask(
    image: Image.Image,
    target_size: Tuple[int, int] | None = None,
) -> np.ndarray:
    """Preprocess a segmentation mask (binary threshold at > 0).

    Resampling: NEAREST (to preserve label boundaries)
    Output shape: (1, H, W, 1)
    """
    target_size = target_size or _IMG_CFG.segmentation_size
    array = _to_grayscale_array(image, target_size, resample=Image.NEAREST)
    array = (array > 0).astype(np.float32)
    return np.expand_dims(array, axis=(0, -1))


# ── GAN ──────────────────────────────────────────────────────────────────

def preprocess_gan(
    image: Image.Image,
    target_size: Tuple[int, int] | None = None,
) -> np.ndarray:
    """Preprocess a single image for GAN training/evaluation.

    Normalization: [-1.0, 1.0]
    Output shape:  (1, H, W, 1)
    """
    target_size = target_size or _IMG_CFG.gan_size
    array = _to_grayscale_array(image, target_size)
    array = array / 127.5 - 1.0
    return np.expand_dims(array, axis=(0, -1))
