"""Runtime configuration helpers: device setup, seeding, and shared utilities.

Kept separate from the individual trainers so device policy, precision
handling, and the JSON/gradient helpers are defined exactly once.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import tensorflow as tf

from config import (
    LATENT_DIM,
    LOW_VRAM_MODE,
    RUNTIME_PROFILE,
    ImageConfig,
    ensure_directories,
)
from models.segmentation import dice_bce_loss, dice_coefficient, iou_metric
from training.reproducibility import seed_state_dict, set_seed

__all__ = [
    "IMG_CFG",
    "SEGMENTATION_CUSTOM_OBJECTS",
    "configure_gpu",
    "seed_state_dict",
    "set_seed",
    "float32_precision",
]

from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Shared image geometry and the custom objects required to deserialize a U-Net.
IMG_CFG = ImageConfig()

# Required to reload a serialized U-Net: the model is compiled with a custom
# loss and two custom metrics, which Keras cannot resolve from a .keras archive
# without them.
SEGMENTATION_CUSTOM_OBJECTS = {
    "dice_bce_loss": dice_bce_loss,
    "dice_coefficient": dice_coefficient,
    "iou_metric": iou_metric,
}



def configure_gpu():
    """Configure TensorFlow for the target GPU."""
    ensure_directories()
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            use_memory_growth = os.getenv("TF_MEMORY_GROWTH", "1").strip().lower() in {"1", "true",
                "yes", "on"}
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, use_memory_growth)
            logger.info(f'TF memory growth: {use_memory_growth}')
            logger.info(f'GPU detected: {[gpu.name for gpu in gpus]}')
            use_mixed_precision = os.getenv("MIXED_PRECISION", "1").strip().lower() in {"1", "true",
                "yes", "on"}
            if use_mixed_precision:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info(f'Mixed precision enabled: {policy.name}')
            else:
                policy = tf.keras.mixed_precision.Policy("float32")
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.warning(f'Mixed precision disabled: {policy.name}')
        except RuntimeError as e:
            logger.warning(f'GPU setup warning: {e}')
    else:
        logger.info('No GPU detected, running on CPU')

    logger.info(f'TensorFlow: {tf.__version__}')
    logger.info(f'CUDA build: {tf.test.is_built_with_cuda()}')
    logger.info(f'Runtime profile: {RUNTIME_PROFILE}')
    if LOW_VRAM_MODE:
        logger.info('Low-VRAM mode enabled for 4GB-class GPUs')
    if gpus:
        logger.info(f'GPU device: {tf.test.gpu_device_name()}')


def _balanced_class_weight_dict(labels):
    """Compute simple balanced class weights for binary labels."""
    labels = np.asarray(labels, dtype=np.int32)
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    total = float(counts.sum())
    num_classes = float(len(counts))
    weights = {}
    for idx, count in enumerate(counts):
        weights[idx] = float(total / max(num_classes * count, 1.0))
    return weights


def _json_safe(value):
    """Recursively convert NumPy values into JSON-safe Python types."""
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sanitize_grads(grads, variables):
    """Replace None/NaN/Inf gradients with finite tensors."""
    fixed = []
    for grad, var in zip(grads, variables, strict=True):
        if grad is None:
            fixed.append(tf.zeros_like(var))
            continue
        fixed.append(tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad)))
    return fixed


def _generator_is_finite(generator, conditional, num_classes, sample_count=16):
    """Quick health probe to ensure generator outputs are finite."""
    z = tf.random.normal([sample_count, LATENT_DIM])
    if conditional:
        idx = tf.random.uniform([sample_count], 0, num_classes, dtype=tf.int32)
        y = tf.one_hot(idx, num_classes)
        out = generator([z, tf.cast(y, tf.float32)], training=False)
    else:
        out = generator(z, training=False)
    return bool(tf.reduce_all(tf.math.is_finite(out)).numpy())


@contextmanager
def float32_precision():
    """Force float32 for the duration of a block, always restoring the policy.

    The GAN trainers previously restored the mixed-precision policy only on the
    normal exit path; any early ``return`` (for example "already reached
    requested epochs") left every subsequent track running in float32, silently
    disabling the configured mixed precision.
    """
    original_policy = tf.keras.mixed_precision.global_policy()
    switched = original_policy.name != "float32"
    if switched:
        logger.info(f'Switching from {original_policy.name} to float32 for this stage')
        tf.keras.mixed_precision.set_global_policy("float32")
    try:
        yield
    finally:
        if switched:
            tf.keras.mixed_precision.set_global_policy(original_policy)
            logger.info(f'Restored mixed precision policy: {original_policy.name}')
