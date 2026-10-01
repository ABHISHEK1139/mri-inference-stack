"""tf.data pipeline builders for each training track, plus synthetic mixing."""

import logging
import os
from collections.abc import Sequence

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import NUM_CLASSES
from data.dataset.loaders import load_images_from_paths
from data.dataset.loading import (
    _finalize_dataset,
    _load_path_image_tf,
    _shuffle,
    augment_image,
)
from data.dataset.splits import _flatten_split, get_figshare_file_index

logger = logging.getLogger(__name__)


def build_segmentation_dataset(
    images: np.ndarray,
    masks: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    augment: bool = True,
    seed: int | None = None,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((images.astype(np.float32),
        masks.astype(np.float32)))
    if shuffle:
        dataset = _shuffle(dataset, len(images), buffer_limit=1024, seed=seed)
    if augment:
        dataset = dataset.map(lambda x, y: augment_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    return _finalize_dataset(dataset, batch_size=batch_size)


def build_segmentation_dataset_from_paths(
    img_paths: Sequence[str | os.PathLike[str]],
    mask_paths: Sequence[str | os.PathLike[str]],
    img_size: tuple[int, int],
    batch_size: int,
    shuffle: bool = True,
    augment: bool = True,
    seed: int | None = None,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((list(img_paths), list(mask_paths)))
    if shuffle:
        dataset = _shuffle(dataset, len(img_paths), buffer_limit=1024, seed=seed)

    def _load_pair(image_path: tf.Tensor, mask_path: tf.Tensor):
        image = _load_path_image_tf(image_path, img_size=img_size)
        mask = _load_path_image_tf(mask_path, img_size=img_size, is_mask=True)
        return image, mask

    dataset = dataset.map(_load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    return _finalize_dataset(dataset, batch_size=batch_size)


def build_gan_dataset(
    images: np.ndarray,
    labels: Sequence[int] | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int | None = None,
) -> tf.data.Dataset:
    """Build a GAN dataset from in-memory images.

    Assumes [0, 1] inputs and maps them to the [-1, 1] range expected by the
    tanh-saturated GAN discriminators. The previous implementation inferred the
    source range from ``images.min()``, which silently mis-scaled batches
    containing no negative values and disagreed with the path-based sibling
    :func:`build_gan_dataset_from_paths`.
    """
    images = _to_minus_one_one(images)
    if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(images)
        if shuffle:
            dataset = _shuffle(dataset, len(images), seed=seed)
        return _finalize_dataset(dataset, batch_size=batch_size)

    label_vectors = tf.one_hot(np.asarray(labels, dtype=np.int32), NUM_CLASSES, dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((images, label_vectors))
    if shuffle:
        dataset = _shuffle(dataset, len(images), seed=seed)
    return _finalize_dataset(dataset, batch_size=batch_size)


def build_gan_dataset_from_paths(
    paths: Sequence[str | os.PathLike[str]],
    labels: Sequence[int] | None = None,
    img_size: tuple[int, int] = (128, 128),
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int | None = None,
) -> tf.data.Dataset:
    paths = list(paths)
    if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(paths)
        if shuffle:
            dataset = _shuffle(dataset, len(paths), seed=seed)
        dataset = dataset.map(
            lambda path: _load_path_image_tf(path, img_size=img_size, normalize="minus_one_one"),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return _finalize_dataset(dataset, batch_size=batch_size)

    labels = np.asarray(labels, dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = _shuffle(dataset, len(labels), seed=seed)
    dataset = dataset.map(
        lambda path, label: (
            _load_path_image_tf(path, img_size=img_size, normalize="minus_one_one"),
            tf.one_hot(label, NUM_CLASSES, dtype=tf.float32),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return _finalize_dataset(dataset, batch_size=batch_size)


def _to_minus_one_one(images: np.ndarray) -> np.ndarray:
    """Map a [0, 1] image batch into [-1, 1] for GAN training."""
    images = np.asarray(images, dtype=np.float32)
    if images.size == 0:
        return images
    return images * 2.0 - 1.0


def mix_real_synthetic(
    real_images: np.ndarray,
    real_labels: Sequence[int],
    synthetic_images: np.ndarray,
    synthetic_labels: Sequence[int],
    ratio: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge real and synthetic samples while keeping the ratio bounded.

    ``ratio`` is the fraction of *real* samples that synthetic data may
    contribute. ``ratio=0`` returns the real set untouched.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be within [0, 1], got {ratio}.")
    if len(synthetic_images) != len(synthetic_labels):
        raise ValueError(
            f"synthetic_images ({len(synthetic_images)}) and synthetic_labels "
            f"({len(synthetic_labels)}) must have the same length."
        )
    if len(real_images) != len(real_labels):
        raise ValueError(
            f"real_images ({len(real_images)}) and real_labels ({len(real_labels)}) "
            "must have the same length."
        )

    rng = np.random.default_rng(seed)
    max_synth = min(len(synthetic_images), int(len(real_images) * ratio))
    if max_synth < len(synthetic_images):
        selected = rng.choice(len(synthetic_images), size=max_synth, replace=False)
        synthetic_images = np.asarray(synthetic_images)[selected]
        synthetic_labels = np.asarray(synthetic_labels)[selected]
    mixed_images = np.concatenate([real_images, synthetic_images], axis=0)
    mixed_labels = np.concatenate([np.asarray(real_labels), np.asarray(synthetic_labels)], axis=0)
    order = rng.permutation(len(mixed_images))
    return mixed_images[order], mixed_labels[order]


def build_detection_dataset(
    images: np.ndarray,
    labels: Sequence[int],
    batch_size: int,
    shuffle: bool = True,
    augment: bool = True,
    seed: int | None = None,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((images.astype(np.float32), np.asarray(labels,
        dtype=np.float32)))
    if shuffle:
        dataset = _shuffle(dataset, len(images), seed=seed)
    if augment:
        dataset = dataset.map(lambda x, y: (augment_image(x), y),
            num_parallel_calls=tf.data.AUTOTUNE)
    return _finalize_dataset(dataset, batch_size=batch_size)


def build_detection_dataset_from_paths(
    paths: Sequence[str | os.PathLike[str]],
    labels: Sequence[int],
    img_size: tuple[int, int],
    batch_size: int,
    shuffle: bool = True,
    augment: bool = True,
    seed: int | None = None,
) -> tf.data.Dataset:
    labels = np.asarray(labels, dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((list(paths), labels))
    if shuffle:
        dataset = _shuffle(dataset, len(labels), seed=seed)
    dataset = dataset.map(
        lambda path, label: (_load_path_image_tf(path, img_size=img_size), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if augment:
        dataset = dataset.map(lambda x, y: (augment_image(x), y),
            num_parallel_calls=tf.data.AUTOTUNE)
    return _finalize_dataset(dataset, batch_size=batch_size)


def build_classifier_dataset(
    images: np.ndarray,
    labels: Sequence[int],
    batch_size: int,
    shuffle: bool = True,
    augment: bool = True,
    seed: int | None = None,
) -> tf.data.Dataset:
    labels = tf.one_hot(np.asarray(labels, dtype=np.int32), NUM_CLASSES,
        dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((images.astype(np.float32), labels))
    if shuffle:
        dataset = _shuffle(dataset, len(images), seed=seed)
    if augment:
        dataset = dataset.map(lambda x, y: (augment_image(x), y),
            num_parallel_calls=tf.data.AUTOTUNE)
    return _finalize_dataset(dataset, batch_size=batch_size)


def build_classifier_dataset_from_paths(
    paths: Sequence[str | os.PathLike[str]],
    labels: Sequence[int],
    img_size: tuple[int, int],
    batch_size: int,
    shuffle: bool = True,
    augment: bool = True,
    seed: int | None = None,
) -> tf.data.Dataset:
    labels = np.asarray(labels, dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((list(paths), labels))
    if shuffle:
        dataset = _shuffle(dataset, len(labels), seed=seed)
    dataset = dataset.map(
        lambda path, label: (
            _load_path_image_tf(path,
                img_size=img_size),
            tf.one_hot(label, NUM_CLASSES, dtype=tf.float32),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if augment:
        dataset = dataset.map(lambda x, y: (augment_image(x), y),
            num_parallel_calls=tf.data.AUTOTUNE)
    return _finalize_dataset(dataset, batch_size=batch_size)


def load_figshare_dataset(
    data_dir: str | os.PathLike[str] | None = None,
    img_size: tuple[int, int] = (224, 224),
) -> tuple[np.ndarray, np.ndarray]:
    """Load the full classification dataset into memory.

    .. warning::
        This function pools all splits (train/val/test) into a single
        array. If the dataset has official splits, they are destroyed.
        Consider using ``get_figshare_train_val_test_split`` or
        ``get_figshare_patient_level_split`` instead.
    """
    index = get_figshare_file_index(data_dir)
    paths: list[str] = []
    labels: list[int] = []
    for split_name in index:
        split_paths, split_labels = _flatten_split(index, split_name)
        paths.extend(split_paths.tolist())
        labels.extend(split_labels.tolist())
    logger.warning(
        "load_figshare_dataset() pools all splits into one array. "
        "Official train/test splits are destroyed. "
        "Use get_figshare_train_val_test_split() or get_figshare_patient_level_split() instead."
    )
    return load_images_from_paths(paths, img_size=img_size), np.asarray(labels, dtype=np.int32)


def split_data(
    images: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray,
    np.ndarray]]:
    """Split arrays into train/val/test partitions."""
    train_val_images, test_images, train_val_labels, test_labels = train_test_split(
        images,
        labels,
        test_size=0.15,
        random_state=seed,
        shuffle=True,
        stratify=labels if len(np.unique(labels)) > 1 else None,
    )
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_val_images,
        train_val_labels,
        test_size=0.17647058823529413,
        random_state=seed,
        shuffle=True,
        stratify=train_val_labels if len(np.unique(train_val_labels)) > 1 else None,
    )
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)
