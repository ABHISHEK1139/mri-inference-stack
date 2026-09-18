"""Portable dataset discovery, loading, and tf.data builders."""

from __future__ import annotations

import os
import re
import urllib.request
import zipfile
import logging
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
from PIL import Image
try:
    import tensorflow as tf
except (ModuleNotFoundError, ImportError):
    tf = None  # type: ignore
from sklearn.model_selection import train_test_split

from config import CLASS_NAMES, DATASET_CONFIG, NUM_CLASSES, RAW_DIR

logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "test": "test",
    "testing": "test",
    "val": "val",
    "valid": "val",
    "validation": "val",
}
CLASS_ALIASES = {
    "glioma": "glioma",
    "gliomatumor": "glioma",
    "meningioma": "meningioma",
    "meningiomatumor": "meningioma",
    "pituitary": "pituitary",
    "pituitarytumor": "pituitary",
    "normal": "normal",
    "other": "normal",
    "notumor": "normal",
    "notumour": "normal",
    "no_tumor": "normal",
    "no_tumour": "normal",
    "healthy": "normal",
}
MASK_HINTS = ("mask", "seg", "label", "tumor", "tumour", "annotation")


def _normalize_token(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _extract_patient_id(path: str) -> str:
    """Extract a patient identifier from a file path.

    Strips trailing slice/frame indicators (e.g., _slice01, _001, _s3)
    to group slices from the same patient. Falls back to the full stem
    when no pattern is detected.
    """
    stem = Path(path).stem.lower()
    # Strip common slice/frame suffixes: _slice01, _s01, _001, _frame3, etc.
    patient_id = re.sub(r"[_\-]?(slice|frame|s|img|image)?[_\-]?\d+$", "", stem)
    # If stripping removed everything, fall back to full stem
    return patient_id if patient_id else stem


def _canonical_split(part: str) -> str | None:
    return SPLIT_ALIASES.get(_normalize_token(part))


def _canonical_class(part: str) -> str | None:
    """Map a path component to a canonical class name via exact alias match."""
    token = _normalize_token(part)
    if token in CLASS_ALIASES:
        return CLASS_ALIASES[token]
    # Fuzzy substring matching removed — it could silently assign wrong labels
    # (e.g., 'normalized' matching 'normal'). Unknown tokens return None.
    return None


def _as_path(path_like: str | os.PathLike[str] | None, default_name: str) -> Path:
    if path_like:
        return Path(path_like)
    return Path(RAW_DIR) / default_name


def _iter_image_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _load_grayscale_array(
    path: str | os.PathLike[str],
    img_size: tuple[int, int],
    normalize: str = "zero_one",
    is_mask: bool = False,
) -> np.ndarray:
    image = Image.open(path).convert("L")
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    image = image.resize((img_size[1], img_size[0]), resample=resample)
    array = np.asarray(image, dtype=np.float32)
    if is_mask:
        array = (array > 0).astype(np.float32)
    else:
        if normalize == "minus_one_one":
            array = array / 127.5 - 1.0
        else:
            array = array / 255.0
    return np.expand_dims(array, axis=-1)


def _load_image_from_bytes(
    path_bytes: bytes,
    img_size: tuple[int, int],
    normalize: str = "zero_one",
    is_mask: bool = False,
) -> np.ndarray:
    path = path_bytes.decode("utf-8")
    return _load_grayscale_array(path, img_size=img_size, normalize=normalize, is_mask=is_mask)


def _load_path_image_tf(
    path_tensor: tf.Tensor,
    img_size: tuple[int, int],
    normalize: str = "zero_one",
    is_mask: bool = False,
) -> tf.Tensor:
    image = tf.py_function(
        lambda value: _load_image_from_bytes(value, img_size=img_size, normalize=normalize, is_mask=is_mask),
        [path_tensor],
        Tout=tf.float32,
    )
    image.set_shape((img_size[0], img_size[1], 1))
    return image


def augment_image(image: tf.Tensor, mask: tf.Tensor | None = None):
    """Apply lightweight augmentations compatible with grayscale medical images."""
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        if mask is not None:
            mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        if mask is not None:
            mask = tf.image.flip_up_down(mask)

    rotations = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, rotations)
    if mask is not None:
        mask = tf.image.rot90(mask, rotations)

    if mask is None:
        image = tf.image.random_brightness(image, max_delta=0.05)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image
    return image, mask


def _finalize_dataset(dataset: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _prepare_index(root: Path) -> dict[str, dict[str, list[str]]]:
    index = {split: {name: [] for name in CLASS_NAMES} for split in ("train", "val", "test", "unsplit")}

    for path in _iter_image_files(root):
        relative_parts = path.relative_to(root).parts
        split_name = None
        class_name = None
        for part in relative_parts[:-1]:
            split_name = split_name or _canonical_split(part)
            class_name = class_name or _canonical_class(part)
        if class_name is None:
            class_name = _canonical_class(path.stem)
        if class_name is None:
            continue
        split_name = split_name or "unsplit"
        index[split_name][class_name].append(str(path))

    cleaned = {}
    for split_name, class_map in index.items():
        if any(class_map.values()):
            cleaned[split_name] = {
                class_name: sorted(paths)
                for class_name, paths in class_map.items()
                if paths
            }
    return cleaned


def _flatten_split(index: dict[str, dict[str, list[str]]], split_name: str) -> tuple[np.ndarray, np.ndarray]:
    paths: list[str] = []
    labels: list[int] = []
    for class_name in CLASS_NAMES:
        class_paths = index.get(split_name, {}).get(class_name, [])
        paths.extend(class_paths)
        labels.extend([CLASS_TO_INDEX[class_name]] * len(class_paths))
    return np.asarray(paths, dtype=object), np.asarray(labels, dtype=np.int32)


def _summarize_split(index: dict[str, dict[str, list[str]]], split_name: str) -> None:
    if split_name not in index:
        return
    for class_name in CLASS_NAMES:
        class_paths = index[split_name].get(class_name, [])
        if class_paths:
            parents = {str(Path(path).parent) for path in class_paths}
            print(f"  {class_name} [{split_name}]: {len(class_paths)} images from {len(parents)} folder(s)")


def _stratified_split(
    paths: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        paths,
        labels,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=labels if len(np.unique(labels)) > 1 else None,
    )


def download_dataset(name: str) -> Path:
    """Download and extract a supported dataset into data/raw."""
    if name not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset name: {name}")

    config = DATASET_CONFIG[name]
    target_dir = Path(RAW_DIR) / name
    target_dir.mkdir(parents=True, exist_ok=True)

    if name == "figshare":
        archive_path = target_dir / "download.zip"
        print(f"Downloading Figshare archive to {archive_path}...")
        urllib.request.urlretrieve(config["url"], archive_path)
        try:
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(target_dir)
        finally:
            archive_path.unlink(missing_ok=True)
        return target_dir

    kaggle_url = config["url"]
    dataset_slug = kaggle_url.split("/datasets/")[-1].strip("/")
    if not dataset_slug or dataset_slug == kaggle_url:
        raise RuntimeError(f"Could not derive Kaggle dataset slug from {kaggle_url}")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError("Kaggle API is required to download the BraTS dataset.") from exc

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_slug, path=str(target_dir), unzip=True)
    for zip_path in target_dir.glob("*.zip"):
        zip_path.unlink(missing_ok=True)
    return target_dir


def get_figshare_file_index(data_dir: str | os.PathLike[str] | None = None) -> dict[str, dict[str, list[str]]]:
    """Index the MRI classification dataset by split and class."""
    root = _as_path(data_dir, "figshare")
    index = _prepare_index(root)
    if not index:
        raise FileNotFoundError(
            f"No supported MRI images were found under {root}. "
            "Expected folders named after the tumour classes."
        )
    return index


def get_figshare_train_val_test_split(
    data_dir: str | os.PathLike[str] | None = None,
    seed: int = 42,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return stratified train/val/test path splits for the classification dataset."""
    index = get_figshare_file_index(data_dir)

    if "train" in index and "test" in index:
        train_paths, train_labels = _flatten_split(index, "train")
        test_paths, test_labels = _flatten_split(index, "test")
        if "val" in index:
            val_paths, val_labels = _flatten_split(index, "val")
        else:
            train_paths, val_paths, train_labels, val_labels = _stratified_split(
                train_paths, train_labels, test_size=0.15, seed=seed
            )
    else:
        available_splits = [name for name in ("train", "val", "test", "unsplit") if name in index]
        combined_paths = np.concatenate([_flatten_split(index, split_name)[0] for split_name in available_splits])
        combined_labels = np.concatenate([_flatten_split(index, split_name)[1] for split_name in available_splits])
        train_val_paths, test_paths, train_val_labels, test_labels = _stratified_split(
            combined_paths, combined_labels, test_size=0.15, seed=seed
        )
        train_paths, val_paths, train_labels, val_labels = _stratified_split(
            train_val_paths,
            train_val_labels,
            test_size=0.17647058823529413,
            seed=seed,
        )

    explicit_index = {
        "train": {class_name: list(train_paths[train_labels == CLASS_TO_INDEX[class_name]]) for class_name in CLASS_NAMES},
        "val": {class_name: list(val_paths[val_labels == CLASS_TO_INDEX[class_name]]) for class_name in CLASS_NAMES},
        "test": {class_name: list(test_paths[test_labels == CLASS_TO_INDEX[class_name]]) for class_name in CLASS_NAMES},
    }
    for split_name in ("train", "val", "test"):
        _summarize_split(explicit_index, split_name)
    print(
        "Using Figshare split: "
        f"Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}"
    )
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def get_figshare_patient_level_split(
    data_dir: str | os.PathLike[str] | None = None,
    seed: int = 42,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return patient-level train/val/test splits for the classification dataset.

    Unlike image-level splitting, this ensures all slices from the same
    patient end up in the same partition, preventing data leakage.
    """
    from sklearn.model_selection import GroupShuffleSplit

    index = get_figshare_file_index(data_dir)

    # Pool all available paths
    available_splits = [name for name in ("train", "val", "test", "unsplit") if name in index]
    all_paths = np.concatenate([_flatten_split(index, s)[0] for s in available_splits])
    all_labels = np.concatenate([_flatten_split(index, s)[1] for s in available_splits])

    # Extract patient IDs for grouping
    groups = np.array([_extract_patient_id(p) for p in all_paths])
    n_unique = len(set(groups))
    print(f"Patient-level split: {n_unique} unique patient IDs from {len(all_paths)} images")

    # First split: train+val vs test (15% test)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(all_paths, all_labels, groups))

    # Second split: train vs val (~17.6% of train+val = ~15% of total)
    tv_paths, tv_labels, tv_groups = all_paths[train_val_idx], all_labels[train_val_idx], groups[train_val_idx]
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.176, random_state=seed)
    train_idx, val_idx = next(gss_val.split(tv_paths, tv_labels, tv_groups))

    train_paths, train_labels = tv_paths[train_idx], tv_labels[train_idx]
    val_paths, val_labels = tv_paths[val_idx], tv_labels[val_idx]
    test_paths, test_labels = all_paths[test_idx], all_labels[test_idx]

    print(
        f"Patient-level split: "
        f"Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}"
    )
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def load_images_from_paths(
    paths: Sequence[str | os.PathLike[str]],
    img_size: tuple[int, int],
    normalize: str = "zero_one",
    is_mask: bool = False,
) -> np.ndarray:
    """Load grayscale images into a single float32 array."""
    paths = list(paths)
    if not paths:
        return np.empty((0, img_size[0], img_size[1], 1), dtype=np.float32)
    images = np.empty((len(paths), img_size[0], img_size[1], 1), dtype=np.float32)
    for idx, path in enumerate(paths):
        images[idx] = _load_grayscale_array(path, img_size=img_size, normalize=normalize, is_mask=is_mask)
    return images


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
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
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


def build_detection_dataset(
    images: np.ndarray,
    labels: Sequence[int],
    batch_size: int,
    shuffle: bool = True,
    augment: bool = True,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((images.astype(np.float32), np.asarray(labels, dtype=np.float32)))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(images), 2048), reshuffle_each_iteration=True)
    if augment:
        dataset = dataset.map(lambda x, y: (augment_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return _finalize_dataset(dataset, batch_size=batch_size)


def build_detection_dataset_from_paths(
    paths: Sequence[str | os.PathLike[str]],
    labels: Sequence[int],
    img_size: tuple[int, int],
    batch_size: int,
    shuffle: bool = True,
    augment: bool = True,
) -> tf.data.Dataset:
    labels = np.asarray(labels, dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((list(paths), labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(labels), 2048), reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda path, label: (_load_path_image_tf(path, img_size=img_size), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if augment:
        dataset = dataset.map(lambda x, y: (augment_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return _finalize_dataset(dataset, batch_size=batch_size)


def build_classifier_dataset(
    images: np.ndarray,
    labels: Sequence[int],
    batch_size: int,
    shuffle: bool = True,
    augment: bool = True,
) -> tf.data.Dataset:
    labels = tf.one_hot(np.asarray(labels, dtype=np.int32), NUM_CLASSES, dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((images.astype(np.float32), labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(images), 2048), reshuffle_each_iteration=True)
    if augment:
        dataset = dataset.map(lambda x, y: (augment_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return _finalize_dataset(dataset, batch_size=batch_size)


def build_classifier_dataset_from_paths(
    paths: Sequence[str | os.PathLike[str]],
    labels: Sequence[int],
    img_size: tuple[int, int],
    batch_size: int,
    shuffle: bool = True,
    augment: bool = True,
) -> tf.data.Dataset:
    labels = np.asarray(labels, dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((list(paths), labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(labels), 2048), reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda path, label: (
            _load_path_image_tf(path, img_size=img_size),
            tf.one_hot(label, NUM_CLASSES, dtype=tf.float32),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if augment:
        dataset = dataset.map(lambda x, y: (augment_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return _finalize_dataset(dataset, batch_size=batch_size)


def _pair_brats_images_and_masks(root: Path) -> tuple[list[str], list[str]]:
    image_candidates: dict[str, str] = {}
    mask_candidates: dict[str, str] = {}

    for path in _iter_image_files(root):
        stem = _normalize_token(path.stem)
        if not stem:
            continue
        is_mask = any(hint in stem for hint in MASK_HINTS)
        canonical_stem = stem
        for hint in MASK_HINTS:
            canonical_stem = canonical_stem.replace(hint, "")
        if is_mask:
            if canonical_stem in mask_candidates:
                logger.warning(
                    "Duplicate mask stem %r: %s will overwrite %s",
                    canonical_stem, path, mask_candidates[canonical_stem],
                )
            mask_candidates[canonical_stem] = str(path)
        else:
            if canonical_stem in image_candidates:
                logger.warning(
                    "Duplicate image stem %r: %s will overwrite %s",
                    canonical_stem, path, image_candidates[canonical_stem],
                )
            image_candidates[canonical_stem] = str(path)

    matched_keys = sorted(set(image_candidates) & set(mask_candidates))
    image_paths = [image_candidates[key] for key in matched_keys]
    mask_paths = [mask_candidates[key] for key in matched_keys]
    return image_paths, mask_paths


def load_brats_paths(
    data_dir: str | os.PathLike[str] | None = None,
    img_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired image and mask paths for the segmentation dataset."""
    root = _as_path(data_dir, "brats")
    image_paths, mask_paths = _pair_brats_images_and_masks(root)
    if not image_paths:
        raise FileNotFoundError(
            f"No paired BraTS image/mask files were found under {root}. "
            "Expected filenames containing mask or seg for the label maps."
        )
    return np.asarray(image_paths, dtype=object), np.asarray(mask_paths, dtype=object)


def load_brats_dataset(
    data_dir: str | os.PathLike[str] | None = None,
    img_size: tuple[int, int] = (256, 256),
) -> tuple[np.ndarray, np.ndarray]:
    """Load the segmentation dataset into memory."""
    image_paths, mask_paths = load_brats_paths(data_dir, img_size=img_size)
    images = load_images_from_paths(image_paths, img_size=img_size)
    masks = load_images_from_paths(mask_paths, img_size=img_size, is_mask=True)
    return images, masks


def build_segmentation_dataset(
    images: np.ndarray,
    masks: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    augment: bool = True,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((images.astype(np.float32), masks.astype(np.float32)))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(images), 1024), reshuffle_each_iteration=True)
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
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((list(img_paths), list(mask_paths)))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(img_paths), 1024), reshuffle_each_iteration=True)

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
) -> tf.data.Dataset:
    images = images.astype(np.float32)
    if images.min() >= 0.0:
        images = images * 2.0 - 1.0
    if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(images)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(len(images), 2048), reshuffle_each_iteration=True)
        return _finalize_dataset(dataset, batch_size=batch_size)

    label_vectors = tf.one_hot(np.asarray(labels, dtype=np.int32), NUM_CLASSES, dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((images, label_vectors))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(images), 2048), reshuffle_each_iteration=True)
    return _finalize_dataset(dataset, batch_size=batch_size)


def build_gan_dataset_from_paths(
    paths: Sequence[str | os.PathLike[str]],
    labels: Sequence[int] | None = None,
    img_size: tuple[int, int] = (128, 128),
    batch_size: int = 32,
    shuffle: bool = True,
) -> tf.data.Dataset:
    paths = list(paths)
    if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(paths)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(len(paths), 2048), reshuffle_each_iteration=True)
        dataset = dataset.map(
            lambda path: _load_path_image_tf(path, img_size=img_size, normalize="minus_one_one"),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return _finalize_dataset(dataset, batch_size=batch_size)

    labels = np.asarray(labels, dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(labels), 2048), reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda path, label: (
            _load_path_image_tf(path, img_size=img_size, normalize="minus_one_one"),
            tf.one_hot(label, NUM_CLASSES, dtype=tf.float32),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return _finalize_dataset(dataset, batch_size=batch_size)


def mix_real_synthetic(
    real_images: np.ndarray,
    real_labels: Sequence[int],
    synthetic_images: np.ndarray,
    synthetic_labels: Sequence[int],
    ratio: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge real and synthetic samples while keeping the ratio bounded."""
    rng = np.random.default_rng(seed)
    max_synth = min(len(synthetic_images), max(1, int(len(real_images) * ratio)))
    if max_synth < len(synthetic_images):
        selected = rng.choice(len(synthetic_images), size=max_synth, replace=False)
        synthetic_images = synthetic_images[selected]
        synthetic_labels = np.asarray(synthetic_labels)[selected]
    mixed_images = np.concatenate([real_images, synthetic_images], axis=0)
    mixed_labels = np.concatenate([np.asarray(real_labels), np.asarray(synthetic_labels)], axis=0)
    order = rng.permutation(len(mixed_images))
    return mixed_images[order], mixed_labels[order]

