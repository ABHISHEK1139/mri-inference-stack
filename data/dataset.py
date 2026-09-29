"""Portable dataset discovery, loading, and tf.data builders."""

from __future__ import annotations

import logging
import os
import re
import urllib.request
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path

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
VOLUME_EXTENSIONS = {".nii", ".nii.gz"}
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

# Suffixes that mark a file as a *label map*. These are only ever matched as a
# trailing token. "tumor"/"tumour" are deliberately absent: BraTS image volumes
# are named `..._brain_tumor_FLAIR` and masks `..._brain_tumor_seg`, so
# substring-matching on "tumor" classified BOTH sides as masks and produced zero
# image/mask pairs.
MASK_SUFFIXES = ("seg", "segmentation", "mask", "label", "labelmap", "gt", "groundtruth",
    "annotation")
# Every entry is already lowercase alphanumeric, so it is directly comparable
# against a token from _TOKEN_SPLIT_RE.
MASK_SUFFIX_TOKENS = frozenset(MASK_SUFFIXES)

# Modality / tissue tokens that may trail a BraTS volume name.
VOLUME_MODALITIES = frozenset(
    {"flair", "t1ce", "t1gd", "t2", "t1", "wt", "et", "tc", "ncr"}
)

# Tokens that mark a slice/frame counter when they trail a filename. A counter
# is stripped only when it is preceded by a separator, so an undelimited trailing
# number such as the "003" in "image003" is preserved: for a per-volume dataset
# that number is the subject ID, and stripping it would collapse every subject
# into a single group, making a patient-level split impossible.
_COUNTER_KEYWORDS = frozenset({"slice", "frame", "img", "image", "fl", "sl"})
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_TRAILING_DIGITS_RE = re.compile(r"^(.*?)(\d+)$")


def _normalize_token(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _has_mask_suffix(stem: str) -> bool:
    """Return True when the final token of ``stem`` is a label-map token."""
    tokens = [t for t in _TOKEN_SPLIT_RE.split(Path(stem).stem.lower()) if t]
    return len(tokens) > 1 and tokens[-1] in MASK_SUFFIX_TOKENS


def _extract_patient_id(path: str) -> str:
    """Extract a patient identifier from a file path.

    Groups slices/volumes belonging to the same subject so that patient-level
    splitting cannot leak data across partitions. The stem is tokenised on
    non-alphanumeric characters and, in order: a trailing label-map token is
    dropped, a trailing modality token is dropped, and a single trailing slice
    counter is dropped. Examples::

        BraTS20_Training_001_brain_tumor_FLAIR -> brats20training001braintumor
        BraTS20_Training_001_brain_tumor_seg    -> brats20training001braintumor
        patient001_slice01.png                   -> patient001
        patient_a_003.png                        -> patienta
        scan_frame3.png                          -> scan
        glioma (12).png                          -> glioma
        image003.png                             -> image003   (subject ID)

    The counter step runs at most once, so ``Case-045_s3`` yields ``case_045_s3``
    rather than losing the patient number. Tokens are rejoined with ``_``, so
    separator style never splits one patient into two groups
    (``Case-045_s3`` and ``Case_045_s3`` agree). A counter is never stripped when
    it would empty the key, so distinct subjects cannot collapse into one group.
    """
    raw = Path(path).stem.lower()
    tokens = [t for t in _TOKEN_SPLIT_RE.split(raw) if t]

    if not tokens:
        return _normalize_token(raw) or raw

    # 1. Trailing label-map token ("seg", "mask", ...).
    if len(tokens) > 1 and tokens[-1] in MASK_SUFFIX_TOKENS:
        tokens.pop()
    # 2. Trailing modality token ("flair", "t1ce", ...).
    if len(tokens) > 1 and tokens[-1] in VOLUME_MODALITIES:
        tokens.pop()
    # 3. A single trailing slice counter ("_003", "_slice01", "frame3").
    if len(tokens) > 1:
        match = _TRAILING_DIGITS_RE.match(tokens[-1])
        if match is not None:
            base, digits = match.groups()
            # "_003" -> bare number; "_slice01" / "frame3" -> keyword + number.
            if digits and (not base or base in _COUNTER_KEYWORDS):
                tokens.pop()

    return "_".join(tokens) or _normalize_token(raw) or raw


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
    """Yield every supported image/volume file under ``root``.

    BraTS ships ``.nii`` / ``.nii.gz`` volumes, so those are discovered here as
    well as regular raster formats. Note that ``Path.suffix`` reports only
    ``.gz`` for a ``.nii.gz`` file, so the double extension is checked first.
    """
    if not root.exists():
        return []
    return (
        path
        for path in root.rglob("*")
        if path.is_file() and _media_suffix(path) in IMAGE_EXTENSIONS | VOLUME_EXTENSIONS
    )


def _media_suffix(path: Path) -> str:
    """Return the lowercase suffix of ``path``, treating ``.nii.gz`` as one unit."""
    lowered = path.name.lower()
    if lowered.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def _load_volume_array(path: str | os.PathLike[str]) -> np.ndarray:
    """Load a NIfTI volume as a 2D uint8 grayscale array.

    The volume is max-projected over the leading axes until a single 2D slice
    remains, then windowed to 8-bit. WINDOW_OVERRIDE-style percentile windowing
    keeps the result visually stable across BraTS intensities.
    """
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Reading BraTS NIfTI volumes requires the optional 'nibabel' dependency. "
            "Install it with `pip install nibabel` or `pip install .[segmentation]`."
        ) from exc

    volume = np.asanyarray(nib.load(str(path)).dataobj, dtype=np.float32)
    volume = np.squeeze(volume)
    if volume.ndim > 2:
        # Average-project the leading (slice) axes down to a single 2D frame.
        volume = volume.mean(axis=tuple(range(volume.ndim - 2)))
    elif volume.ndim != 2:
        raise ValueError(f"Volume {path} has unsupported shape {volume.shape}.")
    if not np.isfinite(volume).all():
        volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)

    lo, hi = np.percentile(volume, (0.5, 99.5))
    if hi <= lo:
        return np.zeros(volume.shape, dtype=np.uint8)
    scaled = (volume - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _load_grayscale_array(
    path: str | os.PathLike[str],
    img_size: tuple[int, int],
    normalize: str = "zero_one",
    is_mask: bool = False,
) -> np.ndarray:
    if _media_suffix(Path(path)) in VOLUME_EXTENSIONS:
        image = Image.fromarray(_load_volume_array(path), mode="L")
    else:
        image = Image.open(path).convert("L")
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
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
    path_value,
    img_size: tuple[int, int],
    normalize: str = "zero_one",
    is_mask: bool = False,
) -> np.ndarray:
    """Load an image given a path supplied either as bytes or as a tensor.

    ``tf.py_function`` hands the callback an ``EagerTensor``, never raw ``bytes``,
    so calling ``.decode()`` on it directly raised
    ``'EagerTensor' object has no attribute 'decode'`` and broke every
    path-based dataset builder (all four training tracks). The tensor is unwrapped
    here and the failing path is reported in the exception.
    """
    if hasattr(path_value, "numpy"):
        path_value = path_value.numpy()
    if isinstance(path_value, (bytes, bytearray)):
        path = path_value.decode("utf-8")
    else:
        path = str(path_value)

    try:
        return _load_grayscale_array(
            path, img_size=img_size, normalize=normalize, is_mask=is_mask
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load image {path!r}: {exc}") from exc


def _load_path_image_tf(
    path_tensor: tf.Tensor,
    img_size: tuple[int, int],
    normalize: str = "zero_one",
    is_mask: bool = False,
) -> tf.Tensor:
    image = tf.py_function(
        lambda value: _load_image_from_bytes(value, img_size=img_size, normalize=normalize,
            is_mask=is_mask),
        [path_tensor],
        Tout=tf.float32,
    )
    image.set_shape((img_size[0], img_size[1], 1))
    return image


def _shuffle(dataset: tf.data.Dataset, size: int, buffer_limit: int = 2048) -> tf.data.Dataset:
    """Shuffle with a valid buffer size.

    ``tf.data`` rejects ``buffer_size < 1`` with a ``ValueError``, which made
    every ``shuffle=True`` builder fail outright on an empty split instead of
    raising a meaningful error.
    """
    if size <= 0:
        raise ValueError(
            "Cannot build a tf.data.Dataset from an empty split. "
            "Check that the source dataset actually contains images for this partition."
        )
    return dataset.shuffle(buffer_size=max(1, min(size, buffer_limit)),
        reshuffle_each_iteration=True)


def augment_image(image: tf.Tensor, mask: tf.Tensor | None = None):
    """Apply lightweight augmentations compatible with grayscale medical images.

    The stochastic branches use ``tf.cond`` rather than a Python ``if``. A
    Python ``if`` on ``tf.random.uniform(()) > 0.5`` works in eager mode but
    raises "Using a symbolic tf.Tensor as a Python bool is not allowed" as soon
    as the function is traced by ``tf.data.Dataset.map``, which is exactly how
    every training builder calls it.

    The same flip decisions and rotation index are applied to the image and the
    mask so that augmentation cannot misalign segmentation pairs.
    """
    flip_lr = tf.random.uniform(()) > 0.5
    flip_ud = tf.random.uniform(()) > 0.5
    rotation_k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)

    image = tf.cond(flip_lr, lambda: tf.image.flip_left_right(image), lambda: image)
    image = tf.cond(flip_ud, lambda: tf.image.flip_up_down(image), lambda: image)
    image = tf.image.rot90(image, rotation_k)

    if mask is None:
        # Inputs are in [0, 1]; only jitter in the image branch so the mask
        # branch keeps its exact label values for the loss.
        image = tf.image.random_brightness(image, max_delta=0.05)
        return tf.clip_by_value(image, 0.0, 1.0)

    mask = tf.cond(flip_lr, lambda: tf.image.flip_left_right(mask), lambda: mask)
    mask = tf.cond(flip_ud, lambda: tf.image.flip_up_down(mask), lambda: mask)
    mask = tf.image.rot90(mask, rotation_k)
    return image, mask


def _finalize_dataset(dataset: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _prepare_index(root: Path) -> dict[str, dict[str,
    list[str]]]:
    index = {split: {name: [] for name in CLASS_NAMES} for split in ("train", "val", "test",
        "unsplit")}

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
        # Also honour a split marker embedded in the filename itself, e.g.
        # "train_glioma_001.png" in a flat directory.
        if split_name is None:
            split_name = _canonical_split(path.stem)
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


def _flatten_split(index: dict[str, dict[str, list[str]]], split_name: str) -> tuple[np.ndarray,
    np.ndarray]:
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
            print(
                f"  {class_name} [{split_name}]: {len(class_paths)} images "
                f"from {len(parents)} folder(s)"
            )


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


def get_figshare_file_index(data_dir: str | os.PathLike[str] | None = None) -> dict[str, dict[str,
    list[str]]]:
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
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray,
    np.ndarray]]:
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
        # Pool anything that could not be attributed to an official split back
        # into train, rather than silently dropping it from the experiment.
        leftover_paths, leftover_labels = _flatten_split(index, "unsplit")
        if len(leftover_paths):
            logger.info(
                "Pooling %d unassigned image(s) from 'unsplit' into the train partition.",
                len(leftover_paths),
            )
            train_paths = np.concatenate([train_paths, leftover_paths])
            train_labels = np.concatenate([train_labels, leftover_labels])
    else:
        available_splits = [name for name in ("train", "val", "test",
            "unsplit") if name in index]
        combined_paths = np.concatenate([_flatten_split(index,
            split_name)[0] for split_name in available_splits])
        combined_labels = np.concatenate([_flatten_split(index,
            split_name)[1] for split_name in available_splits])
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
        "train": {
            class_name: list(train_paths[train_labels == CLASS_TO_INDEX[class_name]])
            for class_name in CLASS_NAMES
        },
        "val": {
            class_name: list(val_paths[val_labels == CLASS_TO_INDEX[class_name]])
            for class_name in CLASS_NAMES
        },
        "test": {
            class_name: list(test_paths[test_labels == CLASS_TO_INDEX[class_name]])
            for class_name in CLASS_NAMES
        },
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
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray,
    np.ndarray]]:
    """Return patient-level train/val/test splits for the classification dataset.

    Unlike image-level splitting, this ensures all slices from the same
    patient end up in the same partition, preventing data leakage.

    Uses :class:`~sklearn.model_selection.StratifiedGroupKFold` so the class
    balance is preserved *and* patients stay isolated. Plain ``GroupShuffleSplit``
    operates on whole groups only and, with few groups per class, regularly
    produced partitions that were missing an entire class.
    """

    index = get_figshare_file_index(data_dir)

    # Pool all available paths
    available_splits = [name for name in ("train", "val", "test", "unsplit") if name in index]
    all_paths = np.concatenate([_flatten_split(index, s)[0] for s in available_splits])
    all_labels = np.concatenate([_flatten_split(index, s)[1] for s in available_splits])

    # Extract patient IDs for grouping
    groups = np.array([_extract_patient_id(p) for p in all_paths])
    n_unique = len(set(groups))
    print(f"Patient-level split: {n_unique} unique patient IDs from {len(all_paths)} images")

    if n_unique < 2:
        raise ValueError(
            f"Patient-level splitting needs at least 2 distinct patient IDs, found {n_unique}. "
            "Filenames must encode a patient identifier distinct from the slice index."
        )

    # Hold out the test partition, then split the remainder into train/val,
    # keeping each stratum as balanced as the group structure allows.
    train_val_idx, test_idx = _stratified_group_partition(
        all_paths, all_labels, groups, test_size=0.15, seed=seed, n_splits=5
    )
    train_idx, val_idx = _stratified_group_partition(
        all_paths[train_val_idx],
        all_labels[train_val_idx],
        groups[train_val_idx],
        test_size=0.17647058823529413,
        seed=seed,
        n_splits=5,
    )

    train_paths = all_paths[train_val_idx][train_idx]
    train_labels = all_labels[train_val_idx][train_idx]
    val_paths, val_labels = all_paths[train_val_idx][val_idx], all_labels[train_val_idx][val_idx]
    test_paths, test_labels = all_paths[test_idx], all_labels[test_idx]

    _validate_split("train", train_labels)
    _validate_split("val", val_labels)
    _validate_split("test", test_labels)

    print(
        f"Patient-level split: "
        f"Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}"
    )
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def _validate_split(split_name: str, labels: np.ndarray) -> None:
    """Fail loudly when a split is empty or has lost a class."""
    if len(labels) == 0:
        raise ValueError(f"Patient-level split produced an empty '{split_name}' partition.")
    present = set(np.unique(labels).tolist())
    missing = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES)) if i not in present]
    if missing:
        raise ValueError(
            f"Patient-level split lost class(es) {missing} from the '{split_name}' partition. "
            "Increase the number of patients or disable patient-level splitting."
        )


def _stratified_group_partition(
    paths: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    seed: int,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_val_idx, holdout_idx) honouring class balance and groups.

    Falls back to a plain grouped shuffle when stratification is impossible
    (e.g. only one class present), and degrades to a non-overlapping random
    partition when the number of distinct groups cannot support ``n_splits``.
    """
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

    n_groups = len(np.unique(groups))
    n_classes = len(np.unique(labels))

    if n_groups > n_splits and n_classes > 1:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(paths, labels, groups))
        n_holdout = int(round(test_size * n_groups))
        # Take the fold whose holdout group count is closest to the target
        # ratio. `item` is (position, (train_idx, holdout_idx)).
        fold_idx = min(
            enumerate(folds),
            key=lambda item: abs(len(np.unique(groups[item[1][1]])) - n_holdout),
        )[0]
        train_val, holdout = folds[fold_idx]
        if len(train_val) and len(holdout):
            return train_val, holdout

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, holdout_idx = next(gss.split(paths, labels, groups))
    if len(train_val_idx) == 0 or len(holdout_idx) == 0:
        raise ValueError(
            f"Patient-level split failed to produce disjoint non-empty partitions "
            f"from {n_groups} group(s). Reduce the split ratio or disable patient-level splitting."
        )
    return train_val_idx, holdout_idx


def load_images_from_paths(
    paths: Sequence[str | os.PathLike[str]],
    img_size: tuple[int, int],
    normalize: str = "zero_one",
    is_mask: bool = False,
) -> np.ndarray:
    """Load grayscale images into a single float32 array."""
    paths = list(paths)
    if not paths:
        return np.empty((0, img_size[0], img_size[1], 1),
            dtype=np.float32)
    images = np.empty((len(paths), img_size[0], img_size[1], 1),
        dtype=np.float32)
    for idx, path in enumerate(paths):
        images[idx] = _load_grayscale_array(path, img_size=img_size, normalize=normalize,
            is_mask=is_mask)
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


def build_detection_dataset(
    images: np.ndarray,
    labels: Sequence[int],
    batch_size: int,
    shuffle: bool = True,
    augment: bool = True,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((images.astype(np.float32), np.asarray(labels,
        dtype=np.float32)))
    if shuffle:
        dataset = _shuffle(dataset, len(images))
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
) -> tf.data.Dataset:
    labels = np.asarray(labels, dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((list(paths), labels))
    if shuffle:
        dataset = _shuffle(dataset, len(labels))
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
) -> tf.data.Dataset:
    labels = tf.one_hot(np.asarray(labels, dtype=np.int32), NUM_CLASSES,
        dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((images.astype(np.float32), labels))
    if shuffle:
        dataset = _shuffle(dataset, len(images))
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
) -> tf.data.Dataset:
    labels = np.asarray(labels, dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((list(paths), labels))
    if shuffle:
        dataset = _shuffle(dataset, len(labels))
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


def _pair_brats_images_and_masks(root: Path) -> tuple[list[str], list[str]]:
    """Pair BraTS image volumes with their segmentation masks.

    A BraTS subject has several image volumes (FLAIR, T1, T1ce, T2) that all
    share a single ``_brain_tumor_seg`` label map. Pairing on an exact
    canonical stem therefore keeps at most one modality, so pairing is keyed on
    the subject identifier and every image volume of a subject is paired with
    that subject's mask.
    """
    images_by_patient: dict[str, list[str]] = {}
    masks_by_patient: dict[str, str] = {}

    for path in _iter_image_files(root):
        if _has_mask_suffix(path.stem):
            patient = _extract_patient_id(path)
            if patient in masks_by_patient:
                logger.warning(
                    "Duplicate mask for subject %r: %s will overwrite %s",
                    patient,
                    masks_by_patient[patient],
                    path,
                )
            masks_by_patient[patient] = str(path)
        else:
            patient = _extract_patient_id(path)
            if not patient:
                continue
            images_by_patient.setdefault(patient, []).append(str(path))

    image_paths: list[str] = []
    mask_paths: list[str] = []
    for patient in sorted(set(images_by_patient) & set(masks_by_patient)):
        mask = masks_by_patient[patient]
        for image in sorted(images_by_patient[patient]):
            image_paths.append(image)
            mask_paths.append(mask)
    return image_paths, mask_paths


def load_brats_paths(
    data_dir: str | os.PathLike[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired image and mask paths for the segmentation dataset."""
    root = _as_path(data_dir, "brats")
    image_paths, mask_paths = _pair_brats_images_and_masks(root)
    if not image_paths:
        raise FileNotFoundError(
            f"No paired BraTS image/mask files were found under {root}. "
            "Expected BraTS volumes named '..._brain_tumor_<MODALITY>' alongside "
            "'..._brain_tumor_seg' label maps."
        )
    return np.asarray(image_paths, dtype=object), np.asarray(mask_paths, dtype=object)


def load_brats_dataset(
    data_dir: str | os.PathLike[str] | None = None,
    img_size: tuple[int, int] = (256, 256),
) -> tuple[np.ndarray, np.ndarray]:
    """Load the segmentation dataset into memory."""
    image_paths, mask_paths = load_brats_paths(data_dir)
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
    dataset = tf.data.Dataset.from_tensor_slices((images.astype(np.float32),
        masks.astype(np.float32)))
    if shuffle:
        dataset = _shuffle(dataset, len(images), buffer_limit=1024)
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
        dataset = _shuffle(dataset, len(img_paths), buffer_limit=1024)

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
            dataset = _shuffle(dataset, len(images))
        return _finalize_dataset(dataset, batch_size=batch_size)

    label_vectors = tf.one_hot(np.asarray(labels, dtype=np.int32), NUM_CLASSES, dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((images, label_vectors))
    if shuffle:
        dataset = _shuffle(dataset, len(images))
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
            dataset = _shuffle(dataset, len(paths))
        dataset = dataset.map(
            lambda path: _load_path_image_tf(path, img_size=img_size, normalize="minus_one_one"),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return _finalize_dataset(dataset, batch_size=batch_size)

    labels = np.asarray(labels, dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = _shuffle(dataset, len(labels))
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

