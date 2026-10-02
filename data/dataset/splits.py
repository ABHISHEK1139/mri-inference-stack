"""Dataset discovery, download, and leakage-free train/val/test splitting."""

import logging
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from config import CLASS_NAMES, DATASET_CONFIG, RAW_DIR, dataset_url
from data.dataset.loading import _iter_image_files
from data.dataset.naming import (
    CLASS_TO_INDEX,
    _as_path,
    _canonical_class,
    _canonical_split,
    _extract_patient_id,
)

logger = logging.getLogger(__name__)


def _prepare_index(root: Path) -> dict[str, dict[str,
    list[str]]]:
    index: dict[str, dict[str, list[str]]] = {
        split: {name: [] for name in CLASS_NAMES}
        for split in ("train", "val", "test", "unsplit")
    }

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

    url = dataset_url(name)
    target_dir = Path(RAW_DIR) / name
    target_dir.mkdir(parents=True, exist_ok=True)

    if name == "figshare":
        archive_path = target_dir / "download.zip"
        print(f"Downloading Figshare archive to {archive_path}...")
        urllib.request.urlretrieve(url, archive_path)
        try:
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(target_dir)
        finally:
            archive_path.unlink(missing_ok=True)
        return target_dir

    kaggle_url = url
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
