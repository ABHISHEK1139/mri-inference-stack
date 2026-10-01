"""Dataset discovery, loading, splitting, and tf.data pipeline construction.

This module is a facade over the :mod:`data.dataset` package. It re-exports
every public name so existing imports (``from data.dataset import
build_classifier_dataset``) keep working, while the implementations live in
focused modules:

===========================  ==========================================
:mod:`data.dataset.naming`    Filename conventions: class and split
                             aliases, patient grouping
:mod:`data.dataset.loading`   Decoding, augmentation, tf.data primitives
:mod:`data.dataset.splits`    Discovery, download, and leakage-free
                             train/val/test partitioning
:mod:`data.dataset.loaders`   In-memory array loaders, BraTS pairing
:mod:`data.dataset.builders`  The per-track tf.data builders and
                             synthetic data mixing
===========================  ==========================================
"""

from data.dataset.builders import (
    build_classifier_dataset,
    build_classifier_dataset_from_paths,
    build_detection_dataset,
    build_detection_dataset_from_paths,
    build_gan_dataset,
    build_gan_dataset_from_paths,
    build_segmentation_dataset,
    build_segmentation_dataset_from_paths,
    load_figshare_dataset,
    mix_real_synthetic,
    split_data,
)
from data.dataset.loaders import (
    _pair_brats_images_and_masks,
    load_brats_dataset,
    load_brats_paths,
    load_images_from_paths,
)
from data.dataset.loading import _load_image_from_bytes, augment_image
from data.dataset.naming import (
    _canonical_class,
    _canonical_split,
    _extract_patient_id,
    _has_mask_suffix,
    _normalize_token,
)
from data.dataset.splits import (
    download_dataset,
    get_figshare_file_index,
    get_figshare_patient_level_split,
    get_figshare_train_val_test_split,
)

__all__ = [
    # Discovery and splits
    "download_dataset",
    "get_figshare_file_index",
    "get_figshare_patient_level_split",
    "get_figshare_train_val_test_split",
    # Loaders
    "load_brats_dataset",
    "load_brats_paths",
    "load_figshare_dataset",
    "load_images_from_paths",
    "split_data",
    # tf.data builders
    "augment_image",
    "build_classifier_dataset",
    "build_classifier_dataset_from_paths",
    "build_detection_dataset",
    "build_detection_dataset_from_paths",
    "build_gan_dataset",
    "build_gan_dataset_from_paths",
    "build_segmentation_dataset",
    "build_segmentation_dataset_from_paths",
    "mix_real_synthetic",
    # Naming helpers (tested directly)
    "_canonical_class",
    "_canonical_split",
    "_extract_patient_id",
    "_has_mask_suffix",
    "_load_image_from_bytes",
    "_normalize_token",
    "_pair_brats_images_and_masks",
]
