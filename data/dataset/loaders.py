"""In-memory array loaders and BraTS image/mask pairing."""

import logging
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from data.dataset.loading import _iter_image_files, _load_grayscale_array
from data.dataset.naming import _as_path, _extract_patient_id, _has_mask_suffix

logger = logging.getLogger(__name__)

# A single filesystem path, in any of the forms callers use.
PathLike = str | os.PathLike[str]
# A collection of paths. NumPy arrays are included because the dataset loaders
# return `np.asarray(...)` results that are then fed straight back into these
# builders; NumPy's stubs do not declare `ndarray` as a `Sequence`, so
# spelling the union out keeps callers honest instead of needing 25 ignores.
PathSequence = Sequence[PathLike] | np.ndarray
# Label columns likewise: tracks hold them in NumPy arrays from the dataset
# loaders, but a plain list of ints is just as valid.
LabelSequence = Sequence[int] | np.ndarray


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


def load_images_from_paths(
    paths: PathSequence,
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
