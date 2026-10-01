"""Image decoding, augmentation, and the tf.data pipeline primitives."""

import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from data.dataset.naming import IMAGE_EXTENSIONS, VOLUME_EXTENSIONS


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


def _shuffle(dataset: tf.data.Dataset, size: int, buffer_limit: int = 2048,
            seed: int | None = None) -> tf.data.Dataset:
    """Shuffle with a valid buffer size.

    ``tf.data`` rejects ``buffer_size < 1`` with a ``ValueError``, which made
    every ``shuffle=True`` builder fail outright on an empty split instead of
    raising a meaningful error.

    ``seed`` makes the shuffle order reproducible for a given TensorFlow version
    and hardware; pass the run seed from :mod:`training.reproducibility`.
    """
    if size <= 0:
        raise ValueError(
            "Cannot build a tf.data.Dataset from an empty split. "
            "Check that the source dataset actually contains images for this partition."
        )
    return dataset.shuffle(
        buffer_size=max(1, min(size, buffer_limit)),
        reshuffle_each_iteration=True,
        seed=seed,
    )


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
