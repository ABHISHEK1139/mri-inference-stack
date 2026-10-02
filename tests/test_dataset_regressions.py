"""Regression tests for bugs fixed in the dataset loading and splitting layer.

Each test here corresponds to a defect that silently corrupted training data,
evaluation results, or split integrity. The docstring on each test names the
original failure so the reason for the assertion stays discoverable.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from data.dataset import (
    _extract_patient_id,
    _has_mask_suffix,
    _pair_brats_images_and_masks,
    build_classifier_dataset,
    build_detection_dataset,
    build_gan_dataset,
    build_segmentation_dataset,
    get_figshare_patient_level_split,
    load_brats_paths,
    mix_real_synthetic,
)

tf = pytest.importorskip("tensorflow")


# ── BraTS image/mask pairing ─────────────────────────────────────────────

class TestBraTSPairing:
    """Regression: 'tumor' in MASK_HINTS classified both sides as masks.

    BraTS files are named ``..._brain_tumor_FLAIR`` (image) and
    ``..._brain_tumor_seg`` (mask). Substring matching on "tumor" made every
    file look like a mask, so ``_pair_brats_images_and_masks`` always returned
    zero pairs and ``load_brats_paths`` raised FileNotFoundError.
    """

    @staticmethod
    def _make_brats(root, n_patients=3, modalities=("FLAIR", "T1", "T1ce", "T2")):
        rng = np.random.default_rng(0)
        for i in range(n_patients):
            for modality in modalities:
                Image.fromarray(
                    rng.integers(0, 256, (8, 8), dtype=np.uint8)
                ).save(root / f"BraTS20_Training_{i:03d}_brain_tumor_{modality}.png")
            Image.fromarray(
                (rng.integers(0, 2, (8, 8)) * 255).astype(np.uint8)
            ).save(root / f"BraTS20_Training_{i:03d}_brain_tumor_seg.png")

    def test_real_brats_naming_yields_all_modalities(self, tmp_path):
        self._make_brats(tmp_path)
        images, masks = _pair_brats_images_and_masks(tmp_path)
        # 3 patients x 4 modalities, each paired with that patient's mask.
        assert len(images) == 12
        assert len(masks) == 12

    def test_every_image_paired_with_its_own_subject_mask(self, tmp_path):
        self._make_brats(tmp_path)
        images, masks = _pair_brats_images_and_masks(tmp_path)
        for image, mask in zip(images, masks, strict=True):
            assert _extract_patient_id(image) == _extract_patient_id(mask)
            assert "seg" in mask

    def test_load_brats_paths_does_not_raise(self, tmp_path):
        self._make_brats(tmp_path)
        image_paths, mask_paths = load_brats_paths(tmp_path)
        assert len(image_paths) == len(mask_paths) == 12

    def test_mask_suffix_detection(self):
        assert _has_mask_suffix("BraTS20_Training_001_brain_tumor_seg")
        assert _has_mask_suffix("patient001_mask")
        assert not _has_mask_suffix("BraTS20_Training_001_brain_tumor_FLAIR")
        assert not _has_mask_suffix("scan000")

    def test_unpaired_images_excluded(self, tmp_path):
        self._make_brats(tmp_path, n_patients=2)
        (tmp_path / "orphan_FLAIR.png").touch()
        images, _ = _pair_brats_images_and_masks(tmp_path)
        assert all("orphan" not in p for p in images)


# ── Patient identifier extraction ────────────────────────────────────────

class TestPatientIdGrouping:
    """Regression: the old regex never grouped anything.

    ``re.sub(r"[_\\-]?(slice|frame|s|img|image)?[_\\-]?\\d+$", "", stem)`` made the
    keyword group optional and kept the separators inside the capture, so
    'glioma (12)' and 'image003' came back unchanged and a BraTS image and its
    mask produced two *different* group keys.
    """

    def test_image_and_mask_share_a_group_key(self):
        image = _extract_patient_id("BraTS20_Training_007_brain_tumor_FLAIR.png")
        mask = _extract_patient_id("BraTS20_Training_007_brain_tumor_seg.png")
        assert image == mask

    def test_all_modalities_share_a_group_key(self):
        keys = {
            _extract_patient_id(f"BraTS20_Training_007_brain_tumor_{m}.png")
            for m in ("FLAIR", "T1", "T1ce", "T2")
        }
        assert len(keys) == 1

    def test_different_subjects_do_not_collide(self):
        assert _extract_patient_id("BraTS20_Training_007_brain_tumor_seg") != _extract_patient_id(
            "BraTS20_Training_008_brain_tumor_seg"
        )

    def test_separator_style_does_not_split_a_subject(self):
        assert _extract_patient_id("Case-045_s3") == _extract_patient_id("Case_045_s3")

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("patient001_slice01.png", "patient001"),
            ("patient_a_003.png", "patient_a"),
            ("scan_frame3.png", "scan"),
            ("glioma (12).png", "glioma"),
        ],
    )
    def test_slice_counters_are_stripped(self, name, expected):
        assert _extract_patient_id(name) == expected

    def test_subject_id_is_not_mistaken_for_a_slice_counter(self):
        # "image003" is one subject, not a slice of "image": stripping it would
        # collapse every subject into a single group.
        assert _extract_patient_id("image003.png") == "image003"

    def test_never_returns_empty(self):
        for name in ["001.png", "_.png", "x.png", "12.png"]:
            assert _extract_patient_id(name)


# ── Synthetic mixing ─────────────────────────────────────────────────────

class TestMixRealSynthetic:
    """Regression: ``max(1, ...)`` injected a synthetic sample at ratio=0."""

    def test_zero_ratio_returns_real_only(self):
        real = np.zeros((10, 4, 4, 1), np.float32)
        synthetic = np.ones((10, 4, 4, 1), np.float32)
        mixed, _ = mix_real_synthetic(
            real, np.zeros(10, np.int32), synthetic, np.ones(10, np.int32), ratio=0.0
        )
        assert len(mixed) == 10

    def test_half_ratio(self):
        real = np.zeros((10, 4, 4, 1), np.float32)
        synthetic = np.ones((10, 4, 4, 1), np.float32)
        mixed, _ = mix_real_synthetic(
            real, np.zeros(10, np.int32), synthetic, np.ones(10, np.int32), ratio=0.5
        )
        assert len(mixed) == 15

    def test_full_ratio_capped_by_synthetic_pool(self):
        real = np.zeros((10, 4, 4, 1), np.float32)
        synthetic = np.ones((3, 4, 4, 1), np.float32)
        mixed, _ = mix_real_synthetic(
            real, np.zeros(10, np.int32), synthetic, np.zeros(3, np.int32), ratio=1.0
        )
        assert len(mixed) == 13

    @pytest.mark.parametrize("ratio", [-0.1, 1.5])
    def test_invalid_ratio_rejected(self, ratio):
        real = np.zeros((4, 2, 2, 1), np.float32)
        with pytest.raises(ValueError, match="ratio"):
            mix_real_synthetic(
                real, np.zeros(4, np.int32), real, np.zeros(4, np.int32), ratio=ratio
            )

    def test_length_mismatch_rejected(self):
        real = np.zeros((4, 2, 2, 1), np.float32)
        with pytest.raises(ValueError, match="same length"):
            mix_real_synthetic(real, np.zeros(4, np.int32), real, np.zeros(3, np.int32))


# ── tf.data builders ─────────────────────────────────────────────────────

class TestDatasetBuilders:
    """Regression: ``shuffle(buffer_size=min(len(x), N))`` raised on empty splits.

    ``tf.data`` rejects ``buffer_size < 1``, so every ``shuffle=True`` builder
    failed with an opaque ``ValueError`` instead of a meaningful message.
    """

    def test_empty_split_raises_meaningful_error(self):
        with pytest.raises(ValueError, match="empty split"):
            build_detection_dataset(np.zeros((0, 8, 8, 1), np.float32), [], batch_size=2)

    def test_gan_dataset_handles_empty_input(self):
        with pytest.raises(ValueError, match="empty split"):
            build_gan_dataset(np.zeros((0, 8, 8, 1), np.float32), batch_size=2)


class TestGanDatasetRange:
    """Regression: range was inferred from ``images.min()``.

    A [0, 1] batch containing no negative values was shifted again, and raw
    0..255 data became -1..509. The GAN path now maps [0, 1] -> [-1, 1]
    unconditionally, matching ``build_gan_dataset_from_paths``.
    """

    def test_zero_one_maps_to_minus_one_one(self):
        dataset = build_gan_dataset(np.zeros((4, 8, 8, 1), np.float32), batch_size=2)
        batch = next(iter(dataset)).numpy()
        assert batch.min() == pytest.approx(-1.0)
        assert batch.max() == pytest.approx(-1.0)

    def test_unit_maximum_maps_to_one(self):
        dataset = build_gan_dataset(np.ones((4, 8, 8, 1), np.float32), batch_size=2)
        batch = next(iter(dataset)).numpy()
        assert batch.min() == pytest.approx(1.0)

    def test_labels_are_one_hot(self):
        images = np.zeros((4, 8, 8, 1), np.float32)
        dataset = build_gan_dataset(images, labels=[0, 1, 2, 3], batch_size=2, shuffle=False)
        labels = np.concatenate([batch.numpy() for _, batch in dataset])
        assert labels.shape == (4, 4)
        assert np.all(labels.sum(axis=1) == 1)


class TestShuffledBuildersRun:
    def test_detection_builder(self):
        images = np.random.default_rng(0).random((6, 8, 8, 1)).astype(np.float32)
        dataset = build_detection_dataset(images, [0, 1, 0, 1, 0, 1], batch_size=3)
        x, y = next(iter(dataset))
        assert tuple(x.shape) == (3, 8, 8, 1)
        assert tuple(y.shape) == (3,)

    def test_classifier_builder_one_hot(self):
        images = np.random.default_rng(0).random((6, 8, 8, 1)).astype(np.float32)
        dataset = build_classifier_dataset(images, [0, 1, 2, 3, 0, 1], batch_size=3)
        x, y = next(iter(dataset))
        assert tuple(x.shape) == (3, 8, 8, 1)
        assert tuple(y.shape) == (3, 4)

    def test_segmentation_builder(self):
        images = np.random.default_rng(0).random((4, 8, 8, 1)).astype(np.float32)
        masks = np.zeros((4, 8, 8, 1), np.float32)
        dataset = build_segmentation_dataset(images, masks, batch_size=2)
        x, y = next(iter(dataset))
        assert tuple(x.shape) == (2, 8, 8, 1)
        assert tuple(y.shape) == (2, 8, 8, 1)


# ── Patient-level splitting ──────────────────────────────────────────────

class TestPathBasedLoaders:
    """Regression: the path-based loaders crashed on the first batch.

    ``_load_path_image_tf`` feeds ``tf.py_function``, which hands the callback
    an ``EagerTensor``. The callback called ``.decode('utf-8')`` on it, raising
    ``'EagerTensor' object has no attribute 'decode'``. Every path-based
    builder is used by all four training tracks, so no track could train at all.
    """

    @staticmethod
    def _write_images(folder, count=4, size=16):
        from PIL import Image

        folder.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        paths = []
        for i in range(count):
            path = folder / f"img{i:03d}.png"
            Image.fromarray(rng.integers(0, 256, (size, size), dtype=np.uint8)).save(path)
            paths.append(str(path))
        return paths

    def test_detection_dataset_from_paths(self, tmp_path):
        from data.dataset import build_detection_dataset_from_paths

        paths = self._write_images(tmp_path)
        dataset = build_detection_dataset_from_paths(
            paths, [0, 1, 0, 1], img_size=(16, 16), batch_size=2
        )
        x, y = next(iter(dataset))
        assert tuple(x.shape) == (2, 16, 16, 1)
        assert tuple(y.shape) == (2,)

    def test_classifier_dataset_from_paths(self, tmp_path):
        from data.dataset import build_classifier_dataset_from_paths

        paths = self._write_images(tmp_path)
        dataset = build_classifier_dataset_from_paths(
            paths, [0, 1, 2, 3], img_size=(16, 16), batch_size=2
        )
        x, y = next(iter(dataset))
        assert tuple(x.shape) == (2, 16, 16, 1)
        assert tuple(y.shape) == (2, 4)

    def test_segmentation_dataset_from_paths(self, tmp_path):
        from data.dataset import build_segmentation_dataset_from_paths

        images = self._write_images(tmp_path / "img")
        masks = self._write_images(tmp_path / "msk")
        dataset = build_segmentation_dataset_from_paths(
            images, masks, img_size=(16, 16), batch_size=2
        )
        x, y = next(iter(dataset))
        assert tuple(x.shape) == (2, 16, 16, 1)
        # Masks are binarized, so only 0.0 and 1.0 may appear.
        assert set(np.unique(y.numpy()).tolist()) <= {0.0, 1.0}

    def test_gan_dataset_from_paths(self, tmp_path):
        from data.dataset import build_gan_dataset_from_paths

        paths = self._write_images(tmp_path)
        dataset = build_gan_dataset_from_paths(
            paths, labels=[0, 1, 2, 3], img_size=(16, 16), batch_size=2
        )
        x, y = next(iter(dataset))
        assert tuple(x.shape) == (2, 16, 16, 1)
        assert tuple(y.shape) == (2, 4)
        # minus_one_one normalisation
        assert -1.0 <= float(x.numpy().min()) and float(x.numpy().max()) <= 1.0

    def test_corrupt_path_reports_the_path(self, tmp_path):
        """The failure must name the offending file, not surface as InvalidArgumentError."""
        from data.dataset import _load_image_from_bytes

        missing = str(tmp_path / "does_not_exist.png")
        with pytest.raises(RuntimeError, match="does_not_exist.png"):
            _load_image_from_bytes(missing, img_size=(16, 16))

    def test_accepts_bytes_and_tensor_forms(self, tmp_path):
        from data.dataset import _load_image_from_bytes

        paths = self._write_images(tmp_path, count=1)
        as_str = _load_image_from_bytes(paths[0], img_size=(16, 16))
        as_bytes = _load_image_from_bytes(paths[0].encode("utf-8"), img_size=(16, 16))
        as_tensor = _load_image_from_bytes(tf.constant(paths[0]), img_size=(16, 16))
        assert np.array_equal(as_str, as_bytes)
        assert np.array_equal(as_str, as_tensor)


class TestPlotDirectoryCreation:
    """Regression: saving a loss curve aborted an otherwise finished run.

    ``config.ensure_directories()`` only creates the top-level directories, so
    ``logs/<track>/`` did not exist and ``plt.savefig`` raised
    ``FileNotFoundError`` at the end of every training run.
    """

    def test_ensure_parent_dir_creates_nested_directories(self, tmp_path):
        from evaluation import _ensure_parent_dir

        target = tmp_path / "a" / "b" / "c" / "plot.png"
        _ensure_parent_dir(target)
        assert target.parent.is_dir()

    def test_is_idempotent(self, tmp_path):
        from evaluation import _ensure_parent_dir

        target = tmp_path / "x" / "plot.png"
        _ensure_parent_dir(target)
        _ensure_parent_dir(target)
        assert target.parent.is_dir()

    def test_none_path_is_a_noop(self):
        from evaluation import _ensure_parent_dir

        _ensure_parent_dir(None)

    def test_loss_curves_write_into_a_new_directory(self, tmp_path):
        from evaluation import plot_loss_curves

        class History:
            history = {"loss": [1.0, 0.5], "val_loss": [1.2, 0.7]}

        target = tmp_path / "logs" / "detection" / "loss_curves.png"
        plot_loss_curves(History(), save_path=target)
        assert target.exists()


class TestPatientLevelSplit:
    """Regression: GroupShuffleSplit is unstratified and can drop whole classes.

    With few groups per class the test partition could contain an entire class
    while the training partition held only the remaining ones, or the second
    split produced an empty partition.
    """

    @staticmethod
    def _make_dataset(root, n_per_class=12):
        rng = np.random.default_rng(0)
        for class_name in ("glioma", "meningioma", "pituitary", "normal"):
            folder = root / "train" / class_name
            folder.mkdir(parents=True)
            for i in range(n_per_class):
                Image.fromarray(
                    rng.integers(0, 256, (8, 8), dtype=np.uint8)
                ).save(folder / f"patient{i:03d}_slice01.png")

    def test_every_class_present_in_every_split(self, tmp_path):
        self._make_dataset(tmp_path)
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = (
            get_figshare_patient_level_split(tmp_path)
        )
        for name, labels in (("train", y_train), ("val", y_val), ("test", y_test)):
            assert len(labels) > 0, f"{name} split is empty"
            assert set(np.unique(labels).tolist()) == {0, 1, 2, 3}, f"{name} lost a class"

    def test_no_patient_spans_two_splits(self, tmp_path):
        self._make_dataset(tmp_path)
        (x_train, _), (x_val, _), (x_test, _) = get_figshare_patient_level_split(tmp_path)
        groups = [
            {_extract_patient_id(p) for p in paths} for paths in (x_train, x_val, x_test)
        ]
        assert not (groups[0] & groups[1]), "patient leaked between train and val"
        assert not (groups[0] & groups[2]), "patient leaked between train and test"
        assert not (groups[1] & groups[2]), "patient leaked between val and test"

    def test_no_duplicate_paths_across_splits(self, tmp_path):
        self._make_dataset(tmp_path)
        (x_train, _), (x_val, _), (x_test, _) = get_figshare_patient_level_split(tmp_path)
        everything = list(x_train) + list(x_val) + list(x_test)
        assert len(everything) == len(set(everything))

    def test_slices_of_one_patient_stay_together(self, tmp_path):
        self._make_dataset(tmp_path, n_per_class=6)
        (x_train, _), (x_val, _), (x_test, _) = get_figshare_patient_level_split(tmp_path)
        # patient000 appears 4x (once per class) and must land in exactly one split.
        owner = []
        for idx, paths in enumerate((x_train, x_val, x_test)):
            owner.extend([idx] * sum("patient000" in p for p in paths))
        assert len(set(owner)) == 1

    def test_single_patient_dataset_raises(self, tmp_path):
        folder = tmp_path / "glioma"
        folder.mkdir()
        Image.fromarray(np.zeros((8, 8), np.uint8)).save(folder / "only.png")
        with pytest.raises(ValueError, match="patient IDs"):
            get_figshare_patient_level_split(tmp_path)
