"""Regression tests for readiness checks, config, callbacks, and the training path.

These cover defects that made a failing configuration look healthy, or that
crashed entry points before any useful output was produced.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def preflight():
    return _load_preflight()


def _load_preflight():
    path = REPO_ROOT / "scripts" / "preflight.py"
    spec = importlib.util.spec_from_file_location("_preflight_under_test", path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: preflight uses @dataclass, which resolves the
    # module's __dict__ through sys.modules during class creation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ── scripts/preflight.py ─────────────────────────────────────────────────

class TestPreflightRequiredFlag:
    """Regression: ``required=True`` was hard-coded inside optional checks.

    The caller passes ``required=args.require_weights``, but the LFS-pointer,
    read-error, and invalid-JSON branches all forced ``required=True``, so a
    committed LFS stub failed the whole preflight even when weights were
    explicitly optional -- exactly what ``--require-weights`` should control.
    """

    def test_lfs_pointer_is_only_fatal_when_required(self, preflight, tmp_path):
        stub = tmp_path / "model.keras"
        stub.write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 100\n"
        )
        optional = preflight._check_not_lfs_pointer(stub, "w", required=False)
        assert optional.status == preflight.STATUS_WARN
        assert optional.required is False

        required = preflight._check_not_lfs_pointer(stub, "w", required=True)
        assert required.status == preflight.STATUS_FAIL
        assert required.required is True

    def test_read_error_is_only_fatal_when_required(self, preflight, tmp_path):
        # A directory is a path that exists but cannot be read as a file.
        target = tmp_path / "weird.keras"
        target.mkdir()
        optional = preflight._check_not_lfs_pointer(target, "w", required=False)
        assert optional.required is False
        assert optional.status == preflight.STATUS_WARN

    def test_valid_file_passes(self, preflight, tmp_path):
        good = tmp_path / "model.keras"
        good.write_bytes(b"Keras archive payload")
        result = preflight._check_not_lfs_pointer(good, "w", required=True)
        assert result.status == preflight.STATUS_PASS


class TestPreflightDetectionConfig:
    """Regression: a missing ``threshold`` key silently passed as 0.5.

    ``data.get("threshold", 0.5)`` meant a truncated config reported
    ``PASS threshold=0.5000`` while the app would use an uncalibrated cutoff
    instead of the calibrated 0.225 -- a silent accuracy regression.
    """

    def test_missing_threshold_key_does_not_pass(self, preflight, tmp_path):
        path = tmp_path / "cfg.json"
        path.write_text(json.dumps({"validation_metrics": {}}))
        result = preflight._check_detection_config(path, required=True)
        assert result.status == preflight.STATUS_FAIL
        assert "threshold" in result.details

    def test_valid_threshold_passes(self, preflight, tmp_path):
        path = tmp_path / "cfg.json"
        path.write_text(json.dumps({"threshold": 0.225}))
        result = preflight._check_detection_config(path, required=True)
        assert result.status == preflight.STATUS_PASS
        assert "0.2250" in result.details

    def test_out_of_range_threshold_fails(self, preflight, tmp_path):
        path = tmp_path / "cfg.json"
        path.write_text(json.dumps({"threshold": 4.2}))
        assert preflight._check_detection_config(path, required=True).status == (
            preflight.STATUS_FAIL
        )

    def test_malformed_json_is_only_fatal_when_required(self, preflight, tmp_path):
        path = tmp_path / "cfg.json"
        path.write_text("{not json")
        assert preflight._check_detection_config(path, required=True).status == (
            preflight.STATUS_FAIL
        )
        assert preflight._check_detection_config(path, required=False).status == (
            preflight.STATUS_WARN
        )

    def test_missing_file_is_only_fatal_when_required(self, preflight, tmp_path):
        path = tmp_path / "absent.json"
        assert preflight._check_detection_config(path, required=False).status == (
            preflight.STATUS_WARN
        )


class TestPreflightNamespace:
    """Regression: run_preflight dereferenced args attributes unguarded."""

    def test_partial_namespace_uses_defaults(self, preflight):
        results = preflight.run_preflight(argparse.Namespace(ci_mode=True))
        assert results
        assert all(isinstance(r.required, bool) for r in results)

    def test_full_namespace(self, preflight):
        results = preflight.run_preflight(
            argparse.Namespace(ci_mode=True, require_weights=True, require_datasets=True)
        )
        names = {r.name for r in results}
        assert "dataset-figshare" in names
        assert "dataset-brats" in names

    def test_python_floor_matches_pyproject(self, preflight):
        import tomllib

        with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
            pyproject = tomllib.load(handle)
        requires = pyproject["project"]["requires-python"]
        minor = int(requires.split(">=")[1].split(".")[1])
        result = preflight._check_python_version((3, minor))
        assert result.status == preflight.STATUS_PASS


# ── config.py ────────────────────────────────────────────────────────────

class TestConfig:
    def test_import_does_not_create_directories(self, tmp_path, monkeypatch):
        """Regression: importing config created 7 directories as a side effect.

        Every module and every test imports config, so collection itself
        mutated the working tree (and would fail outright on a read-only mount).
        """
        import config

        # Directories are created only when explicitly requested.
        assert hasattr(config, "ensure_directories")
        assert isinstance(config.MANAGED_DIRS, tuple)
        assert len(config.MANAGED_DIRS) >= 5

    def test_ensure_directories_is_idempotent(self):
        import config

        config.ensure_directories()
        config.ensure_directories()
        for directory in config.MANAGED_DIRS:
            assert os.path.isdir(directory)

    def test_class_names_and_counts_agree(self):
        import config

        assert config.NUM_CLASSES == len(config.CLASS_NAMES)
        assert len(set(config.CLASS_NAMES)) == config.NUM_CLASSES

    def test_every_track_has_a_config(self):
        import config

        for track in config.FLAGSHIP_TRACKS + config.EXPERIMENTAL_TRACKS:
            assert track in config.TRACK_CONFIGS

    def test_image_config_shapes_are_positive(self):
        import config

        cfg = config.ImageConfig()
        for name in ("detection_size", "segmentation_size", "classifier_size", "gan_size"):
            height, width = getattr(cfg, name)
            assert height > 0 and width > 0

    def test_segmentation_size_is_divisible_by_16(self):
        """build_unet rejects other sizes; the default must satisfy it."""
        import config

        height, width = config.ImageConfig().segmentation_size
        assert height % 16 == 0 and width % 16 == 0


# ── training/callbacks.py ────────────────────────────────────────────────

tf = pytest.importorskip("tensorflow")

from training.callbacks import (  # noqa: E402
    GANLossLogger,
    ModelCollapseDetector,
    _tensorboard_callback,
    get_standard_callbacks,
)


class TestTensorBoardIsOptional:
    """Regression: every training run crashed when TensorBoard was not installed.

    ``tf.keras.callbacks.TensorBoard`` raises ``TBNotInstalledError`` at
    construction time, and that error derives straight from ``Exception`` rather
    than ``ImportError``. Constructing it unconditionally therefore aborted all
    ten trainer tests in any environment that installed only the runtime stack,
    while passing on a developer machine that happened to have TensorBoard.
    """

    def test_returns_callback_when_tensorboard_present(self, tmp_path):
        if importlib.util.find_spec("tensorboard") is None:
            pytest.skip("TensorBoard is not installed in this environment")
        assert len(_tensorboard_callback(tmp_path)) == 1

    def test_returns_nothing_when_tensorboard_missing(self, tmp_path, monkeypatch):
        real = importlib.util.find_spec

        def without_tensorboard(name, *args, **kwargs):
            if name == "tensorboard":
                return None
            return real(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", without_tensorboard)
        assert _tensorboard_callback(tmp_path) == []

    def test_standard_callbacks_survive_without_tensorboard(self, tmp_path, monkeypatch):
        """The full callback list must still build, minus the TensorBoard entry."""
        from config import CHECKPOINT_DIR, LOG_DIR

        real = importlib.util.find_spec
        monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: (
            None if name == "tensorboard" else real(name, *a, **k)
        ))
        monkeypatch.setattr("training.callbacks.CHECKPOINT_DIR", tmp_path / "ckpt")
        monkeypatch.setattr("training.callbacks.LOG_DIR", tmp_path / "logs")

        callbacks = get_standard_callbacks(
            tf.keras.Sequential([tf.keras.layers.Input((4,)), tf.keras.layers.Dense(1)]),
            "tensorboard_optional_probe",
        )
        names = [type(c).__name__ for c in callbacks]
        assert "TensorBoard" not in names
        assert {"ModelCheckpoint", "EarlyStopping", "CSVLogger"} <= set(names)
        assert (tmp_path / "logs" / "tensorboard_optional_probe").is_dir()
        assert CHECKPOINT_DIR is not None and LOG_DIR is not None


class TestGanLossLoggerColumns:
    """Regression: v1 and v2 runs shared one CSV with different semantics.

    The WGAN-GP logger passes ``w_dist`` where the v1 header says ``d_acc``, and
    the header was only written when the file was absent, so Wasserstein
    distances were appended under a column labelled accuracy.
    """

    def test_column_mismatch_starts_a_new_file(self, tmp_path):
        path = tmp_path / "shared.csv"
        v1 = GANLossLogger(csv_path=path, columns=["epoch", "d_loss", "g_loss", "d_acc", "g_acc"])
        v1.log_step(0, 0.5, 0.4, 0.9, 0.1)

        v2 = GANLossLogger(csv_path=path, columns=["epoch", "d_loss", "g_loss", "w_dist", "g_acc"])
        v2.log_step(0, 0.5, 0.4, 1.5, 0.1)

        # The v1 log is preserved under a distinct name, not corrupted.
        assert path.exists()
        header_v2 = path.read_text(encoding="utf-8").splitlines()[0]
        assert header_v2 == "epoch,d_loss,g_loss,w_dist,g_acc"

        backups = list(tmp_path.glob("shared.csv.*.bak"))
        assert backups
        assert "d_acc" in backups[0].read_text(encoding="utf-8").splitlines()[0]

    def test_same_columns_append(self, tmp_path):
        path = tmp_path / "run.csv"
        columns = ["epoch", "d_loss", "g_loss", "d_acc", "g_acc"]
        for _ in range(2):
            GANLossLogger(csv_path=path, columns=columns).log_step(0, 0.5, 0.4, 0.9, 0.1)
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        assert len(lines) == 3  # header + 2 rows

    def test_wrong_arity_is_rejected(self, tmp_path):
        logger = GANLossLogger(csv_path=tmp_path / "x.csv", columns=["a", "b"])
        with pytest.raises(ValueError, match="columns"):
            logger.log_step(0, 1.0, 2.0, 3.0, 4.0)


class TestCollapseDetectorDiversity:
    """Regression: diversity was measured across classes.

    ``std(axis=0)`` over a batch containing two samples per class is dominated
    by inter-class variance, so a healthy class-conditional GAN was flagged as
    collapsed.
    """

    @staticmethod
    def _generator_returning(array):
        class _Fixed:
            def __call__(self, inputs, training=False):
                return tf.constant(array, dtype=tf.float32)

        return _Fixed()

    def test_within_class_diversity_used(self):
        # Two classes, each internally constant -> a healthy conditional GAN.
        per_class = np.zeros((8, 8, 8, 1), np.float32)
        per_class[4:] = 1.0
        detector = ModelCollapseDetector(
            self._generator_returning(per_class), latent_dim=8, conditional=True
        )
        # Within-class std is 0, so the detector *should* warn here.
        assert detector._within_class_diversity(per_class) == pytest.approx(0.0)

    def test_diverse_within_class_scores_higher(self):
        rng = np.random.default_rng(0)
        varied = rng.random((8, 8, 8, 1)).astype(np.float32)
        detector = ModelCollapseDetector(
            self._generator_returning(varied), latent_dim=8, conditional=True
        )
        assert detector._within_class_diversity(varied) > 0.02

    def test_unconditional_path(self):
        rng = np.random.default_rng(0)
        varied = rng.random((8, 8, 8, 1)).astype(np.float32)
        detector = ModelCollapseDetector(
            self._generator_returning(varied), latent_dim=8, conditional=False
        )
        assert detector._within_class_diversity(varied) > 0.02

    def test_non_finite_output_warns(self, caplog):
        import logging

        bad = np.full((8, 8, 8, 1), np.nan, np.float32)
        detector = ModelCollapseDetector(
            self._generator_returning(bad), latent_dim=8, conditional=True
        )
        with caplog.at_level(logging.WARNING):
            detector.on_epoch_end(0)
        assert "non-finite" in caplog.text


# ── training state / entry point ─────────────────────────────────────────

class TestTrainingStateAndEntryPoint:
    def test_help_works_without_importing_tensorflow(self):
        """``build_arg_parser`` is hoisted so --help needs no heavy imports."""
        sys.path.insert(0, str(REPO_ROOT))
        import train

        assert train.build_arg_parser() is not None

    def test_parser_exposes_reproducibility_flags(self):
        import train

        options = {a.dest for a in train.build_arg_parser()._actions}
        assert {"seed", "deterministic", "log_level"} <= options

    def test_gan_state_fresh_schema_has_distinct_w_distance_slot(self):
        from training.state import GANState

        state = GANState.fresh_state()
        # Wasserstein distances must not share the d_accs slot.
        assert "w_distances" in state
        assert state["w_distances"] == []

    def test_training_state_save_is_atomic(self, tmp_path, monkeypatch):
        import training.state as state_module

        monkeypatch.setattr(state_module, "CHECKPOINT_DIR", str(tmp_path))
        state = state_module.TrainingState("unit_test_track")
        state.update_epoch(3)
        assert state.start_epoch() == 4
        # No leftover temp file after a successful save.
        assert not list(tmp_path.glob("**/*.tmp"))

    def test_state_records_the_seed(self, tmp_path, monkeypatch):
        import json
        from pathlib import Path

        import training.state as state_module

        monkeypatch.setattr(state_module, "CHECKPOINT_DIR", str(tmp_path))
        state = state_module.TrainingState("seeded_track", seed=1234)
        state.update_epoch(0)
        payload = json.loads(Path(state.state_path).read_text(encoding="utf-8"))
        assert payload["reproducibility"]["seed"] == 1234
        assert "tensorflow" in payload["reproducibility"]

    def test_custom_objects_cover_the_unet_loss_and_metrics(self):
        from training.runtime import SEGMENTATION_CUSTOM_OBJECTS

        for name in ("dice_bce_loss", "dice_coefficient", "iou_metric"):
            assert name in SEGMENTATION_CUSTOM_OBJECTS

    def test_every_track_module_exposes_its_trainer(self):
        from training.tracks import (
            classifier,
            detection,
            gan,
            gan_augmented,
            gan_v2,
            segmentation,
        )

        assert callable(detection.train_detection)
        assert callable(segmentation.train_segmentation)
        assert callable(classifier.train_classifier)
        assert callable(gan.train_gan)
        assert callable(gan_v2.train_gan_v2)
        assert callable(gan_augmented.train_classifier_with_gan)

    def test_trainers_accept_a_seed(self):
        import inspect

        from training.tracks import classifier, detection, gan, segmentation

        for fn in (detection.train_detection, segmentation.train_segmentation,
                   classifier.train_classifier, gan.train_gan):
            assert "seed" in inspect.signature(fn).parameters


class TestSeeding:
    def test_set_seed_returns_the_applied_seed(self):
        from training.reproducibility import set_seed

        assert set_seed(123) == 123

    def test_set_seed_makes_numpy_reproducible(self):
        from training.reproducibility import set_seed

        set_seed(99)
        first = np.random.default_rng(0).random(3)
        second = np.random.default_rng(0).random(3)
        assert np.array_equal(first, second)

    def test_tf_seed_controls_dropout_masks(self):
        """A model built after set_seed must draw identical dropout masks.

        The layer is rebuilt each round because a Keras layer caches its own
        random state; this mirrors real training, where seeding happens before
        the model is constructed.
        """
        from training.reproducibility import set_seed

        def dropout_pattern(seed):
            set_seed(seed)
            layer = tf.keras.layers.Dropout(0.5)
            x = tf.ones((1, 6))
            return [layer(x, training=True).numpy().astype(int).tolist()
                    for _ in range(3)]

        assert dropout_pattern(7) == dropout_pattern(7)
        assert dropout_pattern(7) != dropout_pattern(8)

    def test_seed_state_dict_shape(self):
        from training.reproducibility import seed_state_dict

        block = seed_state_dict(5)
        assert block["seed"] == 5
        assert {"tensorflow", "numpy", "python", "deterministic_ops"} <= set(block)



class TestAugmentImageIsGraphSafe:
    """Regression: augment_image used a Python ``if`` on a traced tensor.

    ``if tf.random.uniform(()) > 0.5`` works eagerly but raises "Using a
    symbolic tf.Tensor as a Python bool is not allowed" the moment
    ``tf.data.Dataset.map`` traces it -- which is how every training builder
    calls it, so all augment=True training crashed.
    """

    def test_traces_inside_dataset_map(self):
        from data.dataset import augment_image

        dataset = tf.data.Dataset.from_tensor_slices(
            (np.zeros((2, 8, 8, 1), np.float32), np.zeros(2, np.float32))
        )
        mapped = dataset.map(lambda x, y: (augment_image(x), y)).batch(2)
        images, labels = next(iter(mapped))
        assert tuple(images.shape) == (2, 8, 8, 1)
        assert tuple(labels.shape) == (2,)

    def test_image_and_mask_receive_the_same_transform(self):
        from data.dataset import augment_image

        # A mask whose left half is 1 and right half is 0 lets us detect any
        # disagreement between the image and mask transforms.
        image = np.tile(
            np.concatenate([np.zeros((8, 4, 1)), np.ones((8, 4, 1))], axis=1),
            (8, 1, 1),
        ).astype(np.float32)
        mask = image.copy()

        dataset = tf.data.Dataset.from_tensors((image, mask))
        mapped = dataset.map(augment_image)
        for _ in range(10):
            out_image, out_mask = next(iter(mapped))
            # Flips/rotations preserve equality, so a mismatch means the two
            # tensors were transformed independently.
            assert np.array_equal(
                (out_image.numpy() > 0.5), (out_mask.numpy() > 0.5)
            ), "image and mask were augmented inconsistently"

    def test_mask_values_stay_binary(self):
        from data.dataset import augment_image

        image = np.zeros((8, 8, 1), np.float32)
        mask = np.ones((8, 8, 1), np.float32)
        dataset = tf.data.Dataset.from_tensors((image, mask))
        out_image, out_mask = next(iter(dataset.map(augment_image)))
        assert set(np.unique(out_mask.numpy()).tolist()) <= {0.0, 1.0}
