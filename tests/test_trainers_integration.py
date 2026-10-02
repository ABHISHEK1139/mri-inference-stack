"""Integration tests that run each training track end-to-end on synthetic data.

These are the tests that matter for coverage: every other suite exercises
shapes, and a shape test would not have caught the ``EagerTensor.decode``
crash, the ``tf.data`` graph-tracing failure in ``augment_image``, or the
missing output directory that aborted a finished run.

Each track runs a single epoch over a tiny generated dataset. Even so, a full
trainer run costs 40-180s, so every test in this module is marked ``slow`` and
CI runs them in a dedicated job. The fast tier covers shapes, loading, and
regressions; this module covers behaviour.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

tf = pytest.importorskip("tensorflow")

pytestmark = pytest.mark.slow

CLASSES = ("glioma", "meningioma", "pituitary", "normal")
IMG = 32


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Redirect every output path the track modules captured at import time.

    Returns a namespace with ``.root`` (the temp dir) and ``.weights`` (the
    resolved WEIGHTS_DIR) so tests can assert on exported artifacts.
    """
    import types

    import config
    import training.runtime as runtime
    import training.state as state
    from training.tracks import (
        classifier,
        detection,
        gan,
        gan_augmented,
        gan_v2,
        segmentation,
    )

    mapping = {
        "DATA_DIR": "data",
        "RAW_DIR": os.path.join("data", "raw"),
        "PROCESSED_DIR": os.path.join("data", "processed"),
        "CHECKPOINT_DIR": "checkpoints",
        "LOG_DIR": "logs",
        "OUTPUT_DIR": "outputs",
        "WEIGHTS_DIR": "weights",
    }
    created = {}
    for attr, rel in mapping.items():
        path = tmp_path / rel
        path.mkdir(parents=True, exist_ok=True)
        created[attr] = str(path)
        monkeypatch.setattr(config, attr, str(path), raising=False)

    for module in (runtime, state, detection, segmentation, classifier, gan,
                   gan_v2, gan_augmented):
        for attr, value in created.items():
            if hasattr(module, attr):
                monkeypatch.setattr(module, attr, value)

    # Keep the resolution used by the models small so a run stays fast. Every
    # track module imported IMG_CFG by value, so each binding must be patched.
    small = type(detection.IMG_CFG)(detection_size=(IMG, IMG),
                                    classifier_size=(IMG, IMG),
                                    segmentation_size=(IMG, IMG),
                                    gan_size=(IMG, IMG))
    for module in (runtime, detection, segmentation, classifier, gan, gan_v2,
                   gan_augmented):
        if hasattr(module, "IMG_CFG"):
            monkeypatch.setattr(module, "IMG_CFG", small)
    return types.SimpleNamespace(
        root=tmp_path,
        weights=created["WEIGHTS_DIR"],
        checkpoints=created["CHECKPOINT_DIR"],
        logs=created["LOG_DIR"],
    )


@pytest.fixture
def figshare_dir(tmp_path):
    """A tiny Figshare-shaped classification dataset."""
    root = tmp_path / "figshare"
    rng = np.random.default_rng(0)
    for class_name in CLASSES:
        folder = root / "train" / class_name
        folder.mkdir(parents=True)
        # 8 per class keeps the 15% stratified test split (>=4 samples)
        # from being smaller than the class count.
        for i in range(8):
            array = rng.integers(0, 256, (IMG, IMG), dtype=np.uint8)
            Image.fromarray(array).save(folder / f"p{i:03d}_slice01.png")
    return str(root)


@pytest.fixture
def brats_dir(tmp_path):
    """A tiny BraTS-shaped segmentation dataset (4 modalities per subject)."""
    root = tmp_path / "brats"
    root.mkdir()
    rng = np.random.default_rng(1)
    for i in range(3):
        for modality in ("FLAIR", "T1", "T1ce", "T2"):
            Image.fromarray(rng.integers(0, 256, (IMG, IMG), dtype=np.uint8)).save(
                root / f"BraTS20_Training_{i:03d}_brain_tumor_{modality}.png"
            )
        Image.fromarray((rng.integers(0, 2, (IMG, IMG)) * 255).astype(np.uint8)).save(
            root / f"BraTS20_Training_{i:03d}_brain_tumor_seg.png"
        )
    return str(root)


# ── Detection ────────────────────────────────────────────────────────────

class TestDetectionTrainer:
    def test_runs_and_exports_artifacts(self, workspace, figshare_dir):
        from training.tracks.detection import train_detection

        weights = workspace.weights
        model, history, metrics = train_detection(
            data_dir=figshare_dir, epochs=1, resume=False, seed=1234
        )

        assert model.output_shape == (None, 1)
        assert history is not None and "loss" in history.history
        assert set(metrics) >= {"accuracy", "precision", "recall", "f1_score", "auc"}
        assert os.path.exists(os.path.join(weights, "detection_model.keras"))

    def test_writes_a_calibrated_threshold(self, workspace, figshare_dir):
        from training.tracks.detection import train_detection

        weights = workspace.weights
        train_detection(data_dir=figshare_dir, epochs=1, resume=False, seed=1234)

        config = json.loads(
            (Path(weights) / "detection_inference_config.json").read_text(encoding="utf-8")
        )
        assert 0.0 <= config["threshold"] <= 1.0
        assert "recall_floor_met" in config["validation_metrics"]

    def test_resume_skips_retraining(self, workspace, figshare_dir):
        from training.tracks.detection import train_detection

        train_detection(data_dir=figshare_dir, epochs=1, resume=False, seed=7)
        _, history, _ = train_detection(
            data_dir=figshare_dir, epochs=1, resume=True, seed=7
        )
        # The requested epoch budget is already satisfied, so no refit happens.
        assert history is None

    def test_no_resume_starts_fresh(self, workspace, figshare_dir):
        from training.tracks.detection import train_detection

        train_detection(data_dir=figshare_dir, epochs=1, resume=False, seed=7)
        _, history, _ = train_detection(
            data_dir=figshare_dir, epochs=1, resume=False, seed=7
        )
        assert history is not None

    def test_patient_level_split_path(self, workspace, figshare_dir):
        from training.tracks.detection import train_detection

        _, _, metrics = train_detection(
            data_dir=figshare_dir, epochs=1, resume=False, patient_level=True, seed=5
        )
        assert "accuracy" in metrics


# ── Classification ───────────────────────────────────────────────────────

@pytest.mark.slow
class TestClassifierTrainer:
    def test_runs_and_exports_artifacts(self, workspace, figshare_dir):
        from training.tracks.classifier import train_classifier

        weights = workspace.weights
        model, history, metrics = train_classifier(
            data_dir=figshare_dir, epochs=1, resume=False, seed=1234
        )
        assert model.output_shape == (None, 4)
        assert history is not None
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert os.path.exists(os.path.join(weights, "classifier_model.keras"))


# ── Segmentation ─────────────────────────────────────────────────────────

class TestSegmentationTrainer:
    def test_runs_and_exports_artifacts(self, workspace, brats_dir):
        from training.tracks.segmentation import train_segmentation

        weights = workspace.weights
        model, history, metrics = train_segmentation(
            data_dir=brats_dir, epochs=1, resume=False, seed=1234
        )
        assert model.output_shape == (None, IMG, IMG, 1)
        assert history is not None
        assert set(metrics) == {"mean_dice", "mean_iou"}
        assert os.path.exists(os.path.join(weights, "segmentation_model.keras"))

    def test_dice_is_in_range(self, workspace, brats_dir):
        from training.tracks.segmentation import train_segmentation

        _, _, metrics = train_segmentation(
            data_dir=brats_dir, epochs=1, resume=False, seed=1
        )
        assert 0.0 <= metrics["mean_dice"] <= 1.0
        assert 0.0 <= metrics["mean_iou"] <= 1.0

    def test_resume_from_checkpoint(self, workspace, brats_dir):
        """Guards the load_weights(.keras) checkpoint round-trip."""
        from training.tracks.segmentation import train_segmentation

        train_segmentation(data_dir=brats_dir, epochs=1, resume=False, seed=3)
        _, history, metrics = train_segmentation(
            data_dir=brats_dir, epochs=2, resume=True, seed=3
        )
        assert history is not None
        assert "mean_dice" in metrics


# ── GAN ──────────────────────────────────────────────────────────────────

class TestGanTrainer:
    def test_runs_and_exports_artifacts(self, workspace, figshare_dir):
        from training.tracks.gan import train_gan

        weights = workspace.weights
        generator, discriminator, loss_logger, fid, fs = train_gan(
            data_dir=figshare_dir, gan_type="conditional", epochs=1,
            resume=False, seed=1234,
        )
        assert discriminator.output_shape[-1] == 1
        assert loss_logger is not None
        assert isinstance(fid, list) and isinstance(fs, list)
        assert os.path.exists(
            os.path.join(weights, "generator_conditional.keras")
        )

    def test_state_is_persisted(self, workspace, figshare_dir):
        from training.state import GANState
        from training.tracks.gan import train_gan

        assert workspace.weights  # outputs were redirected away from the repo
        train_gan(data_dir=figshare_dir, gan_type="conditional", epochs=1,
                  resume=False, seed=99)

        state = GANState("conditional")
        assert state.state["last_epoch"] >= 0
        assert state.state["reproducibility"]["seed"] == 99
        # Non-empty loss history proves the loop actually stepped.
        assert len(state.state["g_losses"]) >= 1

    @pytest.mark.parametrize("gan_type", ["baseline", "dcgan", "conditional"])
    def test_supported_gan_types(self, workspace, figshare_dir, gan_type):
        from training.tracks.gan import train_gan

        generator, _, _, _, _ = train_gan(
            data_dir=figshare_dir, gan_type=gan_type, epochs=1,
            resume=False, seed=2,
        )
        assert callable(generator)

    def test_unknown_type_rejected(self, workspace, figshare_dir):
        from training.tracks.gan import train_gan

        with pytest.raises(ValueError, match="Unknown GAN type"):
            train_gan(data_dir=figshare_dir, gan_type="nope", epochs=1, resume=False)


# ── v2 GAN ───────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestGanV2Trainer:
    def test_runs_and_persists_ema(self, workspace, figshare_dir):
        from training.state import GANState
        from training.tracks.gan_v2 import train_gan_v2

        weights = workspace.weights
        generator, discriminator, _, _, _ = train_gan_v2(
            data_dir=figshare_dir, epochs=1, resume=False, seed=1234
        )
        # Subclassed models expose no .output_shape in Keras 3; probe the graph.
        logit = discriminator(
            [tf.zeros((1, IMG, IMG, 1)), tf.one_hot([0], 4)], training=False
        )
        assert tuple(logit.shape) == (1, 1)
        assert callable(generator)
        assert os.path.exists(
            os.path.join(weights, "generator_v2.weights.h5")
        )
        assert os.path.exists(GANState("v2").ema_ckpt())

    def test_w_distance_uses_its_own_state_slot(self, workspace, figshare_dir):
        from training.state import GANState
        from training.tracks.gan_v2 import train_gan_v2

        train_gan_v2(data_dir=figshare_dir, epochs=1, resume=False, seed=4)
        state = GANState("v2")
        assert "w_distances" in state.state
        # Accuracies must not be written into the Wasserstein slot.
        assert state.state.get("d_accs") in (None, [])
        assert len(state.state["w_distances"]) >= 1
