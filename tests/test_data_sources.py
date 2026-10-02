"""Tests for dataset acquisition and the GAN-augmented classifier track."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

tf = pytest.importorskip("tensorflow")

# The GAN-augmented track generates synthetic images and trains a classifier on
# them, so these tests belong to the slow tier alongside the trainer suites.
pytestmark = pytest.mark.slow

CLASSES = ("glioma", "meningioma", "pituitary", "normal")
IMG = 32


# ── Dataset acquisition ──────────────────────────────────────────────────

class TestEnsureDatasets:
    @pytest.fixture
    def data_sources(self, tmp_path, monkeypatch):
        import training.data_sources as module

        raw = tmp_path / "raw"
        raw.mkdir()
        monkeypatch.setattr(module, "RAW_DIR", str(raw))
        return module, raw

    def test_existing_dataset_is_not_re_downloaded(self, data_sources):
        module, raw = data_sources
        (raw / "figshare").mkdir()
        (raw / "figshare" / "a.png").touch()

        calls = []
        monkey = module.download_dataset
        module.download_dataset = lambda name: calls.append(name)
        try:
            module.ensure_datasets(download_figshare=True, download_brats=False)
        finally:
            module.download_dataset = monkey
        assert calls == []

    def test_empty_directory_triggers_download(self, data_sources):
        module, raw = data_sources
        (raw / "figshare").mkdir()  # exists but empty

        calls = []

        def fake_download(name):
            calls.append(name)
            (raw / "figshare" / "a.png").touch()

        original = module.download_dataset
        module.download_dataset = fake_download
        try:
            module.ensure_datasets(download_figshare=True, download_brats=False)
        finally:
            module.download_dataset = original
        assert calls == ["figshare"]

    def test_download_failure_falls_back_to_kaggle(self, data_sources):
        """A failing direct download must attempt the Kaggle fallback."""
        module, raw = data_sources
        (raw / "figshare").mkdir()

        def boom(name):
            raise RuntimeError("network down")

        fallback = []
        original = module.download_dataset
        module.download_dataset = boom
        module._download_kaggle_alternative = lambda: fallback.append("kaggle")
        try:
            module.ensure_datasets(download_figshare=True, download_brats=False)
        finally:
            module.download_dataset = original
        assert fallback == ["kaggle"]

    def test_brats_flag_is_independent(self, data_sources):
        module, raw = data_sources
        (raw / "figshare").mkdir()
        (raw / "figshare" / "a.png").touch()

        calls = []
        original = module.download_dataset
        module.download_dataset = lambda name: calls.append(name)
        try:
            module.ensure_datasets(download_figshare=False, download_brats=True)
        finally:
            module.download_dataset = original
        assert calls == ["brats"]


# ── GAN-augmented classifier ─────────────────────────────────────────────

@pytest.mark.slow
class TestGanAugmentedClassifier:
    @pytest.fixture
    def workspace(self, tmp_path, monkeypatch):
        import types

        import config
        import training.runtime as runtime
        import training.state as state
        from training.tracks import classifier, gan_augmented, segmentation

        mapping = {
            "CHECKPOINT_DIR": "checkpoints",
            "LOG_DIR": "logs",
            "OUTPUT_DIR": "outputs",
            "WEIGHTS_DIR": "weights",
            "RAW_DIR": os.path.join("data", "raw"),
            "PROCESSED_DIR": os.path.join("data", "processed"),
        }
        created = {}
        for attr, rel in mapping.items():
            path = tmp_path / rel
            path.mkdir(parents=True, exist_ok=True)
            created[attr] = str(path)
            monkeypatch.setattr(config, attr, str(path), raising=False)
        for module in (runtime, state, classifier, gan_augmented, segmentation):
            for attr, value in created.items():
                if hasattr(module, attr):
                    monkeypatch.setattr(module, attr, value)

        small = type(classifier.IMG_CFG)(detection_size=(IMG, IMG),
                                         classifier_size=(IMG, IMG),
                                         segmentation_size=(IMG, IMG),
                                         gan_size=(IMG, IMG))
        for module in (runtime, classifier, gan_augmented):
            if hasattr(module, "IMG_CFG"):
                monkeypatch.setattr(module, "IMG_CFG", small)
        return types.SimpleNamespace(root=tmp_path, weights=created["WEIGHTS_DIR"])

    @pytest.fixture
    def figshare_dir(self, tmp_path):
        root = tmp_path / "figshare"
        rng = np.random.default_rng(0)
        for class_name in CLASSES:
            folder = root / "train" / class_name
            folder.mkdir(parents=True)
            for i in range(8):
                Image.fromarray(
                    rng.integers(0, 256, (IMG, IMG), dtype=np.uint8)
                ).save(folder / f"p{i:03d}_slice01.png")
        return str(root)

    def test_trains_baseline_and_augmented(self, workspace, figshare_dir):
        from config import LATENT_DIM, NUM_CLASSES
        from models.gan import build_conditional_generator
        from training.tracks.gan_augmented import train_classifier_with_gan

        generator = build_conditional_generator(
            latent_dim=LATENT_DIM, num_classes=NUM_CLASSES, output_shape=(IMG, IMG, 1)
        )
        _, aug_metrics, baseline_metrics = train_classifier_with_gan(
            generator, data_dir=figshare_dir, gan_type="conditional",
            ratio=0.5, epochs=1, resume=False,
        )
        assert 0.0 <= aug_metrics["accuracy"] <= 1.0
        assert 0.0 <= baseline_metrics["accuracy"] <= 1.0
        assert Path(workspace.weights, "classifier_augmented.keras").exists()
        assert Path(workspace.weights, "classifier_baseline.keras").exists()

    def test_rejects_invalid_ratio(self, workspace, figshare_dir):
        from data.dataset import mix_real_synthetic
        from training.tracks.gan_augmented import train_classifier_with_gan

        real = np.zeros((4, IMG, IMG, 1), np.float32)
        with pytest.raises(ValueError, match="ratio"):
            mix_real_synthetic(real, np.zeros(4, np.int32), real,
                               np.zeros(4, np.int32), ratio=1.5)
        assert callable(train_classifier_with_gan)
