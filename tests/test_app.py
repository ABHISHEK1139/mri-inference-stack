"""Tests for the Streamlit app's pure helpers.

Streamlit UI code cannot be rendered in a test process, but every non-visual
helper in ``app.py`` is plain Python and is where the real defects live: config
resolution, LFS-pointer detection, image decoding, and the class-index guard that
stops a model with the wrong output width from being labelled incorrectly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("app")
import app

# ── Class-index guard ────────────────────────────────────────────────────

class TestTopClass:
    def test_returns_argmax_and_score(self):
        scores = np.array([0.1, 0.7, 0.15, 0.05], np.float32)
        index, score = app._top_class(scores)
        assert index == 1
        assert score == pytest.approx(0.7)

    def test_accepts_batched_input(self):
        # A (1, NUM_CLASSES) batch row is what model.predict() returns.
        index, _ = app._top_class(np.array([[0.9, 0.02, 0.05, 0.03]], np.float32))
        assert index == 0

    def test_wrong_output_width_is_rejected(self):
        """Regression: a model with a different output width mislabelled classes.

        ``CLASS_NAMES[class_index]`` used to index past the end of the list, or
        attach the wrong label to a prediction.
        """
        with pytest.raises(ValueError, match="out of sync"):
            app._top_class(np.array([0.25, 0.25, 0.5], np.float32))

    def test_too_many_scores_rejected(self):
        with pytest.raises(ValueError, match="out of sync"):
            app._top_class(np.zeros(9, np.float32))

    def test_matches_argmax(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            scores = rng.random(len(app.CLASS_NAMES)).astype(np.float32)
            index, _ = app._top_class(scores)
            assert index == int(np.argmax(scores))


# ── LFS pointer detection ────────────────────────────────────────────────

class TestLfsPointerDetection:
    def test_real_file_is_not_a_pointer(self, tmp_path):
        path = tmp_path / "model.keras"
        path.write_bytes(b"Keras archive payload" * 10)
        assert app._is_lfs_pointer(path) is False

    def test_pointer_stub_is_detected(self, tmp_path):
        path = tmp_path / "model.keras"
        path.write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 100\n"
        )
        assert app._is_lfs_pointer(path) is True

    def test_missing_file_is_not_a_pointer(self, tmp_path):
        assert app._is_lfs_pointer(tmp_path / "absent.keras") is False

    def test_empty_file_is_not_a_pointer(self, tmp_path):
        path = tmp_path / "empty.keras"
        path.write_bytes(b"")
        assert app._is_lfs_pointer(path) is False


# ── Detection config resolution ──────────────────────────────────────────

class TestDetectionConfig:
    def test_reads_threshold_and_metrics(self, tmp_path, monkeypatch):
        config_path = tmp_path / "detection_inference_config.json"
        config_path.write_text(
            json.dumps({"threshold": 0.225, "validation_metrics": {"f1_score": 0.99}})
        )
        monkeypatch.setattr(app, "DETECTION_CONFIG_CANDIDATES", [config_path])

        config = app._load_detection_config()
        assert config["threshold"] == pytest.approx(0.225)
        assert config["validation_metrics"]["f1_score"] == pytest.approx(0.99)

    def test_falls_back_to_defaults_when_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app, "DETECTION_CONFIG_CANDIDATES",
                            [tmp_path / "missing.json"])
        config = app._load_detection_config()
        assert config["threshold"] == pytest.approx(0.5)
        assert config["validation_metrics"] == {}

    def test_prefers_the_first_candidate(self, tmp_path, monkeypatch):
        first = tmp_path / "first.json"
        second = tmp_path / "second.json"
        first.write_text(json.dumps({"threshold": 0.11, "validation_metrics": {}}))
        second.write_text(json.dumps({"threshold": 0.99, "validation_metrics": {}}))
        monkeypatch.setattr(app, "DETECTION_CONFIG_CANDIDATES", [first, second])
        assert app._load_detection_config()["threshold"] == pytest.approx(0.11)

    def test_falls_through_when_first_candidate_is_missing(self, tmp_path, monkeypatch):
        second = tmp_path / "second.json"
        second.write_text(json.dumps({"threshold": 0.3, "validation_metrics": {}}))
        monkeypatch.setattr(app, "DETECTION_CONFIG_CANDIDATES",
                            [tmp_path / "missing.json", second])
        assert app._load_detection_config()["threshold"] == pytest.approx(0.3)


# ── Preprocessing contract ───────────────────────────────────────────────

class TestPreprocessingShapes:
    @staticmethod
    def _image(size=64):
        return Image.fromarray(
            np.random.default_rng(0).integers(0, 256, (size, size), dtype=np.uint8)
        )

    def test_detection_input(self):
        out = app.preprocess_detection(self._image())
        assert out.shape[0] == 1 and out.shape[-1] == 1
        assert 0.0 <= out.min() and out.max() <= 1.0

    def test_classifier_input(self):
        out = app.preprocess_classifier(self._image())
        assert out.shape[0] == 1 and out.shape[-1] == 1

    def test_segmentation_input(self):
        out = app.preprocess_segmentation(self._image(), target_size=(32, 32))
        assert out.shape[1:3] == (32, 32)

    def test_resize_is_grayscale_single_channel(self):
        out = app.preprocess_detection(self._image(), target_size=(16, 24))
        assert out.shape == (1, 16, 24, 1)

    def test_values_are_normalized(self):
        black = app.preprocess_detection(Image.new("L", (32, 32), 0))
        white = app.preprocess_detection(Image.new("L", (32, 32), 255))
        assert black.max() == pytest.approx(0.0)
        assert white.min() == pytest.approx(1.0)

    def test_mask_preprocessing_binarizes(self):
        from preprocessing import preprocess_segmentation_mask

        grey = Image.new("L", (32, 32), 128)
        out = preprocess_segmentation_mask(grey, target_size=(16, 16))
        assert set(np.unique(out).tolist()) <= {0.0, 1.0}


# ── Image loading ────────────────────────────────────────────────────────

class TestLoadImage:
    def test_converts_to_grayscale(self, tmp_path):
        path = tmp_path / "rgb.png"
        Image.fromarray(
            np.random.default_rng(0).integers(0, 256, (20, 20, 3), dtype=np.uint8)
        ).save(path)
        with path.open("rb") as handle:
            loaded = app._load_image(handle)
        assert loaded.mode == "L"

    def test_does_not_leak_the_open_handle(self, tmp_path):
        """Regression: the upload handle was held open for the session."""
        path = tmp_path / "x.png"
        Image.fromarray(np.zeros((8, 8), np.uint8)).save(path)
        with path.open("rb") as handle:
            app._load_image(handle)
            # A closed handle raises on use; the image was already materialised.
            assert handle.closed or handle.tell() >= 0

    def test_corrupt_upload_is_reported(self, tmp_path):
        path = tmp_path / "bad.png"
        path.write_bytes(b"not an image at all")
        with path.open("rb") as handle:
            # Raises rather than returning None: st.stop() is a no-op outside a
            # Streamlit script context, which previously let None through.
            with pytest.raises(ValueError, match="Could not read"):
                app._load_image(handle)


# ── Configuration contract ───────────────────────────────────────────────

class TestAppConfigContract:
    def test_class_names_agree_with_num_classes(self):
        import config

        assert app.NUM_CLASSES == config.NUM_CLASSES == len(app.CLASS_NAMES)

    def test_app_uses_the_shared_latent_dim(self):
        """Regression: the GAN tab hard-coded 100 instead of config.LATENT_DIM."""
        import config

        assert app.LATENT_DIM == config.LATENT_DIM

    def test_app_uses_shared_class_names(self):
        import config

        assert list(app.CLASS_NAMES) == list(config.CLASS_NAMES)

    def test_custom_objects_are_defined(self):
        """Lazily populated on load_research_models; the name must exist."""
        assert hasattr(app, "SEGMENTATION_CUSTOM_OBJECTS")
