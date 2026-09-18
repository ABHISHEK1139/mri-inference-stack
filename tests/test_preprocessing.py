"""Tests for preprocessing contract consistency across all tracks."""

from __future__ import annotations

import numpy as np
from PIL import Image

from preprocessing import (
    preprocess_classifier,
    preprocess_detection,
    preprocess_gan,
    preprocess_segmentation,
    preprocess_segmentation_mask,
)


def _make_test_image(width: int = 300, height: int = 300) -> Image.Image:
    """Create a synthetic grayscale test image."""
    rng = np.random.default_rng(42)
    array = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    return Image.fromarray(array, mode="L")


class TestDetectionPreprocessor:
    def test_output_shape(self):
        image = _make_test_image()
        result = preprocess_detection(image, target_size=(224, 224))
        assert result.shape == (1, 224, 224, 1)

    def test_output_range(self):
        image = _make_test_image()
        result = preprocess_detection(image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_dtype(self):
        image = _make_test_image()
        result = preprocess_detection(image)
        assert result.dtype == np.float32


class TestClassifierPreprocessor:
    def test_output_shape(self):
        image = _make_test_image()
        result = preprocess_classifier(image, target_size=(224, 224))
        assert result.shape == (1, 224, 224, 1)

    def test_output_range(self):
        image = _make_test_image()
        result = preprocess_classifier(image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestSegmentationPreprocessor:
    def test_output_shape(self):
        image = _make_test_image()
        result = preprocess_segmentation(image, target_size=(256, 256))
        assert result.shape == (1, 256, 256, 1)

    def test_output_range(self):
        image = _make_test_image()
        result = preprocess_segmentation(image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestSegmentationMaskPreprocessor:
    def test_binary_output(self):
        image = _make_test_image()
        result = preprocess_segmentation_mask(image, target_size=(256, 256))
        unique_values = set(np.unique(result))
        assert unique_values <= {0.0, 1.0}

    def test_output_shape(self):
        image = _make_test_image()
        result = preprocess_segmentation_mask(image, target_size=(256, 256))
        assert result.shape == (1, 256, 256, 1)


class TestGANPreprocessor:
    def test_output_shape(self):
        image = _make_test_image()
        result = preprocess_gan(image, target_size=(128, 128))
        assert result.shape == (1, 128, 128, 1)

    def test_output_range(self):
        image = _make_test_image()
        result = preprocess_gan(image)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_output_dtype(self):
        image = _make_test_image()
        result = preprocess_gan(image)
        assert result.dtype == np.float32
