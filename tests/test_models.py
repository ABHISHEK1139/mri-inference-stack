"""Tests for model architecture construction and output shapes.

These tests build each model WITHOUT loading weights, verifying that
architecture code produces the expected output shapes and doesn't crash.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tensorflow")


class TestDetectionModel:
    def test_output_shape(self):
        from models.detection import build_detection_model

        model = build_detection_model(input_shape=(224, 224, 1))
        assert model.output_shape == (None, 1)

    def test_input_shape(self):
        from models.detection import build_detection_model

        model = build_detection_model(input_shape=(224, 224, 1))
        assert model.input_shape == (None, 224, 224, 1)


class TestClassifierModel:
    def test_output_shape(self):
        from models.classifier import build_classifier

        model = build_classifier(num_classes=4, input_shape=(224, 224, 1))
        assert model.output_shape == (None, 4)

    def test_custom_classes(self):
        from models.classifier import build_classifier

        model = build_classifier(num_classes=3, input_shape=(224, 224, 1))
        assert model.output_shape == (None, 3)


class TestSegmentationModel:
    def test_output_shape_default(self):
        from models.segmentation import build_unet

        model = build_unet(input_shape=(256, 256, 1))
        assert model.output_shape == (None, 256, 256, 1)

    def test_output_shape_smaller(self):
        from models.segmentation import build_unet

        model = build_unet(input_shape=(128, 128, 1))
        assert model.output_shape == (None, 128, 128, 1)

    def test_without_attention(self):
        from models.segmentation import build_unet

        model = build_unet(input_shape=(128, 128, 1), use_attention=False)
        assert model.output_shape == (None, 128, 128, 1)

    def test_without_residual(self):
        from models.segmentation import build_unet

        model = build_unet(input_shape=(128, 128, 1), use_residual=False)
        assert model.output_shape == (None, 128, 128, 1)


class TestGeneratorOutputValidation:
    def test_valid_output_shape(self):
        from models.gan import build_v2_generator

        gen = build_v2_generator(output_shape=(128, 128, 1))
        assert gen is not None

    def test_invalid_output_shape_raises(self):
        from models.gan import build_v2_generator

        with pytest.raises(ValueError, match="divisible by 16"):
            build_v2_generator(output_shape=(100, 100, 1))

    def test_stylegan_invalid_shape_raises(self):
        from models.gan import build_stylegan_generator

        with pytest.raises(ValueError, match="divisible by 16"):
            build_stylegan_generator(output_shape=(100, 100, 1))


class TestBaselineModels:
    def test_baseline_classifier_shape(self):
        from models.classifier import build_classifier_baseline

        model = build_classifier_baseline(num_classes=4, input_shape=(224, 224, 1))
        assert model.output_shape == (None, 4)
        assert model.input_shape == (None, 224, 224, 1)

    def test_baseline_detection_shape(self):
        from models.detection import build_detection_baseline

        model = build_detection_baseline(input_shape=(224, 224, 1))
        assert model.output_shape == (None, 1)
        assert model.input_shape == (None, 224, 224, 1)
