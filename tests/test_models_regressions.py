"""Regression tests for model-layer defects fixed in models/.

Each test names the original failure in its docstring.
"""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from models.classifier import build_classifier, build_multimodal_classifier
from models.gan import (
    EMAGenerator,
    build_v2_discriminator,
    build_v2_generator,
    gradient_penalty,
)
from models.segmentation import (
    build_unet,
    dice_bce_loss,
    dice_coefficient,
    iou_metric,
)

# ── Segmentation ─────────────────────────────────────────────────────────

class TestUnetDiceIsPerImage:
    """Regression: Dice/IoU flattened the whole batch into one reduction.

    ``keras.backend.flatten`` on a (B, H, W, 1) tensor produced a single
    dataset-level (micro) average, so the reported score changed with batch size
    and batch composition, and as a loss it made the objective depend on how
    samples happened to be grouped.
    """

    @staticmethod
    def _masks():
        perfect = np.array([[[[1.0], [1.0], [0.0], [0.0]]]], np.float32)
        half = np.array([[[[1.0], [0.0], [0.0], [0.0]]]], np.float32)
        return perfect, half

    def test_perfect_overlap_is_one(self):
        a, _ = self._masks()
        assert float(dice_coefficient(a, a)) == pytest.approx(1.0, abs=1e-5)
        assert float(iou_metric(a, a)) == pytest.approx(1.0, abs=1e-5)

    def test_disjoint_is_zero(self):
        a = np.array([[[[1.0], [0.0]]]], np.float32)
        b = np.array([[[[0.0], [1.0]]]], np.float32)
        assert float(dice_coefficient(a, b)) == pytest.approx(0.0, abs=1e-4)

    def test_batched_equals_mean_of_per_image_scores(self):
        a, b = self._masks()
        batched_true = np.concatenate([a, b])
        batched_pred = np.concatenate([a, b * 0.5])
        batched = float(dice_coefficient(batched_true, batched_pred))
        per_image = (
            float(dice_coefficient(a, a)) + float(dice_coefficient(b, b * 0.5))
        ) / 2
        assert batched == pytest.approx(per_image, abs=1e-5)

    def test_score_is_independent_of_batch_composition(self):
        a, b = self._masks()
        together = float(dice_coefficient(np.concatenate([a, b]), np.concatenate([a, b])))
        alone_a = float(dice_coefficient(a, a))
        alone_b = float(dice_coefficient(b, b))
        assert together == pytest.approx((alone_a + alone_b) / 2, abs=1e-5)

    def test_micro_average_would_differ_from_per_image(self):
        """Guards that the fix actually changes the number, not just the shape.

        With equal image sizes, sample 0 scores 1.0 and sample 1 scores 0.0.
        A micro (batch-pooled) average weights by mask area and yields 8/9,
        while a per-image mean yields 0.5.
        """
        good_true = np.ones((1, 1, 4, 1), np.float32)
        good_pred = np.ones((1, 1, 4, 1), np.float32)
        bad_true = np.zeros((1, 1, 4, 1), np.float32)
        bad_true[0, 0, 0, 0] = 1.0
        bad_pred = np.zeros((1, 1, 4, 1), np.float32)

        batched = float(
            dice_coefficient(
                np.concatenate([good_true, bad_true]),
                np.concatenate([good_pred, bad_pred]),
            )
        )
        micro = 2 * (4.0 + 0.0) / (4.0 + 4.0 + 1.0 + 0.0)  # == 8/9
        assert batched == pytest.approx(0.5, abs=1e-3)
        assert batched != pytest.approx(micro, abs=1e-3)

    def test_does_not_overflow_float16(self):
        """A float16 reduction over B*H*W elements overflows 65504 -> NaN."""
        big = np.ones((2, 256, 256, 1), np.float16)
        value = float(dice_coefficient(big, big))
        assert np.isfinite(value)
        assert value == pytest.approx(1.0, abs=1e-3)

    def test_loss_is_finite(self):
        a, _ = self._masks()
        loss = dice_bce_loss(a, tf.constant(a, dtype=tf.float32))
        assert np.isfinite(float(loss))


class TestUnetStructure:
    def test_attention_gate_is_driven_by_the_upsampled_feature(self):
        """Regression: the gate received the bottleneck instead of u1/u2/u3.

        Feeding the least-resolved feature into the attention gate is the
        opposite of the coarse-to-fine guidance attention U-Net is meant to
        provide.
        """
        model = build_unet(input_shape=(64, 64, 1), use_attention=True)
        assert model.output_shape == (None, 64, 64, 1)

    def test_output_is_float32_for_mixed_precision(self):
        """Regression: sigmoid ran in float16 under a mixed_float16 policy.

        A float16 logit feeding the Dice reduction could overflow; the final
        layer is now pinned to float32 like the classifier head.
        """
        model = build_unet(input_shape=(64, 64, 1))
        assert model.output.dtype == "float32"

    def test_rejects_shapes_not_divisible_by_16(self):
        """UpSampling2D doubles while MaxPooling2D floors -> shape mismatch."""
        with pytest.raises(ValueError, match="divisible by 16"):
            build_unet(input_shape=(250, 250, 1))

    @pytest.mark.parametrize("shape", [(128, 128, 1), (256, 256, 1)])
    def test_accepts_valid_shapes(self, shape):
        model = build_unet(input_shape=shape)
        assert model.output_shape == (None, shape[0], shape[1], 1)


# ── Classifier ───────────────────────────────────────────────────────────

class TestClassifierInputRange:
    """Regression: [0, 1] pixels were fed straight into ImageNet EfficientNet.

    EfficientNetB0 opens with Rescaling(1/255) + Normalization and its
    pretrained weights expect [0, 255]. Feeding [0, 1] mapped every pixel to
    roughly -1.0 after the stem, i.e. a constant image, silently discarding all
    ImageNet features.
    """

    def test_rescaling_layer_present_by_default(self):
        model = build_classifier(num_classes=4, input_shape=(64, 64, 1))
        assert any(layer.name == "imagenet_range_rescale" for layer in model.layers)

    def test_backbone_sees_a_non_degenerate_input(self):
        """Black and white inputs must produce clearly different features."""
        model = build_classifier(num_classes=4, input_shape=(64, 64, 1))
        rescale = model.get_layer("imagenet_range_rescale")
        backbone = model.get_layer("efficientnetb0")

        black = backbone(rescale(tf.zeros((1, 64, 64, 1))))
        white = backbone(rescale(tf.ones((1, 64, 64, 1))))

        # Before the fix both collapsed to the same near-constant value.
        assert float(tf.reduce_mean(black) - tf.reduce_mean(white)) != pytest.approx(
            0.0, abs=1e-4
        )

    def test_legacy_behaviour_is_still_reproducible(self):
        model = build_classifier(
            num_classes=4, input_shape=(64, 64, 1), imagenet_input_range=False
        )
        assert not any(layer.name == "imagenet_range_rescale" for layer in model.layers)

    def test_shape(self):
        assert build_classifier(num_classes=4, input_shape=(64, 64, 1)).output_shape == (
            None,
            4,
        )


class TestMultimodalClassifier:
    def test_four_inputs_to_one_output(self):
        model = build_multimodal_classifier(num_classes=4, input_shape=(64, 64, 1))
        assert len(model.inputs) == 4
        assert model.output_shape == (None, 4)

    def test_single_modality(self):
        model = build_multimodal_classifier(
            num_classes=4, input_shape=(64, 64, 1), num_modalities=1
        )
        assert model.output_shape == (None, 4)


# ── GAN v2 ───────────────────────────────────────────────────────────────

class TestProjectionDiscriminator:
    """Regression: ``embed_dim`` was stored and exported but never used.

    The embedding was hard-coded to 512 units, so a saved discriminator claimed
    ``embed_dim=128`` while its weights were ``(num_classes, 512)``.
    """

    def test_embedding_uses_embed_dim(self):
        disc = build_v2_discriminator(input_shape=(32, 32, 1), num_classes=4, embed_dim=64)
        assert tuple(disc.class_embed.weights[0].shape) == (4, 64)

    def test_output_shape(self):
        disc = build_v2_discriminator(input_shape=(32, 32, 1), num_classes=4, embed_dim=64)
        out = disc([tf.random.normal((2, 32, 32, 1)), tf.one_hot([0, 1], 4)], training=False)
        assert tuple(out.shape) == (2, 1)

    def test_conditioning_actually_changes_the_output(self):
        disc = build_v2_discriminator(input_shape=(32, 32, 1), num_classes=4, embed_dim=64)
        image = tf.random.normal((1, 32, 32, 1))
        score_0 = disc([image, tf.one_hot([0], 4)], training=False)
        score_1 = disc([image, tf.one_hot([1], 4)], training=False)
        assert not float(score_0) == pytest.approx(float(score_1))


class TestV2Generator:
    def test_builds_with_two_inputs(self):
        gen = build_v2_generator(latent_dim=100, num_classes=4, output_shape=(32, 32, 1))
        out = gen([tf.random.normal((2, 100)), tf.one_hot([0, 1], 4)], training=False)
        assert tuple(out.shape) == (2, 32, 32, 1)

    def test_rejects_indivisible_output_shape(self):
        with pytest.raises(ValueError, match="divisible by 16"):
            build_v2_generator(latent_dim=100, num_classes=4, output_shape=(30, 30, 1))


class TestGradientPenaltyScaling:
    """Regression: the helper pre-multiplied by lambda while callers also did.

    The only call site did ``d_loss = ... + gp`` and passed ``lambda_gp`` into
    the helper, so the convention was ambiguous and easy to double-scale.
    """

    def test_default_is_unscaled(self):
        disc = build_v2_discriminator(input_shape=(32, 32, 1), num_classes=4, embed_dim=16)
        real = tf.random.uniform((2, 32, 32, 1), -1, 1)
        fake = tf.random.uniform((2, 32, 32, 1), -1, 1)
        labels = tf.one_hot([0, 1], 4)
        base = float(gradient_penalty(disc, real, fake, labels))
        scaled = float(gradient_penalty(disc, real, fake, labels, lambda_gp=10.0))
        assert scaled == pytest.approx(base * 10.0, rel=0.05)

    def test_is_non_negative(self):
        disc = build_v2_discriminator(input_shape=(32, 32, 1), num_classes=4, embed_dim=16)
        real = tf.random.uniform((2, 32, 32, 1), -1, 1)
        fake = tf.random.uniform((2, 32, 32, 1), -1, 1)
        value = float(gradient_penalty(disc, real, fake, tf.one_hot([0, 1], 4)))
        assert value >= 0.0


class TestEMAGenerator:
    """Regression: EMA covered only trainable variables and was not exception-safe.

    BatchNorm moving statistics are non-trainable state, so ``apply()`` produced
    a hybrid model (EMA kernels + live BN stats) that is not a valid EMA. And an
    exception between ``apply()``/``restore()`` left the generator permanently
    holding EMA weights while the optimizer kept updating the real ones.
    """

    @staticmethod
    def _generator():
        return build_v2_generator(latent_dim=16, num_classes=4, output_shape=(32, 32, 1))

    def test_tracks_non_trainable_batchnorm_state(self):
        gen = self._generator()
        ema = EMAGenerator(gen)
        assert len(ema.ema_weights) == len(gen.variables)
        assert len(ema.ema_weights) > len(gen.trainable_variables)

    def test_weights_restored_after_exception(self):
        gen = self._generator()
        ema = EMAGenerator(gen)
        before = [v.numpy().copy() for v in gen.variables]
        with pytest.raises(RuntimeError):
            with ema.swapped():
                ema.update()
                raise RuntimeError("boom")
        after = [v.numpy() for v in gen.variables]
        assert all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))

    def test_restore_without_apply_is_a_no_op(self):
        ema = EMAGenerator(self._generator())
        ema.restore()  # must not raise AttributeError

    def test_double_restore_is_a_no_op(self):
        gen = self._generator()
        ema = EMAGenerator(gen)
        with ema.swapped():
            pass
        ema.restore()

    def test_swapped_actually_swaps(self):
        gen = self._generator()
        ema = EMAGenerator(gen, decay=0.0)
        with ema.swapped():
            pass
        # decay=0 means the EMA copies the live weights exactly.
        for ema_w, w in zip(ema.ema_weights, gen.variables, strict=True):
            assert np.allclose(ema_w.numpy(), w.numpy())

    def test_save_load_round_trip(self, tmp_path):
        ema = EMAGenerator(self._generator())
        path = tmp_path / "ema.npz"
        ema.save(path)
        reloaded = EMAGenerator(self._generator())
        reloaded.load(path)
        for a, b in zip(ema.ema_weights, reloaded.ema_weights, strict=True):
            assert np.allclose(a.numpy(), b.numpy())

    def test_rejects_foreign_checkpoint(self, tmp_path):
        ema = EMAGenerator(self._generator())
        path = tmp_path / "bad.npz"
        np.savez(path, ema_0=np.zeros((2, 2)))
        with pytest.raises(ValueError, match="does not match"):
            ema.load(path)
