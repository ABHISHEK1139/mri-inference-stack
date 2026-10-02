"""Tests for the GAN trainer configuration object.

The trainer previously read more than a dozen environment variables inline.
These pin the documented defaults, the environment overrides, and the
recovery-mode preset so refactoring the knobs cannot silently change behaviour.
"""

from __future__ import annotations

import pytest

from training.tracks.gan_config import GanTrainerConfig

GAN_VARS = [
    "GAN_FID_EVAL_FREQ",
    "GAN_EARLY_STOP_PATIENCE",
    "GAN_TARGET_FID",
    "GAN_TARGET_FS",
    "GAN_D_STEPS",
    "GAN_G_STEPS",
    "GAN_RECOVERY_MODE",
    "GAN_DIVERSITY_WEIGHT",
    "GAN_CLASS_GUIDANCE_WEIGHT",
    "GAN_PREVIEW_FREQ",
    "GAN_SHAKE_ON_COLLAPSE",
    "GAN_SHAKE_STD",
    "GAN_GRAD_CLIP_NORM",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in GAN_VARS:
        monkeypatch.delenv(name, raising=False)


class TestDefaults:
    def test_matches_documented_defaults(self):
        config = GanTrainerConfig.from_env()
        assert config.fid_eval_freq == 10
        assert config.early_stop_patience == 3
        assert config.d_steps == 1
        assert config.g_steps == 2
        assert config.preview_freq == 5
        assert config.shake_std == pytest.approx(0.0005)
        assert config.grad_clip_norm == pytest.approx(5.0)
        assert config.recovery_mode is False
        assert config.diversity_weight == pytest.approx(0.0)
        assert config.class_guidance_weight == pytest.approx(0.0)

    def test_fid_eval_freq_argument_used_as_fallback(self):
        assert GanTrainerConfig.from_env(fid_eval_freq=25).fid_eval_freq == 25

    def test_describe_is_a_single_line(self):
        text = GanTrainerConfig.from_env().describe()
        assert "\n" not in text
        assert "precision=float32" in text


class TestEnvironmentOverrides:
    def test_reads_each_variable(self, monkeypatch):
        monkeypatch.setenv("GAN_FID_EVAL_FREQ", "7")
        monkeypatch.setenv("GAN_EARLY_STOP_PATIENCE", "9")
        monkeypatch.setenv("GAN_TARGET_FID", "42.5")
        monkeypatch.setenv("GAN_TARGET_FS", "3.5")
        monkeypatch.setenv("GAN_D_STEPS", "3")
        monkeypatch.setenv("GAN_G_STEPS", "4")
        monkeypatch.setenv("GAN_DIVERSITY_WEIGHT", "0.02")
        monkeypatch.setenv("GAN_CLASS_GUIDANCE_WEIGHT", "0.4")
        monkeypatch.setenv("GAN_PREVIEW_FREQ", "2")
        monkeypatch.setenv("GAN_SHAKE_STD", "0.01")
        monkeypatch.setenv("GAN_GRAD_CLIP_NORM", "3.0")

        config = GanTrainerConfig.from_env()
        assert config.fid_eval_freq == 7
        assert config.early_stop_patience == 9
        assert config.target_fid == pytest.approx(42.5)
        assert config.target_fs == pytest.approx(3.5)
        assert config.d_steps == 3
        assert config.g_steps == 4
        assert config.diversity_weight == pytest.approx(0.02)
        assert config.class_guidance_weight == pytest.approx(0.4)
        assert config.preview_freq == 2
        assert config.shake_std == pytest.approx(0.01)
        assert config.grad_clip_norm == pytest.approx(3.0)

    @pytest.mark.parametrize("truthy", ["1", "true", "TRUE", "yes", "on"])
    def test_boolean_truthy_spellings(self, monkeypatch, truthy):
        monkeypatch.setenv("GAN_RECOVERY_MODE", truthy)
        assert GanTrainerConfig.from_env().recovery_mode is True

    @pytest.mark.parametrize("falsy", ["0", "false", "no", "off", ""])
    def test_boolean_falsy_spellings(self, monkeypatch, falsy):
        monkeypatch.setenv("GAN_SHAKE_ON_COLLAPSE", falsy)
        assert GanTrainerConfig.from_env().shake_on_collapse is False

    def test_invalid_int_falls_back_to_default(self, monkeypatch):
        """Regression: a malformed value used to raise ValueError and kill the run."""
        monkeypatch.setenv("GAN_G_STEPS", "not-a-number")
        assert GanTrainerConfig.from_env().g_steps == 2

    def test_invalid_float_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("GAN_SHAKE_STD", "abc")
        assert GanTrainerConfig.from_env().shake_std == pytest.approx(0.0005)

    def test_empty_value_is_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("GAN_TARGET_FID", "")
        assert GanTrainerConfig.from_env().target_fid == pytest.approx(0.0)

    def test_step_counts_are_clamped_to_at_least_one(self, monkeypatch):
        monkeypatch.setenv("GAN_D_STEPS", "0")
        monkeypatch.setenv("GAN_G_STEPS", "-4")
        config = GanTrainerConfig.from_env()
        assert config.d_steps == 1
        assert config.g_steps == 1

    def test_preview_freq_is_clamped_to_at_least_one(self, monkeypatch):
        monkeypatch.setenv("GAN_PREVIEW_FREQ", "0")
        assert GanTrainerConfig.from_env().preview_freq == 1

    def test_negative_fid_freq_is_clamped_to_zero(self, monkeypatch):
        """A negative frequency modulo is never zero, so the guard is required."""
        monkeypatch.setenv("GAN_FID_EVAL_FREQ", "-3")
        assert GanTrainerConfig.from_env().fid_eval_freq == 0


class TestRecoveryModePreset:
    def test_disabled_is_a_no_op(self):
        config = GanTrainerConfig.from_env()
        assert config.with_recovery_overrides() is config

    def test_applies_the_preset(self, monkeypatch):
        monkeypatch.setenv("GAN_RECOVERY_MODE", "1")
        config = GanTrainerConfig.from_env()
        assert config.g_steps >= 4
        assert config.preview_freq == 1
        assert config.shake_on_collapse is True
        assert config.diversity_weight == pytest.approx(0.03)
        assert config.class_guidance_weight == pytest.approx(0.35)

    def test_respects_explicitly_set_values(self, monkeypatch):
        """An explicit setting must win over the preset."""
        monkeypatch.setenv("GAN_RECOVERY_MODE", "1")
        monkeypatch.setenv("GAN_DIVERSITY_WEIGHT", "0.11")
        monkeypatch.setenv("GAN_G_STEPS", "8")
        config = GanTrainerConfig.from_env()
        assert config.diversity_weight == pytest.approx(0.11)
        assert config.g_steps == 8

    def test_from_env_applies_the_preset(self, monkeypatch):
        monkeypatch.setenv("GAN_RECOVERY_MODE", "true")
        assert GanTrainerConfig.from_env().g_steps >= 4


class TestDataclassBehaviour:
    def test_replace_returns_a_new_instance(self):
        from dataclasses import replace

        original = GanTrainerConfig()
        updated = replace(original, g_steps=9)
        assert original.g_steps == 2
        assert updated.g_steps == 9

    def test_config_is_hashable_by_value_not_identity(self):
        # Dataclasses are compared by value, which is what the trainer relies on
        # when it re-derives a config after a resume.
        assert GanTrainerConfig() == GanTrainerConfig()

    def test_every_documented_field_is_present(self):
        fields = {
            "fid_eval_freq",
            "early_stop_patience",
            "target_fid",
            "target_fs",
            "d_steps",
            "g_steps",
            "recovery_mode",
            "diversity_weight",
            "class_guidance_weight",
            "preview_freq",
            "shake_on_collapse",
            "shake_std",
            "grad_clip_norm",
        }
        assert fields == {f.name for f in GanTrainerConfig.__dataclass_fields__.values()}
