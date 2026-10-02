"""Configuration for the v1 conditional GAN trainer.

The trainer previously read more than a dozen environment variables inline,
scattering the knobs across the function body and making it impossible to see
what was tunable without reading the whole loop. Collecting them here gives one
documented surface, and lets a caller construct a config directly in tests
instead of mutating ``os.environ``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring invalid int for %s=%r; using %d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring invalid float for %s=%r; using %s", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in _TRUTHY


@dataclass
class GanTrainerConfig:
    """Tunable knobs for the v1 GAN training loop.

    Every field defaults to the environment value (falling back to the documented
    default), so ``GanTrainerConfig.from_env()`` reproduces the previous
    behaviour exactly.
    """

    fid_eval_freq: int = 10
    early_stop_patience: int = 3
    target_fid: float = 0.0
    target_fs: float = 0.0

    # TTUR: the discriminator takes d_steps, the generator g_steps, per batch.
    d_steps: int = 1
    g_steps: int = 2

    recovery_mode: bool = False
    diversity_weight: float = 0.0
    class_guidance_weight: float = 0.0
    preview_freq: int = 5

    shake_on_collapse: bool = False
    shake_std: float = 0.0005
    grad_clip_norm: float = 5.0

    @classmethod
    def from_env(cls, fid_eval_freq: int = 10) -> GanTrainerConfig:
        """Build a config from environment variables, falling back to defaults."""
        config = cls(
            fid_eval_freq=max(0, _env_int("GAN_FID_EVAL_FREQ", fid_eval_freq)),
            early_stop_patience=_env_int("GAN_EARLY_STOP_PATIENCE", 3),
            target_fid=_env_float("GAN_TARGET_FID", 0.0),
            target_fs=_env_float("GAN_TARGET_FS", 0.0),
            d_steps=max(1, _env_int("GAN_D_STEPS", 1)),
            g_steps=max(1, _env_int("GAN_G_STEPS", 2)),
            recovery_mode=_env_bool("GAN_RECOVERY_MODE", False),
            diversity_weight=_env_float("GAN_DIVERSITY_WEIGHT", 0.0),
            class_guidance_weight=_env_float("GAN_CLASS_GUIDANCE_WEIGHT", 0.0),
            preview_freq=max(1, _env_int("GAN_PREVIEW_FREQ", 5)),
            shake_on_collapse=_env_bool("GAN_SHAKE_ON_COLLAPSE", False),
            shake_std=_env_float("GAN_SHAKE_STD", 0.0005),
            grad_clip_norm=_env_float("GAN_GRAD_CLIP_NORM", 5.0),
        )
        return config.with_recovery_overrides()

    def with_recovery_overrides(self) -> GanTrainerConfig:
        """Apply the recovery-mode preset.

        Recovery mode trades speed for stability: more generator steps, a
        diversity term, class guidance, per-epoch previews and parameter shaking.
        Values already set explicitly (non-zero) are respected.
        """
        if not self.recovery_mode:
            return self
        return replace(
            self,
            g_steps=max(self.g_steps, 4),
            diversity_weight=self.diversity_weight or 0.03,
            class_guidance_weight=self.class_guidance_weight or 0.35,
            preview_freq=1,
            shake_on_collapse=True,
        )

    def describe(self) -> str:
        """One-line summary for the start-of-run log."""
        return (
            f"fid_eval_freq={self.fid_eval_freq} preview_freq={self.preview_freq} "
            f"d_steps={self.d_steps} g_steps={self.g_steps} "
            f"recovery_mode={self.recovery_mode} "
            f"class_guidance_w={self.class_guidance_weight:.3f} "
            f"diversity_w={self.diversity_weight:.3f} "
            f"shake_on_collapse={self.shake_on_collapse} precision=float32"
        )
