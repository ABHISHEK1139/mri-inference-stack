"""Seeding and determinism controls for reproducible training runs.

Without this, ``tf.random.normal``, dropout masks, weight initialisation and
``tf.data`` shuffling all draw from an unseeded generator, so two runs over
identical data produce different weights and the reproducibility guarantees in
``docs/reproducibility_runbook.md`` cannot be honoured.
"""

from __future__ import annotations

import logging
import os
import platform
import random
from typing import Any

import numpy as np

from config import DEFAULT_SEED, DETERMINISTIC_OPS

logger = logging.getLogger(__name__)


def set_seed(seed: int | None = None, deterministic_ops: bool | None = None) -> int:
    """Seed Python, NumPy and TensorFlow from a single integer.

    Args:
        seed: Seed to apply. ``None`` uses :data:`config.DEFAULT_SEED`.
        deterministic_ops: Request deterministic kernels. This is materially
            slower and TensorFlow raises for ops that have no deterministic
            implementation, so it stays opt-in via ``DETERMINISTIC_OPS``.

    Returns:
        The seed that was actually applied, so callers can record it in logs
        and in checkpoint state.
    """
    if seed is None:
        seed = DEFAULT_SEED
    seed = int(seed)
    if deterministic_ops is None:
        deterministic_ops = DETERMINISTIC_OPS

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)

    if deterministic_ops:
        _enable_deterministic_ops(tf)

    logger.info(
        "Seeded run: seed=%d deterministic_ops=%s python=%s numpy=%s tf=%s",
        seed,
        bool(deterministic_ops),
        platform.python_version(),
        np.__version__,
        tf.__version__,
    )
    return seed


def _enable_deterministic_ops(tf: Any) -> None:
    """Request deterministic kernels, tolerating unsupported ops."""
    try:
        tf.config.experimental.enable_op_determinism()
        logger.info("Deterministic ops enabled")
    except (RuntimeError, ValueError) as exc:
        logger.warning(
            "Could not enable full determinism (%s). Some ops have no "
            "deterministic implementation; set DETERMINISTIC_OPS=0 to disable.",
            exc,
        )


def seed_state_dict(seed: int) -> dict[str, Any]:
    """Build the reproducibility block stored alongside checkpoint state."""
    import tensorflow as tf

    return {
        "seed": int(seed),
        "deterministic_ops": bool(DETERMINISTIC_OPS),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }
