"""Training-side GAN utilities: EMA weight tracking and gradient penalty."""

from contextlib import contextmanager

import numpy as np
import tensorflow as tf

# ═══════════════════════════════════════════════════════════════════════
# EMA (Exponential Moving Average) for Generator
# ═══════════════════════════════════════════════════════════════════════

class EMAGenerator:
    """Tracks exponential moving average of generator weights.
    The EMA generator produces smoother, higher-quality outputs for evaluation.
    """

    def __init__(self, generator, decay=0.999):
        self.generator = generator
        self.decay = decay
        # Track *all* variables, not just trainable ones: BatchNorm moving_mean
        # / moving_variance are non-trainable state, and leaving them out
        # produced a hybrid model (EMA kernels + live BN statistics) that is not
        # a valid EMA of the generator.
        self._variables = list(generator.variables)
        self.ema_weights = [
            tf.Variable(w, trainable=False, name=f"ema_{i}") for i, w in enumerate(self._variables)
        ]
        self._backup: list | None = None
        self._swap_depth = 0

    def update(self):
        """Update EMA weights after each generator training step."""
        for ema_w, w in zip(self.ema_weights, self._variables, strict=True):
            ema_w.assign(self.decay * ema_w + (1.0 - self.decay) * w)

    @contextmanager
    def swapped(self):
        """Context manager that temporarily applies EMA weights.

        Replaces the error-prone ``apply()``/``restore()`` pair: an exception
        between the two calls used to leave the generator permanently holding
        EMA weights while the optimizer kept updating the real ones, silently
        corrupting the rest of the run. Re-entrant via a depth counter.
        """
        self.apply()
        try:
            yield self.generator
        finally:
            self.restore()

    def apply(self):
        """Apply EMA weights to generator (for evaluation/preview).

        Prefer :meth:`swapped`, which guarantees :meth:`restore` runs.
        """
        if self._backup is not None:
            self._swap_depth += 1
            return
        self._backup = [tf.identity(w) for w in self._variables]
        for w, ema_w in zip(self._variables, self.ema_weights, strict=True):
            w.assign(ema_w)

    def restore(self):
        """Restore original weights after evaluation."""
        if self._backup is None:
            return
        self._swap_depth -= 1
        if self._swap_depth > 0:
            return
        for w, backup in zip(self._variables, self._backup, strict=True):
            w.assign(backup)
        self._backup = None

    def save(self, path: str) -> None:
        """Save EMA weights to a .npz file for checkpoint resumption."""
        np.savez(
            path,
            **{f"ema_{i}": w.numpy() for i, w in enumerate(self.ema_weights)},
        )

    def load(self, path: str) -> None:
        """Load EMA weights from a .npz file.

        Validates the key set and shapes; previously a partial or foreign
        checkpoint was accepted silently, leaving some variables at their
        initial values and producing a subtly wrong generator.
        """
        with np.load(path) as data:
            expected = {f"ema_{i}" for i in range(len(self.ema_weights))}
            found = set(data.files)
            if found != expected:
                raise ValueError(
                    f"EMA checkpoint {path} does not match this generator: "
                    f"missing={sorted(expected - found)[:5]}, "
                    f"unexpected={sorted(found - expected)[:5]}"
                )
            for i, ema_w in enumerate(self.ema_weights):
                value = data[f"ema_{i}"]
                if tuple(value.shape) != tuple(ema_w.shape):
                    raise ValueError(
                        f"EMA weight 'ema_{i}' has shape {value.shape} but the "
                        f"generator expects {tuple(ema_w.shape)}."
                    )
                ema_w.assign(value)


# ═══════════════════════════════════════════════════════════════════════
# WGAN-GP GRADIENT PENALTY
# ═══════════════════════════════════════════════════════════════════════

def gradient_penalty(discriminator, real_images, fake_images, labels, lambda_gp=1.0):
    """Compute the WGAN-GP gradient penalty.

    Returns ``lambda_gp * E[(||∇D(x̂)||_2 - 1)^2]``. ``lambda_gp`` defaults to
    ``1.0`` so the function is unscaled by default; callers applying
    ``lambda_gp * gradient_penalty(...)`` therefore get the intended penalty
    instead of silently squaring the coefficient (previously this function
    pre-multiplied by 10.0 while its only caller added no factor, so the two
    conventions could not be combined without a 10x divergence).
    """
    batch_size = tf.shape(real_images)[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated = alpha * real_images + (1.0 - alpha) * fake_images

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator([interpolated, labels], training=True)
    grads = tape.gradient(pred, interpolated)
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-8)
    gp = tf.reduce_mean(tf.square(grad_norm - 1.0))
    return lambda_gp * gp
