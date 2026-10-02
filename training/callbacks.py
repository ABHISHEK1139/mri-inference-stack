"""Reusable callbacks for supervised and GAN training."""

from __future__ import annotations

import csv
import importlib.util
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import CHECKPOINT_DIR, CLASS_NAMES, LOG_DIR, NUM_CLASSES

logger = logging.getLogger(__name__)


def get_standard_callbacks(
    model: tf.keras.Model,
    track_name: str,
    resume: bool = False,
    early_stopping_patience: int = 7,
    reduce_lr_patience: int = 3,
) -> list[tf.keras.callbacks.Callback]:
    """Return the standard callback suite used by the training script.

    ``resume=False`` (the default) starts a fresh ``training_log.csv`` for the
    run instead of appending a second run whose epoch numbering restarts at 0
    into the middle of the previous run's history.
    """
    checkpoint_dir = Path(CHECKPOINT_DIR) / track_name
    log_dir = Path(LOG_DIR) / track_name
    tensorboard_dir = log_dir / "tensorboard"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(log_dir / "training_log.csv"), append=resume),
        *_tensorboard_callback(tensorboard_dir),
    ]


def _tensorboard_callback(log_dir: Path) -> list:
    """Return the TensorBoard callback, or nothing when TensorBoard is absent.

    TensorBoard is an observability tool, not something a training run needs to
    be correct, so a missing install must not abort the run. Without this guard
    Keras raises ``TBNotInstalledError`` while constructing the callback, which
    would fail every ``train.py`` invocation in an environment that installed
    only the runtime stack.

    The check is explicit rather than a ``try``/``except`` because
    ``TBNotInstalledError`` derives directly from ``Exception``, not from
    ``ImportError``, so it would not be caught by the obvious handler. The CSV
    logger in :func:`get_standard_callbacks` still records the full per-epoch
    history either way.
    """
    if importlib.util.find_spec("tensorboard") is None:
        logger.warning(
            "TensorBoard is not installed; skipping the TensorBoard callback. "
            "Install it with `pip install tensorboard` to keep the event logs."
        )
        return []
    return [
        tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=0,
            write_graph=False,
            update_freq="epoch",
        )
    ]


class GANLossLogger:
    """Append per-epoch GAN losses to a CSV file.

    The log path is derived from the column set, so a WGAN-GP run
    (``epoch,d_loss,g_loss,w_dist,g_acc``) never appends Wasserstein distances
    underneath a header that labels the column ``d_acc``. Previously both
    variants defaulted to the same filename and the header was written only once.
    """

    def __init__(
        self,
        csv_path: str | os.PathLike[str] | None = None,
        columns: list[str] | None = None,
    ):
        self.columns = list(columns or ["epoch", "d_loss", "g_loss", "d_acc", "g_acc"])
        if csv_path is None:
            is_default = self.columns == ["epoch", "d_loss", "g_loss", "d_acc", "g_acc"]
            variant = "default" if is_default else "alt"
            csv_path = Path(LOG_DIR) / f"gan_training_log_{variant}.csv"
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.csv_path.exists():
            self._write_header()
        else:
            existing = self._read_header()
            if existing and existing != self.columns:
                # Column set changed: start a fresh file instead of corrupting
                # an existing log with a mismatched header.
                suffix = self.csv_path.suffix + "." + "-".join(self.columns) + ".bak"
                backup = self.csv_path.with_suffix(suffix)
                self.csv_path.replace(backup)
                self._write_header()

    def _read_header(self) -> list[str]:
        try:
            with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
                return next(csv.reader(handle), [])
        except (OSError, StopIteration):
            return []

    def _write_header(self) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(self.columns)

    def log_step(self, epoch: int, d_loss: float, g_loss: float, d_acc: float,
        g_acc: float) -> None:
        row = [epoch, d_loss, g_loss, d_acc, g_acc]
        if len(row) != len(self.columns):
            raise ValueError(
                f"log_step produced {len(row)} values but the log has "
                f"{len(self.columns)} columns: {self.columns}"
            )
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)


class GANImageSampler:
    """Generate and save a small image grid from the current generator."""

    def __init__(
        self,
        generator: tf.keras.Model,
        latent_dim: int,
        conditional: bool = False,
        num_classes: int = NUM_CLASSES,
        output_dir: str | os.PathLike[str] | None = None,
    ):
        self.generator = generator
        self.latent_dim = latent_dim
        self.conditional = conditional
        self.num_classes = num_classes
        self.output_dir = Path(output_dir) if output_dir else Path(LOG_DIR) / "gan_samples"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_batch(self, count: int = 16) -> np.ndarray:
        noise = tf.random.normal([count, self.latent_dim])
        if self.conditional:
            labels = tf.one_hot(
                np.arange(count) % self.num_classes,
                depth=self.num_classes,
                dtype=tf.float32,
            )
            generated = self.generator([noise, labels], training=False)
        else:
            generated = self.generator(noise, training=False)
        generated = generated.numpy()
        return np.clip((generated + 1.0) / 2.0, 0.0, 1.0)

    def _generate_and_save(self, epoch: int) -> None:
        images = self._generate_batch()
        count = len(images)
        if count < 1:
            logger.info('  GAN preview skipped: generator produced no samples.')
            return
        cols = min(4, count)
        rows = int(np.ceil(count / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
        axes = np.asarray(axes).reshape(rows, cols)

        for index, axis in enumerate(axes.flat):
            axis.axis("off")
            if index >= len(images):
                continue
            axis.imshow(images[index].squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
            if self.conditional:
                axis.set_title(CLASS_NAMES[index % self.num_classes].title(), fontsize=10)

        fig.tight_layout()
        epoch_path = self.output_dir / f"epoch_{epoch + 1:04d}.png"
        latest_path = self.output_dir / "latest.png"
        fig.savefig(epoch_path, dpi=140, bbox_inches="tight")
        fig.savefig(latest_path, dpi=140, bbox_inches="tight")
        plt.close(fig)


class ModelCollapseDetector:
    """Lightweight detector that warns when GAN outputs lose diversity.

    For a conditional generator, diversity is measured *within* each class.
    Taking ``std`` across the whole batch would mostly measure inter-class
    variance, so a healthy class-conditional GAN (which should produce similar
    images per class) was flagged as collapsed.
    """

    def __init__(
        self,
        generator: tf.keras.Model,
        latent_dim: int,
        conditional: bool = False,
        num_classes: int = NUM_CLASSES,
        min_std_threshold: float = 0.02,
    ):
        self.generator = generator
        self.latent_dim = latent_dim
        self.conditional = conditional
        self.num_classes = num_classes
        self.min_std_threshold = min_std_threshold

    def _within_class_diversity(self, generated: np.ndarray) -> float:
        """Mean of the per-image spatial std, averaged within each class."""
        # (B, H, W, C) -> per-image std over pixels, giving (B,)
        per_image_std = generated.reshape(len(generated), -1).std(axis=1)
        if not self.conditional or self.num_classes < 2:
            return float(per_image_std.mean())
        classes = np.arange(len(generated)) % self.num_classes
        per_class = [per_image_std[classes == c].mean() for c in range(self.num_classes)]
        per_class = [v for v in per_class if np.isfinite(v)]
        return float(np.mean(per_class)) if per_class else 0.0

    def on_epoch_end(self, epoch: int) -> None:
        noise = tf.random.normal([8, self.latent_dim])
        if self.conditional:
            labels = tf.one_hot(np.arange(8) % self.num_classes, self.num_classes, dtype=tf.float32)
            generated = self.generator([noise, labels], training=False)
        else:
            generated = self.generator(noise, training=False)

        generated = np.asarray(generated)
        if not np.isfinite(generated).all():
            logger.warning(
                f"'  Collapse warning: non-finite generat"
                f"or output detected at epoch {epoch + 1}."
            )
            return

        diversity = self._within_class_diversity(generated)
        if diversity < self.min_std_threshold:
            logger.warning(
                f"'  Collapse warning: within-class sample diversit"
                f"y dropped to {diversity:.4f} at epoch {epoch + 1}."
            )
