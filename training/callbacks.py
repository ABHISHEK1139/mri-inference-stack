"""Reusable callbacks for supervised and GAN training."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import CHECKPOINT_DIR, CLASS_NAMES, LOG_DIR, NUM_CLASSES


def get_standard_callbacks(model: tf.keras.Model, track_name: str) -> list[tf.keras.callbacks.Callback]:
    """Return the standard callback suite used by the training script."""
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
            patience=7,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(log_dir / "training_log.csv"), append=True),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=0,
            write_graph=False,
            update_freq="epoch",
        ),
    ]


class GANLossLogger:
    """Append per-epoch GAN losses to a CSV file."""

    def __init__(self, csv_path: str | os.PathLike[str] | None = None):
        self.csv_path = Path(csv_path) if csv_path else Path(LOG_DIR) / "gan_training_log.csv"
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["epoch", "d_loss", "g_loss", "d_acc", "g_acc"])

    def log_step(self, epoch: int, d_loss: float, g_loss: float, d_acc: float, g_acc: float) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([epoch, d_loss, g_loss, d_acc, g_acc])


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
        cols = min(4, len(images))
        rows = int(np.ceil(len(images) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = np.atleast_1d(axes).reshape(rows, cols)

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
    """Lightweight detector that warns when GAN outputs lose diversity."""

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

    def on_epoch_end(self, epoch: int) -> None:
        noise = tf.random.normal([8, self.latent_dim])
        if self.conditional:
            labels = tf.one_hot(np.arange(8) % self.num_classes, self.num_classes, dtype=tf.float32)
            generated = self.generator([noise, labels], training=False)
        else:
            generated = self.generator(noise, training=False)

        generated = generated.numpy()
        if not np.isfinite(generated).all():
            print(f"  Collapse warning: non-finite generator output detected at epoch {epoch + 1}.")
            return

        diversity = float(np.std(generated, axis=0).mean())
        if diversity < self.min_std_threshold:
            print(
                f"  Collapse warning: sample diversity dropped to {diversity:.4f} "
                f"at epoch {epoch + 1}."
            )
