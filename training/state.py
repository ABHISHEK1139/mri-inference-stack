"""Checkpoint state persistence for resumable training runs.

State is written to a temporary file and then atomically renamed, so an
interrupted save cannot leave truncated JSON that silently resets a run on the
next resume.
"""

from __future__ import annotations

import json
import logging
import os

from config import CHECKPOINT_DIR
from training.reproducibility import seed_state_dict

logger = logging.getLogger(__name__)


class TrainingState:
    """Persistent state for resuming non-GAN training."""

    def __init__(self, track_name, seed=None):
        self.track_name = track_name
        self.track_dir = os.path.join(CHECKPOINT_DIR, track_name)
        os.makedirs(self.track_dir, exist_ok=True)
        self.state_path = os.path.join(self.track_dir, "training_state.json")
        self.seed = seed
        self.state = self._load()

    def _load(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, encoding="utf-8") as f:
                    data = json.load(f)
                logger.info("Resuming %s from epoch %s", self.track_name,
                            data.get("last_epoch", 0) + 1)
                return data
            except Exception as e:
                logger.warning("Could not load state for %s: %s", self.track_name, e)
        return {"last_epoch": -1}

    def save(self):
        # Record the run's seed and library versions so a resumed run can be
        # traced back to the exact configuration that produced it.
        if self.seed is not None:
            self.state["reproducibility"] = seed_state_dict(self.seed)
        tmp_path = f"{self.state_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        os.replace(tmp_path, self.state_path)

    def update_epoch(self, epoch):
        self.state["last_epoch"] = int(epoch)
        self.save()

    def start_epoch(self):
        return int(self.state.get("last_epoch", -1)) + 1

    def checkpoint_path(self):
        best_path = os.path.join(self.track_dir, "best_model.keras")
        last_path = os.path.join(self.track_dir, "last_model.keras")
        if os.path.exists(best_path):
            return best_path
        if os.path.exists(last_path):
            return last_path
        return None


class GANState:
    """Persistent state for resuming GAN training."""

    def __init__(self, gan_type, seed=None, weights_only=False):
        self.gan_type = gan_type
        # v2 checkpoints via save_weights/load_weights; v1 via model.save.
        self.weights_only = weights_only or gan_type.endswith("v2")
        self.track_dir = os.path.join(CHECKPOINT_DIR, "gan")
        os.makedirs(self.track_dir, exist_ok=True)
        self.state_path = os.path.join(self.track_dir, f"gan_{gan_type}_state.json")
        self.seed = seed
        self.state = self._load()

    def _load(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, encoding="utf-8") as f:
                    data = json.load(f)
                logger.info("Resuming GAN (%s) from epoch %s", self.gan_type,
                            data.get("last_epoch", 0) + 1)
                return data
            except Exception as e:
                logger.warning("Could not load GAN state: %s", e)
        return self.fresh_state()

    def save(self):
        if self.seed is not None:
            self.state["reproducibility"] = seed_state_dict(self.seed)
        # Write to a temp file then swap, so an interrupted save cannot leave a
        # truncated JSON file that silently resets the run on the next resume.
        tmp_path = f"{self.state_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        os.replace(tmp_path, self.state_path)

    def start_epoch(self):
        return int(self.state.get("last_epoch", -1)) + 1

    def generator_ckpt(self):
        """Checkpoint path for the generator.

        ``gan_type="v2"`` stores raw weights rather than a full model, because the
        WGAN-GP trainer calls ``save_weights``. Keras 3 requires a ``.weights.h5``
        suffix there and raises ``ValueError: The filename must end in
        `.weights.h5``` for a ``.keras`` path, so the extension must follow the
        save style instead of being fixed.
        """
        suffix = ".weights.h5" if self.weights_only else ".keras"
        return os.path.join(self.track_dir, f"generator_{self.gan_type}_last{suffix}")

    def discriminator_ckpt(self):
        suffix = ".weights.h5" if self.weights_only else ".keras"
        return os.path.join(self.track_dir, f"discriminator_{self.gan_type}_last{suffix}")

    def ema_ckpt(self):
        return os.path.join(self.track_dir, f"ema_{self.gan_type}.npz")

    def has_ckpt(self):
        return os.path.exists(self.generator_ckpt()) and os.path.exists(self.discriminator_ckpt())

    @staticmethod
    def fresh_state():
        return {
            "last_epoch": -1,
            "d_losses": [],
            "g_losses": [],
            "d_accs": [],
            "g_accs": [],
            "w_distances": [],
            "fid_scores": [],
            "fs_scores": [],
            "best_quality": float("inf"),
            "gan_no_improve": 0,
            "g_steps": None,
            "d_lr": None,
        }
