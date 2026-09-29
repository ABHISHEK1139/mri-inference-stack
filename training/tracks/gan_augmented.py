from __future__ import annotations

import logging
import os

import numpy as np
import tensorflow as tf

from config import (
    CHECKPOINT_DIR,
    LATENT_DIM,
    NUM_CLASSES,
    TRACK_CONFIGS,
    WEIGHTS_DIR,
)
from data.dataset import (
    build_classifier_dataset,
    load_figshare_dataset,
    mix_real_synthetic,
    split_data,
)
from evaluation.metrics import evaluate_classifier
from models.classifier import build_classifier
from training.callbacks import (
    get_standard_callbacks,
)
from training.runtime import (
    IMG_CFG,
    set_seed,
)
from training.state import TrainingState

logger = logging.getLogger(__name__)



def train_classifier_with_gan(generator, data_dir=None, gan_type="conditional", ratio=0.5,
    seed: int | None = None,
    epochs=None, resume=True):
    seed = set_seed(seed)
    logger.info("Track gan-augmented: seed=%d", seed)
    logger.info('\n' + '=' * 60)
    logger.info('GAN AUGMENTED CLASSIFIER')
    logger.info('=' * 60)

    images, labels = load_figshare_dataset(data_dir,
        img_size=IMG_CFG.classifier_size)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(images, labels)
    cfg = TRACK_CONFIGS["classifier"]

    baseline_state = TrainingState("classifier_baseline", seed=seed)
    if resume and baseline_state.checkpoint_path():
        baseline_model = tf.keras.models.load_model(baseline_state.checkpoint_path())
        baseline_start = baseline_state.start_epoch()
    else:
        baseline_model = build_classifier(num_classes=NUM_CLASSES,
            input_shape=(*IMG_CFG.classifier_size, 1))
        baseline_start = 0

    train_ds = build_classifier_dataset(X_train,
        y_train,
        batch_size=cfg.batch_size, seed=seed)
    val_ds = build_classifier_dataset(X_val, y_val, batch_size=cfg.batch_size, shuffle=False,
        augment=False, seed=seed)

    class BaselineSaver(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            baseline_model.save(os.path.join(CHECKPOINT_DIR, "classifier_baseline",
                "last_model.keras"))
            baseline_state.update_epoch(epoch)

    total_epochs = epochs or cfg.epochs
    if baseline_start < total_epochs:
        baseline_model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=baseline_start,
            epochs=total_epochs,
            shuffle=False,
            callbacks=get_standard_callbacks(baseline_model, "classifier_baseline",
                resume=resume) + [BaselineSaver()],
        )

    y_test_oh = tf.keras.utils.to_categorical(y_test,
        num_classes=NUM_CLASSES)
    baseline_metrics = evaluate_classifier(baseline_model, X_test, y_test_oh,
        track_name="classifier_baseline")

    n_synth = int(len(X_train) * ratio)
    if gan_type == "conditional":
        per_class = max(1, n_synth // NUM_CLASSES)
        syn_imgs = []
        syn_lbls = []
        for c in range(NUM_CLASSES):
            zc = tf.random.normal([per_class, LATENT_DIM])
            lc = tf.one_hot(tf.constant([c] * per_class), NUM_CLASSES)
            gc = generator([zc, lc], training=False).numpy()
            gc = np.clip((gc + 1.0) / 2.0, 0.0,
                1.0)
            gc = tf.image.resize(gc, IMG_CFG.classifier_size).numpy()
            syn_imgs.append(gc)
            syn_lbls.extend([c] * per_class)
        syn_imgs = np.concatenate(syn_imgs, axis=0)
        syn_lbls = np.array(syn_lbls, dtype=np.int32)
    else:
        z = tf.random.normal([n_synth,
            LATENT_DIM])
        gi = generator(z,
            training=False).numpy()
        syn_imgs = np.clip((gi + 1.0) / 2.0, 0.0,
            1.0)
        syn_imgs = tf.image.resize(syn_imgs, IMG_CFG.classifier_size).numpy()
        syn_lbls = np.random.randint(0, NUM_CLASSES, size=len(syn_imgs))

    mixed_X, mixed_y = mix_real_synthetic(X_train, y_train, syn_imgs, syn_lbls, ratio=ratio)

    aug_state = TrainingState("classifier_augmented", seed=seed)
    if resume and aug_state.checkpoint_path():
        aug_model = tf.keras.models.load_model(aug_state.checkpoint_path())
        aug_start = aug_state.start_epoch()
    else:
        aug_model = build_classifier(num_classes=NUM_CLASSES, input_shape=(*IMG_CFG.classifier_size,
            1))
        aug_start = 0

    train_aug_ds = build_classifier_dataset(mixed_X, mixed_y,
        batch_size=cfg.batch_size, seed=seed)

    class AugSaver(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            aug_model.save(os.path.join(CHECKPOINT_DIR, "classifier_augmented",
                "last_model.keras"))
            aug_state.update_epoch(epoch)

    if aug_start < total_epochs:
        aug_model.fit(
            train_aug_ds,
            validation_data=val_ds,
            initial_epoch=aug_start,
            epochs=total_epochs,
            shuffle=False,
            callbacks=get_standard_callbacks(aug_model, "classifier_augmented",
                resume=resume) + [AugSaver()],
        )

    aug_metrics = evaluate_classifier(aug_model, X_test, y_test_oh,
        track_name="classifier_augmented")
    logger.info(f"Baseline acc: {baseline_metrics['accuracy']:.4f}")
    logger.info(f"Augmented acc: {aug_metrics['accuracy']:.4f}")
    logger.info(f"Improvement: {aug_metrics['accuracy'] - baseline_metrics['accuracy']:.4f}")

    aug_model.save(os.path.join(WEIGHTS_DIR, "classifier_augmented.keras"))
    baseline_model.save(os.path.join(WEIGHTS_DIR, "classifier_baseline.keras"))

    return aug_model, aug_metrics, baseline_metrics
