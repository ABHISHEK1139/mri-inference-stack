from __future__ import annotations

import logging
import os

import tensorflow as tf

from config import (
    CHECKPOINT_DIR,
    LOG_DIR,
    NUM_CLASSES,
    TRACK_CONFIGS,
    WEIGHTS_DIR,
)
from data.dataset import (
    build_classifier_dataset_from_paths,
    get_figshare_patient_level_split,
    get_figshare_train_val_test_split,
    load_images_from_paths,
)
from evaluation.classification import evaluate_classifier
from evaluation.plots import plot_loss_curves
from models.classifier import build_classifier, build_classifier_baseline
from training.callbacks import (
    get_standard_callbacks,
)
from training.runtime import (
    IMG_CFG,
    set_seed,
)
from training.state import TrainingState

logger = logging.getLogger(__name__)



def train_classifier(data_dir=None, use_enhanced=True, epochs=None, resume=True,
    seed: int | None = None,
    patient_level=False):
    seed = set_seed(seed)
    logger.info("Track classifier: seed=%d", seed)
    logger.info('\n' + '=' * 60)
    logger.info('TRACK 3 - CLASSIFICATION')
    logger.info('=' * 60)

    if patient_level or os.getenv("PATIENT_LEVEL_SPLIT", "0") == "1":
        logger.info('Using patient-level split (grouped by patient ID)')
        (X_train_paths, y_train), (X_val_paths, y_val), (X_test_paths,
            y_test) = get_figshare_patient_level_split(data_dir)
    else:
        (X_train_paths, y_train), (X_val_paths, y_val), (X_test_paths,
            y_test) = get_figshare_train_val_test_split(data_dir)

    cfg = TRACK_CONFIGS["classifier"]
    train_ds = build_classifier_dataset_from_paths(X_train_paths, y_train,
        img_size=IMG_CFG.classifier_size, batch_size=cfg.batch_size, seed=seed)
    val_ds = build_classifier_dataset_from_paths(
        X_val_paths, y_val, img_size=IMG_CFG.classifier_size, batch_size=cfg.batch_size,
            shuffle=False, augment=False
, seed=seed)

    state = TrainingState("classifier", seed=seed)
    initial_epoch = 0

    if resume and state.checkpoint_path():
        ckpt = state.checkpoint_path()
        logger.info(f'Loading checkpoint: {ckpt}')
        model = tf.keras.models.load_model(ckpt)
        initial_epoch = state.start_epoch()
    else:
        if use_enhanced:
            model = build_classifier(
                num_classes=NUM_CLASSES, input_shape=(*IMG_CFG.classifier_size, 1)
            )
        else:
            model = build_classifier_baseline(
                num_classes=NUM_CLASSES, input_shape=(*IMG_CFG.classifier_size, 1)
            )

    class StateSaver(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            model.save(os.path.join(CHECKPOINT_DIR, "classifier", "last_model.keras"))
            state.update_epoch(epoch)

    total_epochs = epochs or cfg.epochs
    history = None
    if initial_epoch >= total_epochs:
        logger.info('Classifier already reached requested epochs')
    else:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=initial_epoch,
            epochs=total_epochs,
            shuffle=False,
            callbacks=get_standard_callbacks(model, "classifier", resume=resume) + [StateSaver()],
        )

    best_ckpt = state.checkpoint_path()
    if best_ckpt:
        logger.info(f'Reloading best classifier checkpoint: {best_ckpt}')
        model = tf.keras.models.load_model(best_ckpt)

    model.save(os.path.join(WEIGHTS_DIR, "classifier_model.keras"))
    X_test = load_images_from_paths(X_test_paths, img_size=IMG_CFG.classifier_size)
    y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)
    metrics = evaluate_classifier(model, X_test, y_test_oh)
    if history is not None:
        plot_loss_curves(history, save_path=os.path.join(LOG_DIR, "classifier", "loss_curves.png"))
    return model, history, metrics
