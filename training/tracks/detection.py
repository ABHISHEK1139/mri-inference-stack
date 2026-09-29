from __future__ import annotations

import json
import logging
import os

import numpy as np
import tensorflow as tf

from config import (
    CHECKPOINT_DIR,
    LOG_DIR,
    TRACK_CONFIGS,
    WEIGHTS_DIR,
)
from data.dataset import (
    build_detection_dataset_from_paths,
    get_figshare_patient_level_split,
    get_figshare_train_val_test_split,
    load_images_from_paths,
)
from evaluation.detection_eval import calibrate_binary_threshold, evaluate_detection_refined
from evaluation.metrics import plot_loss_curves
from models.detection import build_detection_baseline, build_detection_model
from training.callbacks import (
    get_standard_callbacks,
)
from training.runtime import (
    IMG_CFG,
    _balanced_class_weight_dict,
    _json_safe,
    set_seed,
)
from training.state import TrainingState

logger = logging.getLogger(__name__)



def train_detection(data_dir=None, use_enhanced=True, epochs=None, resume=True,
    seed: int | None = None,
    patient_level=False):
    seed = set_seed(seed)
    logger.info("Track detection: seed=%d", seed)
    logger.info('\n' + '=' * 60)
    logger.info('TRACK 1 - DETECTION')
    logger.info('=' * 60)

    if patient_level or os.getenv("PATIENT_LEVEL_SPLIT", "0") == "1":
        logger.info('Using patient-level split (grouped by patient ID)')
        (X_train_paths, y_train_multiclass), (X_val_paths, y_val_multiclass), (X_test_paths,
            y_test_multiclass) = (
            get_figshare_patient_level_split(data_dir)
        )
    else:
        (X_train_paths, y_train_multiclass), (X_val_paths, y_val_multiclass), (X_test_paths,
            y_test_multiclass) = (
            get_figshare_train_val_test_split(data_dir)
        )
    y_train = (y_train_multiclass < 3).astype(np.int32)
    y_val = (y_val_multiclass < 3).astype(np.int32)
    y_test = (y_test_multiclass < 3).astype(np.int32)

    cfg = TRACK_CONFIGS["detection"]
    train_ds = build_detection_dataset_from_paths(X_train_paths, y_train,
        img_size=IMG_CFG.detection_size, batch_size=cfg.batch_size, seed=seed)
    val_ds = build_detection_dataset_from_paths(
        X_val_paths, y_val, img_size=IMG_CFG.detection_size, batch_size=cfg.batch_size,
            shuffle=False, augment=False
, seed=seed)

    state = TrainingState("detection", seed=seed)
    initial_epoch = 0

    if resume and state.checkpoint_path():
        ckpt = state.checkpoint_path()
        logger.info(f'Loading checkpoint: {ckpt}')
        model = tf.keras.models.load_model(ckpt)
        initial_epoch = state.start_epoch()
    else:
        if use_enhanced:
            model = build_detection_model(input_shape=(*IMG_CFG.detection_size, 1))
        else:
            model = build_detection_baseline(input_shape=(*IMG_CFG.detection_size, 1))

    class_weights = _balanced_class_weight_dict(y_train)
    logger.info(f'Detection class weights: {class_weights}')

    class StateSaver(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            model.save(os.path.join(CHECKPOINT_DIR, "detection", "last_model.keras"))
            state.update_epoch(epoch)

    total_epochs = epochs or cfg.epochs
    history = None
    if initial_epoch >= total_epochs:
        logger.info('Detection already reached requested epochs')
    else:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=initial_epoch,
            epochs=total_epochs,
            class_weight=class_weights,
            shuffle=False,
            callbacks=get_standard_callbacks(model, "detection", resume=resume) + [StateSaver()],
        )

    best_ckpt = state.checkpoint_path()
    if best_ckpt:
        logger.info(f'Reloading best detection checkpoint: {best_ckpt}')
        model = tf.keras.models.load_model(best_ckpt)

    X_val = load_images_from_paths(X_val_paths,
        img_size=IMG_CFG.detection_size)
    X_test = load_images_from_paths(X_test_paths, img_size=IMG_CFG.detection_size)
    val_probs = model.predict(X_val,
        verbose=0).flatten()
    threshold, threshold_metrics = calibrate_binary_threshold(y_val, val_probs, optimize="f1",
        min_recall=0.97)
    logger.info(
        "Detection threshold tuning - threshold=%.3f val_f1=%.4f "
        "val_recall=%.4f val_specificity=%.4f",
        threshold, threshold_metrics["f1_score"], threshold_metrics["recall"],
        threshold_metrics["specificity"],
    )

    model.save(os.path.join(WEIGHTS_DIR, "detection_model.keras"))
    # Primary location — this is what the Streamlit app loads first
    config_path_weights = os.path.join(WEIGHTS_DIR, "detection_inference_config.json")
    with open(config_path_weights, "w", encoding="utf-8") as f:
        json.dump(
            _json_safe({"threshold": threshold, "validation_metrics": threshold_metrics}),
            f,
            indent=2,
        )

    # Secondary backup in checkpoint dir
    config_path_checkpoint = os.path.join(CHECKPOINT_DIR, "detection", "inference_config.json")
    os.makedirs(os.path.dirname(config_path_checkpoint), exist_ok=True)
    with open(config_path_checkpoint, "w", encoding="utf-8") as f:
        json.dump(
            _json_safe({"threshold": threshold, "validation_metrics": threshold_metrics}),
            f,
            indent=2,
        )

    metrics = evaluate_detection_refined(model, X_test, y_test, threshold=threshold)
    if history is not None:
        plot_loss_curves(history, save_path=os.path.join(LOG_DIR, "detection",
            "loss_curves.png"))
    return model, history, metrics
