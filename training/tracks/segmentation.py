from __future__ import annotations

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
    _extract_patient_id,
    build_segmentation_dataset_from_paths,
    load_brats_paths,
)
from evaluation.metrics import evaluate_segmentation, plot_loss_curves
from models.segmentation import build_unet
from training.callbacks import (
    get_standard_callbacks,
)
from training.runtime import (
    IMG_CFG,
    SEGMENTATION_CUSTOM_OBJECTS,
    set_seed,
)
from training.state import TrainingState

logger = logging.getLogger(__name__)



def train_segmentation(data_dir=None, use_attention=True, use_residual=True, epochs=None,
    seed: int | None = None,
    resume=True, patient_level=True):
    seed = set_seed(seed)
    logger.info("Track segmentation: seed=%d", seed)
    logger.info('\n' + '=' * 60)
    logger.info('TRACK 2 - SEGMENTATION')
    logger.info('=' * 60)

    # Memory-efficient: load file paths only, not pixel data
    img_paths, mask_paths = load_brats_paths(data_dir)
    n = len(img_paths)

    groups = np.array([_extract_patient_id(p) for p in img_paths])
    n_unique = len(set(groups))

    if patient_level and n_unique > 3:
        from sklearn.model_selection import GroupShuffleSplit

        gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        train_val_idx, test_idx = next(gss_test.split(img_paths,
            groups=groups))

        tv_img = img_paths[train_val_idx]
        tv_msk = mask_paths[train_val_idx]
        tv_groups = groups[train_val_idx]
        gss_val = GroupShuffleSplit(n_splits=1, test_size=0.17647, random_state=42)
        train_idx, val_idx = next(gss_val.split(tv_img,
            groups=tv_groups))

        train_img, train_msk = tv_img[train_idx], tv_msk[train_idx]
        val_img, val_msk = tv_img[val_idx], tv_msk[val_idx]
        test_img, test_msk = img_paths[test_idx], mask_paths[test_idx]
        logger.info(
            "Patient-level split (%d patients): Train=%d, Val=%d, Test=%d",
            n_unique, len(train_img), len(val_img), len(test_img),
        )
    else:
        rng = np.random.default_rng(42)
        idx = rng.permutation(n)
        split1, split2 = int(0.7 * n), int(0.85 * n)
        train_img = img_paths[idx[:split1]]
        train_msk = mask_paths[idx[:split1]]
        val_img = img_paths[idx[split1:split2]]
        val_msk = mask_paths[idx[split1:split2]]
        test_img = img_paths[idx[split2:]]
        test_msk = mask_paths[idx[split2:]]
        logger.info(f'Split: Train={len(train_img)}, Val={len(val_img)}, Test={len(test_img)}')

    cfg = TRACK_CONFIGS["segmentation"]
    seg_size = IMG_CFG.segmentation_size
    train_ds = build_segmentation_dataset_from_paths(
        train_img, train_msk, img_size=seg_size, batch_size=cfg.batch_size,
        seed=seed,
)
    val_ds = build_segmentation_dataset_from_paths(
        val_img, val_msk, img_size=seg_size, batch_size=cfg.batch_size,
        shuffle=False, augment=False,
        seed=seed,
)

    state = TrainingState("segmentation", seed=seed)
    initial_epoch = 0

    if resume and state.checkpoint_path():
        ckpt = state.checkpoint_path()
        logger.info(f'Loading checkpoint weights: {ckpt}')
        model = build_unet(
            input_shape=(*seg_size, 1),
            use_attention=use_attention,
            use_residual=use_residual,
        )
        try:
            model.load_weights(ckpt)
        except Exception as exc:
            # A .keras archive may have been written by ModelCheckpoint; fall
            # back to a full model load that supplies the custom loss/metrics.
            logger.info(f'load_weights failed ({exc}); retrying via load_model with custom_objects')
            model = tf.keras.models.load_model(ckpt, custom_objects=SEGMENTATION_CUSTOM_OBJECTS)
        initial_epoch = state.start_epoch()
    else:
        model = build_unet(
            input_shape=(*seg_size, 1),
            use_attention=use_attention,
            use_residual=use_residual,
        )

    class StateSaver(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            model.save(os.path.join(CHECKPOINT_DIR, "segmentation", "last_model.keras"))
            state.update_epoch(epoch)

    total_epochs = epochs or cfg.epochs
    if initial_epoch >= total_epochs:
        logger.info('Segmentation already reached requested epochs')
        test_ds = build_segmentation_dataset_from_paths(
            test_img, test_msk, img_size=seg_size, batch_size=cfg.batch_size,
            shuffle=False, augment=False,
            seed=seed,
)
        metrics = evaluate_segmentation(model, test_ds=test_ds)
        return model, None, metrics

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=total_epochs,
        shuffle=False,
        callbacks=get_standard_callbacks(model, "segmentation", resume=resume) + [StateSaver()],
    )

    # Export the best checkpoint, not whatever weights training happened to end on.
    best_ckpt = state.checkpoint_path()
    if best_ckpt:
        logger.info(f'Reloading best segmentation checkpoint: {best_ckpt}')
        try:
            model.load_weights(best_ckpt)
        except Exception as exc:
            logger.info(f'load_weights failed ({exc}); retrying via load_model with custom_objects')
            model = tf.keras.models.load_model(best_ckpt,
                custom_objects=SEGMENTATION_CUSTOM_OBJECTS)

    model.save(os.path.join(WEIGHTS_DIR, "segmentation_model.keras"))
    test_ds = build_segmentation_dataset_from_paths(
        test_img, test_msk, img_size=seg_size, batch_size=cfg.batch_size,
        shuffle=False, augment=False,
        seed=seed,
)
    metrics = evaluate_segmentation(model,
        test_ds=test_ds)
    plot_loss_curves(history, save_path=os.path.join(LOG_DIR, "segmentation", "loss_curves.png"))
    return model, history, metrics
