"""Segmentation evaluation with per-image Dice and IoU."""

import os

import matplotlib.pyplot as plt
import numpy as np

from config import LOG_DIR
from evaluation.plots import _ensure_parent_dir


def evaluate_segmentation(model, X_test=None, y_test=None, test_ds=None, save_dir=None):
    """Evaluate segmentation model with Dice, IoU, etc.

    Supports two modes:
      - Legacy: pass X_test and y_test as numpy arrays
      - Streaming: pass test_ds as a tf.data.Dataset (memory-efficient)
    """
    save_dir = save_dir or os.path.join(LOG_DIR, "segmentation")
    os.makedirs(save_dir, exist_ok=True)

    # Collect predictions — streaming or in-memory
    dice_scores = []
    iou_scores = []
    vis_inputs = []
    vis_truths = []
    vis_preds = []
    vis_dices = []
    n_vis_target = 8

    if test_ds is not None:
        # Streaming mode: iterate batch-by-batch
        for x_batch, y_batch in test_ds:
            pred_batch = np.asarray(model(x_batch, training=False))
            pred_bin = (pred_batch > 0.5).astype(np.float32)
            y_np = y_batch.numpy() if hasattr(y_batch, 'numpy') else np.asarray(y_batch)
            x_np = x_batch.numpy() if hasattr(x_batch, 'numpy') else np.asarray(x_batch)
            for i in range(len(y_np)):
                d = _dice_coef(y_np[i], pred_bin[i])
                iou = _iou_coef(y_np[i], pred_bin[i])
                dice_scores.append(d)
                iou_scores.append(iou)
                if len(vis_inputs) < n_vis_target:
                    vis_inputs.append(x_np[i])
                    vis_truths.append(y_np[i])
                    vis_preds.append(pred_bin[i])
                    vis_dices.append(d)
    else:
        # Legacy in-memory mode
        y_pred = model.predict(X_test, verbose=0)
        y_pred_bin = (y_pred > 0.5).astype(np.float32)
        for i in range(len(y_test)):
            d = _dice_coef(y_test[i], y_pred_bin[i])
            iou = _iou_coef(y_test[i], y_pred_bin[i])
            dice_scores.append(d)
            iou_scores.append(iou)
        vis_inputs = list(X_test[:n_vis_target])
        vis_truths = list(y_test[:n_vis_target])
        vis_preds = list(y_pred_bin[:n_vis_target])
        vis_dices = dice_scores[:n_vis_target]

    if not dice_scores:
        raise ValueError(
            "evaluate_segmentation received an empty evaluation set; "
            "no Dice/IoU scores can be computed."
        )
    mean_dice = float(np.mean(dice_scores))
    mean_iou = float(np.mean(iou_scores))

    # Visualize some predictions
    n_vis = min(n_vis_target, len(vis_inputs))
    if n_vis > 0:
        fig, axes = plt.subplots(3, n_vis, figsize=(3 * n_vis, 9))
        if n_vis == 1:
            axes = axes[:, np.newaxis]
        for i in range(n_vis):
            axes[0, i].imshow(vis_inputs[i].squeeze(), cmap='gray')
            axes[0, i].set_title('Input')
            axes[1, i].imshow(vis_truths[i].squeeze(), cmap='gray')
            axes[1, i].set_title('Ground Truth')
            axes[2, i].imshow(vis_preds[i].squeeze(), cmap='gray')
            axes[2, i].set_title(f'Pred (D={vis_dices[i]:.3f})')
            for row in range(3):
                axes[row,
                    i].axis('off')
        plt.suptitle(f"Segmentation — Mean Dice: {mean_dice:.4f}, Mean IoU: {mean_iou:.4f}")
        _ensure_parent_dir(os.path.join(save_dir, "segmentation_results.png"))
        plt.savefig(os.path.join(save_dir, "segmentation_results.png"), dpi=150,
            bbox_inches='tight', facecolor='white', transparent=False)
        plt.close()

    metrics = {'mean_dice': mean_dice, 'mean_iou': mean_iou}
    print(f"\n  Segmentation — Dice:{mean_dice:.4f} IoU:{mean_iou:.4f}")
    return metrics


def _dice_coef(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


def _iou_coef(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)
