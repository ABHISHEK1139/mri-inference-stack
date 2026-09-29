"""Detection-specific evaluation helpers."""
import os
import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import LOG_DIR
from evaluation.metrics import plot_confusion_matrix


def binary_metrics_at_threshold(y_true, y_pred_proba, threshold=0.5):
    """Compute binary classification metrics for a probability cutoff."""
    y_true = np.asarray(y_true).ravel()
    y_pred_proba = np.asarray(y_pred_proba).ravel()
    if y_true.shape != y_pred_proba.shape:
        raise ValueError(
            f"y_true and y_pred_proba must have the same length, "
            f"got {y_true.shape} and {y_pred_proba.shape}."
        )
    if y_true.size == 0:
        raise ValueError("binary_metrics_at_threshold received an empty evaluation set.")

    y_pred = (y_pred_proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 confusion matrix, got shape {cm.shape}.")
    # cm is indexed [true, pred] with labels=[0, 1], so ravel() yields
    # (tn, fp, fn, tp) in that order.
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    bal_acc = (rec + specificity) / 2.0
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": specificity,
        "balanced_accuracy": bal_acc,
        "f1_score": f1,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": cm,
    }


def calibrate_binary_threshold(
    y_true,
    y_pred_proba,
    thresholds=None,
    optimize="f1",
    min_recall=None,
):
    """
    Pick a validation threshold for binary detection.

    Defaults to maximizing F1 while optionally enforcing a recall floor. Ties
    are broken towards the threshold with the highest specificity, so a flat F1
    plateau does not silently select the most aggressive (lowest) cutoff.

    The returned metrics dict always carries ``recall_floor_met`` so a caller
    can tell a genuine calibration apart from the 0.5 fallback used when no
    candidate satisfies the recall floor.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred_proba = np.asarray(y_pred_proba).ravel()
    if optimize not in {"f1", "balanced_accuracy"}:
        raise ValueError(f"optimize must be 'f1' or 'balanced_accuracy', got {optimize!r}.")

    thresholds = thresholds if thresholds is not None else np.linspace(0.05, 0.95, 37)
    best_key: tuple[float, float, float] | None = None
    best_threshold = 0.5
    best_metrics = None

    for threshold in thresholds:
        metrics = binary_metrics_at_threshold(y_true, y_pred_proba, threshold=float(threshold))
        metrics["threshold"] = float(threshold)
        if min_recall is not None and metrics["recall"] < min_recall:
            continue
        if optimize == "balanced_accuracy":
            score = float(metrics["balanced_accuracy"])
        else:
            score = float(metrics["f1_score"])
        # Maximise the objective, then specificity, then prefer the higher
        # (more conservative) threshold.
        key = (score, float(metrics["specificity"]), float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_metrics is None:
        best_metrics = binary_metrics_at_threshold(y_true, y_pred_proba, threshold=0.5)
        best_metrics["threshold"] = 0.5
        best_threshold = 0.5
        best_metrics["recall_floor_met"] = False
        warnings.warn(
            f"No threshold in [{float(np.min(thresholds)):.3f}, {float(np.max(thresholds)):.3f}] "
            f"reached min_recall={min_recall}; falling back to threshold=0.5. "
            "The model is likely too weak for this recall floor.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        best_metrics["recall_floor_met"] = True

    return best_threshold, best_metrics


def evaluate_detection_refined(model, X_test, y_test, save_dir=None, threshold=0.5):
    """Evaluate a detection model with an explicit threshold and specificity."""
    save_dir = save_dir or os.path.join(LOG_DIR, "detection")
    os.makedirs(save_dir, exist_ok=True)

    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    metrics = binary_metrics_at_threshold(y_test, y_pred_proba, threshold=threshold)

    try:
        auc = float(roc_auc_score(np.asarray(y_test).ravel(), y_pred_proba))
    except ValueError as exc:
        warnings.warn(f"ROC-AUC is undefined for this evaluation set: {exc}", RuntimeWarning,
            stacklevel=2)
        auc = float("nan")

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        ["Normal", "Tumour"],
        save_path=os.path.join(save_dir, "confusion_matrix.png"),
    )

    metrics.update({"auc": auc, "threshold": float(threshold)})
    print(
        f"\n  Detection - "
        f"Thr:{threshold:.3f} "
        f"Acc:{metrics['accuracy']:.4f} "
        f"Prec:{metrics['precision']:.4f} "
        f"Rec:{metrics['recall']:.4f} "
        f"Spec:{metrics['specificity']:.4f} "
        f"F1:{metrics['f1_score']:.4f} "
        f"AUC:{auc:.4f}"
    )
    return metrics
