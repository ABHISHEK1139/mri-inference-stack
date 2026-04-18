"""Detection-specific evaluation helpers."""
import os
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
    y_pred = (y_pred_proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
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
        "confusion_matrix": cm,
    }


def calibrate_binary_threshold(y_true, y_pred_proba, thresholds=None, optimize="f1", min_recall=None):
    """
    Pick a validation threshold for binary detection.
    Defaults to maximizing F1 while optionally enforcing a recall floor.
    """
    thresholds = thresholds if thresholds is not None else np.linspace(0.05, 0.95, 37)
    best_threshold = 0.5
    best_metrics = None
    best_score = float("-inf")

    for threshold in thresholds:
        metrics = binary_metrics_at_threshold(y_true, y_pred_proba, threshold=float(threshold))
        metrics["threshold"] = float(threshold)
        if min_recall is not None and metrics["recall"] < min_recall:
            score = float("-inf")
        elif optimize == "balanced_accuracy":
            score = metrics["balanced_accuracy"]
        else:
            score = metrics["f1_score"]

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_metrics is None:
        best_metrics = binary_metrics_at_threshold(y_true, y_pred_proba, threshold=0.5)
        best_metrics["threshold"] = 0.5
        best_threshold = 0.5

    return best_threshold, best_metrics


def evaluate_detection_refined(model, X_test, y_test, save_dir=None, threshold=0.5):
    """Evaluate a detection model with an explicit threshold and specificity."""
    save_dir = save_dir or os.path.join(LOG_DIR, "detection")
    os.makedirs(save_dir, exist_ok=True)

    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    metrics = binary_metrics_at_threshold(y_test, y_pred_proba, threshold=threshold)

    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except Exception:
        auc = 0.0

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
