"""Classification and binary-detection evaluation."""

import os
import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import CLASS_NAMES, LOG_DIR
from evaluation.plots import plot_confusion_matrix


def _safe_auc(y_true, y_score, **kwargs):
    """Compute ROC-AUC, returning NaN (not a plausible 0.0) when undefined.

    A blanket ``except Exception: auc = 0.0`` used to make a degenerate split or
    a label/shape mismatch look like a genuinely terrible model in printed
    reports. NaN is unambiguous and renders as ``nan`` in the summary.

    Depending on the scikit-learn version, an undefined AUC either raises
    ``ValueError`` or returns NaN with a warning, so both paths are handled.
    """
    try:
        value = float(roc_auc_score(y_true, y_score, **kwargs))
    except ValueError as exc:
        warnings.warn(f"ROC-AUC is undefined for this evaluation set: {exc}",
                      RuntimeWarning, stacklevel=2)
        return float("nan")
    if not np.isfinite(value):
        warnings.warn(
            "ROC-AUC is undefined for this evaluation set "
            "(only one class present, or non-finite scores).",
            RuntimeWarning,
            stacklevel=2,
        )
    return value


def evaluate_classifier(model, X_test, y_test, track_name="classifier", save_dir=None):
    """Evaluate a classifier and generate all metrics + plots."""
    save_dir = save_dir or os.path.join(LOG_DIR, track_name)
    os.makedirs(save_dir, exist_ok=True)

    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Handle one-hot labels
    if len(y_test.shape) > 1 and y_test.shape[-1] > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    auc = _safe_auc(y_test, y_pred_proba, multi_class='ovr', average='weighted')

    # Classification report
    report = classification_report(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES,
            zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    plot_confusion_matrix(cm, CLASS_NAMES, save_path=os.path.join(save_dir, "confusion_matrix.png"))

    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'classification_report': report,
    }

    print(f"\n{'='*50}")
    print(f"  {track_name.upper()} EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"\n{report}")

    return metrics


def evaluate_detection(model, X_test, y_test, save_dir=None):
    """Evaluate binary detection model."""
    save_dir = save_dir or os.path.join(LOG_DIR, "detection")
    os.makedirs(save_dir, exist_ok=True)

    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    auc = _safe_auc(y_test, y_pred_proba)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    class_names_binary = ["Normal",
        "Tumour"]
    plot_confusion_matrix(cm, class_names_binary, save_path=os.path.join(save_dir,
        "confusion_matrix.png"))

    metrics = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'auc': auc}
    print(f"\n  Detection — Acc:{acc:.4f} Prec:{prec:.4f} Rec:{rec:.4f} F1:{f1:.4f} AUC:{auc:.4f}")
    return metrics
