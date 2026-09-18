import numpy as np

from evaluation.metrics import _dice_coef, _iou_coef


def test_dice_and_iou_perfect_overlap():
    """Perfect overlap should return 1.0 for both Dice and IoU."""
    y_true = np.array([[1, 0], [0, 1]], dtype=np.float32)
    y_pred = np.array([[1, 0], [0, 1]], dtype=np.float32)

    assert _dice_coef(y_true, y_pred) == 1.0
    assert _iou_coef(y_true, y_pred) == 1.0


def test_dice_and_iou_no_overlap():
    """No overlap should return near-zero scores."""
    y_true = np.array([[1, 1], [0, 0]], dtype=np.float32)
    y_pred = np.array([[0, 0], [1, 1]], dtype=np.float32)

    dice = _dice_coef(y_true, y_pred)
    iou = _iou_coef(y_true, y_pred)

    assert dice < 1e-5
    assert iou < 1e-5


def test_binary_metrics_all_positive_no_crash():
    """Edge case: all predictions are positive (e.g. at low threshold).
    Must not crash with unpacking error on 1x1 confusion matrix.
    """
    from evaluation.detection_eval import binary_metrics_at_threshold

    y_true = np.array([1, 1, 1, 1])
    y_pred_proba = np.array([0.9, 0.8, 0.95, 0.7])
    metrics = binary_metrics_at_threshold(y_true, y_pred_proba, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["confusion_matrix"].shape == (2, 2)


def test_binary_metrics_all_negative_no_crash():
    """Edge case: all predictions are negative (e.g. at high threshold).
    Must not crash with unpacking error on 1x1 confusion matrix.
    """
    from evaluation.detection_eval import binary_metrics_at_threshold

    y_true = np.array([0, 0, 0, 0])
    y_pred_proba = np.array([0.1, 0.05, 0.2, 0.15])
    metrics = binary_metrics_at_threshold(y_true, y_pred_proba, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["confusion_matrix"].shape == (2, 2)


def test_calibrate_binary_threshold():
    """Calibrate binary threshold finds best operating point."""
    from evaluation.detection_eval import calibrate_binary_threshold

    y_true = np.array([0, 0, 1, 1, 1, 0, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.6, 0.8, 0.7, 0.3, 0.9])

    best_thresh, best_metrics = calibrate_binary_threshold(y_true, y_pred_proba)
    assert 0.0 < best_thresh < 1.0
    assert "accuracy" in best_metrics
    assert "f1_score" in best_metrics
