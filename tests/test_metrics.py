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
