"""Evaluation metrics: Fréchet distances, classification, segmentation, plots.

This module is a facade over the :mod:`evaluation` package. It re-exports every
public name so ``from evaluation import calculate_fid`` keeps working,
while the implementations live in focused modules:

===========================  ==========================================
:mod:`evaluation.frechet`     FID over InceptionV3, relative FS
:mod:`evaluation.classification`  Classifier and binary-detection reports
:mod:`evaluation.segmentation`    Segmentation Dice/IoU reports
:mod:`evaluation.plots`           Confusion matrices and training curves
===========================  ==========================================
"""

from evaluation.classification import (
    _safe_auc,
    evaluate_classifier,
    evaluate_detection,
)
from evaluation.frechet import (
    _calculate_statistics,
    _frechet_distance,
    _matrix_sqrtm,
    _to_unit_range,
    build_inception_feature_extractor,
    calculate_fid,
    calculate_fs,
    preprocess_for_inception,
)
from evaluation.plots import (
    _ensure_parent_dir,
    plot_confusion_matrix,
    plot_fid_fs_vs_epochs,
    plot_gan_losses,
    plot_loss_curves,
)
from evaluation.segmentation import (
    _dice_coef,
    _iou_coef,
    evaluate_segmentation,
)

__all__ = [
    # Fréchet distances
    "build_inception_feature_extractor",
    "calculate_fid",
    "calculate_fs",
    "preprocess_for_inception",
    # Classification
    "evaluate_classifier",
    "evaluate_detection",
    # Segmentation
    "evaluate_segmentation",
    # Plots
    "plot_confusion_matrix",
    "plot_fid_fs_vs_epochs",
    "plot_gan_losses",
    "plot_loss_curves",
    # Helpers
    "_calculate_statistics",
    "_dice_coef",
    "_ensure_parent_dir",
    "_frechet_distance",
    "_iou_coef",
    "_matrix_sqrtm",
    "_safe_auc",
    "_to_unit_range",
]
