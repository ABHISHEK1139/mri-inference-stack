"""Regression tests for evaluation metrics and the detection calibrator.

These cover failures that produced plausible-looking but wrong numbers, or
crashed outright on supported dependency versions.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation import (
    _calculate_statistics,
    _frechet_distance,
    _matrix_sqrtm,
    _safe_auc,
    _to_unit_range,
    plot_fid_fs_vs_epochs,
    plot_gan_losses,
    plot_loss_curves,
)
from evaluation.detection_eval import (
    binary_metrics_at_threshold,
    calibrate_binary_threshold,
)

tf = pytest.importorskip("tensorflow")


# ── Fréchet distance / FID plumbing ──────────────────────────────────────

class TestMatrixSqrtm:
    """Regression: ``sqrtm(..., disp=False)`` raises on SciPy >= 1.16.

    ``scipy.linalg.sqrtm`` removed the ``disp`` argument; the old call raised
    ``TypeError`` on modern SciPy. Since every FID/FS call sat inside a blanket
    ``except Exception``, GAN quality monitoring silently stopped working.
    """

    def test_does_not_raise(self):
        result = _matrix_sqrtm(np.eye(3))
        assert np.allclose(result, np.eye(3), atol=1e-6)

    def test_real_result_for_valid_input(self):
        assert not np.iscomplexobj(_matrix_sqrtm(np.diag([4.0, 9.0])))

    def test_round_trips_a_psd_matrix(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(6, 6))
        psd = a @ a.T
        root = _matrix_sqrtm(psd)
        assert np.allclose(root @ root, psd, atol=1e-6)


class TestStatistics:
    """Regression: ``np.cov`` returns all-NaN for fewer than 2 samples.

    That NaN propagated into FID/FS, and because ``NaN < best`` is False the
    early-stopping counter incremented on every evaluation.
    """

    def test_single_sample_raises_instead_of_returning_nan(self):
        with pytest.raises(ValueError, match="At least 2 samples"):
            _calculate_statistics(np.zeros((1, 8)))

    def test_two_samples_work(self):
        mu, sigma = _calculate_statistics(np.random.default_rng(0).normal(size=(4, 8)))
        assert mu.shape == (8,)
        assert sigma.shape == (8, 8)

    def test_non_finite_features_rejected(self):
        bad = np.array([[1.0, 2.0], [3.0, np.nan]])
        with pytest.raises(ValueError, match="non-finite"):
            _calculate_statistics(bad)

    def test_frechet_distance_of_identical_sets_is_zero(self):
        features = np.random.default_rng(0).normal(size=(32, 8))
        assert _frechet_distance(features, features) == pytest.approx(0.0, abs=1e-6)

    def test_frechet_distance_grows_with_disagreement(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(32, 8))
        near = _frechet_distance(a, a + 0.1)
        far = _frechet_distance(a, a + 5.0)
        assert far > near


class TestUnitRangeConversion:
    def test_minus_one_one_maps_to_zero_one(self):
        result = _to_unit_range(np.array([-1.0, 0.0, 1.0], np.float32))
        assert np.allclose(result, [0.0, 0.5, 1.0])

    def test_zero_one_passes_through(self):
        result = _to_unit_range(np.array([0.0, 0.25, 1.0], np.float32))
        assert np.allclose(result, [0.0, 0.25, 1.0])

    def test_all_zero_batch_is_not_rescaled(self):
        """Regression: a min() heuristic made an all-black batch ambiguous."""
        result = _to_unit_range(np.zeros((2, 2), np.float32))
        assert np.allclose(result, 0.0)

    def test_empty_input_is_safe(self):
        assert _to_unit_range(np.zeros((0, 4), np.float32)).size == 0


class TestSafeAuc:
    """Regression: ``except Exception: auc = 0.0`` looked like a real score."""

    def test_degenerate_input_returns_nan(self):
        with pytest.warns(RuntimeWarning, match="ROC-AUC"):
            value = _safe_auc(np.zeros(10), np.zeros(10))
        assert np.isnan(value)

    def test_nan_is_not_mistaken_for_zero(self):
        with pytest.warns(RuntimeWarning):
            assert not _safe_auc(np.zeros(4), np.zeros(4)) == 0.0

    def test_valid_input_computes_normally(self):
        y = np.array([0, 0, 1, 1])
        assert _safe_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


# ── Detection threshold calibration ──────────────────────────────────────

class TestBinaryMetrics:
    def test_confusion_matrix_ordering(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.0, 0.9, 0.2, 0.8])
        metrics = binary_metrics_at_threshold(y_true, y_pred, threshold=0.5)
        # tn=1 (idx0), fp=1 (idx1), fn=1 (idx2), tp=1 (idx3)
        assert metrics["true_negatives"] == 1
        assert metrics["false_positives"] == 1
        assert metrics["false_negatives"] == 1
        assert metrics["true_positives"] == 1

    def test_threshold_boundary_is_inclusive(self):
        """Regression: metrics.py used ``> 0.5`` while detection_eval used ``>=``.

        A score of exactly 0.5 must count as positive here, otherwise the same
        model reports two different confusion matrices.
        """
        metrics = binary_metrics_at_threshold(np.array([1, 0]), np.array([0.5, 0.4]), 0.5)
        assert metrics["true_positives"] == 1
        assert metrics["true_negatives"] == 1
        assert metrics["false_negatives"] == 0

    def test_specificity_and_balanced_accuracy(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0.0, 0.0, 0.1, 0.9, 0.9])
        m = binary_metrics_at_threshold(y_true, y_pred, 0.5)
        assert m["specificity"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["balanced_accuracy"] == pytest.approx(1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            binary_metrics_at_threshold(np.array([0, 1]), np.array([0.5]), 0.5)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            binary_metrics_at_threshold(np.array([]), np.array([]), 0.5)


class TestCalibrateThreshold:
    def test_finds_a_reasonable_threshold(self):
        y_true = np.array([0] * 50 + [1] * 50)
        y_pred = np.array([0.1] * 50 + [0.9] * 50)
        threshold, metrics = calibrate_binary_threshold(y_true, y_pred)
        # Every threshold above 0.1 and below 0.9 separates the classes perfectly,
        # so the F1 plateau is broken towards the most conservative cutoff.
        assert threshold == pytest.approx(0.9)
        assert metrics["f1_score"] == pytest.approx(1.0)

    def test_tie_break_prefers_specificity(self):
        """Regression: ``>`` kept the lowest threshold on an F1 plateau.

        With perfectly separated data every threshold in a wide range scores
        F1 = 1.0. The tie-break now selects the most conservative one.
        """
        y_true = np.array([0] * 20 + [1] * 20)
        y_pred = np.array([0.05] * 20 + [0.95] * 20)
        threshold, metrics = calibrate_binary_threshold(y_true, y_pred)
        assert threshold == pytest.approx(0.95)
        assert metrics["recall_floor_met"] is True

    def test_recall_floor_is_enforced(self):
        y_true = np.array([0] * 10 + [1] * 10)
        y_pred = np.array([0.9] * 10 + [0.95] * 10)  # everything looks positive
        threshold, metrics = calibrate_binary_threshold(
            y_true, y_pred, min_recall=0.5
        )
        assert metrics["recall"] >= 0.5

    def test_unreachable_recall_floor_is_flagged(self):
        """Regression: the 0.5 fallback was indistinguishable from a real result."""
        y_true = np.zeros(20, dtype=int)  # no positives at all
        y_pred = np.linspace(0, 1, 20)
        with pytest.warns(RuntimeWarning, match="min_recall"):
            threshold, metrics = calibrate_binary_threshold(
                y_true, y_pred, min_recall=0.97
            )
        assert threshold == pytest.approx(0.5)
        assert metrics["recall_floor_met"] is False

    def test_unknown_objective_rejected(self):
        with pytest.raises(ValueError, match="optimize"):
            calibrate_binary_threshold(np.array([0, 1]), np.array([0.1, 0.9]),
                                       optimize="accuracy")

    def test_balanced_accuracy_objective(self):
        y_true = np.array([0] * 10 + [1] * 10)
        y_pred = np.array([0.4] * 10 + [0.6] * 10)
        _, metrics = calibrate_binary_threshold(y_true, y_pred,
                                                optimize="balanced_accuracy")
        assert "balanced_accuracy" in metrics


# ── Plotting helpers ─────────────────────────────────────────────────────

class _FakeHistory:
    def __init__(self, **kwargs):
        self.history = kwargs


class TestPlots:
    def test_loss_curves_without_accuracy_metric(self):
        """Regression: ``history.history['dice_coefficient']`` raised KeyError."""
        figure = plot_loss_curves(_FakeHistory(loss=[1.0, 0.5], val_loss=[1.1, 0.6]))
        assert figure is not None

    def test_loss_curves_with_dice(self):
        figure = plot_loss_curves(
            _FakeHistory(
                loss=[1.0],
                val_loss=[1.1],
                dice_coefficient=[0.4],
                val_dice_coefficient=[0.3],
            )
        )
        assert figure is not None

    def test_loss_curves_with_no_metrics_at_all(self):
        figure = plot_loss_curves(_FakeHistory())
        assert figure is not None

    def test_gan_losses_without_accs(self):
        """Regression: the dead ternary still created a blank second axis."""
        figure = plot_gan_losses([0.1], [0.2])
        assert figure is not None

    def test_gan_losses_with_accs(self):
        figure = plot_gan_losses([0.1], [0.2], [0.9], [0.1])
        assert figure is not None

    def test_gan_losses_custom_label(self):
        figure = plot_gan_losses([0.1], [0.2], [1.5], None, d_label="W-Distance")
        assert figure is not None

    def test_fid_fs_length_mismatch_raises(self):
        """Regression: mismatched lengths silently mislabelled the x-axis."""
        with pytest.raises(ValueError, match="same length"):
            plot_fid_fs_vs_epochs([1.0, 2.0], [1.0])
