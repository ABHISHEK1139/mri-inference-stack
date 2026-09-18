"""Tests for dataset discovery, label parsing, and mask pairing logic."""

from __future__ import annotations

# Import the functions under test
from data.dataset import (
    _canonical_class,
    _extract_patient_id,
    _normalize_token,
    _pair_brats_images_and_masks,
)


# ── _normalize_token ─────────────────────────────────────────────────────

class TestNormalizeToken:
    def test_lowercases(self):
        assert _normalize_token("Glioma") == "glioma"

    def test_strips_non_alnum(self):
        assert _normalize_token("no_tumor") == "notumor"
        assert _normalize_token("glioma-tumor") == "gliomatumor"

    def test_empty(self):
        assert _normalize_token("") == ""


# ── _canonical_class ─────────────────────────────────────────────────────

class TestCanonicalClass:
    """Tests that class matching is strict (exact alias only)."""

    def test_exact_matches(self):
        assert _canonical_class("glioma") == "glioma"
        assert _canonical_class("Glioma") == "glioma"
        assert _canonical_class("meningioma") == "meningioma"
        assert _canonical_class("pituitary") == "pituitary"
        assert _canonical_class("normal") == "normal"
        assert _canonical_class("notumor") == "normal"
        assert _canonical_class("no_tumor") == "normal"
        assert _canonical_class("healthy") == "normal"

    def test_compound_aliases(self):
        assert _canonical_class("GliomaTumor") == "glioma"
        assert _canonical_class("Meningioma_Tumor") == "meningioma"
        assert _canonical_class("Pituitary-Tumor") == "pituitary"

    def test_unknown_returns_none(self):
        assert _canonical_class("unknown") is None
        assert _canonical_class("astrocytoma") is None

    def test_fuzzy_substring_no_longer_matches(self):
        """Substring matching was removed to prevent silent mislabeling.

        Previously, 'normalized' would match 'normal' via substring.
        This must now return None.
        """
        assert _canonical_class("normalized") is None
        assert _canonical_class("normalized_scan_001") is None
        assert _canonical_class("meningioma_normalized") is None


# ── _extract_patient_id ──────────────────────────────────────────────────

class TestExtractPatientId:
    def test_strips_slice_suffix(self):
        assert _extract_patient_id("/data/patient001_slice01.png") == "patient001"
        assert _extract_patient_id("/data/patient001_slice12.png") == "patient001"

    def test_strips_numeric_suffix(self):
        assert _extract_patient_id("/data/patient_a_003.png") == "patient_a"

    def test_strips_frame_suffix(self):
        assert _extract_patient_id("/data/scan_frame3.png") == "scan"

    def test_no_suffix_returns_stem(self):
        assert _extract_patient_id("/data/standalone.png") == "standalone"

    def test_only_digits_returns_stem(self):
        # Edge case: filename is only digits
        result = _extract_patient_id("/data/001.png")
        assert result == "001"


# ── _pair_brats_images_and_masks ─────────────────────────────────────────

class TestPairBratsImagesAndMasks:
    def test_basic_pairing(self, tmp_path):
        (tmp_path / "patient001.png").touch()
        (tmp_path / "patient001_mask.png").touch()
        images, masks = _pair_brats_images_and_masks(tmp_path)
        assert len(images) == 1
        assert len(masks) == 1
        assert "patient001" in images[0]
        assert "mask" in masks[0]

    def test_unmatched_files_excluded(self, tmp_path):
        (tmp_path / "patient001.png").touch()
        (tmp_path / "patient002_mask.png").touch()  # No matching image
        images, masks = _pair_brats_images_and_masks(tmp_path)
        assert len(images) == 0
        assert len(masks) == 0

    def test_multiple_pairs(self, tmp_path):
        for i in range(3):
            (tmp_path / f"scan{i:03d}.png").touch()
            (tmp_path / f"scan{i:03d}_seg.png").touch()
        images, masks = _pair_brats_images_and_masks(tmp_path)
        assert len(images) == 3
        assert len(masks) == 3
