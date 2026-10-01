"""Filename conventions: class aliases, split aliases, patient grouping.

Separated from I/O so the parsing rules can be reasoned about and tested
without touching the filesystem.
"""

import os
import re
from pathlib import Path

from config import CLASS_NAMES, RAW_DIR

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VOLUME_EXTENSIONS = {".nii", ".nii.gz"}
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "test": "test",
    "testing": "test",
    "val": "val",
    "valid": "val",
    "validation": "val",
}
CLASS_ALIASES = {
    "glioma": "glioma",
    "gliomatumor": "glioma",
    "meningioma": "meningioma",
    "meningiomatumor": "meningioma",
    "pituitary": "pituitary",
    "pituitarytumor": "pituitary",
    "normal": "normal",
    "other": "normal",
    "notumor": "normal",
    "notumour": "normal",
    "no_tumor": "normal",
    "no_tumour": "normal",
    "healthy": "normal",
}

# Suffixes that mark a file as a *label map*. These are only ever matched as a
# trailing token. "tumor"/"tumour" are deliberately absent: BraTS image volumes
# are named `..._brain_tumor_FLAIR` and masks `..._brain_tumor_seg`, so
# substring-matching on "tumor" classified BOTH sides as masks and produced zero
# image/mask pairs.
MASK_SUFFIXES = ("seg", "segmentation", "mask", "label", "labelmap", "gt", "groundtruth",
    "annotation")
# Every entry is already lowercase alphanumeric, so it is directly comparable
# against a token from _TOKEN_SPLIT_RE.
MASK_SUFFIX_TOKENS = frozenset(MASK_SUFFIXES)

# Modality / tissue tokens that may trail a BraTS volume name.
VOLUME_MODALITIES = frozenset(
    {"flair", "t1ce", "t1gd", "t2", "t1", "wt", "et", "tc", "ncr"}
)

# Tokens that mark a slice/frame counter when they trail a filename. A counter
# is stripped only when it is preceded by a separator, so an undelimited trailing
# number such as the "003" in "image003" is preserved: for a per-volume dataset
# that number is the subject ID, and stripping it would collapse every subject
# into a single group, making a patient-level split impossible.
_COUNTER_KEYWORDS = frozenset({"slice", "frame", "img", "image", "fl", "sl"})
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_TRAILING_DIGITS_RE = re.compile(r"^(.*?)(\d+)$")


def _normalize_token(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _has_mask_suffix(stem: str) -> bool:
    """Return True when the final token of ``stem`` is a label-map token."""
    tokens = [t for t in _TOKEN_SPLIT_RE.split(Path(stem).stem.lower()) if t]
    return len(tokens) > 1 and tokens[-1] in MASK_SUFFIX_TOKENS


def _extract_patient_id(path: str) -> str:
    """Extract a patient identifier from a file path.

    Groups slices/volumes belonging to the same subject so that patient-level
    splitting cannot leak data across partitions. The stem is tokenised on
    non-alphanumeric characters and, in order: a trailing label-map token is
    dropped, a trailing modality token is dropped, and a single trailing slice
    counter is dropped. Examples::

        BraTS20_Training_001_brain_tumor_FLAIR -> brats20training001braintumor
        BraTS20_Training_001_brain_tumor_seg    -> brats20training001braintumor
        patient001_slice01.png                   -> patient001
        patient_a_003.png                        -> patienta
        scan_frame3.png                          -> scan
        glioma (12).png                          -> glioma
        image003.png                             -> image003   (subject ID)

    The counter step runs at most once, so ``Case-045_s3`` yields ``case_045_s3``
    rather than losing the patient number. Tokens are rejoined with ``_``, so
    separator style never splits one patient into two groups
    (``Case-045_s3`` and ``Case_045_s3`` agree). A counter is never stripped when
    it would empty the key, so distinct subjects cannot collapse into one group.
    """
    raw = Path(path).stem.lower()
    tokens = [t for t in _TOKEN_SPLIT_RE.split(raw) if t]

    if not tokens:
        return _normalize_token(raw) or raw

    # 1. Trailing label-map token ("seg", "mask", ...).
    if len(tokens) > 1 and tokens[-1] in MASK_SUFFIX_TOKENS:
        tokens.pop()
    # 2. Trailing modality token ("flair", "t1ce", ...).
    if len(tokens) > 1 and tokens[-1] in VOLUME_MODALITIES:
        tokens.pop()
    # 3. A single trailing slice counter ("_003", "_slice01", "frame3").
    if len(tokens) > 1:
        match = _TRAILING_DIGITS_RE.match(tokens[-1])
        if match is not None:
            base, digits = match.groups()
            # "_003" -> bare number; "_slice01" / "frame3" -> keyword + number.
            if digits and (not base or base in _COUNTER_KEYWORDS):
                tokens.pop()

    return "_".join(tokens) or _normalize_token(raw) or raw


def _canonical_split(part: str) -> str | None:
    return SPLIT_ALIASES.get(_normalize_token(part))


def _canonical_class(part: str) -> str | None:
    """Map a path component to a canonical class name via exact alias match."""
    token = _normalize_token(part)
    if token in CLASS_ALIASES:
        return CLASS_ALIASES[token]
    # Fuzzy substring matching removed — it could silently assign wrong labels
    # (e.g., 'normalized' matching 'normal'). Unknown tokens return None.
    return None


def _as_path(path_like: str | os.PathLike[str] | None, default_name: str) -> Path:
    if path_like:
        return Path(path_like)
    return Path(RAW_DIR) / default_name
