"""End-to-end tests that actually render the Streamlit app.

Streamlit ships a testing harness (``streamlit.testing.v1.AppTest``) that runs
the real script, so these exercise the widget tree, the tab structure and the
upload flow rather than only the pure helpers. That is what catches a broken
label, a missing widget key, or a tab that raises on render -- defects a helper
test cannot see.

Model loading is stubbed, so the suite stays fast and does not depend on the
Git LFS weights being present.
"""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
APP = REPO_ROOT / "app.py"

tf = pytest.importorskip("tensorflow")

# The app builds three tabs: flagship workflow, classifier only, research.
EXPECTED_TABS = 3
# Two uploaders render up front. The segmentation uploader lives inside the
# research tab behind an opt-in checkbox, so it only appears once enabled.
EXPECTED_UPLOADERS = 2


class _StubModel:
    """Stands in for a loaded Keras model, returning fixed-width outputs.

    Returning the configured vector (rather than a random one) means the app's
    shape checks and class-index guards run for real.
    """

    def __init__(self, outputs):
        self._outputs = np.asarray(outputs, dtype=np.float32)

    def predict(self, x, verbose=0):
        batch = np.asarray(x).shape[0]
        return np.repeat(self._outputs[None, :], batch, axis=0)


_SEGMENTATION_STUB = SimpleNamespace(
    input_shape=(None, 128, 128, 1),
    predict=lambda x, verbose=0: np.zeros(
        (np.asarray(x).shape[0], 128, 128, 1), np.float32
    ),
)
_GENERATOR_STUB = _StubModel(np.zeros((128, 128, 1), np.float32))


def _png_bytes(size=(64, 64), seed=0):
    """A minimal valid PNG for the upload widgets."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    buffer = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8)).save(buffer, "PNG")
    return buffer.getvalue()


def _values(*element_lists) -> list[str]:
    """Flatten one or more AppTest element lists into rendered strings.

    AppTest returns ``ElementList`` objects, which do not support ``+``, so
    callers pass the lists as separate arguments.
    """
    out = []
    for elements in element_lists:
        for element in elements:
            value = getattr(element, "value", element)
            if isinstance(value, (list, tuple)):
                out.extend(str(v) for v in value)
            else:
                out.append(str(value))
    return out


@pytest.fixture(autouse=True)
def stub_models(monkeypatch):
    """Return stubs for model loading so the app renders without the weights.

    ``AppTest.from_file`` executes app.py as a *script*, not as an imported
    module, so patching attributes on an ``import app`` handle has no effect on
    the code under test. The patch therefore has to sit below the script, at
    ``tf.keras.models.load_model``, which is what app.py actually calls.

    ``load_core_models`` is wrapped in ``@st.cache_resource``, so the cache must
    be cleared around every test. Without this the first test's stubs are reused
    for the rest of the session and a later test cannot inject a failing loader.
    """
    import streamlit as st

    st.cache_resource.clear()

    def fake_load_model(path, *args, **kwargs):
        name = Path(str(path)).stem
        if "segment" in name:
            return _SEGMENTATION_STUB
        if "generator" in name:
            return _GENERATOR_STUB
        if "detection" in name:
            return _StubModel([0.90])                     # above the threshold
        if "classifier" in name:
            return _StubModel([0.05, 0.80, 0.10, 0.05])  # -> meningioma
        raise AssertionError(f"unexpected model load: {path}")

    monkeypatch.setattr(tf.keras.models, "load_model", fake_load_model)
    yield fake_load_model
    st.cache_resource.clear()


@pytest.fixture
def at():
    return AppTest.from_file(str(APP), default_timeout=180)


# ── Rendering ────────────────────────────────────────────────────────────

class TestAppRenders:
    def test_script_runs_without_exception(self, at):
        at.run()
        assert not at.exception, at.exception

    def test_title_is_present(self, at):
        at.run()
        assert any("MRI" in t for t in _values(at.title))

    def test_all_three_tabs_render(self, at):
        at.run()
        assert len(at.tabs) == EXPECTED_TABS

    def test_tab_labels_are_distinct(self, at):
        at.run()
        labels = [tab.label for tab in at.tabs]
        assert len(set(labels)) == len(labels)

    def test_threshold_is_surfaced_in_the_sidebar(self, at):
        at.run()
        labels = [str(m.label) for m in at.sidebar.metric]
        assert any("threshold" in label.lower() for label in labels)

    def test_sidebar_reports_loaded_models(self, at):
        at.run()
        captions = " ".join(_values(at.sidebar.caption))
        assert "detection" in captions.lower()

    def test_medical_disclaimer_is_rendered(self, at):
        """The scope disclaimer must survive any UI refactor."""
        at.run()
        body = " ".join(_values(at.info, at.warning, at.error, at.caption))
        lowered = body.lower()
        assert "clinical" in lowered or "diagnostic" in lowered

    def test_2d_slice_scope_is_stated(self, at):
        at.run()
        body = " ".join(_values(at.info, at.warning, at.error, at.markdown))
        assert "2d" in body.lower()


# ── Uploads ──────────────────────────────────────────────────────────────

class TestUploadWidgets:
    def test_uploaders_exist(self, at):
        at.run()
        assert len(at.file_uploader) == EXPECTED_UPLOADERS

    def test_upload_widgets_are_distinct(self, at):
        at.run()
        keys = {u.key for u in at.file_uploader}
        assert len(keys) == EXPECTED_UPLOADERS

    def test_segmentation_uploader_appears_when_research_enabled(self, at):
        """The research tab's uploader is gated behind the opt-in checkbox."""
        at.run()
        assert len(at.file_uploader) == EXPECTED_UPLOADERS
        checkbox = next(c for c in at.checkbox if "experimental" in str(c.label).lower())
        checkbox.set_value(True)
        at.run()
        assert len(at.file_uploader) > EXPECTED_UPLOADERS

    def test_no_upload_is_a_safe_no_op(self, at):
        at.run()
        assert not at.exception

    def test_detection_only_still_screens(self, at, monkeypatch):
        import app as app_module

        monkeypatch.setattr(
            app_module, "load_core_models",
            lambda: {"detection": _StubModel([0.95])},
        )
        at.run()
        assert not at.exception


# ── Research tab ─────────────────────────────────────────────────────────

class TestResearchTab:
    def test_research_models_stay_unloaded_by_default(self, at):
        """Regression: eagerly loading 250 MB of research weights.

        The opt-in checkbox guards that, so it must exist and default to off.
        """
        at.run()
        checkboxes = [c for c in at.checkbox if "experimental" in str(c.label).lower()]
        assert checkboxes, "expected an opt-in control for the research models"
        assert all(not c.value for c in checkboxes)

    def test_enabling_research_does_not_raise(self, at):
        at.run()
        checkbox = next(c for c in at.checkbox if "experimental" in str(c.label).lower())
        checkbox.set_value(True)
        at.run()
        assert not at.exception, at.exception


# ── Error paths ──────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_missing_weights_shows_guidance(self, monkeypatch):
        """A missing core model must tell the user what to do, not fail silently."""
        def no_models(path, *args, **kwargs):
            raise FileNotFoundError(path)

        monkeypatch.setattr(tf.keras.models, "load_model", no_models)
        fresh = AppTest.from_file(str(APP), default_timeout=180)
        fresh.run()
        # Load failures surface immediately, before any upload.
        errors = " ".join(_values(fresh.error)).lower()
        assert "failed to load" in errors
        # And the upload path explains how to fix it.
        fresh.file_uploader[0].upload("scan.png", _png_bytes(), "image/png")
        fresh.run()
        assert not fresh.exception, fresh.exception
        body = " ".join(_values(fresh.error)).lower()
        assert "lfs" in body or "weights" in body

    def test_mismatched_class_output_does_not_crash(self, monkeypatch):
        """A model with the wrong output width must not silently mislabel.

        Regression: ``CLASS_NAMES[class_index]`` used to index past the end of
        the list, or attach the wrong label, when a checkpoint's output width
        disagreed with the configured class count.
        """

        def wrong_width(path, *args, **kwargs):
            name = Path(str(path)).stem
            if "classifier" in name:
                return _StubModel([0.5, 0.5])  # 2 scores vs 4 classes
            return _StubModel([0.99])

        monkeypatch.setattr(tf.keras.models, "load_model", wrong_width)
        fresh = AppTest.from_file(str(APP), default_timeout=180)
        fresh.run()
        fresh.file_uploader[0].upload("scan.png", _png_bytes(), "image/png")
        fresh.run()
        assert not fresh.exception, fresh.exception
        body = " ".join(_values(fresh.error)).lower()
        assert "out of sync" in body, f"expected a sync error, got: {body[:200]}"


# ── Contract with config ─────────────────────────────────────────────────

class TestConfigContract:
    def test_app_uses_the_shared_preprocessing_contracts(self):
        source = APP.read_text(encoding="utf-8")
        for fn in ("preprocess_detection", "preprocess_classifier", "preprocess_segmentation"):
            assert fn in source, f"{fn} is not used by the app"

    def test_no_hardcoded_latent_dimension(self):
        """Regression: the GAN tab hard-coded 100 instead of config.LATENT_DIM."""
        source = APP.read_text(encoding="utf-8")
        assert "tf.random.normal([1, 100])" not in source
        assert "LATENT_DIM" in source

    def test_module_imports_cleanly(self):
        import app as app_module

        assert callable(app_module.main)
        assert callable(app_module._top_class)
        assert callable(app_module._load_image)
