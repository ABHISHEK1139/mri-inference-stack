"""Streamlit demo for the Brain MRI intelligence system."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

from config import (
    CHECKPOINT_DIR,
    CLASS_NAMES,
    LATENT_DIM,
    NUM_CLASSES,
    PROJECT_NAME,
    WEIGHTS_DIR,
    ensure_directories,
)
from preprocessing import (
    preprocess_classifier,
    preprocess_detection,
    preprocess_segmentation,
)

logger = logging.getLogger(__name__)


DETECTION_CONFIG_CANDIDATES = [
    Path(WEIGHTS_DIR) / "detection_inference_config.json",
    Path(CHECKPOINT_DIR) / "detection" / "inference_config.json",
]

SEGMENTATION_CUSTOM_OBJECTS = None  # populated lazily in load_research_models


def _load_image(image_file) -> Image.Image:
    """Decode an upload into a grayscale PIL image.

    The buffer is read inside a ``with`` block so Streamlit's upload handle is
    not held open for the lifetime of the session, and the image is fully
    materialised before the handle is released.

    Raises:
        ValueError: The upload could not be decoded. Raising rather than calling
            ``st.stop()`` keeps the failure observable; ``st.stop`` is a no-op
            outside a Streamlit script context, which would otherwise let
            ``None`` reach the caller.
    """
    try:
        with Image.open(image_file) as image:
            return image.convert("L")
    except Exception as exc:
        logger.exception("Could not decode the uploaded image")
        raise ValueError(f"Could not read that image: {exc}") from exc


def _load_detection_config() -> dict:
    for config_path in DETECTION_CONFIG_CANDIDATES:
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    return {"threshold": 0.5, "validation_metrics": {}}


def _is_lfs_pointer(path: Path) -> bool:
    """Check if a file is a Git LFS pointer instead of actual content."""
    try:
        with open(path, "rb") as f:
            header = f.read(48)
        return header.startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


@st.cache_resource
def load_core_models() -> dict[str, Any]:
    import tensorflow as tf

    # Configure GPU memory growth before loading any models
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # Memory growth must be set before GPUs are initialized

    models: dict[str, Any] = {}
    detection_path = Path(WEIGHTS_DIR) / "detection_model.keras"
    classifier_path = Path(WEIGHTS_DIR) / "classifier_model.keras"

    for name, path in [("detection", detection_path), ("classifier", classifier_path)]:
        if not path.exists():
            continue
        if _is_lfs_pointer(path):
            msg = (
                f"{name.title()} weights at {path} are a Git LFS pointer, "
                f"not the actual model. Run `git lfs pull` to download."
            )
            logger.error(msg)
            st.error(msg)
            continue
        try:
            models[name] = tf.keras.models.load_model(path, compile=False)
        except Exception as exc:
            logger.exception("Could not load %s model from %s", name, path)
            st.error(f"Failed to load {name} model: {exc}")

    return models


@st.cache_resource
def load_research_models() -> dict[str, Any]:
    import tensorflow as tf

    from models.segmentation import dice_bce_loss, dice_coefficient, iou_metric

    global SEGMENTATION_CUSTOM_OBJECTS
    SEGMENTATION_CUSTOM_OBJECTS = {
        "dice_bce_loss": dice_bce_loss,
        "dice_coefficient": dice_coefficient,
        "iou_metric": iou_metric,
    }

    models: dict[str, Any] = {}
    segmentation_path = Path(WEIGHTS_DIR) / "segmentation_model.keras"
    generator_path = Path(WEIGHTS_DIR) / "generator_conditional.keras"

    if segmentation_path.exists():
        if _is_lfs_pointer(segmentation_path):
            logger.error("Segmentation weights are a Git LFS pointer. Run `git lfs pull`.")
        else:
            try:
                models["segmentation"] = tf.keras.models.load_model(
                    segmentation_path,
                    compile=False,
                    custom_objects=SEGMENTATION_CUSTOM_OBJECTS,
                )
            except Exception as exc:
                logger.exception("Could not load segmentation model")
                st.warning(f"Could not load segmentation model: {exc}")

    if generator_path.exists():
        if _is_lfs_pointer(generator_path):
            logger.error("Generator weights are a Git LFS pointer. Run `git lfs pull`.")
        else:
            try:
                models["generator"] = tf.keras.models.load_model(generator_path, compile=False)
            except Exception as exc:
                logger.exception("Could not load generator model")
                st.warning(f"Could not load generator model: {exc}")

    return models


def _top_class(predictions: np.ndarray) -> tuple[int, float]:
    """Return (argmax index, softmax score), validated against CLASS_NAMES.

    A model whose output width disagrees with ``CLASS_NAMES`` used to index past
    the end of the list (or silently mislabel a class), so the mismatch is
    surfaced instead.
    """
    scores = np.asarray(predictions).ravel()
    if scores.size != NUM_CLASSES:
        raise ValueError(
            f"Model returned {scores.size} class scores but CLASS_NAMES has {NUM_CLASSES} "
            f"entries. The weights and config are out of sync."
        )
    index = int(np.argmax(scores))
    return index, float(scores[index])


def render_sidebar(core_models: dict[str, Any], detection_config: dict) -> None:
    st.sidebar.header("Repo Focus")
    st.sidebar.markdown(
        "Flagship workflow: calibrated tumour screening plus tumour type classification."
    )
    st.sidebar.markdown(
        "Research extensions: segmentation and synthetic MRI generation remain available, "
        "but they are not the primary public promise of the repo."
    )

    threshold = float(detection_config.get("threshold", 0.5))
    val_metrics = detection_config.get("validation_metrics", {})
    st.sidebar.metric("Detection threshold", f"{threshold:.3f}")
    if val_metrics:
        st.sidebar.metric("Validation F1", f"{val_metrics.get('f1_score', 0.0):.4f}")
        st.sidebar.metric("Validation recall", f"{val_metrics.get('recall', 0.0):.4f}")

    available = ", ".join(sorted(core_models)) if core_models else "none"
    st.sidebar.caption(f"Loaded core models: {available}")


def render_flagship_workflow(core_models: dict[str, Any], detection_config: dict) -> None:
    st.subheader("Flagship Workflow")
    st.info(
        "⚠️ **Scope**: This model accepts preprocessed 2D axial brain MRI slices only. "
        "It does not handle volumetric formats (NIfTI, DICOM). "
        "**This is not a diagnostic tool** and should not be used for clinical decisions."
    )
    st.write(
        "Upload a brain MRI slice to run the calibrated screening model. "
        "If tumour likelihood is above the saved operating threshold, "
        "the tumour type classifier runs next."
    )

    uploaded = st.file_uploader(
        "Upload an MRI image",
        type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
        key="pipeline_upload",
    )
    if uploaded is None:
        return

    try:
        image = _load_image(uploaded)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.image(image, caption="Uploaded grayscale MRI", width=320)

    if "detection" not in core_models:
        st.error(
            "Detection weights are not available. Pull the Git LFS files before running the demo."
        )
        return

    threshold = float(detection_config.get("threshold", 0.5))
    input_tensor = preprocess_detection(image)
    probability = float(core_models["detection"].predict(input_tensor, verbose=0)[0][0])
    tumour_likely = probability >= threshold

    if tumour_likely:
        st.warning("Screening model: elevated tumour probability")
    else:
        st.success("Screening model: below detection threshold")
    st.write(f"Detection probability: `{probability:.4f}`")
    st.write(f"Decision threshold: `{threshold:.4f}`")
    st.caption("This is a model prediction, not a clinical diagnosis.")

    if not tumour_likely:
        st.info(
            "Tumour type classification is skipped because the screening model "
            "stayed below threshold."
        )
        return

    if "classifier" not in core_models:
        st.warning("Classifier weights are not available, so only screening is shown.")
        return

    predictions = np.asarray(core_models["classifier"].predict(preprocess_classifier(image),
        verbose=0)[0])
    try:
        class_index, _ = _top_class(predictions)
    except ValueError as exc:
        st.error(str(exc))
        return
    predicted_label = CLASS_NAMES[class_index]

    st.write(f"Predicted tumour type: `{predicted_label.title()}`")
    st.bar_chart({CLASS_NAMES[idx].title(): float(score) for idx, score in enumerate(predictions)})


def render_classifier_only(core_models: dict[str, Any]) -> None:
    st.subheader("Tumour Type Classifier")
    st.write(
        "Use the multi-class classifier directly when you already know the slice "
        "contains a tumour."
    )

    uploaded = st.file_uploader(
        "Upload an MRI image for tumour type classification",
        type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
        key="classifier_upload",
    )
    if uploaded is None:
        return

    try:
        image = _load_image(uploaded)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.image(image, caption="Uploaded grayscale MRI", width=320)

    if "classifier" not in core_models:
        st.error(
            "Classifier weights are not available. Pull the Git LFS files before "
            "running the demo."
        )
        return

    predictions = np.asarray(core_models["classifier"].predict(preprocess_classifier(image),
        verbose=0)[0])
    try:
        class_index, _ = _top_class(predictions)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.success(f"Predicted tumour type: {CLASS_NAMES[class_index].title()}")
    st.bar_chart({CLASS_NAMES[idx].title(): float(score) for idx, score in enumerate(predictions)})


def render_research_extensions() -> None:
    st.subheader("Research Extensions")
    st.warning(
        "These modules are intentionally marked experimental. They stay in the repo "
        "as research tracks, not as the default production demo."
    )

    enable_research = st.checkbox("Load experimental models", value=False)
    if not enable_research:
        st.info(
            "Experimental models stay unloaded by default to keep the core demo "
            "focused and lightweight."
        )
        return

    research_models = load_research_models()
    subtab_seg, subtab_gan = st.tabs(["Segmentation", "Synthetic MRI"])

    with subtab_seg:
        st.write("Attention U-Net segmentation preview.")
        uploaded = st.file_uploader(
            "Upload an MRI image for segmentation",
            type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
            key="segmentation_upload",
        )
        if uploaded is None:
            st.info("Upload an MRI slice to preview segmentation.")
        elif "segmentation" not in research_models:
            st.error("Segmentation weights are not available in this repo snapshot.")
        else:
            try:
                image = _load_image(uploaded)
            except ValueError as exc:
                st.error(str(exc))
                return
            seg_model = research_models["segmentation"]
            input_shape = seg_model.input_shape
            tensor = preprocess_segmentation(image, target_size=(input_shape[1], input_shape[2]))
            pred_mask = seg_model.predict(tensor, verbose=0)[0, :, :, 0]
            pred_mask = (pred_mask > 0.5).astype(np.float32)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original MRI", use_container_width=True)
            with col2:
                overlay = np.asarray(image.resize((input_shape[2], input_shape[1])),
                    dtype=np.float32) / 255.0
                overlay_rgb = np.stack([overlay, overlay, overlay], axis=-1)
                overlay_rgb[..., 0] = np.maximum(overlay_rgb[..., 0],
                    pred_mask)
                st.image(overlay_rgb, caption="Predicted mask overlay", use_container_width=True,
                    clamp=True)

    with subtab_gan:
        st.write("Conditional GAN preview for synthetic MRI slices.")
        if "generator" not in research_models:
            st.error("Generator weights are not available in this repo snapshot.")
        else:
            import tensorflow as tf

            target_class = st.selectbox("Condition class", CLASS_NAMES, key="gan_class")
            if st.button("Generate synthetic MRI", key="gan_generate"):
                class_index = CLASS_NAMES.index(target_class)
                noise = tf.random.normal([1, LATENT_DIM])
                label = tf.one_hot([class_index], NUM_CLASSES)
                generated = research_models["generator"]([noise, label], training=False)[0, :, :, 0]
                generated = (generated + 1.0) / 2.0
                st.image(
                    generated.numpy(),
                    caption=f"Synthetic {target_class.title()} MRI",
                    width=320,
                    clamp=True,
                )


def main() -> None:
    st.set_page_config(page_title=PROJECT_NAME, layout="wide")
    ensure_directories()
    detection_config = _load_detection_config()
    core_models = load_core_models()

    st.title(PROJECT_NAME)
    st.caption(
        "A resource-constrained brain MRI pipeline centered on calibrated tumour detection "
        "and multi-class subtype classification."
    )
    render_sidebar(core_models, detection_config)

    flagship_tab, classifier_tab, research_tab = st.tabs(
        ["Flagship Workflow", "Tumour Type Classifier", "Research Extensions"]
    )

    with flagship_tab:
        render_flagship_workflow(core_models, detection_config)

    with classifier_tab:
        render_classifier_only(core_models)

    with research_tab:
        render_research_extensions()


if __name__ == "__main__":
    main()
