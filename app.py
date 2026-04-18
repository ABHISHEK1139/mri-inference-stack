"""Streamlit demo for the Brain MRI intelligence system."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

from config import CHECKPOINT_DIR, CLASS_NAMES, PROJECT_NAME, WEIGHTS_DIR


DETECTION_CONFIG_CANDIDATES = [
    Path(WEIGHTS_DIR) / "detection_inference_config.json",
    Path(CHECKPOINT_DIR) / "detection" / "inference_config.json",
]


def _load_image(image_file) -> Image.Image:
    return Image.open(image_file).convert("L")


def _preprocess(image: Image.Image, target_size: tuple[int, int]) -> np.ndarray:
    resized = image.resize((target_size[1], target_size[0]))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=(0, -1))


def _load_detection_config() -> dict:
    for config_path in DETECTION_CONFIG_CANDIDATES:
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    return {"threshold": 0.5, "validation_metrics": {}}


@st.cache_resource
def load_core_models() -> dict[str, Any]:
    import tensorflow as tf

    models: dict[str, Any] = {}
    detection_path = Path(WEIGHTS_DIR) / "detection_model.keras"
    classifier_path = Path(WEIGHTS_DIR) / "classifier_model.keras"

    if detection_path.exists():
        try:
            models["detection"] = tf.keras.models.load_model(detection_path, compile=False)
        except Exception as exc:  # pragma: no cover - defensive UI fallback
            print(f"Could not load detection model: {exc}")
    if classifier_path.exists():
        try:
            models["classifier"] = tf.keras.models.load_model(classifier_path, compile=False)
        except Exception as exc:  # pragma: no cover - defensive UI fallback
            print(f"Could not load classifier model: {exc}")
    return models


@st.cache_resource
def load_research_models() -> dict[str, Any]:
    import tensorflow as tf
    from models.segmentation import dice_bce_loss, dice_coefficient, iou_metric

    models: dict[str, Any] = {}
    segmentation_path = Path(WEIGHTS_DIR) / "segmentation_model.keras"
    generator_path = Path(WEIGHTS_DIR) / "generator_conditional.keras"

    if segmentation_path.exists():
        try:
            models["segmentation"] = tf.keras.models.load_model(
                segmentation_path,
                compile=False,
                custom_objects={
                    "dice_bce_loss": dice_bce_loss,
                    "dice_coefficient": dice_coefficient,
                    "iou_metric": iou_metric,
                },
            )
        except Exception as exc:  # pragma: no cover - defensive UI fallback
            print(f"Could not load segmentation model: {exc}")
    if generator_path.exists():
        try:
            models["generator"] = tf.keras.models.load_model(generator_path, compile=False)
        except Exception as exc:  # pragma: no cover - defensive UI fallback
            print(f"Could not load generator model: {exc}")
    return models


def render_sidebar(core_models: dict[str, Any], detection_config: dict) -> None:
    st.sidebar.header("Repo Focus")
    st.sidebar.markdown(
        "Flagship workflow: calibrated tumour screening plus subtype classification."
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
    st.write(
        "Upload a brain MRI slice to run the calibrated screening model. "
        "If tumour likelihood is above the saved operating threshold, the subtype classifier runs next."
    )

    uploaded = st.file_uploader(
        "Upload an MRI image",
        type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
        key="pipeline_upload",
    )
    if uploaded is None:
        return

    image = _load_image(uploaded)
    st.image(image, caption="Uploaded grayscale MRI", width=320)

    if "detection" not in core_models:
        st.error("Detection weights are not available. Pull the Git LFS files before running the demo.")
        return

    threshold = float(detection_config.get("threshold", 0.5))
    input_tensor = _preprocess(image, target_size=(224, 224))
    probability = float(core_models["detection"].predict(input_tensor, verbose=0)[0][0])
    tumour_likely = probability >= threshold

    if tumour_likely:
        st.warning("Tumour likely")
    else:
        st.success("No tumour detected")
    st.write(f"Detection probability: `{probability:.4f}`")
    st.write(f"Decision threshold: `{threshold:.4f}`")

    if not tumour_likely:
        st.info("Subtype classification is skipped because the screening model stayed below threshold.")
        return

    if "classifier" not in core_models:
        st.warning("Classifier weights are not available, so only screening is shown.")
        return

    classifier_input = _preprocess(image, target_size=(224, 224))
    predictions = core_models["classifier"].predict(classifier_input, verbose=0)[0]
    class_index = int(np.argmax(predictions))
    predicted_label = CLASS_NAMES[class_index]

    st.write(f"Predicted subtype: `{predicted_label.title()}`")
    st.bar_chart({CLASS_NAMES[idx].title(): float(score) for idx, score in enumerate(predictions)})


def render_classifier_only(core_models: dict[str, Any]) -> None:
    st.subheader("Subtype Classifier")
    st.write("Use the multi-class classifier directly when you already know the slice contains a tumour.")

    uploaded = st.file_uploader(
        "Upload an MRI image for subtype classification",
        type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
        key="classifier_upload",
    )
    if uploaded is None:
        return

    image = _load_image(uploaded)
    st.image(image, caption="Uploaded grayscale MRI", width=320)

    if "classifier" not in core_models:
        st.error("Classifier weights are not available. Pull the Git LFS files before running the demo.")
        return

    predictions = core_models["classifier"].predict(_preprocess(image, (224, 224)), verbose=0)[0]
    class_index = int(np.argmax(predictions))
    st.success(f"Predicted subtype: {CLASS_NAMES[class_index].title()}")
    st.bar_chart({CLASS_NAMES[idx].title(): float(score) for idx, score in enumerate(predictions)})


def render_research_extensions() -> None:
    st.subheader("Research Extensions")
    st.warning(
        "These modules are intentionally marked experimental. They stay in the repo as research tracks, "
        "not as the default production demo."
    )

    enable_research = st.checkbox("Load experimental models", value=False)
    if not enable_research:
        st.info("Experimental models stay unloaded by default to keep the core demo focused and lightweight.")
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
            image = _load_image(uploaded)
            seg_model = research_models["segmentation"]
            input_shape = seg_model.input_shape
            tensor = _preprocess(image, target_size=(input_shape[1], input_shape[2]))
            pred_mask = seg_model.predict(tensor, verbose=0)[0, :, :, 0]
            pred_mask = (pred_mask > 0.5).astype(np.float32)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original MRI", use_container_width=True)
            with col2:
                overlay = np.asarray(image.resize((input_shape[2], input_shape[1])), dtype=np.float32) / 255.0
                overlay_rgb = np.stack([overlay, overlay, overlay], axis=-1)
                overlay_rgb[..., 0] = np.maximum(overlay_rgb[..., 0], pred_mask)
                st.image(overlay_rgb, caption="Predicted mask overlay", use_container_width=True, clamp=True)

    with subtab_gan:
        st.write("Conditional GAN preview for synthetic MRI slices.")
        if "generator" not in research_models:
            st.error("Generator weights are not available in this repo snapshot.")
        else:
            import tensorflow as tf

            target_class = st.selectbox("Condition class", CLASS_NAMES, key="gan_class")
            if st.button("Generate synthetic MRI", key="gan_generate"):
                class_index = CLASS_NAMES.index(target_class)
                noise = tf.random.normal([1, 100])
                label = tf.one_hot([class_index], len(CLASS_NAMES))
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
    detection_config = _load_detection_config()
    core_models = load_core_models()

    st.title(PROJECT_NAME)
    st.caption(
        "A resource-constrained brain MRI pipeline centered on calibrated tumour detection "
        "and multi-class subtype classification."
    )
    render_sidebar(core_models, detection_config)

    flagship_tab, classifier_tab, research_tab = st.tabs(
        ["Flagship Workflow", "Subtype Classifier", "Research Extensions"]
    )

    with flagship_tab:
        render_flagship_workflow(core_models, detection_config)

    with classifier_tab:
        render_classifier_only(core_models)

    with research_tab:
        render_research_extensions()


if __name__ == "__main__":
    main()
