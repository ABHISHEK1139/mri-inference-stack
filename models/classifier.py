"""
Track 3 — Tumour Type Classification Model
Multi-class: Glioma, Meningioma, Pituitary, Other
Enhanced with EfficientNet backbone, attention, and multi-modal fusion support.
"""
import tensorflow as tf
from tensorflow.keras import layers, models

from config import NUM_CLASSES


def build_classifier(
    num_classes=NUM_CLASSES,
    input_shape=(224, 224, 1),
    imagenet_input_range=True,
):
    """Enhanced classifier using EfficientNetB0 with a custom attention head.

    The dataset pipeline emits pixels in ``[0, 1]``, but ``EfficientNet`` starts
    with ``Rescaling(1/255)`` + ``Normalization`` and its pretrained weights
    expect raw ``[0, 255]`` pixels. Feeding ``[0, 1]`` straight through maps
    every pixel to roughly ``-1.0`` after the stem, i.e. a constant image that
    discards the ImageNet features entirely. ``imagenet_input_range=True``
    rescales ``[0, 1]`` up to ``[0, 255]`` so transfer learning actually works.

    .. note::
       Set ``imagenet_input_range=False`` to reproduce the legacy behaviour when
       loading checkpoints that were trained on unscaled ``[0, 1]`` inputs. The
       flag is stored in the saved ``.keras`` config, so old checkpoints keep
       their original behaviour.
    """
    inputs = layers.Input(shape=input_shape)

    # Convert grayscale to 3 channels for pretrained backbone
    x = layers.Conv2D(3, 1, padding='same')(inputs)  # 1ch -> 3ch

    if imagenet_input_range:
        # EfficientNet's internal stem rescales by 1/255; undo our [0,1] scaling
        # first so the pretrained normalisation sees the range it was trained on.
        x = layers.Rescaling(255.0, name="imagenet_range_rescale")(x)

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",  # Transfer learning from ImageNet features
        input_shape=(input_shape[0], input_shape[1], 3),
    )
    x = base(x)

    # ── Attention pooling ──────────────────────────────────────────────
    # Instead of simple GAP, use channel attention (squeeze-and-excitation).
    channels = int(x.shape[-1])
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(max(1, channels // 16), activation='relu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, channels))(se)
    x = layers.Multiply()([x, se])

    x = layers.GlobalAveragePooling2D()(x)

    # ── Classification Head ─────────────────────────────────────────────
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # Keep final probabilities in float32 so mixed precision does not break
    # metric ops such as TopKCategoricalAccuracy.
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = models.Model(inputs, outputs, name="tumour_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_acc')],
    )
    return model


def build_classifier_baseline(num_classes=NUM_CLASSES, input_shape=(224, 224, 1)):
    """Original baseline classifier from the challenge."""
    inputs = layers.Input(shape=input_shape)

    # Adapt 1-channel grayscale to 3 channels required by EfficientNet
    if input_shape[-1] == 1:
        x = layers.Conv2D(3, (1, 1), padding='same')(inputs)
        eff_shape = (input_shape[0], input_shape[1], 3)
    else:
        x = inputs
        eff_shape = input_shape

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=eff_shape,
    )
    x = base(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = models.Model(inputs, outputs, name="classifier_baseline")
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def build_multimodal_classifier(num_classes=NUM_CLASSES, input_shape=(224, 224, 1),
    num_modalities=4):
    """
    Multi-modal fusion classifier.
    Accepts multiple MRI modalities (T1, T2, FLAIR, T1ce) as separate inputs
    and fuses them for classification.
    """
    # ── Per-modality encoder ───────────────────────────────────────────
    # NOTE: these encoders are NOT weight-shared — each modality gets its own
    # independent copy of the conv stack. Sharing would require calling one
    # sub-model on several inputs; keeping them independent is the simpler and
    # more expressive choice for this dataset size.
    modality_inputs = []
    modality_features = []

    for i in range(num_modalities):
        inp = layers.Input(shape=input_shape, name=f"modality_{i}")
        modality_inputs.append(inp)

        x = layers.Conv2D(32, 3, activation='relu', padding='same')(inp)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.GlobalAveragePooling2D()(x)
        modality_features.append(x)

    # ── Fusion ─────────────────────────────────────────────────────────
    if num_modalities > 1:
        fused = layers.Concatenate()(modality_features)
    else:
        fused = modality_features[0]

    # ── Classification Head ─────────────────────────────────────────────
    x = layers.Dense(512, activation='relu')(fused)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = models.Model(modality_inputs, outputs, name="multimodal_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model
