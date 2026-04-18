"""
Track 3 — Tumour Type Classification Model
Multi-class: Glioma, Meningioma, Pituitary, Other
Enhanced with EfficientNet backbone, attention, and multi-modal fusion support.
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from config import NUM_CLASSES


def build_classifier(num_classes=NUM_CLASSES, input_shape=(224, 224, 1)):
    """Enhanced classifier using EfficientNetB0 with custom head."""
    # For grayscale input, we need to adapt EfficientNet
    inputs = layers.Input(shape=input_shape)

    # Convert grayscale to 3 channels for pretrained backbone
    x = layers.Conv2D(3, 1, padding='same')(inputs)  # 1ch -> 3ch

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,  # No pretrained weights for medical domain
        input_shape=(input_shape[0], input_shape[1], 3),
    )
    x = base(x)

    # ── Attention pooling ──────────────────────────────────────────────
    # Instead of simple GAP, use channel + spatial attention
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(x.shape[-1] // 16, activation='relu')(se)
    se = layers.Dense(x.shape[-1], activation='sigmoid')(se)
    se = layers.Reshape((1, 1, x.shape[-1]))(se)
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

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=input_shape,
    )

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = models.Model(base.input, outputs, name="classifier_baseline")
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def build_multimodal_classifier(num_classes=NUM_CLASSES, input_shape=(224, 224, 1), num_modalities=4):
    """
    Multi-modal fusion classifier.
    Accepts multiple MRI modalities (T1, T2, FLAIR, T1ce) as separate inputs
    and fuses them for classification.
    """
    # ── Separate encoder per modality ──────────────────────────────────
    modality_inputs = []
    modality_features = []

    for i in range(num_modalities):
        inp = layers.Input(shape=input_shape, name=f"modality_{i}")
        modality_inputs.append(inp)

        # Shared encoder backbone
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
