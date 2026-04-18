"""
Track 1 — Tumour Detection Model
Binary classification: Tumour vs Normal
Enhanced baseline with residual blocks, dropout, and attention.
"""
import tensorflow as tf
from tensorflow.keras import layers, models


def residual_block(x, filters, stride=1):
    """Residual block with skip connection."""
    shortcut = x
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv2D(filters, 3, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def channel_attention(x, reduction_ratio=16):
    """Squeeze-and-Excitation channel attention."""
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(channels // reduction_ratio, activation='relu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, channels))(se)
    return layers.Multiply()([x, se])


def build_detection_model(input_shape=(224, 224, 1)):
    """Enhanced tumour detection model with residual blocks and attention."""
    inputs = layers.Input(shape=input_shape)

    # ── Stem ──────────────────────────────────────────────────────────
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # ── Residual Block 1 ──────────────────────────────────────────────
    x = residual_block(x, 64)
    x = channel_attention(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)

    # ── Residual Block 2 ──────────────────────────────────────────────
    x = residual_block(x, 128)
    x = channel_attention(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    # ── Residual Block 3 ──────────────────────────────────────────────
    x = residual_block(x, 256)
    x = channel_attention(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.4)(x)

    # ── Head ──────────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    model = models.Model(inputs, outputs, name="detection_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    return model


def build_detection_baseline(input_shape=(224, 224, 1)):
    """Original baseline detection model from the challenge."""
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs, name="detection_baseline")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC'],
    )
    return model
