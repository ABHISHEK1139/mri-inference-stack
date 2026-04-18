"""
Track 2 — Tumour Segmentation Model
Enhanced U-Net with attention gates and residual connections.
"""
import tensorflow as tf
from tensorflow.keras import layers, models


def conv_block(x, filters, kernel_size=3, use_bn=True):
    """Double convolution block."""
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def residual_conv_block(x, filters):
    """Residual convolution block."""
    shortcut = layers.Conv2D(filters, 1, padding='same')(x)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def attention_gate(x, gating, inter_filters):
    """Attention gate for skip connections."""
    # x: skip connection feature map
    # gating: feature map from decoder
    theta_x = layers.Conv2D(inter_filters, 1, strides=1, padding='same')(x)
    phi_g = layers.Conv2D(inter_filters, 1, strides=1, padding='same')(gating)

    # Upsample gating to match x dimensions
    phi_g = layers.UpSampling2D(size=(2, 2))(phi_g)

    add = layers.Add()([theta_x, phi_g])
    act = layers.Activation('relu')(add)
    psi = layers.Conv2D(1, 1, padding='same')(act)
    psi = layers.Activation('sigmoid')(psi)

    return layers.Multiply()([x, psi])


def build_unet(input_shape=(256, 256, 1), use_attention=True, use_residual=True):
    """Enhanced U-Net with optional attention gates and residual blocks."""
    inputs = layers.Input(input_shape)

    # ── Encoder ────────────────────────────────────────────────────────
    if use_residual:
        c1 = residual_conv_block(inputs, 64)
    else:
        c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D()(c1)
    p1 = layers.Dropout(0.1)(p1)

    if use_residual:
        c2 = residual_conv_block(p1, 128)
    else:
        c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D()(c2)
    p2 = layers.Dropout(0.2)(p2)

    if use_residual:
        c3 = residual_conv_block(p2, 256)
    else:
        c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D()(c3)
    p3 = layers.Dropout(0.3)(p3)

    # ── Bottleneck ──────────────────────────────────────────────────────
    if use_residual:
        b = residual_conv_block(p3, 512)
    else:
        b = conv_block(p3, 512)
    b = layers.Dropout(0.4)(b)

    # ── Decoder ─────────────────────────────────────────────────────────
    u1 = layers.UpSampling2D()(b)
    if use_attention:
        c3_att = attention_gate(c3, b, 128)
        u1 = layers.Concatenate()([u1, c3_att])
    else:
        u1 = layers.Concatenate()([u1, c3])
    if use_residual:
        c4 = residual_conv_block(u1, 256)
    else:
        c4 = conv_block(u1, 256)

    u2 = layers.UpSampling2D()(c4)
    if use_attention:
        c2_att = attention_gate(c2, c4, 64)
        u2 = layers.Concatenate()([u2, c2_att])
    else:
        u2 = layers.Concatenate()([u2, c2])
    if use_residual:
        c5 = residual_conv_block(u2, 128)
    else:
        c5 = conv_block(u2, 128)

    u3 = layers.UpSampling2D()(c5)
    if use_attention:
        c1_att = attention_gate(c1, c5, 32)
        u3 = layers.Concatenate()([u3, c1_att])
    else:
        u3 = layers.Concatenate()([u3, c1])
    if use_residual:
        c6 = residual_conv_block(u3, 64)
    else:
        c6 = conv_block(u3, 64)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c6)

    model = models.Model(inputs, outputs, name="attention_unet" if use_attention else "unet")

    # Dice loss + BCE combined
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=dice_bce_loss,
        metrics=['accuracy', dice_coefficient, iou_metric],
    )
    return model


def build_unet_baseline(input_shape=(256, 256, 1)):
    """Original baseline U-Net from the challenge."""
    inputs = layers.Input(input_shape)

    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D()(c3)

    b = conv_block(p3, 512)

    u1 = layers.UpSampling2D()(b)
    u1 = layers.Concatenate()([u1, c3])
    c4 = conv_block(u1, 256)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = conv_block(u2, 128)

    u3 = layers.UpSampling2D()(c5)
    u3 = layers.Concatenate()([u3, c1])
    c6 = conv_block(u3, 64)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c6)

    model = models.Model(inputs, outputs, name="unet_baseline")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ── Custom Losses & Metrics ──────────────────────────────────────────────
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient metric."""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred, smooth=1e-6):
    """Dice loss."""
    return 1.0 - dice_coefficient(y_true, y_pred, smooth)


def dice_bce_loss(y_true, y_pred):
    """Combined Dice + BCE loss."""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice


def iou_metric(y_true, y_pred, smooth=1e-6):
    """Intersection over Union metric."""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)