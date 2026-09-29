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
    """Attention gate for skip connections.

    Dynamically resizes the gating signal to match x's spatial dimensions,
    rather than assuming a fixed 2x ratio.
    """
    theta_x = layers.Conv2D(inter_filters, 1, strides=1, padding='same')(x)
    phi_g = layers.Conv2D(inter_filters, 1, strides=1, padding='same')(gating)

    # Dynamically resize gating signal to match skip connection dimensions.
    # Keras 3 removed keras.backend.int_shape; use the static tensor shape,
    # which is known here because x comes from a fixed-size Input.
    height, width = x.shape[1], x.shape[2]
    if height is not None and width is not None:
        phi_g = layers.Resizing(
            height=height,
            width=width,
            interpolation='bilinear',
            name=f"gate_resize_{inter_filters}",
        )(phi_g)
    else:
        # Fully dynamic shapes: align the gate with an explicit target shape.
        raise ValueError(
            "attention_gate requires statically known spatial dimensions on the "
            "skip connection. Build the U-Net with a fixed input_shape."
        )

    add = layers.Add()([theta_x, phi_g])
    act = layers.Activation('relu')(add)
    psi = layers.Conv2D(1, 1, padding='same')(act)
    psi = layers.Activation('sigmoid')(psi)

    return layers.Multiply()([x, psi])


def build_unet(input_shape=(256, 256, 1), use_attention=True, use_residual=True):
    """Enhanced U-Net with optional attention gates and residual blocks.

    ``input_shape`` must be divisible by 16: the encoder halves resolution three
    times via ``MaxPooling2D`` (which floors), while the decoder doubles it via
    ``UpSampling2D``, so non-multiple-of-16 inputs produce mismatched
    ``Concatenate`` shapes deep inside the graph.
    """
    height, width = input_shape[0], input_shape[1]
    if height % 16 or width % 16:
        raise ValueError(
            f"input_shape {input_shape} must be divisible by 16 so that the U-Net "
            "encoder/decoder resolutions line up, "
            f"got {height}x{width}."
        )

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
        c3_att = attention_gate(c3, u1, 128)
        u1 = layers.Concatenate()([u1, c3_att])
    else:
        u1 = layers.Concatenate()([u1, c3])
    if use_residual:
        c4 = residual_conv_block(u1, 256)
    else:
        c4 = conv_block(u1, 256)

    u2 = layers.UpSampling2D()(c4)
    if use_attention:
        c2_att = attention_gate(c2, u2, 64)
        u2 = layers.Concatenate()([u2, c2_att])
    else:
        u2 = layers.Concatenate()([u2, c2])
    if use_residual:
        c5 = residual_conv_block(u2, 128)
    else:
        c5 = conv_block(u2, 128)

    u3 = layers.UpSampling2D()(c5)
    if use_attention:
        c1_att = attention_gate(c1, u3, 32)
        u3 = layers.Concatenate()([u3, c1_att])
    else:
        u3 = layers.Concatenate()([u3, c1])
    if use_residual:
        c6 = residual_conv_block(u3, 64)
    else:
        c6 = conv_block(u3, 64)

    # Force float32: under a mixed_float16 policy the logits reach these metrics in
    # float16, where summing over B*H*W mask elements overflows the 65504 limit.
    outputs = layers.Conv2D(1, 1, activation='sigmoid', dtype='float32')(c6)

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

    # Force float32: under a mixed_float16 policy the logits reach these metrics in
    # float16, where summing over B*H*W mask elements overflows the 65504 limit.
    outputs = layers.Conv2D(1, 1, activation='sigmoid', dtype='float32')(c6)

    model = models.Model(inputs, outputs, name="unet_baseline")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ── Custom Losses & Metrics ──────────────────────────────────────────────
def _flatten_per_sample(y):
    """Reshape (B, H, W, C) to (B, -1) so reductions keep a batch axis.

    ``tf.reshape(x, [-1])`` collapses the batch axis away, so a following
    ``reduce_sum(axis=-1)`` yields one scalar for the whole batch -- the
    dataset-level micro average this function is meant to avoid.
    """
    y = tf.cast(y, tf.float32)
    return tf.reshape(y, [tf.shape(y)[0], -1])


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Per-image Dice coefficient, averaged over the batch.

    The previous implementation flattened ``(B, H, W, 1)`` into one vector and
    took a single sum, which reports a *micro* (dataset-level) average: the
    value silently changed with batch size and batch composition, and as a loss
    it made the optimisation objective depend on how samples were grouped.
    Reductions are also forced to float32 because ``sum`` over ``B*H*W`` elements
    overflows float16's 65504 maximum under a mixed-precision policy.
    """
    y_true_f = _flatten_per_sample(y_true)
    y_pred_f = _flatten_per_sample(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=-1)
    denominator = tf.reduce_sum(y_true_f, axis=-1) + tf.reduce_sum(y_pred_f, axis=-1)
    return tf.reduce_mean((2.0 * intersection + smooth) / (denominator + smooth))


def dice_loss(y_true, y_pred, smooth=1e-6):
    """Dice loss."""
    return 1.0 - dice_coefficient(y_true, y_pred, smooth)


def dice_bce_loss(y_true, y_pred):
    """Combined Dice + BCE loss."""
    y_true_c = tf.cast(y_true, tf.float32)
    y_pred_c = tf.cast(y_pred, tf.float32)
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true_c, y_pred_c))
    dice = dice_loss(y_true_c, y_pred_c)
    return bce + dice


def iou_metric(y_true, y_pred, smooth=1e-6):
    """Per-image Intersection over Union, averaged over the batch."""
    y_true_f = _flatten_per_sample(y_true)
    y_pred_f = _flatten_per_sample(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=-1)
    union = tf.reduce_sum(y_true_f, axis=-1) + tf.reduce_sum(y_pred_f, axis=-1) - intersection
    return tf.reduce_mean((intersection + smooth) / (union + smooth))
