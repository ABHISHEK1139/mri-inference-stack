"""Building blocks shared by the v2 generator and discriminator."""


import tensorflow as tf
from tensorflow.keras import layers

# ═══════════════════════════════════════════════════════════════════════
# BUILDING BLOCKS (v2)
# ═══════════════════════════════════════════════════════════════════════

def _spectral_norm(layer):
    """Wrap a layer with spectral normalization for training stability."""
    return tf.keras.layers.SpectralNormalization(layer)


class ConditionalBatchNorm(layers.Layer):
    """Conditional Batch Normalization (CBN).
    Learns per-class scale (gamma) and shift (beta) via linear projections
    from a class embedding vector. Essential for class-conditional generation.
    """

    def __init__(self, num_features, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features

    def build(self, input_shape):
        self.bn = layers.BatchNormalization(
            center=False, scale=False, epsilon=1e-5
        )
        self.gamma_proj = layers.Dense(self.num_features, kernel_initializer="ones")
        self.beta_proj = layers.Dense(self.num_features, kernel_initializer="zeros")
        super().build(input_shape)

    def call(self, x, class_embed, training=None):
        out = self.bn(x, training=training)
        gamma = self.gamma_proj(class_embed)
        beta = self.beta_proj(class_embed)
        # Reshape for broadcasting: (batch, 1, 1, features)
        gamma = tf.reshape(gamma, [-1, 1, 1, self.num_features])
        beta = tf.reshape(beta, [-1, 1, 1, self.num_features])
        return out * (1.0 + gamma) + beta

    def get_config(self):
        config = super().get_config()
        config["num_features"] = self.num_features
        return config


class SelfAttention(layers.Layer):
    """Self-attention layer for capturing long-range spatial dependencies.
    Applied at intermediate resolutions (e.g., 16×16) where it's most effective.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.ch = channels
        reduced = max(channels // 8, 1)
        self.query = _spectral_norm(layers.Conv2D(reduced, 1, use_bias=False))
        self.key = _spectral_norm(layers.Conv2D(reduced, 1, use_bias=False))
        self.value = _spectral_norm(layers.Conv2D(channels, 1, use_bias=False))
        self.gamma = self.add_weight(
            name="sa_gamma", shape=(1,), initializer="zeros", trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        batch, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], self.ch
        hw = h * w

        q = tf.reshape(self.query(x), [batch, hw, -1])   # (B, HW, C/8)
        k = tf.reshape(self.key(x), [batch, hw, -1])      # (B, HW, C/8)
        v = tf.reshape(self.value(x), [batch, hw, c])      # (B, HW, C)

        # Scale logits by 1/sqrt(d) (scaled dot-product attention). Without this
        # the logits have a large spread, the softmax saturates, and the layer
        # degenerates into a near-uniform global average with vanishing grads.
        attn = tf.matmul(q, k, transpose_b=True)           # (B, HW, HW)
        attn = attn / tf.math.sqrt(tf.cast(tf.shape(q)[-1], attn.dtype))
        attn = tf.nn.softmax(attn, axis=-1)

        out = tf.matmul(attn, v)                            # (B, HW, C)
        out = tf.reshape(out, [batch, h, w, c])

        return x + self.gamma * out


class GenResBlock(layers.Layer):
    """Generator residual block with conditional batch norm and upsampling."""

    def __init__(self, filters, upsample=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.upsample = upsample

    def build(self, x_shape, class_embed_shape=None):
        # `call` receives two tensors (x, class_embed), so Keras 3 hands us one
        # shape per call argument. Older/legacy paths may still pass a single
        # nested list of shapes; unwrap that form defensively. Note that
        # `x_shape[-1]` must index the *feature map* -- reading it off the
        # embedding shape built a Dense with a TensorShape unit and failed.
        if (
            isinstance(x_shape, (list, tuple))
            and x_shape
            and isinstance(x_shape[0], (list, tuple, tf.TensorShape))
        ):
            x_shape = x_shape[0]
        channels = int(x_shape[-1])
        self.cbn1 = ConditionalBatchNorm(channels)
        self.conv1 = _spectral_norm(
            layers.Conv2D(self.filters, 3, padding="same", use_bias=False,
                          kernel_initializer="he_normal")
        )
        self.cbn2 = ConditionalBatchNorm(self.filters)
        self.conv2 = _spectral_norm(
            layers.Conv2D(self.filters, 3, padding="same", use_bias=False,
                          kernel_initializer="he_normal")
        )

        # Shortcut conv if channels change
        if channels != self.filters:
            self.shortcut = _spectral_norm(
                layers.Conv2D(self.filters, 1, use_bias=False)
            )
        else:
            self.shortcut = None

        if self.upsample:
            self.up = layers.UpSampling2D(size=(2, 2), interpolation="nearest")
        super().build(x_shape)

    def call(self, x, class_embed, training=None):
        h = self.cbn1(x, class_embed, training=training)
        h = tf.nn.relu(h)
        if self.upsample:
            h = self.up(h)
        h = self.conv1(h)

        h = self.cbn2(h, class_embed, training=training)
        h = tf.nn.relu(h)
        h = self.conv2(h)

        # Shortcut
        sc = x
        if self.upsample:
            sc = self.up(sc)
        if self.shortcut is not None:
            sc = self.shortcut(sc)

        return h + sc


class DiscResBlock(layers.Layer):
    """Discriminator residual block with spectral norm and optional downsampling."""

    def __init__(self, filters, downsample=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.downsample = downsample

    def build(self, input_shape):
        in_ch = input_shape[-1]
        self.conv1 = _spectral_norm(
            layers.Conv2D(self.filters, 3, padding="same",
                          kernel_initializer="he_normal")
        )
        self.conv2 = _spectral_norm(
            layers.Conv2D(self.filters, 3, padding="same",
                          kernel_initializer="he_normal")
        )

        if in_ch != self.filters or self.downsample:
            self.shortcut = _spectral_norm(
                layers.Conv2D(self.filters, 1, use_bias=False)
            )
        else:
            self.shortcut = None

        if self.downsample:
            self.pool = layers.AveragePooling2D(pool_size=(2, 2))
        super().build(input_shape)

    def call(self, x):
        h = tf.nn.relu(x)
        h = self.conv1(h)
        h = tf.nn.relu(h)
        h = self.conv2(h)
        if self.downsample:
            h = self.pool(h)

        sc = x
        if self.shortcut is not None:
            sc = self.shortcut(sc)
        if self.downsample:
            sc = self.pool(sc)

        return h + sc
