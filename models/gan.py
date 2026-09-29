"""
Track 4 — GAN Augmentation Models
v2: Research-grade cGAN with ResNet blocks, spectral normalization,
    projection discriminator, self-attention, and WGAN-GP compatibility.

Includes legacy builders for backward compatibility.
"""
from contextlib import contextmanager

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from config import LATENT_DIM, NUM_CLASSES

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


# ═══════════════════════════════════════════════════════════════════════
# V2 GENERATOR — ResNet + Self-Attention + Conditional BN
# ═══════════════════════════════════════════════════════════════════════

class ResNetGenerator(tf.keras.Model):
    """Research-grade conditional generator.

    Architecture: z + class_embed → Dense → 8×8×512
      → ResBlock(512, up) → 16×16
      → SelfAttention
      → ResBlock(256, up) → 32×32
      → ResBlock(128, up) → 64×64
      → ResBlock(64, up)  → 128×128
      → BN → ReLU → Conv → tanh
    """

    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES,
                 embed_dim=128, output_shape=(128, 128, 1), **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self._output_shape_target = output_shape
        h, w, c = output_shape
        if h % 16 != 0 or w % 16 != 0:
            raise ValueError(
                f"output_shape spatial dimensions must be divisible by 16, "
                f"got ({h}, {w}). The generator uses 4 upsampling stages (×2 each)."
            )
        self.init_h = h // 16
        self.init_w = w // 16

        # Class embedding
        self.class_embed = layers.Embedding(num_classes, embed_dim)
        # Noise → spatial
        self.fc = _spectral_norm(
            layers.Dense(self.init_h * self.init_w * 512, use_bias=False)
        )

        # ResNet blocks with upsampling
        self.res1 = GenResBlock(512, upsample=True)   # 8→16
        self.attn = SelfAttention()                     # at 16×16
        self.res2 = GenResBlock(256, upsample=True)   # 16→32
        self.res3 = GenResBlock(128, upsample=True)   # 32→64
        self.res4 = GenResBlock(64, upsample=True)    # 64→128

        # Output head
        self.bn_out = layers.BatchNormalization()
        self.conv_out = _spectral_norm(
            layers.Conv2D(c, 3, padding="same", kernel_initializer="he_normal")
        )

    def call(self, inputs, training=None):
        # inputs = [noise, label_indices_or_onehot]
        z, labels = inputs

        # Get class embedding
        if len(labels.shape) > 1 and labels.shape[-1] == self.num_classes:
            # One-hot → index
            class_idx = tf.argmax(labels, axis=-1)
        else:
            class_idx = tf.cast(labels, tf.int32)
        class_emb = self.class_embed(class_idx)  # (B, embed_dim)

        # Combine noise and class info
        h = tf.concat([z, class_emb], axis=-1)
        h = self.fc(h)
        h = tf.reshape(h, [-1, self.init_h, self.init_w, 512])

        # ResNet synthesis with class conditioning
        h = self.res1(h, class_emb, training=training)
        h = self.attn(h)
        h = self.res2(h, class_emb, training=training)
        h = self.res3(h, class_emb, training=training)
        h = self.res4(h, class_emb, training=training)

        # Output
        h = self.bn_out(h, training=training)
        h = tf.nn.relu(h)
        h = self.conv_out(h)
        return tf.nn.tanh(h)

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "num_classes": self.num_classes,
            "embed_dim": self.embed_dim,
            "output_shape": self._output_shape_target,
        })
        return config


# ═══════════════════════════════════════════════════════════════════════
# V2 DISCRIMINATOR — Projection + Spectral Norm + ResNet
# ═══════════════════════════════════════════════════════════════════════

class ProjectionDiscriminator(tf.keras.Model):
    """Research-grade projection discriminator (Miyato & Koyama, 2018).

    Uses inner product of class embedding with feature vector for conditioning,
    which is mathematically superior to concatenation-based conditioning.

    Architecture: img(128×128×1)
      → ResBlock(64, down) → 64×64
      → ResBlock(128, down) → 32×32
      → ResBlock(256, down) → 16×16
      → SelfAttention
      → ResBlock(512, down) → 8×8
      → ResBlock(512, no_down) → 8×8
      → ReLU → GlobalSumPool → 512
      → projection(class_embed) + linear → scalar
    """

    def __init__(self, input_shape=(128, 128, 1), num_classes=NUM_CLASSES,
                 embed_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self._input_shape_config = input_shape

        # Initial conv (no activation, spectral norm)
        self.conv_in = _spectral_norm(
            layers.Conv2D(64, 3, padding="same", kernel_initializer="he_normal")
        )

        # ResNet blocks with downsampling
        self.res1 = DiscResBlock(64, downsample=True)    # →64×64
        self.res2 = DiscResBlock(128, downsample=True)   # →32×32
        self.res3 = DiscResBlock(256, downsample=True)   # →16×16
        self.attn = SelfAttention()                        # at 16×16
        self.res4 = DiscResBlock(512, downsample=True)   # →8×8
        self.res5 = DiscResBlock(512, downsample=False)  # →8×8

        # Output. Spectral normalisation is deliberately NOT applied here: it is
        # the standard WGAN practice to leave the critic's scalar output
        # unconstrained, since normalising it fights the projection term and
        # produces non-monotone/"sign-flipping" critic losses.
        self.linear = layers.Dense(1)

        # Projection: class embedding for projection discriminator.
        # `embed_dim` was previously stored and exported in get_config() but
        # ignored, hard-coding a 512-wide embedding instead.
        self.class_embed = layers.Embedding(num_classes, embed_dim)

        # Projects the pooled feature map into the embedding space so the inner
        # product is a genuine projection rather than a sum of unnormalised
        # 8x8x512 activations (which dominated the conditional term).
        self.feature_proj = layers.Dense(embed_dim, use_bias=False)

    def call(self, inputs, training=None):
        # inputs = [image, label_indices_or_onehot]
        img, labels = inputs

        # Get class index
        if len(labels.shape) > 1 and labels.shape[-1] == self.num_classes:
            class_idx = tf.argmax(labels, axis=-1)
        else:
            class_idx = tf.cast(labels, tf.int32)

        h = self.conv_in(img)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.attn(h)
        h = self.res4(h)
        h = self.res5(h)

        # Global average pooling (magnitude is resolution-independent, unlike
        # the previous global *sum*).
        h = tf.nn.relu(h)
        features = tf.reduce_mean(h, axis=[1, 2])  # (B, 512)
        features = self.feature_proj(features)    # (B, embed_dim)

        # Unconditional output
        out = self.linear(features)  # (B, 1)

        # Projection: inner product with class embedding
        class_emb = tf.cast(self.class_embed(class_idx), features.dtype)  # (B, embed_dim)
        projection = tf.reduce_sum(features * class_emb, axis=1, keepdims=True)

        return out + projection  # Raw logit (no sigmoid for WGAN-GP)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_shape": self._input_shape_config,
            "num_classes": self.num_classes,
            "embed_dim": self.embed_dim,
        })
        return config


# ═══════════════════════════════════════════════════════════════════════
# V2 BUILDER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def build_v2_generator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES,
                       output_shape=(128, 128, 1)):
    """Build the v2 ResNet conditional generator."""
    h, w, c = output_shape
    if h % 16 != 0 or w % 16 != 0:
        raise ValueError(
            f"output_shape spatial dimensions must be divisible by 16, "
            f"got ({h}, {w}). The generator uses 4 upsampling stages (×2 each)."
        )
    gen = ResNetGenerator(
        latent_dim=latent_dim, num_classes=num_classes,
        output_shape=output_shape, name="resnet_generator"
    )
    # Build the model by calling it with dummy data
    dummy_z = tf.zeros((1, latent_dim))
    dummy_labels = tf.zeros((1, num_classes))
    _ = gen([dummy_z, dummy_labels], training=False)
    print(f"  V2 Generator params: {gen.count_params():,}")
    return gen


def build_v2_discriminator(input_shape=(128, 128, 1), num_classes=NUM_CLASSES, embed_dim=128):
    """Build the v2 projection discriminator."""
    disc = ProjectionDiscriminator(
        input_shape=input_shape, num_classes=num_classes, embed_dim=embed_dim,
        name="projection_discriminator"
    )
    # Build the model by calling it with eye() labels so every class embedding
    # row is exercised. All-zero labels made argmax always return class 0,
    # leaving the remaining embedding rows untouched at build time.
    dummy_img = tf.zeros((1, *input_shape))
    dummy_labels = tf.eye(num_classes)[:1]
    _ = disc([dummy_img, dummy_labels], training=False)
    print(f"  V2 Discriminator params: {disc.count_params():,}")
    return disc


# ═══════════════════════════════════════════════════════════════════════
# EMA (Exponential Moving Average) for Generator
# ═══════════════════════════════════════════════════════════════════════

class EMAGenerator:
    """Tracks exponential moving average of generator weights.
    The EMA generator produces smoother, higher-quality outputs for evaluation.
    """

    def __init__(self, generator, decay=0.999):
        self.generator = generator
        self.decay = decay
        # Track *all* variables, not just trainable ones: BatchNorm moving_mean
        # / moving_variance are non-trainable state, and leaving them out
        # produced a hybrid model (EMA kernels + live BN statistics) that is not
        # a valid EMA of the generator.
        self._variables = list(generator.variables)
        self.ema_weights = [
            tf.Variable(w, trainable=False, name=f"ema_{i}") for i, w in enumerate(self._variables)
        ]
        self._backup: list | None = None
        self._swap_depth = 0

    def update(self):
        """Update EMA weights after each generator training step."""
        for ema_w, w in zip(self.ema_weights, self._variables, strict=True):
            ema_w.assign(self.decay * ema_w + (1.0 - self.decay) * w)

    @contextmanager
    def swapped(self):
        """Context manager that temporarily applies EMA weights.

        Replaces the error-prone ``apply()``/``restore()`` pair: an exception
        between the two calls used to leave the generator permanently holding
        EMA weights while the optimizer kept updating the real ones, silently
        corrupting the rest of the run. Re-entrant via a depth counter.
        """
        self.apply()
        try:
            yield self.generator
        finally:
            self.restore()

    def apply(self):
        """Apply EMA weights to generator (for evaluation/preview).

        Prefer :meth:`swapped`, which guarantees :meth:`restore` runs.
        """
        if self._backup is not None:
            self._swap_depth += 1
            return
        self._backup = [tf.identity(w) for w in self._variables]
        for w, ema_w in zip(self._variables, self.ema_weights, strict=True):
            w.assign(ema_w)

    def restore(self):
        """Restore original weights after evaluation."""
        if self._backup is None:
            return
        self._swap_depth -= 1
        if self._swap_depth > 0:
            return
        for w, backup in zip(self._variables, self._backup, strict=True):
            w.assign(backup)
        self._backup = None

    def save(self, path: str) -> None:
        """Save EMA weights to a .npz file for checkpoint resumption."""
        np.savez(
            path,
            **{f"ema_{i}": w.numpy() for i, w in enumerate(self.ema_weights)},
        )

    def load(self, path: str) -> None:
        """Load EMA weights from a .npz file.

        Validates the key set and shapes; previously a partial or foreign
        checkpoint was accepted silently, leaving some variables at their
        initial values and producing a subtly wrong generator.
        """
        with np.load(path) as data:
            expected = {f"ema_{i}" for i in range(len(self.ema_weights))}
            found = set(data.files)
            if found != expected:
                raise ValueError(
                    f"EMA checkpoint {path} does not match this generator: "
                    f"missing={sorted(expected - found)[:5]}, "
                    f"unexpected={sorted(found - expected)[:5]}"
                )
            for i, ema_w in enumerate(self.ema_weights):
                value = data[f"ema_{i}"]
                if tuple(value.shape) != tuple(ema_w.shape):
                    raise ValueError(
                        f"EMA weight 'ema_{i}' has shape {value.shape} but the "
                        f"generator expects {tuple(ema_w.shape)}."
                    )
                ema_w.assign(value)


# ═══════════════════════════════════════════════════════════════════════
# WGAN-GP GRADIENT PENALTY
# ═══════════════════════════════════════════════════════════════════════

def gradient_penalty(discriminator, real_images, fake_images, labels, lambda_gp=1.0):
    """Compute the WGAN-GP gradient penalty.

    Returns ``lambda_gp * E[(||∇D(x̂)||_2 - 1)^2]``. ``lambda_gp`` defaults to
    ``1.0`` so the function is unscaled by default; callers applying
    ``lambda_gp * gradient_penalty(...)`` therefore get the intended penalty
    instead of silently squaring the coefficient (previously this function
    pre-multiplied by 10.0 while its only caller added no factor, so the two
    conventions could not be combined without a 10x divergence).
    """
    batch_size = tf.shape(real_images)[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated = alpha * real_images + (1.0 - alpha) * fake_images

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator([interpolated, labels], training=True)
    grads = tape.gradient(pred, interpolated)
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-8)
    gp = tf.reduce_mean(tf.square(grad_norm - 1.0))
    return lambda_gp * gp


# ═══════════════════════════════════════════════════════════════════════
# LEGACY BUILDERS (backward compatibility)
# ═══════════════════════════════════════════════════════════════════════

def build_generator(latent_dim=LATENT_DIM, output_shape=(128, 128, 1)):
    """Enhanced DCGAN generator with spectral normalization."""
    h, w, c = output_shape
    if h % 16 != 0 or w % 16 != 0:
        raise ValueError(
            f"output_shape spatial dimensions must be divisible by 16, "
            f"got ({h}, {w}). The generator uses 4 upsampling stages (×2 each)."
        )
    init_h, init_w = h // 16, w // 16

    model = models.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(init_h * init_w * 512, use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Reshape((init_h, init_w, 512)),

        layers.Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(32, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2D(c, 3, padding='same', activation='tanh'),
    ], name="generator")
    return model


def build_discriminator(input_shape=(128, 128, 1)):
    """Enhanced DCGAN discriminator with dropout and spectral norm."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(64, 4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Conv2D(128, 4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Conv2D(256, 4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Conv2D(512, 4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1, activation='sigmoid'),
    ], name="discriminator")
    return model


def build_gan(generator, discriminator, latent_dim=LATENT_DIM, lr=2e-4):
    """Assemble GAN for training."""
    discriminator.compile(
        optimizer=tf.keras.optimizers.Adam(lr, beta_1=0.5),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    discriminator.trainable = False
    z = layers.Input(shape=(latent_dim,))
    img = generator(z)
    validity = discriminator(img)
    gan = models.Model(z, validity, name="dcgan")
    gan.compile(
        optimizer=tf.keras.optimizers.Adam(lr, beta_1=0.5),
        loss='binary_crossentropy',
    )
    discriminator.trainable = True
    return gan


def build_conditional_generator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES, output_shape=(128,
    128, 1)):
    """Conditional GAN generator — generates images conditioned on tumour type."""
    h, w, c = output_shape
    if h % 16 != 0 or w % 16 != 0:
        raise ValueError(
            f"output_shape spatial dimensions must be divisible by 16, "
            f"got ({h}, {w}). The generator uses 4 upsampling stages (×2 each)."
        )
    init_h, init_w = h // 16, w // 16

    z_input = layers.Input(shape=(latent_dim,), name="noise_input")
    label_input = layers.Input(shape=(num_classes,), name="label_input")

    label_embed = layers.Dense(latent_dim, activation='relu')(label_input)
    mult = layers.Multiply()([z_input, label_embed])
    combined = layers.Add()([mult, z_input])

    x = layers.Dense(init_h * init_w * 512, use_bias=False)(combined)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Reshape((init_h, init_w, 512))(x)

    label_spatial = layers.Dense(init_h * init_w * 64, activation='relu')(label_input)
    label_spatial = layers.Reshape((init_h, init_w, 64))(label_spatial)
    x = layers.Concatenate()([x, label_spatial])

    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(32, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    output = layers.Conv2D(c, 3, padding='same', activation='tanh')(x)

    model = models.Model([z_input, label_input], output, name="conditional_generator")
    return model


def build_conditional_discriminator(input_shape=(128, 128, 1), num_classes=NUM_CLASSES):
    """Conditional GAN discriminator — classifies real/fake conditioned on label.
    Uses spectral normalization on conv layers for training stability."""
    img_input = layers.Input(shape=input_shape, name="image_input")
    label_input = layers.Input(shape=(num_classes,), name="label_input")

    label_spatial = layers.Dense(input_shape[0] * input_shape[1] * 1,
        activation='relu')(label_input)
    label_spatial = layers.Reshape((input_shape[0], input_shape[1], 1))(label_spatial)

    x = layers.Concatenate()([img_input, label_spatial])

    x = _spectral_norm(layers.Conv2D(64, 4, strides=2, padding='same'))(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.25)(x)

    x = _spectral_norm(layers.Conv2D(128, 4, strides=2, padding='same'))(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.25)(x)

    x = _spectral_norm(layers.Conv2D(256, 4, strides=2, padding='same'))(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.25)(x)

    x = _spectral_norm(layers.Conv2D(512, 4, strides=2, padding='same'))(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    validity = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model([img_input, label_input], validity, name="conditional_discriminator")
    return model


def build_conditional_gan(generator, discriminator, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES,
    lr=2e-4):
    """Assemble conditional GAN."""
    discriminator.compile(
        optimizer=tf.keras.optimizers.Adam(lr, beta_1=0.5),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    discriminator.trainable = False

    z_input = layers.Input(shape=(latent_dim,), name="noise_input")
    label_input = layers.Input(shape=(num_classes,), name="label_input")

    img = generator([z_input, label_input])
    validity = discriminator([img, label_input])

    gan = models.Model([z_input, label_input], validity, name="cgan")
    gan.compile(
        optimizer=tf.keras.optimizers.Adam(lr, beta_1=0.5),
        loss='binary_crossentropy',
    )
    discriminator.trainable = True
    return gan


# ═══════════════════════════════════════════════════════════════════════
# STYLEGAN-INSPIRED GENERATOR
# ═══════════════════════════════════════════════════════════════════════
def build_stylegan_generator(latent_dim=LATENT_DIM, output_shape=(128, 128, 1)):
    """
    StyleGAN-inspired generator with:
    - Mapping network (z -> w)
    - Style modulation (FiLM-like)
    - Progressive upsampling synthesis
    """
    h, w, c = output_shape
    if h % 16 != 0 or w % 16 != 0:
        raise ValueError(
            f"output_shape spatial dimensions must be divisible by 16, "
            f"got ({h}, {w}). The generator uses 4 upsampling stages (×2 each)."
        )
    init_h, init_w = h // 16, w // 16

    z_input = layers.Input(shape=(latent_dim,), name="z_input")

    # Mapping network
    style = layers.Dense(512)(z_input)
    style = layers.LeakyReLU(0.2)(style)
    style = layers.Dense(512)(style)
    style = layers.LeakyReLU(0.2)(style)
    style = layers.Dense(512)(style)

    # Synthesis network
    x = layers.Dense(init_h * init_w * 512, use_bias=False)(style)
    x = layers.Reshape((init_h, init_w, 512))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    def style_mod_block(x_in, style_vec, filters):
        x_out = layers.Conv2DTranspose(filters, 4, strides=2, padding='same', use_bias=False)(x_in)
        x_out = layers.BatchNormalization()(x_out)

        gamma = layers.Dense(filters, activation='sigmoid')(style_vec)
        beta = layers.Dense(filters)(style_vec)
        gamma = layers.Reshape((1, 1, filters))(gamma)
        beta = layers.Reshape((1, 1, filters))(beta)

        x_out = layers.Multiply()([x_out, gamma])
        x_out = layers.Add()([x_out, beta])
        x_out = layers.LeakyReLU(0.2)(x_out)
        return x_out

    for filters in [256, 128, 64, 32]:
        x = style_mod_block(x, style, filters)

    output = layers.Conv2D(c, 3, padding='same', activation='tanh')(x)

    model = models.Model(z_input, output, name="stylegan_generator")
    return model


# ═══════════════════════════════════════════════════════════════════════
# BASELINE GAN (from challenge spec)
# ═══════════════════════════════════════════════════════════════════════
def build_baseline_generator(latent_dim=100):
    """Original baseline generator from the challenge."""
    model = models.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(8 * 8 * 256, use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Reshape((8, 8, 256)),

        layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(1, 4, strides=2, padding='same', activation='tanh'),
    ], name="baseline_generator")
    return model


def build_baseline_discriminator(input_shape=(64, 64, 1)):
    """Original baseline discriminator from the challenge."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(64, 4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),

        layers.Conv2D(128, 4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),

        layers.Flatten(),
        layers.Dense(1, activation='sigmoid'),
    ], name="baseline_discriminator")
    return model
