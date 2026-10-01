"""v2 conditional GAN: ResNet generator with projection discriminator.

Research-grade cGAN using residual blocks, conditional batch norm,
self-attention and spectral normalization, trained with WGAN-GP.
"""

import tensorflow as tf
from tensorflow.keras import layers

from config import LATENT_DIM, NUM_CLASSES
from models.gan.layers import (
    DiscResBlock,
    GenResBlock,
    SelfAttention,
    _spectral_norm,
)

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
