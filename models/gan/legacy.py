"""Legacy GAN builders retained for the v1 and stylegan training tracks.

These are the original architectures from before the v2 rewrite. They are
kept because `--gan_type dcgan|conditional|stylegan|baseline` still selects
them, and because saved v1 checkpoints must remain loadable.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from config import LATENT_DIM, NUM_CLASSES
from models.gan.layers import _spectral_norm

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
