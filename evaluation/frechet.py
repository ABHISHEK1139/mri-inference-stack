"""Fréchet distances: FID over InceptionV3 and the relative FS metric.

Both reduce to the same Fréchet formula over extracted feature moments,
so they share the statistics and matrix-square-root helpers.
"""

import warnings

import numpy as np
from scipy import linalg

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None


def build_inception_feature_extractor():
    """Build InceptionV3 model for feature extraction.
    Uses pretrained ImageNet weights for meaningful feature representations.
    """
    if tf is None:
        raise ImportError("TensorFlow is required for FID feature extraction.")

    inception = tf.keras.applications.InceptionV3(
        include_top=False,
        pooling='avg',
        input_shape=(299, 299, 3),
        weights='imagenet',
    )
    inception.trainable = False
    return inception


def _to_unit_range(images):
    """Map an image batch into [0, 1].

    Accepts either [0, 1] or [-1, 1] inputs. The source range is inferred from
    the actual value range rather than from a single global minimum, so a batch
    that happens to contain no negative values is never mis-scaled.
    """
    if images.size == 0:
        return images
    if float(np.min(images)) < 0.0:
        images = (images + 1.0) / 2.0
    return np.clip(images, 0.0, 1.0)


def preprocess_for_inception(images, target_size=(299, 299)):
    """Preprocess images for InceptionV3 feature extraction."""
    images = _to_unit_range(images) * 255.0

    # Resize to InceptionV3 input size (299x299)
    images = tf.image.resize(images, target_size)

    # Convert grayscale to 3 channels
    if images.shape[-1] == 1:
        images = tf.image.grayscale_to_rgb(images)

    # Preprocess for InceptionV3
    images = tf.keras.applications.inception_v3.preprocess_input(images)
    return images


def calculate_fid(real_images, generated_images, batch_size=64):
    """
    Calculate Fréchet Inception Distance between real and generated images.
    FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*sqrt(Sigma_r * Sigma_g))
    """
    feature_extractor = build_inception_feature_extractor()

    # Extract features
    real_features = _extract_features(real_images, feature_extractor, batch_size)
    gen_features = _extract_features(generated_images, feature_extractor, batch_size)

    return _frechet_distance(real_features, gen_features)


def _frechet_distance(real_features, gen_features):
    """Compute the Fréchet distance between two sets of extracted features."""
    mu_real, sigma_real = _calculate_statistics(real_features)
    mu_gen, sigma_gen = _calculate_statistics(gen_features)

    diff = mu_real - mu_gen
    covmean = _matrix_sqrtm(sigma_real @ sigma_gen)

    return float(diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean))


def _matrix_sqrtm(matrix):
    """Principal square root of a matrix, tolerant of SciPy API changes.

    ``scipy.linalg.sqrtm`` removed the ``disp`` argument in SciPy 1.16; calling it
    with that keyword raises ``TypeError`` on modern SciPy. Complex results are
    returned as their real part, which is the standard FID convention.
    """
    covmean = linalg.sqrtm(matrix)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.imag(covmean), 0.0, atol=1e-3):
            warnings.warn(
                "sqrtm returned a strongly complex result; using the real part.",
                RuntimeWarning,
                stacklevel=2,
            )
        covmean = covmean.real
    return covmean


def _extract_features(images, feature_extractor, batch_size):
    """Extract features from images using the feature extractor.

    Preprocessing and inference are done in chunks so peak memory stays bounded
    by ``batch_size`` 299x299x3 tensors instead of materialising the whole set.
    """
    images = np.asarray(images)
    if len(images) == 0:
        raise ValueError("Cannot extract features from an empty image batch.")
    if batch_size is None or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}.")

    chunks = []
    for start in range(0, len(images), batch_size):
        chunk = preprocess_for_inception(images[start : start + batch_size])
        chunks.append(np.asarray(feature_extractor.predict(chunk, batch_size=batch_size,
            verbose=0)))
    return np.concatenate(chunks, axis=0)


def _calculate_statistics(features):
    """Calculate mean and covariance of features.

    ``np.cov`` returns an all-NaN matrix when fewer than two samples are given
    (degrees of freedom <= 0), which silently poisons every downstream score.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {features.shape}.")
    if features.shape[0] < 2:
        raise ValueError(
            f"At least 2 samples are needed for a covariance matrix, got {features.shape[0]}. "
            "Increase the FID/FS evaluation sample count."
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("Extracted features contain non-finite values; check the model outputs.")

    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


# ═══════════════════════════════════════════════════════════════════════
# FS — Fréchet Score (simplified version for medical images)
# ═══════════════════════════════════════════════════════════════════════
def calculate_fs(real_images, generated_images, batch_size=64, seed=1234):
    """
    Calculate Fréchet Score — a Fréchet distance computed in the feature space
    of a fixed, randomly-initialised CNN rather than InceptionV3.
    Lower is better.

    .. note::
       The feature extractor is *not* trained. It is a fixed random projection
       whose weights are seeded deterministically so that scores remain
       comparable across epochs and runs. Treat FS as a cheap relative progress
       indicator only; FID is the primary absolute metric.
    """
    feature_extractor = _build_medical_feature_extractor(
        np.asarray(real_images).shape[1:], seed=seed
    )

    real_features = _extract_features_custom(real_images, feature_extractor, batch_size)
    gen_features = _extract_features_custom(generated_images, feature_extractor, batch_size)

    return _frechet_distance(real_features, gen_features)


def _build_medical_feature_extractor(input_shape, seed=1234):
    """Build a fixed (untrained) CNN feature extractor for the FS metric."""
    if tf is None:
        raise ImportError("TensorFlow is required for FS feature extraction.")

    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inputs, x)
    return model


def _extract_features_custom(images, feature_extractor, batch_size):
    """Extract features using a custom (non-Inception) feature extractor."""
    images = _to_unit_range(np.asarray(images))
    if batch_size is None or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}.")
    chunks = []
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        chunks.append(
            np.asarray(feature_extractor.predict(chunk, batch_size=batch_size, verbose=0))
        )
    return np.concatenate(chunks, axis=0)
