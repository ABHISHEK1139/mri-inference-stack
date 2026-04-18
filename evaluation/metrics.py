"""
Evaluation metrics: FID (Fréchet Inception Distance), FS (Fréchet Score),
confusion matrix, classification reports, and segmentation metrics.
"""
import os
import numpy as np
try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None
from scipy import linalg
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from config import CLASS_NAMES, LOG_DIR


# ═══════════════════════════════════════════════════════════════════════
# FID — Fréchet Inception Distance
# ═══════════════════════════════════════════════════════════════════════
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


def preprocess_for_inception(images, target_size=(299, 299)):
    """Preprocess images for InceptionV3 feature extraction."""
    # Denormalize from [-1,1] to [0,1] if needed
    if images.min() < 0:
        images = (images + 1.0) / 2.0
    images = np.clip(images, 0.0, 1.0)
    images = images * 255.0

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

    # Calculate statistics
    mu_real, sigma_real = _calculate_statistics(real_features)
    mu_gen, sigma_gen = _calculate_statistics(gen_features)

    # Calculate FID
    diff = mu_real - mu_gen
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_gen, disp=False)

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return float(fid)


def _extract_features(images, feature_extractor, batch_size):
    """Extract features from images using the feature extractor."""
    images = preprocess_for_inception(images)
    features = feature_extractor.predict(images, batch_size=batch_size, verbose=0)
    return features


def _calculate_statistics(features):
    """Calculate mean and covariance of features."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


# ═══════════════════════════════════════════════════════════════════════
# FS — Fréchet Score (simplified version for medical images)
# ═══════════════════════════════════════════════════════════════════════
def calculate_fs(real_images, generated_images, batch_size=64):
    """
    Calculate Fréchet Score — a simplified metric using a custom CNN
    feature extractor trained on the real data distribution.
    Lower is better.
    """
    # Build a simple feature extractor
    feature_extractor = _build_medical_feature_extractor(real_images.shape[1:])

    # Extract features
    real_features = _extract_features_custom(real_images, feature_extractor, batch_size)
    gen_features = _extract_features_custom(generated_images, feature_extractor, batch_size)

    # Calculate Fréchet distance on these features
    mu_real, sigma_real = _calculate_statistics(real_features)
    mu_gen, sigma_gen = _calculate_statistics(gen_features)

    diff = mu_real - mu_gen
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_gen, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fs = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return float(fs)


def _build_medical_feature_extractor(input_shape):
    """Build a simple CNN feature extractor for medical images."""
    if tf is None:
        raise ImportError("TensorFlow is required for FS feature extraction.")

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
    """Extract features using custom feature extractor."""
    # Normalize to [0,1]
    if images.min() < 0:
        images = (images + 1.0) / 2.0
    images = np.clip(images, 0.0, 1.0)
    features = feature_extractor.predict(images, batch_size=batch_size, verbose=0)
    return features


# ═══════════════════════════════════════════════════════════════════════
# CLASSIFICATION METRICS
# ═══════════════════════════════════════════════════════════════════════
def evaluate_classifier(model, X_test, y_test, track_name="classifier", save_dir=None):
    """Evaluate a classifier and generate all metrics + plots."""
    save_dir = save_dir or os.path.join(LOG_DIR, track_name)
    os.makedirs(save_dir, exist_ok=True)

    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Handle one-hot labels
    if len(y_test.shape) > 1 and y_test.shape[-1] > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    try:
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    except Exception:
        auc = 0.0

    # Classification report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig_cm = plot_confusion_matrix(cm, CLASS_NAMES, save_path=os.path.join(save_dir, "confusion_matrix.png"))

    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'classification_report': report,
    }

    print(f"\n{'='*50}")
    print(f"  {track_name.upper()} EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"\n{report}")

    return metrics


def evaluate_detection(model, X_test, y_test, save_dir=None):
    """Evaluate binary detection model."""
    save_dir = save_dir or os.path.join(LOG_DIR, "detection")
    os.makedirs(save_dir, exist_ok=True)

    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except Exception:
        auc = 0.0

    cm = confusion_matrix(y_test, y_pred)
    class_names_binary = ["Normal", "Tumour"]
    plot_confusion_matrix(cm, class_names_binary, save_path=os.path.join(save_dir, "confusion_matrix.png"))

    metrics = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'auc': auc}
    print(f"\n  Detection — Acc:{acc:.4f} Prec:{prec:.4f} Rec:{rec:.4f} F1:{f1:.4f} AUC:{auc:.4f}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════
# SEGMENTATION METRICS
# ═══════════════════════════════════════════════════════════════════════
def evaluate_segmentation(model, X_test=None, y_test=None, test_ds=None, save_dir=None):
    """Evaluate segmentation model with Dice, IoU, etc.

    Supports two modes:
      - Legacy: pass X_test and y_test as numpy arrays
      - Streaming: pass test_ds as a tf.data.Dataset (memory-efficient)
    """
    save_dir = save_dir or os.path.join(LOG_DIR, "segmentation")
    os.makedirs(save_dir, exist_ok=True)

    # Collect predictions — streaming or in-memory
    dice_scores = []
    iou_scores = []
    vis_inputs = []
    vis_truths = []
    vis_preds = []
    vis_dices = []
    n_vis_target = 8

    if test_ds is not None:
        # Streaming mode: iterate batch-by-batch
        for x_batch, y_batch in test_ds:
            pred_batch = model.predict(x_batch, verbose=0)
            pred_bin = (pred_batch > 0.5).astype(np.float32)
            y_np = y_batch.numpy() if hasattr(y_batch, 'numpy') else np.asarray(y_batch)
            x_np = x_batch.numpy() if hasattr(x_batch, 'numpy') else np.asarray(x_batch)
            for i in range(len(y_np)):
                d = _dice_coef(y_np[i], pred_bin[i])
                iou = _iou_coef(y_np[i], pred_bin[i])
                dice_scores.append(d)
                iou_scores.append(iou)
                if len(vis_inputs) < n_vis_target:
                    vis_inputs.append(x_np[i])
                    vis_truths.append(y_np[i])
                    vis_preds.append(pred_bin[i])
                    vis_dices.append(d)
    else:
        # Legacy in-memory mode
        y_pred = model.predict(X_test, verbose=0)
        y_pred_bin = (y_pred > 0.5).astype(np.float32)
        for i in range(len(y_test)):
            d = _dice_coef(y_test[i], y_pred_bin[i])
            iou = _iou_coef(y_test[i], y_pred_bin[i])
            dice_scores.append(d)
            iou_scores.append(iou)
        vis_inputs = list(X_test[:n_vis_target])
        vis_truths = list(y_test[:n_vis_target])
        vis_preds = list(y_pred_bin[:n_vis_target])
        vis_dices = dice_scores[:n_vis_target]

    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)

    # Visualize some predictions
    n_vis = min(n_vis_target, len(vis_inputs))
    if n_vis > 0:
        fig, axes = plt.subplots(3, n_vis, figsize=(3 * n_vis, 9))
        if n_vis == 1:
            axes = axes[:, np.newaxis]
        for i in range(n_vis):
            axes[0, i].imshow(vis_inputs[i].squeeze(), cmap='gray')
            axes[0, i].set_title('Input')
            axes[1, i].imshow(vis_truths[i].squeeze(), cmap='gray')
            axes[1, i].set_title('Ground Truth')
            axes[2, i].imshow(vis_preds[i].squeeze(), cmap='gray')
            axes[2, i].set_title(f'Pred (D={vis_dices[i]:.3f})')
            for row in range(3):
                axes[row, i].axis('off')
        plt.suptitle(f"Segmentation — Mean Dice: {mean_dice:.4f}, Mean IoU: {mean_iou:.4f}")
        plt.savefig(os.path.join(save_dir, "segmentation_results.png"), dpi=150, bbox_inches='tight', facecolor='white', transparent=False)
        plt.close()

    metrics = {'mean_dice': mean_dice, 'mean_iou': mean_iou}
    print(f"\n  Segmentation — Dice:{mean_dice:.4f} IoU:{mean_iou:.4f}")
    return metrics


def _dice_coef(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


def _iou_coef(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot and optionally save confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_loss_curves(history, save_path=None, title="Training Loss"):
    """Plot training and validation loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    acc_key = 'accuracy' if 'accuracy' in history.history else 'dice_coefficient'
    val_acc_key = f'val_{acc_key}'
    ax2.plot(history.history[acc_key], label=f'Train {acc_key}')
    if val_acc_key in history.history:
        ax2.plot(history.history[val_acc_key], label=f'Val {acc_key}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(acc_key)
    ax2.set_title(f'{acc_key.title()} Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_gan_losses(d_losses, g_losses, d_accs=None, g_accs=None, save_path=None):
    """Plot GAN training losses and accuracies."""
    fig, axes = plt.subplots(1, 2 if d_accs is None else 2, figsize=(14, 5))

    axes[0].plot(d_losses, label='D Loss')
    axes[0].plot(g_losses, label='G Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('GAN Losses')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if d_accs is not None and len(d_accs) > 0:
        axes[1].plot(d_accs, label='D Accuracy')
        if g_accs is not None:
            axes[1].plot(g_accs, label='G Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('GAN Accuracies')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_fid_fs_vs_epochs(fid_scores, fs_scores, save_path=None):
    """Plot FID and FS scores over training epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(fid_scores, 'b-o', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('FID')
    ax1.set_title('FID Score vs Epochs')
    ax1.grid(True, alpha=0.3)

    ax2.plot(fs_scores, 'r-o', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('FS')
    ax2.set_title('FS Score vs Epochs')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig