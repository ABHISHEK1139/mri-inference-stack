"""Plotting helpers shared by the evaluation reports."""

import os

import matplotlib.pyplot as plt
import seaborn as sns


def _ensure_parent_dir(save_path) -> None:
    """Create the directory a plot will be written into.

    ``logs/<track>/`` is not created by ``config.ensure_directories()`` (which
    only makes the top-level directories), so saving a loss curve raised
    ``FileNotFoundError`` at the very end of an otherwise successful training
    run and aborted it.
    """
    if not save_path:
        return
    parent = os.path.dirname(os.path.abspath(str(save_path)))
    if parent:
        os.makedirs(parent, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot and optionally save confusion matrix."""
    fig, ax = plt.subplots(figsize=(8,
        6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
        yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    _ensure_parent_dir(save_path)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_loss_curves(history, save_path=None, title="Training Loss"):
    """Plot available training/validation curves.

    Only metrics actually present in ``history.history`` are drawn, so a model
    compiled without a secondary metric no longer raises ``KeyError`` and loses
    the loss plot entirely.
    """
    recorded = getattr(history, "history", None) or {}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    if 'loss' in recorded:
        ax1.plot(recorded['loss'], label='Train Loss')
        if 'val_loss' in recorded:
            ax1.plot(recorded['val_loss'], label='Val Loss')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'No loss recorded', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.grid(True, alpha=0.3)

    # Secondary metric
    acc_key = next(
        (k for k in ('accuracy', 'dice_coefficient', 'iou_metric', 'f1_score') if k in recorded),
        None,
    )
    if acc_key is not None:
        ax2.plot(recorded[acc_key], label=f'Train {acc_key}')
        val_acc_key = f'val_{acc_key}'
        if val_acc_key in recorded:
            ax2.plot(recorded[val_acc_key], label=f'Val {acc_key}')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No metric recorded', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(acc_key or 'Metric')
    ax2.set_title(f'{(acc_key or "Metric").title()} Curves')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    _ensure_parent_dir(save_path)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_gan_losses(d_losses, g_losses, d_accs=None, g_accs=None, save_path=None, d_label="D Acc"):
    """Plot GAN training losses and (optionally) per-epoch accuracies."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(d_losses, label='D Loss')
    axes[0].plot(g_losses, label='G Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('GAN Losses')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if d_accs is not None and len(d_accs) > 0:
        axes[1].plot(d_accs, label=d_label)
        if g_accs is not None and len(g_accs) > 0:
            axes[1].plot(g_accs, label='G Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(d_label)
        axes[1].set_title('GAN Accuracies')
        axes[1].legend()
        axes[1].grid(True,
            alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No accuracy recorded', ha='center', va='center',
            transform=axes[1].transAxes)
        axes[1].set_axis_off()

    plt.tight_layout()
    _ensure_parent_dir(save_path)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_fid_fs_vs_epochs(fid_scores, fs_scores, save_path=None):
    """Plot FID and FS scores over training epochs."""
    if len(fid_scores) != len(fs_scores):
        raise ValueError(
            f"FID and FS must have the same length, got {len(fid_scores)} and {len(fs_scores)}."
        )

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
    _ensure_parent_dir(save_path)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig
