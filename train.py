"""
Main training script — All 4 Tracks with GPU acceleration.
Optimized for NVIDIA RTX 3050 (4GB VRAM).
Features: checkpoint resumption, automatic dataset download, and monitoring.
"""
import argparse
import os
import sys
import json
import time
import numpy as np
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Brain Tumour GAN Challenge Training")
    parser.add_argument("--track", type=str, default="all", choices=["all", "detection", "segmentation", "classifier", "gan", "gan_v2", "gan_augmented"])
    parser.add_argument("--gan_type", type=str, default="conditional", choices=["baseline", "dcgan", "conditional", "stylegan"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--no_resume", action="store_true", help="Disable checkpoint resume and start fresh")
    parser.add_argument("--download_figshare", action="store_true", help="Force/ensure Figshare download")
    parser.add_argument("--download_brats", action="store_true", help="Force/ensure BraTS download")
    parser.add_argument("--only_download", action="store_true", help="Download datasets only and exit")
    return parser


if __name__ == "__main__" and any(flag in {"-h", "--help"} for flag in sys.argv[1:]):
    build_arg_parser().print_help()
    sys.exit(0)


import tensorflow as tf

from config import (
    RAW_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
    WEIGHTS_DIR,
    NUM_CLASSES,
    LATENT_DIM,
    TRACK_CONFIGS,
    ImageConfig,
    RUNTIME_PROFILE,
    LOW_VRAM_MODE,
)
from data.dataset import (
    download_dataset,
    load_figshare_dataset,
    load_brats_dataset,
    load_brats_paths,
    load_images_from_paths,
    get_figshare_file_index,
    get_figshare_train_val_test_split,
    split_data,
    build_detection_dataset,
    build_detection_dataset_from_paths,
    build_segmentation_dataset,
    build_segmentation_dataset_from_paths,
    build_classifier_dataset,
    build_classifier_dataset_from_paths,
    build_gan_dataset,
    build_gan_dataset_from_paths,
    mix_real_synthetic,
)
from models.detection import build_detection_model, build_detection_baseline
from models.segmentation import build_unet, dice_bce_loss, dice_coefficient, iou_metric
from models.classifier import build_classifier, build_classifier_baseline
from models.gan import (
    build_generator,
    build_discriminator,
    build_gan,
    build_conditional_generator,
    build_conditional_discriminator,
    build_conditional_gan,
    build_stylegan_generator,
    build_baseline_generator,
    build_baseline_discriminator,
    build_v2_generator,
    build_v2_discriminator,
    gradient_penalty,
    EMAGenerator,
)
from training.callbacks import (
    get_standard_callbacks,
    GANLossLogger,
    GANImageSampler,
    ModelCollapseDetector,
)
from evaluation.detection_eval import calibrate_binary_threshold, evaluate_detection_refined
from evaluation.metrics import (
    calculate_fid,
    calculate_fs,
    evaluate_classifier,
    evaluate_segmentation,
    plot_loss_curves,
    plot_gan_losses,
    plot_fid_fs_vs_epochs,
)

IMG_CFG = ImageConfig()


def configure_gpu():
    """Configure TensorFlow for RTX 3050."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            use_memory_growth = os.getenv("TF_MEMORY_GROWTH", "1").strip().lower() in {"1", "true", "yes", "on"}
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, use_memory_growth)
            print(f"TF memory growth: {use_memory_growth}")
            print(f"GPU detected: {[gpu.name for gpu in gpus]}")
            use_mixed_precision = os.getenv("MIXED_PRECISION", "1").strip().lower() in {"1", "true", "yes", "on"}
            if use_mixed_precision:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
                print(f"Mixed precision enabled: {policy.name}")
            else:
                policy = tf.keras.mixed_precision.Policy("float32")
                tf.keras.mixed_precision.set_global_policy(policy)
                print(f"Mixed precision disabled: {policy.name}")
        except RuntimeError as e:
            print(f"GPU setup warning: {e}")
    else:
        print("No GPU detected, running on CPU")

    print(f"TensorFlow: {tf.__version__}")
    print(f"CUDA build: {tf.test.is_built_with_cuda()}")
    print(f"Runtime profile: {RUNTIME_PROFILE}")
    if LOW_VRAM_MODE:
        print("Low-VRAM mode enabled for 4GB-class GPUs")
    if gpus:
        print(f"GPU device: {tf.test.gpu_device_name()}")


def _balanced_class_weight_dict(labels):
    """Compute simple balanced class weights for binary labels."""
    labels = np.asarray(labels, dtype=np.int32)
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    total = float(counts.sum())
    num_classes = float(len(counts))
    weights = {}
    for idx, count in enumerate(counts):
        weights[idx] = float(total / max(num_classes * count, 1.0))
    return weights


def _json_safe(value):
    """Recursively convert NumPy values into JSON-safe Python types."""
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sanitize_grads(grads, variables):
    """Replace None/NaN/Inf gradients with finite tensors."""
    fixed = []
    for grad, var in zip(grads, variables):
        if grad is None:
            fixed.append(tf.zeros_like(var))
            continue
        fixed.append(tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad)))
    return fixed


def _generator_is_finite(generator, conditional, num_classes, sample_count=16):
    """Quick health probe to ensure generator outputs are finite."""
    z = tf.random.normal([sample_count, LATENT_DIM])
    if conditional:
        idx = tf.random.uniform([sample_count], 0, num_classes, dtype=tf.int32)
        y = tf.one_hot(idx, num_classes)
        out = generator([z, tf.cast(y, tf.float32)], training=False)
    else:
        out = generator(z, training=False)
    return bool(tf.reduce_all(tf.math.is_finite(out)).numpy())


class TrainingState:
    """Persistent state for resuming non-GAN training."""

    def __init__(self, track_name):
        self.track_name = track_name
        self.track_dir = os.path.join(CHECKPOINT_DIR, track_name)
        os.makedirs(self.track_dir, exist_ok=True)
        self.state_path = os.path.join(self.track_dir, "training_state.json")
        self.state = self._load()

    def _load(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"Resuming {self.track_name} from epoch {data.get('last_epoch', 0) + 1}")
                return data
            except Exception as e:
                print(f"Could not load state for {self.track_name}: {e}")
        return {"last_epoch": -1}

    def save(self):
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def update_epoch(self, epoch):
        self.state["last_epoch"] = int(epoch)
        self.save()

    def start_epoch(self):
        return int(self.state.get("last_epoch", -1)) + 1

    def checkpoint_path(self):
        best_path = os.path.join(self.track_dir, "best_model.keras")
        last_path = os.path.join(self.track_dir, "last_model.keras")
        if os.path.exists(best_path):
            return best_path
        if os.path.exists(last_path):
            return last_path
        return None


class GANState:
    """Persistent state for resuming GAN training."""

    def __init__(self, gan_type):
        self.gan_type = gan_type
        self.track_dir = os.path.join(CHECKPOINT_DIR, "gan")
        os.makedirs(self.track_dir, exist_ok=True)
        self.state_path = os.path.join(self.track_dir, f"gan_{gan_type}_state.json")
        self.state = self._load()

    def _load(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"Resuming GAN ({self.gan_type}) from epoch {data.get('last_epoch', 0) + 1}")
                return data
            except Exception as e:
                print(f"Could not load GAN state: {e}")
        return {
            "last_epoch": -1,
            "d_losses": [],
            "g_losses": [],
            "d_accs": [],
            "g_accs": [],
            "fid_scores": [],
            "fs_scores": [],
        }

    def save(self):
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def start_epoch(self):
        return int(self.state.get("last_epoch", -1)) + 1

    def generator_ckpt(self):
        return os.path.join(self.track_dir, f"generator_{self.gan_type}_last.keras")

    def discriminator_ckpt(self):
        return os.path.join(self.track_dir, f"discriminator_{self.gan_type}_last.keras")

    def has_ckpt(self):
        return os.path.exists(self.generator_ckpt()) and os.path.exists(self.discriminator_ckpt())

    @staticmethod
    def fresh_state():
        return {
            "last_epoch": -1,
            "d_losses": [],
            "g_losses": [],
            "d_accs": [],
            "g_accs": [],
            "fid_scores": [],
            "fs_scores": [],
        }


def _download_kaggle_alternative():
    """Fallback dataset downloader for classification MRI data."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    import glob

    out_dir = os.path.join(RAW_DIR, "figshare")
    os.makedirs(out_dir, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        "masoudnickparvar/brain-tumor-mri-dataset",
        path=out_dir,
        unzip=True,
    )
    for zf in glob.glob(os.path.join(out_dir, "*.zip")):
        try:
            os.remove(zf)
        except OSError:
            pass


def ensure_datasets(download_figshare=True, download_brats=True):
    """Ensure required datasets exist. Download when missing."""
    def has_files(path):
        return os.path.exists(path) and any(os.scandir(path))

    if download_figshare:
        figshare_dir = os.path.join(RAW_DIR, "figshare")
        if not has_files(figshare_dir):
            print("Downloading Figshare dataset...")
            try:
                download_dataset("figshare")
                if has_files(figshare_dir):
                    print("Figshare download complete")
                else:
                    print("Figshare download did not produce files; manual download may be needed")
            except BaseException as e:
                print(f"Figshare direct download failed: {e}")
                print("Trying Kaggle fallback...")
                try:
                    _download_kaggle_alternative()
                    if has_files(figshare_dir):
                        print("Kaggle fallback download complete")
                    else:
                        print("Kaggle fallback finished but no files found")
                except BaseException as ke:
                    print(f"Kaggle fallback failed: {ke}")
                    print("Please manually place Figshare/Kaggle MRI dataset under data/raw/figshare")
        else:
            print(f"Figshare dataset found: {figshare_dir}")

    if download_brats:
        brats_dir = os.path.join(RAW_DIR, "brats")
        if not has_files(brats_dir):
            print("Downloading BraTS dataset (requires Kaggle API credentials)...")
            try:
                download_dataset("brats")
                if has_files(brats_dir):
                    print("BraTS download complete")
                else:
                    print("BraTS download did not produce files; manual download may be needed")
            except BaseException as e:
                print(f"BraTS auto-download failed: {e}")
                print("Please manually place BraTS under data/raw/brats")
        else:
            print(f"BraTS dataset found: {brats_dir}")


def train_detection(data_dir=None, use_enhanced=True, epochs=None, resume=True):
    print("\n" + "=" * 60)
    print("TRACK 1 - DETECTION")
    print("=" * 60)

    (X_train_paths, y_train_multiclass), (X_val_paths, y_val_multiclass), (X_test_paths, y_test_multiclass) = get_figshare_train_val_test_split(data_dir)
    y_train = (y_train_multiclass < 3).astype(np.int32)
    y_val = (y_val_multiclass < 3).astype(np.int32)
    y_test = (y_test_multiclass < 3).astype(np.int32)

    cfg = TRACK_CONFIGS["detection"]
    train_ds = build_detection_dataset_from_paths(X_train_paths, y_train, img_size=IMG_CFG.detection_size, batch_size=cfg.batch_size)
    val_ds = build_detection_dataset_from_paths(
        X_val_paths, y_val, img_size=IMG_CFG.detection_size, batch_size=cfg.batch_size, shuffle=False, augment=False
    )

    state = TrainingState("detection")
    initial_epoch = 0

    if resume and state.checkpoint_path():
        ckpt = state.checkpoint_path()
        print(f"Loading checkpoint: {ckpt}")
        model = tf.keras.models.load_model(ckpt)
        initial_epoch = state.start_epoch()
    else:
        model = build_detection_model(input_shape=(*IMG_CFG.detection_size, 1)) if use_enhanced else build_detection_baseline(input_shape=(*IMG_CFG.detection_size, 1))

    class_weights = _balanced_class_weight_dict(y_train)
    print(f"Detection class weights: {class_weights}")

    class StateSaver(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            model.save(os.path.join(CHECKPOINT_DIR, "detection", "last_model.keras"))
            state.update_epoch(epoch)

    total_epochs = epochs or cfg.epochs
    history = None
    if initial_epoch >= total_epochs:
        print("Detection already reached requested epochs")
    else:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=initial_epoch,
            epochs=total_epochs,
            class_weight=class_weights,
            callbacks=get_standard_callbacks(model, "detection") + [StateSaver()],
        )

    best_ckpt = state.checkpoint_path()
    if best_ckpt:
        print(f"Reloading best detection checkpoint: {best_ckpt}")
        model = tf.keras.models.load_model(best_ckpt)

    X_val = load_images_from_paths(X_val_paths, img_size=IMG_CFG.detection_size)
    X_test = load_images_from_paths(X_test_paths, img_size=IMG_CFG.detection_size)
    val_probs = model.predict(X_val, verbose=0).flatten()
    threshold, threshold_metrics = calibrate_binary_threshold(y_val, val_probs, optimize="f1", min_recall=0.97)
    print(
        "Detection threshold tuning - "
        f"threshold={threshold:.3f} "
        f"val_f1={threshold_metrics['f1_score']:.4f} "
        f"val_recall={threshold_metrics['recall']:.4f} "
        f"val_specificity={threshold_metrics['specificity']:.4f}"
    )

    model.save(os.path.join(WEIGHTS_DIR, "detection_model.keras"))
    with open(os.path.join(CHECKPOINT_DIR, "detection", "inference_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            _json_safe({"threshold": threshold, "validation_metrics": threshold_metrics}),
            f,
            indent=2,
        )

    metrics = evaluate_detection_refined(model, X_test, y_test, threshold=threshold)
    if history is not None:
        plot_loss_curves(history, save_path=os.path.join(LOG_DIR, "detection", "loss_curves.png"))
    return model, history, metrics


def train_segmentation(data_dir=None, use_attention=True, use_residual=True, epochs=None, resume=True):
    print("\n" + "=" * 60)
    print("TRACK 2 - SEGMENTATION")
    print("=" * 60)

    # Memory-efficient: load file paths only, not pixel data
    img_paths, mask_paths = load_brats_paths(data_dir, img_size=IMG_CFG.segmentation_size)
    n = len(img_paths)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    split1, split2 = int(0.7 * n), int(0.85 * n)
    train_img = img_paths[idx[:split1]]
    train_msk = mask_paths[idx[:split1]]
    val_img = img_paths[idx[split1:split2]]
    val_msk = mask_paths[idx[split1:split2]]
    test_img = img_paths[idx[split2:]]
    test_msk = mask_paths[idx[split2:]]
    print(f"Split: Train={len(train_img)}, Val={len(val_img)}, Test={len(test_img)}")

    cfg = TRACK_CONFIGS["segmentation"]
    seg_size = IMG_CFG.segmentation_size
    train_ds = build_segmentation_dataset_from_paths(
        train_img, train_msk, img_size=seg_size, batch_size=cfg.batch_size,
    )
    val_ds = build_segmentation_dataset_from_paths(
        val_img, val_msk, img_size=seg_size, batch_size=cfg.batch_size,
        shuffle=False, augment=False,
    )

    state = TrainingState("segmentation")
    initial_epoch = 0

    if resume and state.checkpoint_path():
        ckpt = state.checkpoint_path()
        print(f"Loading checkpoint weights: {ckpt}")
        model = build_unet(
            input_shape=(*seg_size, 1),
            use_attention=use_attention,
            use_residual=use_residual,
        )
        model.load_weights(ckpt)
        initial_epoch = state.start_epoch()
    else:
        model = build_unet(
            input_shape=(*seg_size, 1),
            use_attention=use_attention,
            use_residual=use_residual,
        )

    class StateSaver(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            model.save(os.path.join(CHECKPOINT_DIR, "segmentation", "last_model.keras"))
            state.update_epoch(epoch)

    total_epochs = epochs or cfg.epochs
    if initial_epoch >= total_epochs:
        print("Segmentation already reached requested epochs")
        test_ds = build_segmentation_dataset_from_paths(
            test_img, test_msk, img_size=seg_size, batch_size=cfg.batch_size,
            shuffle=False, augment=False,
        )
        metrics = evaluate_segmentation(model, test_ds=test_ds)
        return model, None, metrics

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=total_epochs,
        callbacks=get_standard_callbacks(model, "segmentation") + [StateSaver()],
    )

    model.save(os.path.join(WEIGHTS_DIR, "segmentation_model.keras"))
    test_ds = build_segmentation_dataset_from_paths(
        test_img, test_msk, img_size=seg_size, batch_size=cfg.batch_size,
        shuffle=False, augment=False,
    )
    metrics = evaluate_segmentation(model, test_ds=test_ds)
    plot_loss_curves(history, save_path=os.path.join(LOG_DIR, "segmentation", "loss_curves.png"))
    return model, history, metrics


def train_classifier(data_dir=None, use_enhanced=True, epochs=None, resume=True):
    print("\n" + "=" * 60)
    print("TRACK 3 - CLASSIFICATION")
    print("=" * 60)

    (X_train_paths, y_train), (X_val_paths, y_val), (X_test_paths, y_test) = get_figshare_train_val_test_split(data_dir)

    cfg = TRACK_CONFIGS["classifier"]
    train_ds = build_classifier_dataset_from_paths(X_train_paths, y_train, img_size=IMG_CFG.classifier_size, batch_size=cfg.batch_size)
    val_ds = build_classifier_dataset_from_paths(
        X_val_paths, y_val, img_size=IMG_CFG.classifier_size, batch_size=cfg.batch_size, shuffle=False, augment=False
    )

    state = TrainingState("classifier")
    initial_epoch = 0

    if resume and state.checkpoint_path():
        ckpt = state.checkpoint_path()
        print(f"Loading checkpoint: {ckpt}")
        model = tf.keras.models.load_model(ckpt)
        initial_epoch = state.start_epoch()
    else:
        model = build_classifier(num_classes=NUM_CLASSES, input_shape=(*IMG_CFG.classifier_size, 1)) if use_enhanced else build_classifier_baseline(num_classes=NUM_CLASSES, input_shape=(*IMG_CFG.classifier_size, 1))

    class StateSaver(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            model.save(os.path.join(CHECKPOINT_DIR, "classifier", "last_model.keras"))
            state.update_epoch(epoch)

    total_epochs = epochs or cfg.epochs
    history = None
    if initial_epoch >= total_epochs:
        print("Classifier already reached requested epochs")
    else:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=initial_epoch,
            epochs=total_epochs,
            callbacks=get_standard_callbacks(model, "classifier") + [StateSaver()],
        )

    best_ckpt = state.checkpoint_path()
    if best_ckpt:
        print(f"Reloading best classifier checkpoint: {best_ckpt}")
        model = tf.keras.models.load_model(best_ckpt)

    model.save(os.path.join(WEIGHTS_DIR, "classifier_model.keras"))
    X_test = load_images_from_paths(X_test_paths, img_size=IMG_CFG.classifier_size)
    y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)
    metrics = evaluate_classifier(model, X_test, y_test_oh)
    if history is not None:
        plot_loss_curves(history, save_path=os.path.join(LOG_DIR, "classifier", "loss_curves.png"))
    return model, history, metrics


def train_gan(data_dir=None, gan_type="conditional", epochs=None, fid_eval_freq=10, resume=True):
    print("\n" + "=" * 60)
    print(f"TRACK 4 - GAN ({gan_type.upper()})")
    print("=" * 60)

    (X_train_paths, y_train_labels), (X_val_paths, y_val_labels), _ = get_figshare_train_val_test_split(data_dir)
    cfg = TRACK_CONFIGS["gan"]
    fid_eval_freq = int(os.getenv("GAN_FID_EVAL_FREQ", str(fid_eval_freq)))
    fid_eval_freq = max(0, fid_eval_freq)
    img_shape = (*IMG_CFG.gan_size, 1)

    conditional = False
    if gan_type == "baseline":
        generator = build_baseline_generator(latent_dim=LATENT_DIM)
        discriminator = build_baseline_discriminator(input_shape=(64, 64, 1))
        images_for_train = X_train_paths
        gan = build_gan(generator, discriminator, latent_dim=LATENT_DIM, lr=cfg.learning_rate)
    elif gan_type == "dcgan":
        generator = build_generator(latent_dim=LATENT_DIM, output_shape=img_shape)
        discriminator = build_discriminator(input_shape=img_shape)
        images_for_train = X_train_paths
        gan = build_gan(generator, discriminator, latent_dim=LATENT_DIM, lr=cfg.learning_rate)
    elif gan_type == "conditional":
        generator = build_conditional_generator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES, output_shape=img_shape)
        discriminator = build_conditional_discriminator(input_shape=img_shape, num_classes=NUM_CLASSES)
        images_for_train = X_train_paths
        gan = build_conditional_gan(generator, discriminator, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES, lr=cfg.learning_rate)
        conditional = True
    elif gan_type == "stylegan":
        generator = build_stylegan_generator(latent_dim=LATENT_DIM, output_shape=img_shape)
        discriminator = build_discriminator(input_shape=img_shape)
        images_for_train = X_train_paths
        gan = build_gan(generator, discriminator, latent_dim=LATENT_DIM, lr=cfg.learning_rate)
    else:
        raise ValueError(f"Unknown GAN type: {gan_type}")

    state = GANState(gan_type)
    initial_epoch = 0

    if not resume:
        state.state = GANState.fresh_state()

    if resume and state.has_ckpt():
        print("Loading GAN checkpoints...")
        generator = tf.keras.models.load_model(state.generator_ckpt())
        discriminator = tf.keras.models.load_model(state.discriminator_ckpt())
        if conditional:
            gan = build_conditional_gan(generator, discriminator, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES, lr=cfg.learning_rate)
        else:
            gan = build_gan(generator, discriminator, latent_dim=LATENT_DIM, lr=cfg.learning_rate)
        initial_epoch = state.start_epoch()

    gan_img_size = (64, 64) if gan_type == "baseline" else IMG_CFG.gan_size
    dataset = build_gan_dataset_from_paths(
        images_for_train,
        labels=y_train_labels if conditional else None,
        img_size=gan_img_size,
        batch_size=cfg.batch_size,
    )
    loss_logger = GANLossLogger()
    sampler = GANImageSampler(generator, latent_dim=LATENT_DIM, conditional=conditional)
    collapse_detector = ModelCollapseDetector(
        generator,
        latent_dim=LATENT_DIM,
        conditional=conditional,
        num_classes=NUM_CLASSES,
    )

    if initial_epoch > 0:
        try:
            sampler._generate_and_save(initial_epoch - 1)
            print(f"Refreshed GAN preview for epoch {initial_epoch}")
        except Exception as e:
            print(f"GAN preview refresh error: {e}")

    def _filter_finite(lst):
        return [v for v in lst if isinstance(v, (int, float)) and np.isfinite(v)]

    d_losses = _filter_finite(state.state.get("d_losses", []))
    g_losses = _filter_finite(state.state.get("g_losses", []))
    d_accs = _filter_finite(state.state.get("d_accs", []))
    g_accs = _filter_finite(state.state.get("g_accs", []))
    fid_scores = _filter_finite(state.state.get("fid_scores", []))
    fs_scores = _filter_finite(state.state.get("fs_scores", []))
    best_quality_raw = state.state.get("best_quality", float("inf"))
    best_quality = best_quality_raw if np.isfinite(best_quality_raw) else float("inf")
    gan_no_improve = int(state.state.get("gan_no_improve", 0))
    gan_early_stop_patience = int(os.getenv("GAN_EARLY_STOP_PATIENCE", "3"))
    gan_target_fid = float(os.getenv("GAN_TARGET_FID", "0") or 0)
    gan_target_fs = float(os.getenv("GAN_TARGET_FS", "0") or 0)
    gan_d_steps = max(1, int(os.getenv("GAN_D_STEPS", "1")))
    gan_g_steps = max(1, int(os.getenv("GAN_G_STEPS", "2")))
    gan_recovery_mode = os.getenv("GAN_RECOVERY_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}
    gan_diversity_weight = float(os.getenv("GAN_DIVERSITY_WEIGHT", "0.0") or 0.0)
    gan_class_guidance_weight = float(os.getenv("GAN_CLASS_GUIDANCE_WEIGHT", "0.0") or 0.0)
    gan_preview_freq = max(1, int(os.getenv("GAN_PREVIEW_FREQ", "5")))
    gan_shake_on_collapse = os.getenv("GAN_SHAKE_ON_COLLAPSE", "0").strip().lower() in {"1", "true", "yes", "on"}
    gan_shake_std = float(os.getenv("GAN_SHAKE_STD", "0.0005") or 0.0005)
    gan_grad_clip_norm = float(os.getenv("GAN_GRAD_CLIP_NORM", "5.0") or 5.0)
    if gan_recovery_mode:
        gan_g_steps = max(gan_g_steps, 4)
        if gan_diversity_weight <= 0.0:
            gan_diversity_weight = 0.03
        if gan_class_guidance_weight <= 0.0:
            gan_class_guidance_weight = 0.35
        gan_preview_freq = 1
        if not gan_shake_on_collapse:
            gan_shake_on_collapse = True

    original_policy = tf.keras.mixed_precision.global_policy()
    if original_policy.name != "float32":
        print(f"GAN: switching from {original_policy.name} to float32 (BCE + low VRAM = NaN risk)")
        tf.keras.mixed_precision.set_global_policy("float32")

    bce = tf.keras.losses.BinaryCrossentropy()

    # --- Create fresh optimizers with TTUR (Two-Timescale Update Rule) ---
    # D learns slower than G to prevent discriminator domination.
    d_lr_base = cfg.learning_rate * 0.5   # 1e-4 for D
    g_lr_base = cfg.learning_rate * 1.5   # 3e-4 for G
    if gan_recovery_mode:
        d_lr_base *= 0.5
        g_lr_base *= 1.25
    d_optimizer = tf.keras.optimizers.Adam(d_lr_base, beta_1=0.5, beta_2=0.999)
    g_optimizer = tf.keras.optimizers.Adam(g_lr_base, beta_1=0.5, beta_2=0.999)

    classifier_guidance_model = None
    if conditional and gan_class_guidance_weight > 0:
        classifier_path = os.path.join(WEIGHTS_DIR, "classifier_model.keras")
        if os.path.exists(classifier_path):
            try:
                classifier_guidance_model = tf.keras.models.load_model(classifier_path, compile=False)
                classifier_guidance_model.trainable = False
                print(f"GAN class-guidance enabled from: {classifier_path}")
            except Exception as e:
                print(f"GAN class-guidance disabled (load error): {e}")
                classifier_guidance_model = None
        else:
            print(f"GAN class-guidance disabled (missing classifier): {classifier_path}")

    @tf.function
    def train_d(real_images, real_labels=None, instance_noise_std=0.0):
        bs = tf.shape(real_images)[0]
        noise = tf.random.normal([bs, LATENT_DIM])
        with tf.GradientTape() as tape:
            if conditional and real_labels is not None:
                fake = generator([noise, real_labels], training=True)
                fake = tf.where(tf.math.is_finite(fake), fake, tf.zeros_like(fake))
                if instance_noise_std > 0:
                    real_noise_std = tf.cast(instance_noise_std, real_images.dtype)
                    fake_noise_std = tf.cast(instance_noise_std, fake.dtype)
                    real_images_noisy = tf.clip_by_value(
                        real_images + tf.random.normal(tf.shape(real_images), stddev=real_noise_std, dtype=real_images.dtype),
                        tf.cast(-1.0, real_images.dtype),
                        tf.cast(1.0, real_images.dtype),
                    )
                    fake_noisy = tf.clip_by_value(
                        fake + tf.random.normal(tf.shape(fake), stddev=fake_noise_std, dtype=fake.dtype),
                        tf.cast(-1.0, fake.dtype),
                        tf.cast(1.0, fake.dtype),
                    )
                else:
                    real_images_noisy = real_images
                    fake_noisy = fake
                real_out = discriminator([real_images_noisy, real_labels], training=True)
                fake_out = discriminator([fake_noisy, real_labels], training=True)
            else:
                fake = generator(noise, training=True)
                fake = tf.where(tf.math.is_finite(fake), fake, tf.zeros_like(fake))
                if instance_noise_std > 0:
                    real_noise_std = tf.cast(instance_noise_std, real_images.dtype)
                    fake_noise_std = tf.cast(instance_noise_std, fake.dtype)
                    real_images_noisy = tf.clip_by_value(
                        real_images + tf.random.normal(tf.shape(real_images), stddev=real_noise_std, dtype=real_images.dtype),
                        tf.cast(-1.0, real_images.dtype),
                        tf.cast(1.0, real_images.dtype),
                    )
                    fake_noisy = tf.clip_by_value(
                        fake + tf.random.normal(tf.shape(fake), stddev=fake_noise_std, dtype=fake.dtype),
                        tf.cast(-1.0, fake.dtype),
                        tf.cast(1.0, fake.dtype),
                    )
                else:
                    real_images_noisy = real_images
                    fake_noisy = fake
                real_out = discriminator(real_images_noisy, training=True)
                fake_out = discriminator(fake_noisy, training=True)

            real_out = tf.where(tf.math.is_finite(real_out), real_out, tf.zeros_like(real_out))
            fake_out = tf.where(tf.math.is_finite(fake_out), fake_out, tf.zeros_like(fake_out))

            real_targets = tf.random.uniform(tf.shape(real_out), minval=0.85, maxval=1.0)
            fake_targets = tf.random.uniform(tf.shape(fake_out), minval=0.0, maxval=0.15)
            d_loss = bce(real_targets, real_out) + bce(fake_targets, fake_out)
            d_loss = tf.where(tf.math.is_finite(d_loss), d_loss, tf.cast(1e6, d_loss.dtype))
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        grads = _sanitize_grads(grads, discriminator.trainable_variables)
        if gan_grad_clip_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, gan_grad_clip_norm)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
        return d_loss, real_out, fake_out

    @tf.function
    def train_g(batch_size, labels=None):
        noise = tf.random.normal([batch_size, LATENT_DIM])
        with tf.GradientTape() as tape:
            if conditional and labels is not None:
                fake = generator([noise, labels], training=True)
                fake = tf.where(tf.math.is_finite(fake), fake, tf.zeros_like(fake))
                fake_out = discriminator([fake, labels], training=True)
            else:
                fake = generator(noise, training=True)
                fake = tf.where(tf.math.is_finite(fake), fake, tf.zeros_like(fake))
                fake_out = discriminator(fake, training=True)
            fake_out = tf.where(tf.math.is_finite(fake_out), fake_out, tf.zeros_like(fake_out))
            adv_loss = bce(tf.ones_like(fake_out), fake_out)
            g_loss = adv_loss

            if conditional and labels is not None and classifier_guidance_model is not None and gan_class_guidance_weight > 0:
                fake_for_cls = tf.clip_by_value((fake + 1.0) / 2.0, 0.0, 1.0)
                fake_for_cls = tf.image.resize(fake_for_cls, IMG_CFG.classifier_size)
                cls_probs = classifier_guidance_model(fake_for_cls, training=False)
                cls_targets = tf.cast(labels, cls_probs.dtype)
                cls_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(cls_targets, cls_probs))
                g_loss = g_loss + tf.cast(gan_class_guidance_weight, g_loss.dtype) * tf.cast(cls_loss, g_loss.dtype)

            if gan_diversity_weight > 0:
                diversity_score = tf.reduce_mean(tf.math.reduce_std(fake, axis=0))
                g_loss = g_loss - tf.cast(gan_diversity_weight, g_loss.dtype) * tf.cast(diversity_score, g_loss.dtype)
            g_loss = tf.where(tf.math.is_finite(g_loss), g_loss, tf.cast(1e6, g_loss.dtype))
        grads = tape.gradient(g_loss, generator.trainable_variables)
        grads = _sanitize_grads(grads, generator.trainable_variables)
        if gan_grad_clip_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, gan_grad_clip_norm)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        return g_loss

    total_epochs = epochs or cfg.epochs
    if initial_epoch >= total_epochs:
        print("GAN already reached requested epochs")
        return generator, discriminator, loss_logger, fid_scores, fs_scores

    print(
        "GAN settings: "
        f"batch_size={cfg.batch_size}, "
        f"fid_eval_freq={fid_eval_freq}, "
        f"preview_freq={gan_preview_freq}, "
        f"d_steps={gan_d_steps}, g_steps={gan_g_steps}, "
        f"d_lr={d_lr_base:.2e}, "
        f"g_lr={g_lr_base:.2e}, "
        f"recovery_mode={gan_recovery_mode}, "
        f"class_guidance_w={gan_class_guidance_weight:.3f}, "
        f"diversity_w={gan_diversity_weight:.3f}, "
        f"shake_on_collapse={gan_shake_on_collapse}, "
        f"precision=float32"
    )
    print(f"Starting GAN training: epoch {initial_epoch} -> {total_epochs}")

    # Track consecutive collapse epochs for stronger rescue
    consecutive_collapse_epochs = 0
    d_frozen_this_epoch = False

    for epoch in range(initial_epoch, total_epochs):
        start = time.time()
        d_epoch = []
        g_epoch = []
        d_acc_epoch = []
        g_acc_epoch = []
        bad_batch_count = 0
        progress = epoch / max(total_epochs - 1, 1)
        base_instance_noise = 0.12 if gan_recovery_mode else 0.08
        instance_noise_std = max(0.015, base_instance_noise * (1.0 - progress))
        # Convert to tf.constant to avoid @tf.function retracing
        instance_noise_tf = tf.constant(instance_noise_std, dtype=tf.float32)

        for batch in dataset:
            if conditional:
                real_images, labels = batch
            else:
                real_images = batch
                labels = None
            bs = tf.shape(real_images)[0]

            # Skip D training if frozen due to severe collapse
            if not d_frozen_this_epoch:
                for _ in range(gan_d_steps):
                    d_loss, real_out, fake_out = train_d(real_images, labels, instance_noise_tf)
            else:
                # Still need d_loss/real_out/fake_out for logging
                noise_probe = tf.random.normal([bs, LATENT_DIM])
                if conditional and labels is not None:
                    fake_probe = generator([noise_probe, labels], training=False)
                    fake_probe = tf.where(tf.math.is_finite(fake_probe), fake_probe, tf.zeros_like(fake_probe))
                    real_out = discriminator([real_images, labels], training=False)
                    fake_out = discriminator([fake_probe, labels], training=False)
                else:
                    fake_probe = generator(noise_probe, training=False)
                    fake_probe = tf.where(tf.math.is_finite(fake_probe), fake_probe, tf.zeros_like(fake_probe))
                    real_out = discriminator(real_images, training=False)
                    fake_out = discriminator(fake_probe, training=False)
                d_loss = bce(tf.ones_like(real_out), real_out) + bce(tf.zeros_like(fake_out), fake_out)

            g_step_losses = []
            for _ in range(gan_g_steps):
                g_step_losses.append(train_g(bs, labels))
            g_loss = tf.add_n(g_step_losses) / float(len(g_step_losses))

            d_val = float(d_loss)
            g_val = float(g_loss)
            if not np.isfinite(d_val) or not np.isfinite(g_val):
                bad_batch_count += 1
                continue

            d_epoch.append(d_val)
            g_epoch.append(g_val)
            d_acc_epoch.append(float(tf.reduce_mean(tf.cast(real_out > 0.5, tf.float32))))
            g_acc_epoch.append(float(tf.reduce_mean(tf.cast(fake_out > 0.5, tf.float32))))

        if not d_epoch or not g_epoch:
            print(
                f"Epoch {epoch + 1}/{total_epochs} produced no finite GAN losses; "
                "stopping early to avoid corrupted previews/checkpoints."
            )
            break

        d_avg = float(np.mean(d_epoch))
        g_avg = float(np.mean(g_epoch))
        d_acc = float(np.mean(d_acc_epoch)) if d_acc_epoch else 0.0
        g_acc = float(np.mean(g_acc_epoch)) if g_acc_epoch else 0.0

        if bad_batch_count > 0:
            print(f"  Skipped {bad_batch_count} unstable batches with non-finite losses")

        if not np.isfinite(d_avg) or not np.isfinite(g_avg):
            print(
                f"Epoch {epoch + 1}/{total_epochs} produced non-finite mean losses "
                f"(D={d_avg}, G={g_avg}); stopping to preserve last good checkpoint."
            )
            break

        d_losses.append(d_avg)
        g_losses.append(g_avg)
        d_accs.append(d_acc)
        g_accs.append(g_acc)
        loss_logger.log_step(epoch, d_avg, g_avg, d_acc, g_acc)

        elapsed = time.time() - start
        extra_info = " [D frozen]" if d_frozen_this_epoch else ""
        print(f"Epoch {epoch + 1}/{total_epochs} [{elapsed:.1f}s] D:{d_avg:.4f} G:{g_avg:.4f} Dacc:{d_acc:.4f} Gacc:{g_acc:.4f}{extra_info}")

        # --- Stronger anti-collapse rescue ---
        d_frozen_this_epoch = False
        if d_acc > 0.95 and g_acc < 0.05:
            consecutive_collapse_epochs += 1
            print(
                f"  ⚠ Discriminator domination detected (streak: {consecutive_collapse_epochs})"
            )

            if consecutive_collapse_epochs >= 3:
                # Severe collapse: freeze D for next epoch and reset its final dense layer
                d_frozen_this_epoch = True
                print("  🔧 Severe collapse: freezing D for next epoch + resetting D final layer")
                for var in discriminator.trainable_variables:
                    if 'dense' in var.name.lower() and ('kernel' in var.name.lower() or 'weight' in var.name.lower()):
                        if var.shape[-1] == 1:
                            var.assign(tf.random.truncated_normal(var.shape, stddev=0.02, dtype=var.dtype))
                consecutive_collapse_epochs = 0

            # Always apply standard rescue measures
            gan_g_steps = min(6, gan_g_steps + 1)
            try:
                new_d_lr = max(1e-6, float(tf.keras.backend.get_value(d_optimizer.learning_rate)) * 0.5)
                d_optimizer.learning_rate = new_d_lr
            except Exception:
                pass
            if gan_shake_on_collapse and gan_shake_std > 0:
                for var in generator.trainable_variables:
                    var.assign_add(tf.random.normal(tf.shape(var), stddev=gan_shake_std, dtype=var.dtype))
            print(
                f"  Anti-collapse: g_steps={gan_g_steps}, "
                f"d_lr={float(tf.keras.backend.get_value(d_optimizer.learning_rate)):.2e}"
            )
        else:
            consecutive_collapse_epochs = 0

        generator_is_finite = _generator_is_finite(generator, conditional, NUM_CLASSES)
        if not generator_is_finite:
            print("Generator health probe failed (non-finite output); skipping preview/checkpoint for this epoch.")

        if generator_is_finite and ((epoch + 1) % 5 == 0 or epoch == total_epochs - 1):
            generator.save(state.generator_ckpt())
            discriminator.save(state.discriminator_ckpt())

        if generator_is_finite and ((epoch + 1) % gan_preview_freq == 0 or epoch == total_epochs - 1):
            sampler._generate_and_save(epoch)

        stop_gan_early = False
        if fid_eval_freq > 0 and (epoch + 1) % fid_eval_freq == 0:
            real_eval_paths = X_val_paths if len(X_val_paths) > 0 else images_for_train
            real_eval_labels = y_val_labels if len(X_val_paths) > 0 else y_train_labels
            n_eval = min(64 if LOW_VRAM_MODE else 256, len(real_eval_paths))
            if n_eval > 0 and generator_is_finite:
                # Batch generator inference to avoid OOM on low-VRAM GPUs
                gen_batch_size = 16 if LOW_VRAM_MODE else 64
                gen_chunks = []
                for i in range(0, n_eval, gen_batch_size):
                    chunk_size = min(gen_batch_size, n_eval - i)
                    z_chunk = tf.random.normal([chunk_size, LATENT_DIM])
                    if conditional:
                        eval_labels_chunk = tf.one_hot(real_eval_labels[i:i+chunk_size], NUM_CLASSES)
                        eval_labels_chunk = tf.cast(eval_labels_chunk, tf.float32)
                        gen_chunk = generator([z_chunk, eval_labels_chunk], training=False)
                    else:
                        gen_chunk = generator(z_chunk, training=False)
                    gen_chunk = tf.where(tf.math.is_finite(gen_chunk), gen_chunk, tf.zeros_like(gen_chunk))
                    gen_chunks.append(gen_chunk.numpy())
                gen = np.concatenate(gen_chunks, axis=0)
                gen = np.clip((gen + 1.0) / 2.0, 0.0, 1.0)
                real_eval = load_images_from_paths(real_eval_paths[:n_eval], img_size=gan_img_size)
                try:
                    fid = calculate_fid(real_eval, gen)
                    fs = calculate_fs(real_eval, gen)
                    fid_scores.append(float(fid))
                    fs_scores.append(float(fs))
                    print(f"FID: {fid:.2f} FS: {fs:.2f}")

                    quality = float(fid + fs)
                    if quality < best_quality - 1e-3:
                        best_quality = quality
                        gan_no_improve = 0
                    else:
                        gan_no_improve += 1

                    if gan_target_fid > 0 and gan_target_fs > 0 and fid <= gan_target_fid and fs <= gan_target_fs:
                        print(f"GAN quality target reached (FID<={gan_target_fid}, FS<={gan_target_fs}); stopping early.")
                        stop_gan_early = True
                    elif gan_early_stop_patience > 0 and gan_no_improve >= gan_early_stop_patience:
                        print(f"GAN quality plateau detected for {gan_no_improve} evaluation cycles; stopping early.")
                        stop_gan_early = True
                except Exception as e:
                    print(f"FID/FS computation error: {e}")

        collapse_detector.on_epoch_end(epoch)

        # --- Save state with NaN/Inf filtering ---
        state.state["last_epoch"] = epoch
        state.state["d_losses"] = _filter_finite(d_losses)
        state.state["g_losses"] = _filter_finite(g_losses)
        state.state["d_accs"] = _filter_finite(d_accs)
        state.state["g_accs"] = _filter_finite(g_accs)
        state.state["fid_scores"] = _filter_finite(fid_scores)
        state.state["fs_scores"] = _filter_finite(fs_scores)
        state.state["best_quality"] = best_quality if np.isfinite(best_quality) else 1e9
        state.state["gan_no_improve"] = gan_no_improve
        state.save()

        if stop_gan_early:
            break

    generator.save(os.path.join(WEIGHTS_DIR, f"generator_{gan_type}.keras"))
    discriminator.save(os.path.join(WEIGHTS_DIR, f"discriminator_{gan_type}.keras"))

    gan_log_dir = os.path.join(LOG_DIR, "gan")
    os.makedirs(gan_log_dir, exist_ok=True)
    plot_gan_losses(d_losses, g_losses, d_accs, g_accs, save_path=os.path.join(gan_log_dir, "loss_curves.png"))
    if fid_scores:
        plot_fid_fs_vs_epochs(fid_scores, fs_scores, save_path=os.path.join(gan_log_dir, "fid_fs_curves.png"))

    # Restore original mixed precision policy for other tracks
    if original_policy.name != "float32":
        tf.keras.mixed_precision.set_global_policy(original_policy)
        print(f"Restored mixed precision policy: {original_policy.name}")

    return generator, discriminator, loss_logger, fid_scores, fs_scores


def train_classifier_with_gan(generator, data_dir=None, gan_type="conditional", ratio=0.5, epochs=None, resume=True):
    print("\n" + "=" * 60)
    print("GAN AUGMENTED CLASSIFIER")
    print("=" * 60)

    images, labels = load_figshare_dataset(data_dir, img_size=IMG_CFG.classifier_size)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(images, labels)
    cfg = TRACK_CONFIGS["classifier"]

    baseline_state = TrainingState("classifier_baseline")
    if resume and baseline_state.checkpoint_path():
        baseline_model = tf.keras.models.load_model(baseline_state.checkpoint_path())
        baseline_start = baseline_state.start_epoch()
    else:
        baseline_model = build_classifier(num_classes=NUM_CLASSES, input_shape=(*IMG_CFG.classifier_size, 1))
        baseline_start = 0

    train_ds = build_classifier_dataset(X_train, y_train, batch_size=cfg.batch_size)
    val_ds = build_classifier_dataset(X_val, y_val, batch_size=cfg.batch_size, shuffle=False, augment=False)

    class BaselineSaver(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            baseline_model.save(os.path.join(CHECKPOINT_DIR, "classifier_baseline", "last_model.keras"))
            baseline_state.update_epoch(epoch)

    total_epochs = epochs or cfg.epochs
    if baseline_start < total_epochs:
        baseline_model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=baseline_start,
            epochs=total_epochs,
            callbacks=get_standard_callbacks(baseline_model, "classifier_baseline") + [BaselineSaver()],
        )

    y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)
    baseline_metrics = evaluate_classifier(baseline_model, X_test, y_test_oh, track_name="classifier_baseline")

    n_synth = int(len(X_train) * ratio)
    if gan_type == "conditional":
        per_class = max(1, n_synth // NUM_CLASSES)
        syn_imgs = []
        syn_lbls = []
        for c in range(NUM_CLASSES):
            zc = tf.random.normal([per_class, LATENT_DIM])
            lc = tf.one_hot(tf.constant([c] * per_class), NUM_CLASSES)
            gc = generator([zc, lc], training=False).numpy()
            gc = np.clip((gc + 1.0) / 2.0, 0.0, 1.0)
            gc = tf.image.resize(gc, IMG_CFG.classifier_size).numpy()
            syn_imgs.append(gc)
            syn_lbls.extend([c] * per_class)
        syn_imgs = np.concatenate(syn_imgs, axis=0)
        syn_lbls = np.array(syn_lbls, dtype=np.int32)
    else:
        z = tf.random.normal([n_synth, LATENT_DIM])
        gi = generator(z, training=False).numpy()
        syn_imgs = np.clip((gi + 1.0) / 2.0, 0.0, 1.0)
        syn_imgs = tf.image.resize(syn_imgs, IMG_CFG.classifier_size).numpy()
        syn_lbls = np.random.randint(0, NUM_CLASSES, size=len(syn_imgs))

    mixed_X, mixed_y = mix_real_synthetic(X_train, y_train, syn_imgs, syn_lbls, ratio=ratio)

    aug_state = TrainingState("classifier_augmented")
    if resume and aug_state.checkpoint_path():
        aug_model = tf.keras.models.load_model(aug_state.checkpoint_path())
        aug_start = aug_state.start_epoch()
    else:
        aug_model = build_classifier(num_classes=NUM_CLASSES, input_shape=(*IMG_CFG.classifier_size, 1))
        aug_start = 0

    train_aug_ds = build_classifier_dataset(mixed_X, mixed_y, batch_size=cfg.batch_size)

    class AugSaver(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            aug_model.save(os.path.join(CHECKPOINT_DIR, "classifier_augmented", "last_model.keras"))
            aug_state.update_epoch(epoch)

    if aug_start < total_epochs:
        aug_model.fit(
            train_aug_ds,
            validation_data=val_ds,
            initial_epoch=aug_start,
            epochs=total_epochs,
            callbacks=get_standard_callbacks(aug_model, "classifier_augmented") + [AugSaver()],
        )

    aug_metrics = evaluate_classifier(aug_model, X_test, y_test_oh, track_name="classifier_augmented")
    print(f"Baseline acc: {baseline_metrics['accuracy']:.4f}")
    print(f"Augmented acc: {aug_metrics['accuracy']:.4f}")
    print(f"Improvement: {aug_metrics['accuracy'] - baseline_metrics['accuracy']:.4f}")

    aug_model.save(os.path.join(WEIGHTS_DIR, "classifier_augmented.keras"))
    baseline_model.save(os.path.join(WEIGHTS_DIR, "classifier_baseline.keras"))

    return aug_model, aug_metrics, baseline_metrics


def train_gan_v2(data_dir=None, epochs=None, fid_eval_freq=10, resume=True):
    """V2 GAN training: ResNet generator + Projection discriminator + WGAN-GP.

    This is a research-grade conditional GAN with:
    - ResNet blocks with conditional batch norm and self-attention
    - Projection discriminator with spectral normalization
    - WGAN-GP loss (Wasserstein + gradient penalty)
    - EMA generator for smoother previews
    - 5:1 discriminator:generator step ratio
    """
    print("\n" + "=" * 60)
    print("TRACK 4 - GAN V2 (WGAN-GP + ResNet + Projection)")
    print("=" * 60)

    (X_train_paths, y_train_labels), (X_val_paths, y_val_labels), _ = get_figshare_train_val_test_split(data_dir)
    cfg = TRACK_CONFIGS["gan"]
    fid_eval_freq = int(os.getenv("GAN_FID_EVAL_FREQ", str(fid_eval_freq)))
    fid_eval_freq = max(0, fid_eval_freq)
    img_shape = (*IMG_CFG.gan_size, 1)

    # Build v2 models
    generator = build_v2_generator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES, output_shape=img_shape)
    discriminator = build_v2_discriminator(input_shape=img_shape, num_classes=NUM_CLASSES)

    state = GANState("v2")
    initial_epoch = 0

    if not resume:
        state.state = GANState.fresh_state()

    if resume and state.has_ckpt():
        print("Loading GAN v2 checkpoints...")
        try:
            generator.load_weights(state.generator_ckpt())
            discriminator.load_weights(state.discriminator_ckpt())
            initial_epoch = state.start_epoch()
            print(f"Resumed from epoch {initial_epoch}")
        except Exception as e:
            print(f"Could not load v2 checkpoints: {e}")
            print("Starting fresh training")
            initial_epoch = 0
            state.state = GANState.fresh_state()

    gan_img_size = IMG_CFG.gan_size
    dataset = build_gan_dataset_from_paths(
        X_train_paths,
        labels=y_train_labels,
        img_size=gan_img_size,
        batch_size=cfg.batch_size,
    )

    # EMA for generator
    ema = EMAGenerator(generator, decay=0.999)

    loss_logger = GANLossLogger()
    sampler = GANImageSampler(generator, latent_dim=LATENT_DIM, conditional=True, num_classes=NUM_CLASSES)
    collapse_detector = ModelCollapseDetector(
        generator, latent_dim=LATENT_DIM, conditional=True, num_classes=NUM_CLASSES,
    )

    def _filter_finite(lst):
        return [v for v in lst if isinstance(v, (int, float)) and np.isfinite(v)]

    d_losses = _filter_finite(state.state.get("d_losses", []))
    g_losses = _filter_finite(state.state.get("g_losses", []))
    d_accs = _filter_finite(state.state.get("d_accs", []))
    g_accs = _filter_finite(state.state.get("g_accs", []))
    fid_scores = _filter_finite(state.state.get("fid_scores", []))
    fs_scores = _filter_finite(state.state.get("fs_scores", []))
    best_quality_raw = state.state.get("best_quality", float("inf"))
    best_quality = best_quality_raw if np.isfinite(best_quality_raw) else float("inf")
    gan_no_improve = int(state.state.get("gan_no_improve", 0))
    gan_early_stop_patience = int(os.getenv("GAN_EARLY_STOP_PATIENCE", "5"))
    gan_d_steps = max(1, int(os.getenv("GAN_D_STEPS", "5")))
    gan_g_steps = max(1, int(os.getenv("GAN_G_STEPS", "1")))
    gan_preview_freq = max(1, int(os.getenv("GAN_PREVIEW_FREQ", "5")))
    gan_grad_clip_norm = float(os.getenv("GAN_GRAD_CLIP_NORM", "0") or 0)
    lambda_gp = float(os.getenv("GAN_LAMBDA_GP", "10.0") or 10.0)

    # Ensure float32 for WGAN-GP
    original_policy = tf.keras.mixed_precision.global_policy()
    if original_policy.name != "float32":
        print(f"GAN v2: switching from {original_policy.name} to float32")
        tf.keras.mixed_precision.set_global_policy("float32")

    # WGAN-GP optimizers: beta1=0, beta2=0.9 (standard)
    d_lr = float(os.getenv("GAN_D_LR", "1e-4") or 1e-4)
    g_lr = float(os.getenv("GAN_G_LR", "1e-4") or 1e-4)
    d_optimizer = tf.keras.optimizers.Adam(d_lr, beta_1=0.0, beta_2=0.9)
    g_optimizer = tf.keras.optimizers.Adam(g_lr, beta_1=0.0, beta_2=0.9)

    @tf.function
    def train_d_step(real_images, real_labels):
        bs = tf.shape(real_images)[0]
        noise = tf.random.normal([bs, LATENT_DIM])
        with tf.GradientTape() as tape:
            fake = generator([noise, real_labels], training=True)
            real_out = discriminator([real_images, real_labels], training=True)
            fake_out = discriminator([fake, real_labels], training=True)

            # WGAN losses
            d_loss_real = -tf.reduce_mean(real_out)
            d_loss_fake = tf.reduce_mean(fake_out)

            # Gradient penalty
            gp = gradient_penalty(discriminator, real_images, fake, real_labels, lambda_gp=lambda_gp)

            d_loss = d_loss_real + d_loss_fake + gp

        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        grads = _sanitize_grads(grads, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        # Wasserstein distance estimate (for logging)
        w_dist = -(d_loss_real + d_loss_fake)
        return d_loss, w_dist, real_out, fake_out

    @tf.function
    def train_g_step(batch_size, labels):
        noise = tf.random.normal([batch_size, LATENT_DIM])
        with tf.GradientTape() as tape:
            fake = generator([noise, labels], training=True)
            fake_out = discriminator([fake, labels], training=True)
            g_loss = -tf.reduce_mean(fake_out)  # Generator wants high scores
        grads = tape.gradient(g_loss, generator.trainable_variables)
        grads = _sanitize_grads(grads, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        return g_loss

    total_epochs = epochs or int(os.getenv("GAN_EPOCHS", "300"))
    if initial_epoch >= total_epochs:
        print("GAN v2 already reached requested epochs")
        return generator, discriminator, loss_logger, fid_scores, fs_scores

    print(
        "GAN v2 settings: "
        f"batch_size={cfg.batch_size}, "
        f"fid_eval_freq={fid_eval_freq}, "
        f"preview_freq={gan_preview_freq}, "
        f"d_steps={gan_d_steps}, g_steps={gan_g_steps}, "
        f"d_lr={d_lr:.2e}, g_lr={g_lr:.2e}, "
        f"lambda_gp={lambda_gp}, "
        f"precision=float32"
    )
    print(f"Starting GAN v2 training: epoch {initial_epoch} -> {total_epochs}")

    for epoch in range(initial_epoch, total_epochs):
        start = time.time()
        d_epoch = []
        g_epoch = []
        w_dist_epoch = []
        bad_batch_count = 0

        for batch in dataset:
            real_images, labels = batch
            bs = tf.shape(real_images)[0]

            # Train discriminator (5 steps per G step for WGAN-GP)
            for _ in range(gan_d_steps):
                d_loss, w_dist, real_out, fake_out = train_d_step(real_images, labels)

            d_val = float(d_loss)
            w_val = float(w_dist)
            if not np.isfinite(d_val):
                bad_batch_count += 1
                continue

            # Train generator
            for _ in range(gan_g_steps):
                g_loss = train_g_step(bs, labels)
                ema.update()

            g_val = float(g_loss)
            if not np.isfinite(g_val):
                bad_batch_count += 1
                continue

            d_epoch.append(d_val)
            g_epoch.append(g_val)
            w_dist_epoch.append(w_val)

        if not d_epoch or not g_epoch:
            print(
                f"Epoch {epoch + 1}/{total_epochs} produced no finite losses; "
                "stopping early."
            )
            break

        d_avg = float(np.mean(d_epoch))
        g_avg = float(np.mean(g_epoch))
        w_avg = float(np.mean(w_dist_epoch))

        if bad_batch_count > 0:
            print(f"  Skipped {bad_batch_count} unstable batches")

        if not np.isfinite(d_avg) or not np.isfinite(g_avg):
            print(f"Epoch {epoch + 1}/{total_epochs} non-finite losses; stopping.")
            break

        d_losses.append(d_avg)
        g_losses.append(g_avg)
        d_accs.append(w_avg)  # Store W-distance in d_accs slot for logging
        g_accs.append(0.0)
        loss_logger.log_step(epoch, d_avg, g_avg, w_avg, 0.0)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch + 1}/{total_epochs} [{elapsed:.1f}s] "
            f"D:{d_avg:.4f} G:{g_avg:.4f} W-dist:{w_avg:.4f}"
        )

        # Check generator health
        generator_healthy = _generator_is_finite(generator, True, NUM_CLASSES)
        if not generator_healthy:
            print("  Generator health probe failed; skipping checkpoint/preview.")

        # Save checkpoints every 5 epochs
        if generator_healthy and ((epoch + 1) % 5 == 0 or epoch == total_epochs - 1):
            generator.save_weights(state.generator_ckpt())
            discriminator.save_weights(state.discriminator_ckpt())

        # Preview with EMA generator
        if generator_healthy and ((epoch + 1) % gan_preview_freq == 0 or epoch == total_epochs - 1):
            ema.apply()
            sampler._generate_and_save(epoch)
            ema.restore()

        # FID evaluation
        stop_early = False
        if fid_eval_freq > 0 and (epoch + 1) % fid_eval_freq == 0:
            real_eval_paths = X_val_paths if len(X_val_paths) > 0 else X_train_paths
            real_eval_labels = y_val_labels if len(X_val_paths) > 0 else y_train_labels
            n_eval = min(64 if LOW_VRAM_MODE else 256, len(real_eval_paths))
            if n_eval > 0 and generator_healthy:
                ema.apply()
                gen_batch_size = 16 if LOW_VRAM_MODE else 64
                gen_chunks = []
                for i in range(0, n_eval, gen_batch_size):
                    chunk_size = min(gen_batch_size, n_eval - i)
                    z_chunk = tf.random.normal([chunk_size, LATENT_DIM])
                    eval_labels_chunk = tf.one_hot(real_eval_labels[i:i+chunk_size], NUM_CLASSES)
                    eval_labels_chunk = tf.cast(eval_labels_chunk, tf.float32)
                    gen_chunk = generator([z_chunk, eval_labels_chunk], training=False)
                    gen_chunk = tf.where(tf.math.is_finite(gen_chunk), gen_chunk, tf.zeros_like(gen_chunk))
                    gen_chunks.append(gen_chunk.numpy())
                ema.restore()
                gen = np.concatenate(gen_chunks, axis=0)
                gen = np.clip((gen + 1.0) / 2.0, 0.0, 1.0)
                real_eval = load_images_from_paths(real_eval_paths[:n_eval], img_size=gan_img_size)
                try:
                    fid = calculate_fid(real_eval, gen)
                    fs = calculate_fs(real_eval, gen)
                    fid_scores.append(float(fid))
                    fs_scores.append(float(fs))
                    print(f"  FID: {fid:.2f} FS: {fs:.2f}")

                    quality = float(fid + fs)
                    if quality < best_quality - 1e-3:
                        best_quality = quality
                        gan_no_improve = 0
                    else:
                        gan_no_improve += 1

                    if gan_early_stop_patience > 0 and gan_no_improve >= gan_early_stop_patience:
                        print(f"  GAN quality plateau for {gan_no_improve} cycles; stopping.")
                        stop_early = True
                except Exception as e:
                    print(f"  FID/FS error: {e}")

        collapse_detector.on_epoch_end(epoch)

        # Save state
        state.state["last_epoch"] = epoch
        state.state["d_losses"] = _filter_finite(d_losses)
        state.state["g_losses"] = _filter_finite(g_losses)
        state.state["d_accs"] = _filter_finite(d_accs)
        state.state["g_accs"] = _filter_finite(g_accs)
        state.state["fid_scores"] = _filter_finite(fid_scores)
        state.state["fs_scores"] = _filter_finite(fs_scores)
        state.state["best_quality"] = best_quality if np.isfinite(best_quality) else 1e9
        state.state["gan_no_improve"] = gan_no_improve
        state.save()

        if stop_early:
            break

    # Save final weights
    ema.apply()
    generator.save_weights(os.path.join(WEIGHTS_DIR, "generator_v2.weights.h5"))
    ema.restore()
    discriminator.save_weights(os.path.join(WEIGHTS_DIR, "discriminator_v2.weights.h5"))

    gan_log_dir = os.path.join(LOG_DIR, "gan")
    os.makedirs(gan_log_dir, exist_ok=True)
    plot_gan_losses(d_losses, g_losses, d_accs, g_accs, save_path=os.path.join(gan_log_dir, "v2_loss_curves.png"))
    if fid_scores:
        plot_fid_fs_vs_epochs(fid_scores, fs_scores, save_path=os.path.join(gan_log_dir, "v2_fid_fs_curves.png"))

    if original_policy.name != "float32":
        tf.keras.mixed_precision.set_global_policy(original_policy)

    return generator, discriminator, loss_logger, fid_scores, fs_scores


def main():
    ensure_datasets(download_figshare=True, download_brats=True)

    figshare_dir = os.path.join(RAW_DIR, "figshare")
    brats_dir = os.path.join(RAW_DIR, "brats")

    train_detection(data_dir=figshare_dir, resume=True)
    train_classifier(data_dir=figshare_dir, resume=True)
    generator, _, _, _, _ = train_gan(data_dir=figshare_dir, gan_type="conditional", resume=True)
    if LOW_VRAM_MODE:
        print("Skipping GAN-augmented classifier stage in low-VRAM mode")
    else:
        train_classifier_with_gan(generator, data_dir=figshare_dir, gan_type="conditional", resume=True)

    if os.path.exists(brats_dir) and len(os.listdir(brats_dir)) > 0:
        train_segmentation(data_dir=brats_dir, resume=True)
    else:
        print("BraTS not available, skipping segmentation track")

    print("All requested training stages completed")


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    configure_gpu()

    if args.download_figshare or args.download_brats:
        ensure_datasets(download_figshare=args.download_figshare, download_brats=args.download_brats)

    if args.only_download:
        if args.download_figshare or args.download_brats:
            dl_figshare = args.download_figshare
            dl_brats = args.download_brats
        else:
            dl_figshare = True
            dl_brats = True
        ensure_datasets(
            download_figshare=dl_figshare,
            download_brats=dl_brats,
        )
        print("Dataset download stage completed")
        sys.exit(0)

    resume = not args.no_resume

    if args.track == "all":
        main()
    elif args.track == "detection":
        ddir = args.data_dir or os.path.join(RAW_DIR, "figshare")
        ensure_datasets(download_figshare=True, download_brats=False)
        train_detection(data_dir=ddir, epochs=args.epochs, resume=resume)
    elif args.track == "segmentation":
        ddir = args.data_dir or os.path.join(RAW_DIR, "brats")
        ensure_datasets(download_figshare=False, download_brats=True)
        train_segmentation(data_dir=ddir, epochs=args.epochs, resume=resume)
    elif args.track == "classifier":
        ddir = args.data_dir or os.path.join(RAW_DIR, "figshare")
        ensure_datasets(download_figshare=True, download_brats=False)
        train_classifier(data_dir=ddir, epochs=args.epochs, resume=resume)
    elif args.track == "gan":
        ddir = args.data_dir or os.path.join(RAW_DIR, "figshare")
        ensure_datasets(download_figshare=True, download_brats=False)
        train_gan(data_dir=ddir, gan_type=args.gan_type, epochs=args.epochs, resume=resume)
    elif args.track == "gan_v2":
        ddir = args.data_dir or os.path.join(RAW_DIR, "figshare")
        ensure_datasets(download_figshare=True, download_brats=False)
        train_gan_v2(data_dir=ddir, epochs=args.epochs, resume=resume)
    elif args.track == "gan_augmented":
        ddir = args.data_dir or os.path.join(RAW_DIR, "figshare")
        ensure_datasets(download_figshare=True, download_brats=False)
        if LOW_VRAM_MODE:
            print("GAN-augmented classifier training is disabled in low-VRAM mode")
            sys.exit(0)
        generator, _, _, _, _ = train_gan(data_dir=ddir, gan_type=args.gan_type, epochs=args.epochs, resume=resume)
        train_classifier_with_gan(generator, data_dir=ddir, gan_type=args.gan_type, epochs=args.epochs, resume=resume)
