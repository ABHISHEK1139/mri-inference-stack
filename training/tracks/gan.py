from __future__ import annotations

import logging
import os
import time

import numpy as np
import tensorflow as tf

from config import (
    LATENT_DIM,
    LOG_DIR,
    LOW_VRAM_MODE,
    NUM_CLASSES,
    TRACK_CONFIGS,
    WEIGHTS_DIR,
)
from data.dataset import (
    build_gan_dataset_from_paths,
    get_figshare_train_val_test_split,
    load_images_from_paths,
)
from evaluation.metrics import calculate_fid, calculate_fs, plot_fid_fs_vs_epochs, plot_gan_losses
from models.gan import (
    build_baseline_discriminator,
    build_baseline_generator,
    build_conditional_discriminator,
    build_conditional_generator,
    build_discriminator,
    build_generator,
    build_stylegan_generator,
)
from training.callbacks import (
    GANImageSampler,
    GANLossLogger,
    ModelCollapseDetector,
)
from training.runtime import (
    IMG_CFG,
    _generator_is_finite,
    _sanitize_grads,
    float32_precision,
    set_seed,
)
from training.state import GANState

logger = logging.getLogger(__name__)



def train_gan(data_dir=None, gan_type="conditional", epochs=None, fid_eval_freq=10, resume=True,
        seed: int | None = None):
    """Train a GAN under a float32 precision policy that is always restored."""
    with float32_precision():
        return _train_gan_impl(
            data_dir=data_dir,
            gan_type=gan_type,
            epochs=epochs,
            fid_eval_freq=fid_eval_freq,
            resume=resume,
            seed=seed,
        )


def _train_gan_impl(data_dir=None, gan_type="conditional", epochs=None, fid_eval_freq=10,
    seed: int | None = None,
    resume=True):
    seed = set_seed(seed)
    logger.info("Track gan: seed=%d", seed)
    logger.info('\n' + '=' * 60)
    logger.info(f'TRACK 4 - GAN ({gan_type.upper()})')
    logger.info('=' * 60)

    (X_train_paths, y_train_labels), (X_val_paths,
        y_val_labels), _ = get_figshare_train_val_test_split(data_dir)
    cfg = TRACK_CONFIGS["gan"]
    fid_eval_freq = int(os.getenv("GAN_FID_EVAL_FREQ", str(fid_eval_freq)))
    fid_eval_freq = max(0, fid_eval_freq)
    img_shape = (*IMG_CFG.gan_size, 1)

    # Note: the combined `build_gan` / `build_conditional_gan` wrapper model is
    # deliberately not used here. It flips `discriminator.trainable` on and off
    # and compiles a joint graph, while this loop drives the two networks with
    # explicit TTUR `train_d` / `train_g` steps instead.
    conditional = False
    if gan_type == "baseline":
        generator = build_baseline_generator(latent_dim=LATENT_DIM)
        discriminator = build_baseline_discriminator(input_shape=(64, 64, 1))
        images_for_train = X_train_paths
    elif gan_type == "dcgan":
        generator = build_generator(latent_dim=LATENT_DIM,
            output_shape=img_shape)
        discriminator = build_discriminator(input_shape=img_shape)
        images_for_train = X_train_paths
    elif gan_type == "conditional":
        generator = build_conditional_generator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES,
            output_shape=img_shape)
        discriminator = build_conditional_discriminator(input_shape=img_shape,
            num_classes=NUM_CLASSES)
        images_for_train = X_train_paths
        conditional = True
    elif gan_type == "stylegan":
        generator = build_stylegan_generator(latent_dim=LATENT_DIM, output_shape=img_shape)
        discriminator = build_discriminator(input_shape=img_shape)
        images_for_train = X_train_paths
    else:
        raise ValueError(f"Unknown GAN type: {gan_type}")

    state = GANState(gan_type, seed=seed)
    initial_epoch = 0

    if not resume:
        state.state = GANState.fresh_state()

    if resume and state.has_ckpt():
        logger.info('Loading GAN checkpoints...')
        generator = tf.keras.models.load_model(state.generator_ckpt())
        discriminator = tf.keras.models.load_model(state.discriminator_ckpt())
        initial_epoch = state.start_epoch()

    gan_img_size = (64, 64) if gan_type == "baseline" else IMG_CFG.gan_size
    dataset = build_gan_dataset_from_paths(
        images_for_train,
        labels=y_train_labels if conditional else None,
        img_size=gan_img_size,
        batch_size=cfg.batch_size,
        seed=seed,
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
            logger.info(f'Refreshed GAN preview for epoch {initial_epoch}')
        except Exception as e:
            logger.info(f'GAN preview refresh error: {e}')

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
    gan_d_steps = max(1, int(os.getenv("GAN_D_STEPS",
        "1")))
    gan_g_steps = max(1, int(os.getenv("GAN_G_STEPS",
        "2")))
    # Restore the anti-collapse step/LR tuning that was in effect when the run
    # was interrupted; it is part of the optimiser state, not a fresh default.
    saved_g_steps = state.state.get("g_steps")
    if resume and saved_g_steps:
        gan_g_steps = max(1, int(saved_g_steps))
    gan_recovery_mode = os.getenv("GAN_RECOVERY_MODE", "0").strip().lower() in {"1", "true", "yes",
        "on"}
    gan_diversity_weight = float(os.getenv("GAN_DIVERSITY_WEIGHT",
        "0.0") or 0.0)
    gan_class_guidance_weight = float(os.getenv("GAN_CLASS_GUIDANCE_WEIGHT", "0.0") or 0.0)
    gan_preview_freq = max(1, int(os.getenv("GAN_PREVIEW_FREQ",
        "5")))
    gan_shake_on_collapse = os.getenv("GAN_SHAKE_ON_COLLAPSE", "0").strip().lower() in {"1", "true",
        "yes", "on"}
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

    bce = tf.keras.losses.BinaryCrossentropy()

    # --- Create fresh optimizers with TTUR (Two-Timescale Update Rule) ---
    # D learns slower than G to prevent discriminator domination.
    d_lr_base = cfg.learning_rate * 0.5   # 1e-4 for D
    g_lr_base = cfg.learning_rate * 1.5   # 3e-4 for G
    if gan_recovery_mode:
        d_lr_base *= 0.5
        g_lr_base *= 1.25
    d_optimizer = tf.keras.optimizers.Adam(d_lr_base, beta_1=0.5,
        beta_2=0.999)
    g_optimizer = tf.keras.optimizers.Adam(g_lr_base, beta_1=0.5,
        beta_2=0.999)

    classifier_guidance_model = None
    if conditional and gan_class_guidance_weight > 0:
        classifier_path = os.path.join(WEIGHTS_DIR, "classifier_model.keras")
        if os.path.exists(classifier_path):
            try:
                classifier_guidance_model = tf.keras.models.load_model(classifier_path,
                    compile=False)
                classifier_guidance_model.trainable = False
                logger.info(f'GAN class-guidance enabled from: {classifier_path}')
            except Exception as e:
                logger.warning(f'GAN class-guidance disabled (load error): {e}')
                classifier_guidance_model = None
        else:
            logger.warning(f'GAN class-guidance disabled (missing classifier): {classifier_path}')

    @tf.function
    def train_d(real_images, real_labels=None, instance_noise_std=0.0):
        bs = tf.shape(real_images)[0]
        noise = tf.random.normal([bs,
            LATENT_DIM])
        with tf.GradientTape() as tape:
            if conditional and real_labels is not None:
                fake = generator([noise, real_labels], training=True)
                fake = tf.where(tf.math.is_finite(fake), fake,
                    tf.zeros_like(fake))
                if instance_noise_std > 0:
                    real_noise_std = tf.cast(instance_noise_std,
                        real_images.dtype)
                    fake_noise_std = tf.cast(instance_noise_std, fake.dtype)
                    real_images_noisy = tf.clip_by_value(
                        real_images + tf.random.normal(tf.shape(real_images), stddev=real_noise_std,
                            dtype=real_images.dtype),
                        tf.cast(-1.0,
                            real_images.dtype),
                        tf.cast(1.0, real_images.dtype),
                    )
                    fake_noisy = tf.clip_by_value(
                        fake + tf.random.normal(tf.shape(fake), stddev=fake_noise_std,
                            dtype=fake.dtype),
                        tf.cast(-1.0, fake.dtype),
                        tf.cast(1.0, fake.dtype),
                    )
                else:
                    real_images_noisy = real_images
                    fake_noisy = fake
                real_out = discriminator([real_images_noisy, real_labels], training=True)
                fake_out = discriminator([fake_noisy, real_labels],
                    training=True)
            else:
                fake = generator(noise,
                    training=True)
                fake = tf.where(tf.math.is_finite(fake),
                    fake,
                    tf.zeros_like(fake))
                if instance_noise_std > 0:
                    real_noise_std = tf.cast(instance_noise_std, real_images.dtype)
                    fake_noise_std = tf.cast(instance_noise_std,
                        fake.dtype)
                    real_images_noisy = tf.clip_by_value(
                        real_images + tf.random.normal(tf.shape(real_images), stddev=real_noise_std,
                            dtype=real_images.dtype),
                        tf.cast(-1.0, real_images.dtype),
                        tf.cast(1.0, real_images.dtype),
                    )
                    fake_noisy = tf.clip_by_value(
                        fake + tf.random.normal(tf.shape(fake), stddev=fake_noise_std,
                            dtype=fake.dtype),
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
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables, strict=True))
        return d_loss, real_out, fake_out

    @tf.function
    def train_g(batch_size, labels=None):
        noise = tf.random.normal([batch_size, LATENT_DIM])
        with tf.GradientTape() as tape:
            if conditional and labels is not None:
                fake = generator([noise, labels], training=True)
                fake = tf.where(tf.math.is_finite(fake), fake, tf.zeros_like(fake))
                fake_out = discriminator([fake, labels],
                    training=True)
            else:
                fake = generator(noise,
                    training=True)
                fake = tf.where(tf.math.is_finite(fake),
                    fake,
                    tf.zeros_like(fake))
                fake_out = discriminator(fake,
                    training=True)
            fake_out = tf.where(tf.math.is_finite(fake_out), fake_out, tf.zeros_like(fake_out))
            adv_loss = bce(tf.ones_like(fake_out),
                fake_out)
            g_loss = adv_loss

            if (
                conditional
                and labels is not None
                and classifier_guidance_model is not None
                and gan_class_guidance_weight > 0
            ):
                fake_for_cls = tf.clip_by_value((fake + 1.0) / 2.0, 0.0,
                    1.0)
                fake_for_cls = tf.image.resize(fake_for_cls, IMG_CFG.classifier_size)
                cls_probs = classifier_guidance_model(fake_for_cls,
                    training=False)
                cls_targets = tf.cast(labels,
                    cls_probs.dtype)
                cls_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(cls_targets,
                    cls_probs))
                g_loss = g_loss + tf.cast(gan_class_guidance_weight,
                    g_loss.dtype) * tf.cast(cls_loss, g_loss.dtype)

            if gan_diversity_weight > 0:
                diversity_score = tf.reduce_mean(tf.math.reduce_std(fake,
                    axis=0))
                g_loss = g_loss - tf.cast(gan_diversity_weight,
                    g_loss.dtype) * tf.cast(diversity_score, g_loss.dtype)
            g_loss = tf.where(tf.math.is_finite(g_loss), g_loss, tf.cast(1e6, g_loss.dtype))
        grads = tape.gradient(g_loss, generator.trainable_variables)
        grads = _sanitize_grads(grads, generator.trainable_variables)
        if gan_grad_clip_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, gan_grad_clip_norm)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables, strict=True))
        return g_loss

    total_epochs = epochs or cfg.epochs
    if initial_epoch >= total_epochs:
        logger.info('GAN already reached requested epochs')
        return generator, discriminator, loss_logger, fid_scores, fs_scores

    logger.info(
        "GAN settings: batch_size=%d fid_eval_freq=%d preview_freq=%d "
        "d_steps=%d g_steps=%d",
        cfg.batch_size, fid_eval_freq, gan_preview_freq, gan_d_steps, gan_g_steps,
    )
    logger.info(
        "  d_lr=%.2e g_lr=%.2e recovery_mode=%s class_guidance_w=%.3f "
        "diversity_w=%.3f shake_on_collapse=%s precision=float32",
        d_lr_base, g_lr_base, gan_recovery_mode, gan_class_guidance_weight,
        gan_diversity_weight, gan_shake_on_collapse,
    )
    logger.info(f'Starting GAN training: epoch {initial_epoch} -> {total_epochs}')

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
                    d_loss, real_out, fake_out = train_d(real_images, labels,
                        instance_noise_tf)
            else:
                # Still need d_loss/real_out/fake_out for logging
                noise_probe = tf.random.normal([bs,
                    LATENT_DIM])
                if conditional and labels is not None:
                    fake_probe = generator([noise_probe,
                        labels],
                        training=False)
                    fake_probe = tf.where(tf.math.is_finite(fake_probe), fake_probe,
                        tf.zeros_like(fake_probe))
                    real_out = discriminator([real_images, labels], training=False)
                    fake_out = discriminator([fake_probe, labels],
                        training=False)
                else:
                    fake_probe = generator(noise_probe,
                        training=False)
                    fake_probe = tf.where(tf.math.is_finite(fake_probe), fake_probe,
                        tf.zeros_like(fake_probe))
                    real_out = discriminator(real_images,
                        training=False)
                    fake_out = discriminator(fake_probe,
                        training=False)
                d_loss = bce(tf.ones_like(real_out), real_out) + bce(tf.zeros_like(fake_out),
                    fake_out)

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
            logger.warning(
                f"'Epoch {epoch + 1}/{total_epochs} produced no finite GAN los"
                f"ses; stopping early to avoid corrupted previews/checkpoints."
            )
            break

        d_avg = float(np.mean(d_epoch))
        g_avg = float(np.mean(g_epoch))
        d_acc = float(np.mean(d_acc_epoch)) if d_acc_epoch else 0.0
        g_acc = float(np.mean(g_acc_epoch)) if g_acc_epoch else 0.0

        if bad_batch_count > 0:
            logger.warning(f'  Skipped {bad_batch_count} unstable batches with non-finite losses')

        if not np.isfinite(d_avg) or not np.isfinite(g_avg):
            logger.warning(
                f"'Epoch {epoch + 1}/{total_epochs} produced non-finite mean losses"
                f"(D={d_avg}, G={g_avg}); stopping to preserve last good checkpoint."
            )
            break

        d_losses.append(d_avg)
        g_losses.append(g_avg)
        d_accs.append(d_acc)
        g_accs.append(g_acc)
        loss_logger.log_step(epoch, d_avg, g_avg, d_acc,
            g_acc)

        elapsed = time.time() - start
        extra_info = " [D frozen]" if d_frozen_this_epoch else ""
        logger.info(
            "Epoch %d/%d [%.1fs] D:%.4f G:%.4f Dacc:%.4f Gacc:%.4f%s",
            epoch + 1, total_epochs, elapsed, d_avg, g_avg, d_acc, g_acc, extra_info,
        )

        # --- Stronger anti-collapse rescue ---
        d_frozen_this_epoch = False
        if d_acc > 0.95 and g_acc < 0.05:
            consecutive_collapse_epochs += 1
            logger.info(
                f"'  [!] Discriminator domination detected"
                f"(streak: {consecutive_collapse_epochs})"
            )

            if consecutive_collapse_epochs >= 3:
                # Severe collapse: freeze D for next epoch and reset its final dense layer
                d_frozen_this_epoch = True
                logger.info(
                    "  [*] Severe collapse: freezing D for"
                    "next epoch + resetting D final layer"
                )
                for var in discriminator.trainable_variables:
                    name = var.name.lower()
                    if "dense" in name and ("kernel" in name or "weight" in name):
                        if var.shape[-1] == 1:
                            var.assign(tf.random.truncated_normal(var.shape, stddev=0.02,
                                dtype=var.dtype))
                consecutive_collapse_epochs = 0

            # Always apply standard rescue measures
            gan_g_steps = min(6,
                gan_g_steps + 1)
            try:
                new_d_lr = max(1e-6,
                    float(tf.keras.backend.get_value(d_optimizer.learning_rate)) * 0.5)
                d_optimizer.learning_rate = new_d_lr
            except Exception:
                pass
            if gan_shake_on_collapse and gan_shake_std > 0:
                for var in generator.trainable_variables:
                    var.assign_add(tf.random.normal(tf.shape(var), stddev=gan_shake_std,
                        dtype=var.dtype))
            logger.info(
                "  Anti-collapse: g_steps=%d, d_lr=%.2e",
                gan_g_steps, float(tf.keras.backend.get_value(d_optimizer.learning_rate)),
            )
        else:
            consecutive_collapse_epochs = 0

        generator_is_finite = _generator_is_finite(generator, conditional, NUM_CLASSES)
        if not generator_is_finite:
            logger.warning(
                "Generator health probe failed (non-finite outpu"
                "t); skipping preview/checkpoint for this epoch."
            )

        if generator_is_finite and ((epoch + 1) % 5 == 0 or epoch == total_epochs - 1):
            generator.save(state.generator_ckpt())
            discriminator.save(state.discriminator_ckpt())

        preview_due = (epoch + 1) % gan_preview_freq == 0 or epoch == total_epochs - 1
        if generator_is_finite and preview_due:
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
                        eval_labels_chunk = tf.one_hot(real_eval_labels[i:i+chunk_size],
                            NUM_CLASSES)
                        eval_labels_chunk = tf.cast(eval_labels_chunk, tf.float32)
                        gen_chunk = generator([z_chunk, eval_labels_chunk],
                            training=False)
                    else:
                        gen_chunk = generator(z_chunk,
                            training=False)
                    gen_chunk = tf.where(tf.math.is_finite(gen_chunk), gen_chunk,
                        tf.zeros_like(gen_chunk))
                    gen_chunks.append(gen_chunk.numpy())
                gen = np.concatenate(gen_chunks, axis=0)
                gen = np.clip((gen + 1.0) / 2.0, 0.0, 1.0)
                real_eval = load_images_from_paths(real_eval_paths[:n_eval],
                    img_size=gan_img_size)
                try:
                    fid = calculate_fid(real_eval, gen)
                    fs = calculate_fs(real_eval,
                        gen)
                    fid_scores.append(float(fid))
                    fs_scores.append(float(fs))
                    logger.info(f'FID: {fid:.2f} FS: {fs:.2f}')

                    quality = float(fid + fs)
                    if quality < best_quality - 1e-3:
                        best_quality = quality
                        gan_no_improve = 0
                    else:
                        gan_no_improve += 1

                    if (
                        gan_target_fid > 0
                        and gan_target_fs > 0
                        and fid <= gan_target_fid
                        and fs <= gan_target_fs
                    ):
                        logger.info(
                            "GAN quality target reached (FID<=%s, FS<=%s); stopping early.",
                            gan_target_fid, gan_target_fs,
                        )
                        stop_gan_early = True
                    elif gan_early_stop_patience > 0 and gan_no_improve >= gan_early_stop_patience:
                        logger.warning(
                            "GAN quality plateau detected for %d evaluation cycles; "
                            "stopping early.",
                            gan_no_improve,
                        )
                        stop_gan_early = True
                except Exception as e:
                    logger.info(f'FID/FS computation error: {e}')

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

    generator.save(os.path.join(WEIGHTS_DIR,
        f"generator_{gan_type}.keras"))
    discriminator.save(os.path.join(WEIGHTS_DIR,
        f"discriminator_{gan_type}.keras"))

    gan_log_dir = os.path.join(LOG_DIR,
        "gan")
    os.makedirs(gan_log_dir,
        exist_ok=True)
    plot_gan_losses(d_losses, g_losses, d_accs, g_accs, save_path=os.path.join(gan_log_dir,
        "loss_curves.png"))
    if fid_scores:
        plot_fid_fs_vs_epochs(fid_scores, fs_scores, save_path=os.path.join(gan_log_dir,
            "fid_fs_curves.png"))

    return generator, discriminator, loss_logger, fid_scores, fs_scores
