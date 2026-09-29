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
    EMAGenerator,
    build_v2_discriminator,
    build_v2_generator,
    gradient_penalty,
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



def train_gan_v2(data_dir=None, epochs=None, fid_eval_freq=10, resume=True,
    seed: int | None = None):
    """Train the v2 WGAN-GP under a float32 policy that is always restored."""
    with float32_precision():
        return _train_gan_v2_impl(
            data_dir=data_dir, epochs=epochs, fid_eval_freq=fid_eval_freq, resume=resume
        )


def _train_gan_v2_impl(data_dir=None, epochs=None, fid_eval_freq=10, seed: int | None = None,
    resume=True):
    seed = set_seed(seed)
    logger.info("Track gan_v2: seed=%d", seed)
    """V2 GAN training: ResNet generator + Projection discriminator + WGAN-GP.

    This is a research-grade conditional GAN with:
    - ResNet blocks with conditional batch norm and self-attention
    - Projection discriminator with spectral normalization
    - WGAN-GP loss (Wasserstein + gradient penalty)
    - EMA generator for smoother previews
    - 5:1 discriminator:generator step ratio
    """
    logger.info('\n' + '=' * 60)
    logger.info('TRACK 4 - GAN V2 (WGAN-GP + ResNet + Projection)')
    logger.info('=' * 60)

    (X_train_paths, y_train_labels), (X_val_paths,
        y_val_labels), _ = get_figshare_train_val_test_split(data_dir)
    cfg = TRACK_CONFIGS["gan"]
    fid_eval_freq = int(os.getenv("GAN_FID_EVAL_FREQ", str(fid_eval_freq)))
    fid_eval_freq = max(0,
        fid_eval_freq)
    img_shape = (*IMG_CFG.gan_size,
        1)

    # Build v2 models
    generator = build_v2_generator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES,
        output_shape=img_shape)
    discriminator = build_v2_discriminator(input_shape=img_shape, num_classes=NUM_CLASSES)

    state = GANState("v2", seed=seed)
    initial_epoch = 0

    if not resume:
        state.state = GANState.fresh_state()

    if resume and state.has_ckpt():
        logger.info('Loading GAN v2 checkpoints...')
        try:
            generator.load_weights(state.generator_ckpt())
            discriminator.load_weights(state.discriminator_ckpt())
            initial_epoch = state.start_epoch()
            logger.info(f'Resumed from epoch {initial_epoch}')
        except Exception as e:
            logger.warning(f'Could not load v2 checkpoints: {e}')
            logger.info('Starting fresh training')
            initial_epoch = 0
            state.state = GANState.fresh_state()

    gan_img_size = IMG_CFG.gan_size
    dataset = build_gan_dataset_from_paths(
        X_train_paths,
        labels=y_train_labels,
        img_size=gan_img_size,
        batch_size=cfg.batch_size,
        seed=seed,
)

    # EMA for generator
    ema = EMAGenerator(generator,
        decay=0.999)

    loss_logger = GANLossLogger(
        columns=["epoch", "d_loss", "g_loss", "w_dist",
            "g_acc"],
    )
    sampler = GANImageSampler(generator, latent_dim=LATENT_DIM, conditional=True,
        num_classes=NUM_CLASSES)
    collapse_detector = ModelCollapseDetector(
        generator, latent_dim=LATENT_DIM, conditional=True, num_classes=NUM_CLASSES,
    )

    def _filter_finite(lst):
        return [v for v in lst if isinstance(v, (int, float)) and np.isfinite(v)]

    d_losses = _filter_finite(state.state.get("d_losses", []))
    g_losses = _filter_finite(state.state.get("g_losses", []))
    # Wasserstein distances get their own state slot. They used to be written
    # into "d_accs", so resuming a v2 run after a v1 run read accuracies in as
    # distances (and vice versa).
    w_distances = _filter_finite(state.state.get("w_distances", []))
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

            # Gradient penalty (unscaled helper; lambda applied here)
            gp = lambda_gp * gradient_penalty(discriminator, real_images, fake, real_labels)

            d_loss = d_loss_real + d_loss_fake + gp

        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        grads = _sanitize_grads(grads, discriminator.trainable_variables)
        if gan_grad_clip_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, gan_grad_clip_norm)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables, strict=True))

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
        if gan_grad_clip_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, gan_grad_clip_norm)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables, strict=True))
        return g_loss

    total_epochs = epochs or int(os.getenv("GAN_EPOCHS", "300"))
    if initial_epoch >= total_epochs:
        logger.info('GAN v2 already reached requested epochs')
        return generator, discriminator, loss_logger, fid_scores, fs_scores

    logger.info(
        "GAN v2 settings: batch_size=%d fid_eval_freq=%d preview_freq=%d "
        "d_steps=%d g_steps=%d d_lr=%.2e g_lr=%.2e lambda_gp=%s precision=float32",
        cfg.batch_size, fid_eval_freq, gan_preview_freq, gan_d_steps, gan_g_steps,
        d_lr, g_lr, lambda_gp,
    )
    logger.info(f'Starting GAN v2 training: epoch {initial_epoch} -> {total_epochs}')

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
            logger.warning(
                f"'Epoch {epoch + 1}/{total_epochs} prod"
                f"uced no finite losses; stopping early."
            )
            break

        d_avg = float(np.mean(d_epoch))
        g_avg = float(np.mean(g_epoch))
        w_avg = float(np.mean(w_dist_epoch))

        if bad_batch_count > 0:
            logger.warning(f'  Skipped {bad_batch_count} unstable batches')

        if not np.isfinite(d_avg) or not np.isfinite(g_avg):
            logger.warning(f'Epoch {epoch + 1}/{total_epochs} non-finite losses; stopping.')
            break

        d_losses.append(d_avg)
        g_losses.append(g_avg)
        w_distances.append(w_avg)
        g_accs.append(0.0)
        loss_logger.log_step(epoch, d_avg, g_avg, w_avg, 0.0)

        elapsed = time.time() - start
        logger.info(
            "Epoch %d/%d [%.1fs] D:%.4f G:%.4f W-dist:%.4f",
            epoch + 1, total_epochs, elapsed, d_avg, g_avg, w_avg,
        )

        # Check generator health
        generator_healthy = _generator_is_finite(generator, True, NUM_CLASSES)
        if not generator_healthy:
            logger.warning('  Generator health probe failed; skipping checkpoint/preview.')

        # Save checkpoints every 5 epochs
        if generator_healthy and ((epoch + 1) % 5 == 0 or epoch == total_epochs - 1):
            generator.save_weights(state.generator_ckpt())
            discriminator.save_weights(state.discriminator_ckpt())
            # Persist EMA weights too, otherwise a resumed run loses them.
            ema.save(state.ema_ckpt())

        # Preview with EMA generator
        if generator_healthy and ((epoch + 1) % gan_preview_freq == 0 or epoch == total_epochs - 1):
            with ema.swapped():
                sampler._generate_and_save(epoch)

        # FID evaluation
        stop_early = False
        if fid_eval_freq > 0 and (epoch + 1) % fid_eval_freq == 0:
            real_eval_paths = X_val_paths if len(X_val_paths) > 0 else X_train_paths
            real_eval_labels = y_val_labels if len(X_val_paths) > 0 else y_train_labels
            n_eval = min(64 if LOW_VRAM_MODE else 256,
                len(real_eval_paths))
            if n_eval > 0 and generator_healthy:
                gen_batch_size = 16 if LOW_VRAM_MODE else 64
                gen_chunks = []
                with ema.swapped():
                    for i in range(0,
                        n_eval,
                        gen_batch_size):
                        chunk_size = min(gen_batch_size,
                            n_eval - i)
                        z_chunk = tf.random.normal([chunk_size, LATENT_DIM])
                        eval_labels_chunk = tf.one_hot(real_eval_labels[i:i+chunk_size],
                            NUM_CLASSES)
                        eval_labels_chunk = tf.cast(eval_labels_chunk, tf.float32)
                        gen_chunk = generator([z_chunk, eval_labels_chunk],
                            training=False)
                        gen_chunk = tf.where(tf.math.is_finite(gen_chunk), gen_chunk,
                            tf.zeros_like(gen_chunk))
                        gen_chunks.append(gen_chunk.numpy())
                gen = np.concatenate(gen_chunks, axis=0)
                gen = np.clip((gen + 1.0) / 2.0, 0.0, 1.0)
                real_eval = load_images_from_paths(real_eval_paths[:n_eval], img_size=gan_img_size)
                try:
                    fid = calculate_fid(real_eval, gen)
                    fs = calculate_fs(real_eval, gen)
                    fid_scores.append(float(fid))
                    fs_scores.append(float(fs))
                    logger.info(f'  FID: {fid:.2f} FS: {fs:.2f}')

                    quality = float(fid + fs)
                    if quality < best_quality - 1e-3:
                        best_quality = quality
                        gan_no_improve = 0
                    else:
                        gan_no_improve += 1

                    if gan_early_stop_patience > 0 and gan_no_improve >= gan_early_stop_patience:
                        logger.warning(
                            "  GAN quality plateau for %d cycles; stopping.", gan_no_improve
                        )
                        stop_early = True
                except Exception as e:
                    logger.info(f'  FID/FS error: {e}')

        collapse_detector.on_epoch_end(epoch)

        # Save state
        state.state["last_epoch"] = epoch
        state.state["d_losses"] = _filter_finite(d_losses)
        state.state["g_losses"] = _filter_finite(g_losses)
        state.state["w_distances"] = _filter_finite(w_distances)
        state.state["d_accs"] = []
        state.state["g_accs"] = _filter_finite(g_accs)
        state.state["fid_scores"] = _filter_finite(fid_scores)
        state.state["fs_scores"] = _filter_finite(fs_scores)
        state.state["best_quality"] = best_quality if np.isfinite(best_quality) else 1e9
        state.state["gan_no_improve"] = gan_no_improve
        state.save()

        if stop_early:
            break

    # Save final weights
    with ema.swapped():
        generator.save_weights(os.path.join(WEIGHTS_DIR, "generator_v2.weights.h5"))
    discriminator.save_weights(os.path.join(WEIGHTS_DIR, "discriminator_v2.weights.h5"))
    ema.save(state.ema_ckpt())

    gan_log_dir = os.path.join(LOG_DIR,
        "gan")
    os.makedirs(gan_log_dir,
        exist_ok=True)
    plot_gan_losses(
        d_losses, g_losses, w_distances,
            g_accs,
        save_path=os.path.join(gan_log_dir, "v2_loss_curves.png"), d_label="W-Distance",
    )
    if fid_scores:
        plot_fid_fs_vs_epochs(fid_scores, fs_scores, save_path=os.path.join(gan_log_dir,
            "v2_fid_fs_curves.png"))

    return generator, discriminator, loss_logger, fid_scores, fs_scores
