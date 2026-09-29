"""Training entry point for the Brain MRI intelligence system.

This module is intentionally thin: argument parsing, device/seed setup, dataset
acquisition, and dispatch to the per-track trainers that live in
:mod:`training.tracks`. The parser and the ``--help`` short-circuit are defined
before any heavy import so the CLI stays usable without TensorFlow installed.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Brain Tumour GAN Challenge Training")
    parser.add_argument("--track", type=str, default="all",
        choices=["all", "detection", "segmentation", "classifier", "gan", "gan_v2",
            "gan_augmented"])
    parser.add_argument("--gan_type", type=str, default="conditional",
        choices=["baseline", "dcgan", "conditional", "stylegan"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--no_resume", action="store_true",
        help="Disable checkpoint resume and start fresh")
    parser.add_argument("--download_figshare", action="store_true",
        help="Force/ensure Figshare download")
    parser.add_argument("--download_brats", action="store_true",
        help="Force/ensure BraTS download")
    parser.add_argument("--only_download", action="store_true",
        help="Download datasets only and exit")
    parser.add_argument(
        "--patient_level",
        action="store_true",
        help="Use patient-level grouping for train/val/test splits to prevent data leakage",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible runs (default: config.DEFAULT_SEED, env SEED)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic TensorFlow kernels (slower; may be unsupported for some ops)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging verbosity",
    )
    return parser


if __name__ == "__main__" and any(flag in {"-h", "--help"} for flag in sys.argv[1:]):
    build_arg_parser().print_help()
    sys.exit(0)


from config import LOW_VRAM_MODE, RAW_DIR  # noqa: E402
from training.data_sources import ensure_datasets  # noqa: E402
from training.reproducibility import set_seed  # noqa: E402
from training.runtime import configure_gpu  # noqa: E402
from training.tracks.classifier import train_classifier  # noqa: E402
from training.tracks.detection import train_detection  # noqa: E402
from training.tracks.gan import train_gan  # noqa: E402
from training.tracks.gan_augmented import train_classifier_with_gan  # noqa: E402
from training.tracks.gan_v2 import train_gan_v2  # noqa: E402
from training.tracks.segmentation import train_segmentation  # noqa: E402

logger = logging.getLogger("train")


def run_all() -> None:
    """Run every track in dependency order."""
    ensure_datasets(download_figshare=True, download_brats=True)

    figshare_dir = os.path.join(RAW_DIR, "figshare")
    brats_dir = os.path.join(RAW_DIR, "brats")

    train_detection(data_dir=figshare_dir, resume=True)
    train_classifier(data_dir=figshare_dir, resume=True)
    generator, _, _, _, _ = train_gan(
        data_dir=figshare_dir, gan_type="conditional", resume=True
    )
    if LOW_VRAM_MODE:
        logger.info("Skipping GAN-augmented classifier stage in low-VRAM mode")
    else:
        train_classifier_with_gan(
            generator, data_dir=figshare_dir, gan_type="conditional", resume=True
        )

    if os.path.exists(brats_dir) and len(os.listdir(brats_dir)) > 0:
        train_segmentation(data_dir=brats_dir, resume=True)
    else:
        logger.info("BraTS not available, skipping segmentation track")

    logger.info("All requested training stages completed")


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    configure_gpu()
    seed = set_seed(args.seed, deterministic_ops=True if args.deterministic else None)
    logger.info("Run seed: %d", seed)

    if args.download_figshare or args.download_brats:
        ensure_datasets(
            download_figshare=args.download_figshare,
            download_brats=args.download_brats,
        )

    if args.only_download:
        ensure_datasets(
            download_figshare=args.download_figshare or not (
                args.download_figshare or args.download_brats
            ),
            download_brats=args.download_brats or not (
                args.download_figshare or args.download_brats
            ),
        )
        logger.info("Dataset download stage completed")
        return

    resume = not args.no_resume
    track = args.track

    if track == "all":
        run_all()
        return

    if track == "segmentation":
        ddir = args.data_dir or os.path.join(RAW_DIR, "brats")
        ensure_datasets(download_figshare=False, download_brats=True)
        train_segmentation(
            data_dir=ddir, epochs=args.epochs, resume=resume,
            patient_level=args.patient_level,
        )
    elif track == "gan_v2":
        ddir = args.data_dir or os.path.join(RAW_DIR, "figshare")
        ensure_datasets(download_figshare=True, download_brats=False)
        train_gan_v2(data_dir=ddir, epochs=args.epochs, resume=resume)
    elif track == "gan_augmented":
        ddir = args.data_dir or os.path.join(RAW_DIR, "figshare")
        ensure_datasets(download_figshare=True, download_brats=False)
        if LOW_VRAM_MODE:
            logger.info("GAN-augmented classifier training is disabled in low-VRAM mode")
            return
        generator, _, _, _, _ = train_gan(
            data_dir=ddir, gan_type=args.gan_type, epochs=args.epochs, resume=resume
        )
        train_classifier_with_gan(
            generator, data_dir=ddir, gan_type=args.gan_type,
            epochs=args.epochs, resume=resume,
        )
    else:
        ddir = args.data_dir or os.path.join(RAW_DIR, "figshare")
        ensure_datasets(download_figshare=True, download_brats=False)
        if track == "detection":
            train_detection(
                data_dir=ddir, epochs=args.epochs, resume=resume,
                patient_level=args.patient_level,
            )
        elif track == "classifier":
            train_classifier(
                data_dir=ddir, epochs=args.epochs, resume=resume,
                patient_level=args.patient_level,
            )
        elif track == "gan":
            train_gan(
                data_dir=ddir, gan_type=args.gan_type,
                epochs=args.epochs, resume=resume,
            )
        else:  # pragma: no cover - argparse restricts the choices
            raise ValueError(f"Unknown track: {track}")


if __name__ == "__main__":
    main()
