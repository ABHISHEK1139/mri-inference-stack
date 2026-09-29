"""Dataset acquisition for the training pipeline."""

from __future__ import annotations

import logging
import os

from config import RAW_DIR
from data.dataset import download_dataset

logger = logging.getLogger(__name__)


def _download_kaggle_alternative():
    """Fallback dataset downloader for classification MRI data."""
    import glob

    from kaggle.api.kaggle_api_extended import KaggleApi

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
            logger.info('Downloading Figshare dataset...')
            try:
                download_dataset("figshare")
                if has_files(figshare_dir):
                    logger.info('Figshare download complete')
                else:
                    logger.info(
                        "Figshare download did not produce f"
                        "iles; manual download may be needed"
                    )
            except BaseException as e:
                logger.info(f'Figshare direct download failed: {e}')
                logger.info('Trying Kaggle fallback...')
                try:
                    _download_kaggle_alternative()
                    if has_files(figshare_dir):
                        logger.info('Kaggle fallback download complete')
                    else:
                        logger.info('Kaggle fallback finished but no files found')
                except BaseException as ke:
                    logger.info(f'Kaggle fallback failed: {ke}')
                    logger.info(
                        "Please manually place Figshare/Kagg"
                        "le MRI data under data/raw/figshare"
                    )
        else:
            logger.info(f'Figshare dataset found: {figshare_dir}')

    if download_brats:
        brats_dir = os.path.join(RAW_DIR, "brats")
        if not has_files(brats_dir):
            logger.info('Downloading BraTS dataset (requires Kaggle API credentials)...')
            try:
                download_dataset("brats")
                if has_files(brats_dir):
                    logger.info('BraTS download complete')
                else:
                    logger.info(
                        "BraTS download did not produce fi"
                        "les; manual download may be needed"
                    )
            except BaseException as e:
                logger.info(f'BraTS auto-download failed: {e}')
                logger.info('Please manually place BraTS under data/raw/brats')
        else:
            logger.info(f'BraTS dataset found: {brats_dir}')
