# Dataset Notes

## Primary classification dataset

The repo expects a local Figshare-style brain MRI dataset with four classes:

- `glioma`
- `meningioma`
- `pituitary`
- `normal`

The dataset loader accepts common aliases such as `other`, `healthy`, and `no_tumor`, and it can discover both explicit split folders and unsplit class folders.

## Expected local layout

```text
data/
  raw/
    figshare/
      train/
        glioma/
        meningioma/
        pituitary/
        normal/
      test/
        ...
```

If there is no explicit validation split, the loader creates one with stratified sampling.

## Segmentation dataset

The BraTS-style segmentation track expects paired images and masks under `data/raw/brats`. The loader looks for image files plus matching mask filenames containing terms such as:

- `mask`
- `seg`
- `label`

## Formats

Supported image formats include:

- `png`
- `jpg`
- `jpeg`
- `bmp`
- `tif`
- `tiff`

## Data policy

- Raw medical data is not committed to git
- Download helpers exist, but local placement is the expected workflow for serious use
- The repo is safe to publish because only code, docs, curated visuals, and LFS-managed core weights are retained

