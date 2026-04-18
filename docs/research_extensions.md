# Research Extensions

The repository keeps two research-oriented tracks beyond the flagship screening pipeline.

## Segmentation

The segmentation module uses an attention-enabled U-Net style architecture with residual blocks. Current tracked evidence shows:

- validation Dice around `0.4808`
- validation IoU around `0.4216`

This is useful for technical discussion and future refinement, but it is not marketed as the strongest production-like result in the repo.

## Synthetic MRI generation

The GAN track explores class-conditional MRI synthesis and includes:

- checkpoint resumption
- sample generation
- FID / FS logging hooks
- a second, more research-heavy WGAN-GP path

The repo intentionally treats GAN output as exploratory:

- sample grids are retained as qualitative evidence
- the core README does not frame GAN results as stable clinical-grade performance

## Why keep them

These modules still add value because they show:

- broader modeling range
- comfort with training stability issues
- attention to evaluation and monitoring
- the ability to separate core deliverables from research exploration

