# Weights Policy

This repository uses a hybrid artifact strategy:

- `weights/detection_model.keras` is tracked with Git LFS
- `weights/classifier_model.keras` is tracked with Git LFS
- `weights/segmentation_model.keras` is tracked with Git LFS
- `weights/generator_conditional.keras` is tracked with Git LFS
- `weights/discriminator_conditional.keras` is tracked with Git LFS
- `weights/detection_inference_config.json` stores the calibrated screening threshold
- checkpoints and logs stay out of git

If the demo does not load the core models after cloning, run:

```powershell
git lfs install
git lfs pull
```

This repo now keeps the deployable research weights too, so segmentation and GAN tabs can run immediately after `git lfs pull`.
