# Innovation Notes

## Engineering focus

The strongest idea in this project is not just the model choice. It is the combination of:

- a **low-VRAM aware training profile**
- a **calibrated screening threshold**
- a **clear separation between stable demo workflow and experimental research tracks**

That combination makes the repo more credible than an oversized local experiment dump.

## Low-resource design

The training stack is designed to adapt to constrained hardware using:

- environment-driven image size and batch size selection
- resumable checkpoints
- optional mixed precision
- separate settings for detection, classification, segmentation, and GAN training

## Calibrated inference

The detector does not just expose a raw sigmoid score. It persists an inference configuration with:

- selected decision threshold
- validation metrics at that operating point

This makes the app demo more realistic and gives the screening stage a defensible deployment posture.

## Repo-level improvement

The project was upgraded into a portfolio-grade repository by:

- rebuilding missing source modules
- removing hard-coded local paths
- shrinking the public promise to the strongest validated workflow
- pushing only the core inference weights through Git LFS

