# Code Reading Guide

This guide is for reading the repository top to bottom without getting lost. It
explains what each package is for, which file to open first, and why the code is
split the way it is.

## The short version

Four commands cover almost everything:

| Command | What it does |
| --- | --- |
| `streamlit run app.py` | Runs the web app for inference |
| `python train.py --track detection` | Trains one model |
| `python scripts/preflight.py` | Checks the environment before you start |
| `python -m pytest tests/ -m "not slow"` | Runs the fast test suite |

## How to read this repository

Start at `config.py`, then follow the numbers in this order:

1. **`config.py`** — every hyperparameter in one place. Image sizes, class
   names, the latent dimension, the decision threshold. No other module invents
   a default value, so this file answers "what exactly is being built?"
2. **`preprocessing.py`** — turns a raw uploaded image into the exact tensor a
   model expects. Each model has its own function (`preprocess_detection`,
   `preprocess_classifier`, `preprocess_segmentation`) because they use
   different normalisation.
3. **`models/`** — model definitions only. `build_detection_model()`,
   `build_classifier()`, `build_unet()`, and `models/gan/` for the generators.
   These functions construct Keras models; they never load data.
4. **`data/dataset/`** — turns files on disk into `tf.data` pipelines.
   - `naming.py` works out which patient a file belongs to.
   - `splits.py` assigns files to train/val/test.
   - `loaders.py` reads images into arrays.
   - `builders.py` assembles the `tf.data.Dataset` objects.
   - `loading.py` holds the low-level decode and augmentation helpers.
5. **`training/`** — the training machinery, shared by every track.
   - `reproducibility.py` sets seeds and deterministic behaviour.
   - `state.py` saves and restores checkpoints.
   - `callbacks.py` holds the Keras callbacks (early stopping, logging).
   - `runtime.py` is the shared per-epoch loop.
   - `data_sources.py` points a track at its dataset on disk.
   - `tracks/` has one file per experiment, and each `train_*` function is a
     complete training run for that model.
6. **`evaluation/`** — metrics and plots, one module per model family.
7. **`app.py`** — the Streamlit UI. It loads models, calls preprocessing, and
   displays results. It contains no model definitions and no metrics.

## Why `train.py` is small

`train.py` only parses arguments and dispatches. Opening it should take under a
minute. All the real work lives in `training/tracks/`, so you can read one track
without wading through the others.

Each track file follows the same shape, which makes them interchangeable to read:

```
train_detection(...)      ->  training/tracks/detection.py
train_classifier(...)     ->  training/tracks/classifier.py
train_segmentation(...)   ->  training/tracks/segmentation.py
train_gan(...)            ->  training/tracks/gan.py
train_gan_v2(...)         ->  training/tracks/gan_v2.py
train_classifier_with_gan(...) -> training/tracks/gan_augmented.py
```

Import them from the module that owns them, for example
`from training.tracks.detection import train_detection`.

## Why the `data/dataset` and `models/gan` split

Both of these started as single files that grew past 400 lines. They were split
by responsibility, which means:

- a bug in image loading cannot hide inside model definition;
- a file name tells you what it does before you open it;
- each module can be tested on its own.

`models/gan/legacy.py` holds the original generator architecture and
`models/gan/v2.py` the conditional variant with the projection discriminator.
`gan_config.py` in `training/tracks/` holds the hyperparameters for both.

## Where things are configured

| Question | Answer |
| --- | --- |
| Class names | `config.CLASS_NAMES` |
| Image size per model | `config.ImageConfig` fields, used as `IMG_CFG.detection_size`, `IMG_CFG.classifier_size`, `IMG_CFG.segmentation_size`, `IMG_CFG.gan_size` |
| Latent vector size | `config.LATENT_DIM` |
| Decision threshold | `weights/detection_inference_config.json`, read by the app through `_load_detection_config()` |
| Dataset URLs | `config.DATASET_CONFIG` |
| Weights directory | `config.WEIGHTS_DIR` |

`IMG_CFG` is the single `ImageConfig` instance. `config.py` declares the shape
of the object, and `training/runtime.py` creates it once so that every track
and the preprocessing helpers read the same values.

The app never hard-codes a value that also appears in `config.py`. That is why a
change to `LATENT_DIM` or `CLASS_NAMES` cannot leave one half of the system
disagreeing with the other.

## Tests

```bash
python -m pytest tests/ -m "not slow"   # fast: shapes, loading, regressions
python -m pytest tests/ -m slow          # slow: trains every model for one epoch
```

The split is deliberate. A single trainer run costs 40-180 seconds, so those
tests are marked `slow` and CI runs them in a separate step. If you only want a
quick "did I break something" signal, use the fast tier.

`tests/test_app_ui.py` renders the real Streamlit app with Streamlit's own
`AppTest` harness, so the widget tree and tab structure are actually exercised
rather than assumed.

## Where to make a change

| You want to... | Edit |
| --- | --- |
| Change a hyperparameter | `config.py` |
| Change how an image is prepared | `preprocessing.py` |
| Change a model's architecture | `models/` |
| Change the data pipeline | `data/dataset/` |
| Change how training loops | `training/runtime.py`, `training/callbacks.py` |
| Add a new experiment | a new file in `training/tracks/` |
| Change a metric | `evaluation/` |
| Change the UI | `app.py` |