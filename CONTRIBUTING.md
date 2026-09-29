# Contributing Guide

## Scope

The repository centres on a production-minded flagship workflow:

1. Calibrated tumour detection
2. Tumour type classification

Segmentation and conditional synthesis are maintained as research tracks.

## Setup

```powershell
git lfs install && git lfs pull
python -m pip install -r requirements.lock
python -m pip install -r requirements-dev.txt
```

`requirements.lock` pins the stack the tests were validated against. Use
`requirements.txt` only when you deliberately want floating ranges.

## Before opening a pull request

```powershell
# 1. Structural readiness
python scripts/preflight.py

# 2. Lint and compile
ruff check .
python -m compileall app.py train.py config.py preprocessing.py data models training evaluation scripts tests -q

# 3. Fast test suite (unit + regression)
python -m pytest tests/ -m "not slow" -q

# 4. Trainer integration tests (real single-epoch runs)
python -m pytest tests/ -m slow -q

# 5. Coverage
python -m pytest tests/ -m "not slow" --cov=. --cov-report=term-missing
```

All five must pass. CI runs the same commands.

## Code layout

`train.py` is a thin CLI. The engine is a package:

```
training/
  runtime.py          device policy, precision, shared helpers
  state.py            atomic checkpoint state
  reproducibility.py  seeding and determinism
  data_sources.py     dataset acquisition
  callbacks.py        checkpointing, logging, collapse detection
  tracks/             one module per training track
```

Adding a track means adding `training/tracks/<name>.py`, exporting its trainer
from that module, and adding a case to the `train.py` dispatcher.

## Writing tests

- Bug fixes require a regression test whose docstring names the original
  failure. `tests/test_*_regressions.py` follow this pattern.
- Anything touching the training engine needs a test in
  `tests/test_trainers_integration.py` that runs a real single-epoch run against
  a generated fixture. Shape assertions will not catch data-pipeline and
  checkpoint-format defects.
- Tests must not write outside `tmp_path`. The `workspace` fixture in
  `tests/test_trainers_integration.py` redirects the output paths the track
  modules captured at import time; use it rather than monkeypatching ad hoc.

## Reproducibility

- Every run is seeded. Pass `--seed` or set `SEED`.
- The seed and library versions are recorded in checkpoint state; include them
  in bug reports.
- If you change a dependency, regenerate `requirements.lock` deliberately and
  re-run the suite before committing.

## Artifacts

- Keep runtime model files in `weights/` via Git LFS.
- Do not commit raw datasets, checkpoints, logs, or coverage output.
- Do not change code that would make the app depend on `training/`; the Docker
  image is inference-only by design.

## Reporting issues

Include the command, the seed, the environment (`python --version`,
`pip freeze | Select-String tensorflow`), and the full traceback.
