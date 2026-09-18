# Contributing Guide

## Scope

This project prioritizes a production-minded flagship workflow:

1. Calibrated tumour detection
2. Subtype classification

Segmentation and GAN modules are maintained as experimental tracks.

## Development Standards

Before opening a pull request:

1. Run the readiness preflight:
   - `python scripts/preflight.py`
2. Run lint and compile checks:
   - `ruff check .`
   - `python -m compileall app.py train.py config.py data models training evaluation scripts`
3. Ensure docs stay aligned with behavior changes.

## Pull Request Expectations

- Keep changes focused and explain the technical reason.
- Avoid broad refactors without a clear operational benefit.
- Include updates to README or docs when user-facing behavior changes.

## Artifact Policy

- Keep runtime-ready model files in `weights/` through Git LFS.
- Do not commit raw datasets.
- Do not commit transient logs, checkpoints, or scratch outputs.

## Reporting Issues

When reporting a bug, include:

- reproduction steps
- command used
- relevant environment details
- stack trace or error output

This helps keep triage fast and actionable.
