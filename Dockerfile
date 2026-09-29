# syntax=docker/dockerfile:1

# Both stages share one pinned base so the builder and runtime cannot drift.
# Replace the tag with a digest (e.g. python:3.12.9-slim-bookworm@sha256:...) for
# byte-reproducible builds; a tag keeps local development convenient.
ARG PYTHON_IMAGE=python:3.12.9-slim-bookworm

# ── Builder ──────────────────────────────────────────────────────────────
FROM ${PYTHON_IMAGE} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install from the fully pinned lock file so the image contains exactly the
# versions the test suite was validated against. Floating ranges in a container
# build are how "works on my machine" becomes "works until the next rebuild".
COPY requirements.txt requirements.lock /build/
RUN python -m pip install --upgrade pip \
    && python -m pip install --require-hashes=false -r /build/requirements.lock

# ── Runtime ──────────────────────────────────────────────────────────────
FROM ${PYTHON_IMAGE}

# Run as a non-root user. Nothing in the image needs root, and a container that
# never runs as root removes an entire class of container-escape impact.
ARG APP_UID=10001
ARG APP_GID=10001

RUN groupadd --gid ${APP_GID} app \
    && useradd --uid ${APP_UID} --gid ${APP_GID} --create-home --shell /usr/sbin/nologin app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    HOME=/home/app

WORKDIR /app

# Copy installed packages from the builder stage.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Application code (respects .dockerignore).
COPY --chown=${APP_UID}:${APP_GID} . /app

# Fail the build if the committed weights are Git LFS pointer stubs rather than
# real archives, so a broken image never reaches a cluster.
RUN python -c "\
import sys; from pathlib import Path; \
bad = []; \
[bad.append(p.name) for p in Path('weights').glob('*.keras') \
   if p.read_bytes()[:48].startswith(b'version https://git-lfs.github.com/spec/v1')]; \
print('LFS POINTER: ' + ', '.join(bad)) if bad else print('weights: OK'); \
sys.exit(1) if bad else None \
"

# The app only reads weights/ and writes to a scratch dir, so the root
# filesystem can stay read-only at runtime (see k8s/deployment.yaml).
RUN mkdir -p /tmp/streamlit && chown -R ${APP_UID}:${APP_GID} /app /tmp/streamlit

USER ${APP_UID}:${APP_GID}

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "app.py"]
