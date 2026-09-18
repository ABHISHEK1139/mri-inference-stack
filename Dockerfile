FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY requirements.txt /build/requirements.txt
RUN pip install --upgrade pip && pip install -r /build/requirements.txt

# ── Runtime stage ────────────────────────────────────────────────────────
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code (respects .dockerignore)
COPY . /app

# Validate that .keras files are not Git LFS pointers
RUN python -c "\
import sys; from pathlib import Path; \
for p in Path('weights').glob('*.keras'): \
    header = p.read_bytes()[:48]; \
    lfs = header.startswith(b'version https://git-lfs.github.com/spec/v1'); \
    print(f'  {p.name}: {\"LFS POINTER\" if lfs else \"OK\"}'); \
    sys.exit(1) if lfs else None \
" || { echo 'ERROR: Git LFS pointers found in weights/. Run git lfs pull before building.'; exit 1; }

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "app.py"]
