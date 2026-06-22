# =============================================================================
# EDGE-RAG — CPU inference / retrieval / evaluation image
# =============================================================================
# Scope: the CPU-only inference + retrieval + evaluation path ONLY. The GPU
# extraction stage (Phase 2 GLiNER + REBEL, run once on Colab) is NOT in this
# image — its output (the LanceDB / KuzuDB / BM25 stores) is shipped as DATA and
# mounted at runtime (see docker-compose.yml volumes + scripts/fetch_data.sh).
#
# Ollama is a SIDECAR container (see docker-compose.yml), not baked in here, so
# the app image stays free of the model weights and the LLM endpoint is reached
# over the Compose network at http://ollama:11434.
#
# Python 3.12-slim matches the frozen reproducibility interpreter (3.12.3,
# pinned in requirements_frozen.txt).
# =============================================================================

FROM python:3.12-slim

# Reproducibility / quality-of-life:
#   PYTHONUNBUFFERED  — stream logs straight to the container stdout.
#   PYTHONUTF8        — the eval CLIs require UTF-8 (the Windows host uses
#                       `python -X utf8`; on Linux this env var is the equivalent
#                       and avoids locale-dependent UnicodeEncodeError on the
#                       diagnostic / progress output).
#   PYTHONDONTWRITEBYTECODE — no stray .pyc in the read-only-ish layer.
ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONDONTWRITEBYTECODE=1

# LanceDB / KuzuDB / torch ship manylinux wheels, so the build needs no compiler
# in the common case. build-essential is kept as a fallback for any sdist-only
# transitive dep; curl + ca-certificates are used by the data-fetch helper.
# Remove build-essential here if you confirm every wheel resolves binary-only.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Dependency layer (cached unless the pinned set changes) -----------------
# Use the PINNED frozen set so the image reproduces the reported numbers
# byte-for-byte. requirements_frozen.txt also pulls the spaCy en_core_web_sm/md
# wheels and sentence-transformers (the cross-encoder reranker dep) directly, so
# no separate `python -m spacy download` step is needed.
#
# Two-step install: lancedb / pylance are pinned in requirements_frozen_nodeps.txt
# and installed with --no-deps. lancedb 0.6.13's pylance dependency over-pins
# pyarrow<15.0.1 in its metadata, which conflicts with the pyarrow 23.x that
# datasets 4.5.0 requires; the pin is stale (lancedb 0.6.13 runs fine on
# pyarrow 23 — verified against the reported stores), so we honour the working
# reproduced configuration rather than pip's over-strict refusal of the combo.
COPY requirements_frozen.txt requirements_frozen_nodeps.txt ./
RUN pip install --no-cache-dir -r requirements_frozen.txt \
    && pip install --no-cache-dir --no-deps -r requirements_frozen_nodeps.txt

# --- Source + config (NOT the big indices / datasets — mounted at runtime) ----
COPY src/ ./src/
COPY config/ ./config/

# Runtime wiring. These are read by:
#   OLLAMA_HOST  -> _settings_loader._apply_env_overrides (llm + embeddings base_url)
#   CONFIG_PATH  -> _settings_loader._default_settings_path
#   INDEX_DIR    -> benchmark_datasets._resolve_data_root (StoreManager base)
# DATASET_DIR is accepted as an alias for INDEX_DIR by _resolve_data_root.
ENV OLLAMA_HOST=http://ollama:11434 \
    CONFIG_PATH=/app/config/frozen_paper.yaml \
    INDEX_DIR=/app/data/indices \
    DATASET_DIR=/app/data/datasets

# Pre-built stores + datasets land here via the compose bind-mounts.
RUN mkdir -p /app/data/indices /app/data/datasets

# Model/HF caches go on a writable path so a read-only rootfs still works and the
# reranker download persists across runs when /app/cache is a mounted volume.
ENV HF_HOME=/app/cache/hf \
    XDG_CACHE_HOME=/app/cache

ENTRYPOINT ["python", "-m", "src.thesis_evaluations.benchmark_datasets"]
CMD ["--help"]
