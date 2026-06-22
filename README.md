# Edge-RAG: Empirical Characterization of When Knowledge-Graph Lanes Add Value in CPU-Only Hybrid Retrieval

[![tests](https://github.com/nietzard-science/edge-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/nietzard-science/edge-rag/actions/workflows/tests.yml)
[![docker](https://github.com/nietzard-science/edge-rag/actions/workflows/docker.yml/badge.svg)](https://github.com/nietzard-science/edge-rag/actions/workflows/docker.yml)

Reference implementation accompanying the paper *"Edge-RAG: Empirical
Characterization of When Knowledge-Graph Lanes Add Value in CPU-Only Hybrid
Retrieval."*

The system combines three retrieval modalities — dense vector search, sparse
BM25, and structured knowledge-graph traversal — fused by Reciprocal Rank
Fusion and mediated by a three-agent reasoning pipeline
(**Planner → Navigator → Verifier**), all running on a single commodity
machine with no GPU at inference time and no cloud dependency.

The codebase is the design-science artifact and is organised into three
independently testable layers:

| Artifact | Layer | Responsibility |
|---|---|---|
| **A** | Data (`src/data_layer/`) | LanceDB vectors + KuzuDB graph, batched embeddings with cache, hybrid retrieval, optional cross-encoder reranking. |
| **B** | Logic (`src/logic_layer/`, `src/pipeline/`) | Query classification & decomposition (Planner), pre-generative retrieval filtering (Navigator), pre-validation + quantised-SLM generation with optional self-correction (Verifier). |
| **C** | Evaluation (`src/thesis_evaluations/`) | Multi-dataset benchmarking, retrieval-only and end-to-end modes, ablation suites, paired-bootstrap significance testing. |

## Quickstart

```bash
# 1. Dependencies (pinned set used for the reported numbers)
pip install -r requirements_frozen.txt
python -m spacy download en_core_web_sm

# 2. Local model server (separate terminal)
ollama serve
ollama pull qwen2:1.5b        # answer-generation SLM
ollama pull nomic-embed-text  # embedding model

# 3. Verify the installation (no model weights required)
python -X utf8 -m pytest test_system/ -m "not nightly" -q
```

Full ingestion + evaluation steps — including input-artifact verification and
the headline benchmark — are in **[`REPRODUCE.md`](REPRODUCE.md)**.

## Reproduce with Docker

A Dockerized artifact reproduces the **inference and retrieval pipeline on
CPU-only hardware**. The runtime container is constrained to the **2 GB / 4-core
target envelope** via Compose resource limits, and Ollama runs as a sidecar
(model weights are pulled automatically, not baked into the image). The GPU
extraction stage is *not* containerized — its output (the LanceDB / KuzuDB /
BM25 stores) ships as a data artifact and is mounted at runtime.

```bash
# 1. Fetch the pre-built stores into ./data/indices (Zenodo artifact).
#    Set the artifact URL once the DOI is minted:
ZENODO_URL="https://zenodo.org/records/<ID>/files/edge-rag-stores.tar.gz" \
    ./scripts/fetch_data.sh          # Windows host: .\scripts\fetch_data.ps1

# 2. Bring up Ollama (healthy), auto-pull qwen2:1.5b + nomic-embed-text, idle app.
docker compose up -d

# 3a. One-command demo — full Planner -> Navigator -> Verifier trace + envelope:
docker compose run --rm app demo --question "Who directed the film Ed Wood?"

# 3b. Reproduce a result slice (constrained to the 2 GB / 4-core envelope):
docker compose run --rm app evaluate --dataset hotpotqa --range 0-5
```

The `mem_limit: 2g` in [`docker-compose.yml`](docker-compose.yml) is the
envelope evidence: a run that ever exceeds 2 GB is OOM-killed by the runtime.
Runtime wiring is via env vars — `OLLAMA_HOST` (LLM + embedding endpoint),
`CONFIG_PATH` (settings YAML), `INDEX_DIR` / `DATASET_DIR` (store base) — so the
container needs no code edits to relocate Ollama or the mounted stores.

## Repository map

```
src/data_layer/        Artifact A — storage, embeddings, retrieval, extraction
src/logic_layer/       Artifact B — Planner / Navigator / Verifier agents
src/pipeline/          Artifact B — AgentPipeline orchestrator
src/thesis_evaluations/ Artifact C — benchmark runner + ablation suites
config/settings.yaml   Single source of truth for every runtime hyperparameter
test_system/           Test suite (738 tests; 720 in the default CI selection)
docs/figures/          Generated paper figures
local_importingestion.py  Phase-3 ingestion entry point (CPU)
colab_extraction.py       Phase-2 GLiNER + REBEL extraction (GPU, offline)
diagnose_*.py             Diagnostic / single-query trace tools
```

## Documentation

- **[`REPRODUCE.md`](REPRODUCE.md)** — step-by-step reproduction protocol (the reproducibility contract).
- **[`TECHNICAL_ARCHITECTURE.md`](TECHNICAL_ARCHITECTURE.md)** — full architecture, design decisions (§11), and academic grounding.
- **[`config/settings.yaml`](config/settings.yaml)** — all hyperparameters; no tuning constant is hardcoded in a production path.

## Requirements

Python 3.12 · Ollama ≥ 0.4 · ~8 GB RAM (16 GB recommended) · CPU-only at
inference. Tested on Windows 11.

## License

Released under the MIT License (see [`LICENSE`](LICENSE)). If you use this
system, please cite the accompanying paper *"Edge-RAG: Empirical
Characterization of When Knowledge-Graph Lanes Add Value in CPU-Only Hybrid
Retrieval."*
