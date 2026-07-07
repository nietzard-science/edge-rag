# Technical Architecture

**Project:** Edge-RAG: Empirical Characterization of When Knowledge-Graph
Lanes Add Value in CPU-Only Hybrid Retrieval
**Document version:** 5.8 (last refresh 2026-06-18; §1/§6/§7 sync — MuSiQue
added as the third multi-hop modality-ablation dataset, the graph-rescue
contribution analysis (`graph_rescue_analysis.py`) and `modality_ablation.py`
documented in §7.2, the `hub_fanout_cap` / `search_budget_seconds` graph knobs
added to the §6 config inventory, and the `tools/` diagnostics relocation +
`repro-guard` CI job (§8.4) reflected. No reported number changed.
Previously, 5.7 (2026-06-13; §3.4 + §4.3 sync with
the storage/planner/verifier code review pass — the triple-confidence error
sentinel and sort-floor (§3.4); the settings-overridable `HUB_FANOUT_CAP` /
graph-search budget (§3.4); the unified `_bounded_retry` engine, the hoisted
retry-prompt constants now captured by `export_prompts.py`, and the
best-answer-prefers-substantive selection rule (§4.3); and the
`_ensure_nlp` config-driven SpaCy model load (§6). These are correctness/
reproducibility refactors — no science or reported number changed; the
metrics stand on 5.5. §6.1 + §8.4 (added in 5.6) document the container/CI
deployment surface — `OLLAMA_HOST` / `CONFIG_PATH` / `INDEX_DIR` env
overrides, the CPU-only Dockerfile + compose with the 2 GB/4-core envelope,
GitHub Actions, and the `demo` sub-command.) The central empirical
finding of the 5.5 release is a rigorously-ablated **negative result**: at
the 1.5 B edge scale, post-hoc agentic verification does not raise answer
EM over single-pass retrieval — the dominant failure (retrieval-miss) is
unreachable by answer-checking, and verifier and generator share failure
modes. The verifier's contribution is re-framed as calibration / selective
prediction (trust), not accuracy. The methodology, opt-in mechanisms, and
re-retrieval loop are documented in §4.4, §11.16, §11.17, and §12.2.

> This document describes the system as it stands at the paper-release
> milestone. Earlier versions tracked an incremental change-log; that
> log has been removed because it does not describe the artifact being
> evaluated. Where a design choice deviates from a textbook approach, the
> deviation is justified in §11 (Design Decisions) with academic
> citations only — no internal version markers.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Structure](#2-repository-structure)
3. [Data Layer — Artifact A](#3-data-layer--artifact-a)
   - 3.1 [Embeddings](#31-embeddings)
   - 3.2 [Chunking](#32-chunking)
   - 3.3 [Entity & Relation Extraction](#33-entity--relation-extraction)
   - 3.4 [Storage: Vectors + Graph](#34-storage-vectors--graph)
   - 3.5 [Hybrid Retriever](#35-hybrid-retriever)
   - 3.6 [Graph Quality](#36-graph-quality)
4. [Logic Layer — Artifact B](#4-logic-layer--artifact-b)
   - 4.1 [Planner (S_P)](#41-planner-s_p)
   - 4.2 [Navigator (S_N)](#42-navigator-s_n)
   - 4.3 [Verifier (S_V)](#43-verifier-s_v)
   - 4.4 [Pipeline Orchestrator (`AgentPipeline`)](#44-pipeline-orchestrator-agentpipeline)
   - 4.5 [Static Helper Namespace (`AgenticController`)](#45-static-helper-namespace-agenticcontroller)
5. [Ingestion: Decoupled Three-Phase Architecture](#5-ingestion-decoupled-three-phase-architecture)
6. [Configuration System](#6-configuration-system)
   - 6.1 [Environment-variable overrides (container / CI surface)](#61-environment-variable-overrides-container--ci-surface)
7. [Evaluation Framework — Artifact C](#7-evaluation-framework--artifact-c)
8. [Technology Stack](#8-technology-stack)
   - 8.4 [Containerization and continuous integration](#84-containerization-and-continuous-integration)
9. [Data Flows](#9-data-flows)
10. [Non-Functional Requirements](#10-non-functional-requirements)
11. [Design Decisions and Academic Grounding](#11-design-decisions-and-academic-grounding)
12. [Known Limitations and Future Work](#12-known-limitations-and-future-work)

---

## 1. System Overview

This system implements a hybrid Retrieval-Augmented Generation (RAG)
architecture optimised for deployment on edge devices with strict CPU and
memory budgets. The architecture was built to test a two-part hypothesis,
and the document is written to report what the evaluation found for each
part — **not** to advocate the original hope. The two parts dissociate:

1. **Retrieval hypothesis** — combining three retrieval modalities (dense
   vector, sparse BM25, knowledge-graph traversal) increases answer
   fidelity over any single modality, particularly for multi-hop questions
   requiring an unnamed intermediate referent. **Supported** for the
   retrieval-vs-none contribution (the only component effect that survives
   family-wise significance correction); the per-modality "hybrid-vs-dense"
   decomposition that the word *hybrid* strictly requires is evaluated
   separately (`modality_ablation.py`, §7).
2. **Agentic-verification hypothesis** — a three-agent Planner → Navigator
   → Verifier pipeline, with self-correction, raises answer EM over
   single-pass retrieval at the 1.5 B edge scale. **Refuted** for accuracy:
   §11.17 reports a rigorously-ablated negative result. The verifier's
   defensible contribution is re-framed as **trust** (calibration /
   selective prediction), not accuracy.

The reader should hold both outcomes in view from the outset: this is a
paper whose central agentic-verification claim is a *negative* result, and
every section below is written to that conclusion rather than around it.
Where an earlier draft phrased the pipeline as simply "increasing
fidelity", that phrasing has been corrected to the part-1/part-2
dissociation above.

The architecture is organised into three independently testable artifact
layers.

| Artifact | Layer | Responsibility |
|---|---|---|
| **A** | Data | Dual-index storage (LanceDB vectors + KuzuDB property graph), batched embeddings with persistent cache, hybrid retrieval via Reciprocal Rank Fusion of vector / BM25 / graph paths, optional cross-encoder reranking. |
| **B** | Logic | Three-agent reasoning pipeline: query classification and decomposition (S_P), pre-generative retrieval orchestration and filtering (S_N), pre-validation followed by quantised-SLM answer generation with optional self-correction (S_V). |
| **C** | Evaluation | Multi-dataset benchmarking (HotpotQA, 2WikiMultiHopQA, MuSiQue, StrategyQA), retrieval-only and end-to-end modes, per-pattern diagnostic JSONL, ablation suites for components and hyperparameters. The retrieval-modality ablation (§7.2) runs on the three multi-hop sets (HotpotQA / 2WikiMultiHopQA / MuSiQue); StrategyQA is used for the boolean-reasoning LLM-only probe. |

**End-to-end runtime path (one query).** A question enters
`AgentPipeline.process()` (FIFO result cache, SHA-256 keyed) and runs the
`S_P → S_N → S_V` chain; the companion `ARCHITECTURE_DIAGRAM.md` renders the
full wiring and query lifecycle. The current component summary:

| Stage | Component | What it does now |
|---|---|---|
| **S_P** | `Planner` (rule-based, < 10 ms, no LLM) | Classifies the query (six types) and decomposes multi-hop questions via SpaCy dependency-parse patterns (E relational-noun, F passive-agent, G/L relative-clause bridge, H chained-attribution) plus two closed-class pre-empts (I boolean conjunction, J implicit bridge). Loads its SpaCy model from `ingestion.spacy_model` via `_ensure_nlp` (§4.1). |
| **S_N** | `Navigator` | Dispatches each sub-query to the `HybridRetriever`, fuses with RRF (cross-source corroboration boost), optionally cross-encoder-reranks, then runs a six-filter chain (relevance / redundancy / contradiction / entity-overlap / entity-mention / context-shrinkage) + a per-anchor fairness cap that guarantees both comparison conjuncts survive the budget (§4.2). |
| **S_V** | `Verifier` | Pre-validation (entity-path, credibility *informational-only* by default, contradiction off by default), question-relevance reorder, quantised-SLM generation (`qwen2:1.5b`, temp 0), four keep-best-guarded bounded retries through one `_bounded_retry` engine, deterministic claim verification, and a calibrated confidence/abstention signal — best-answer selection always prefers a substantive answer over a disclaimer (§4.3). |
| **Data (A)** | `HybridRetriever` + `HybridStore` | Three retrieval lanes — dense vector (LanceDB, cosine ANN), sparse BM25, and bounded multi-hop graph traversal (KuzuDB) with hub suppression (`HUB_MENTION_CAP`/`HUB_FANOUT_CAP`), relation-typed triple-frequency confidence (cooccurs 0.25 / SVO 0.6 / REBEL 1.0, error → −1.0 sentinel that sorts last), and a per-call graph-search time budget (§3.4–§3.5). |

All retrieval knobs, the per-source RRF weights, and the cross-source boost
`β` are settings-overridable (`settings.yaml`); the latent dataclass defaults
are documented emergency fallbacks only (§6, §11.16.4).

**Edge-deployment constraints (binding throughout the design):**

The term *edge device* in this work refers to a single-host commodity
machine without server-grade GPU acceleration and without cloud
dependency at inference time. The concrete operating envelope used as
the design target throughout this paper is:

| Resource | Budget | Rationale |
|---|---|---|
| Host RAM | ≤ 16 GB total; system targets ≤ 8 GB working set | Covers laptops, mini-PCs (Intel NUC / Mac mini class), and industrial single-board computers (Jetson Orin Nano 8 GB, Raspberry Pi 5 8 GB). |
| Compute | CPU-only at inference; x86-64 or ARM64; no CUDA / ROCm requirement | GPU is available only for the GPU-bound extraction phase (Phase 2, run offline in Colab); production inference must run without it. |
| Disk | ≤ 10 GB for code + databases + caches per dataset | Fits a typical embedded device storage profile. |
| Per-query latency (target) | < 60 s end-to-end on the reference hardware | Matches the budget profiled in `latency_memory_profile.py`. |
| External network | Not required at evaluation time | Ollama runs locally on `localhost:11434`; no cloud API calls. |
| Reference hardware (development & evaluation) | Intel-class CPU, 16 GB RAM, Windows 11 / Linux | Hardware on which the reported numbers were produced. |

These limits are not aspirational — they constrain every component
design choice in §3 (embedded stores; no GPU at runtime; quantised
generation), §4 (small generation models, single-pass orchestration,
bounded self-correction), and §10 (memory profiling).

- Every persistence backend is *embedded*: no separate database server.
  LanceDB stores vectors as Apache Arrow files; KuzuDB stores the graph
  as memory-mapped column store files; SQLite stores the embedding cache.
- All language models are served locally via Ollama; no cloud API
  dependency at evaluation time.
- Generation uses 4-bit GGUF quantisation (llama.cpp backend) via Ollama.

### 1.1 Position relative to prior work

Three recent lines of work are the closest neighbours; each is cited
throughout this document where its mechanism is reused.

**vs. IRCoT** (Trivedi et al. 2023, ACL; arXiv:2212.10509) —
*Interleaving Retrieval with Chain-of-Thought.* IRCoT interleaves
free-form CoT reasoning with retrieval steps on large LLMs (GPT-3,
Flan-T5 XXL). This work adopts the *iterative bridge-grounding* idea
(step-N retrieved entities feed step-N+1 query, see §4.4 and
`AgentPipeline._iterative_navigate`) but replaces free-form CoT with a
**structured Planner that emits a finite, parse-derived hop graph**
(Patterns E/F/G/H/I/J) executable by a 1.5B-parameter quantised SLM.
Free-form CoT is not robust at this model size; the dependency-parse
backbone supplies the reasoning skeleton externally.

**vs. HippoRAG** (Gutiérrez et al. 2024, NeurIPS; arXiv:2405.14831) —
*Neurobiologically inspired long-term memory.* HippoRAG runs personalised
PageRank over a passage graph at query time using a GPT-3.5 OpenIE
extractor. This work shares the typed-vs-untyped relation-weighting
intuition (cooccurrence edges down-weighted, semantic edges full
weight; §11.6) and the use of named-entity bridges as retrieval
anchors, but (i) extracts relations *offline* with GLiNER + REBEL
(open-source, edge-feasible), (ii) replaces personalised PageRank with
**bounded multi-hop Cypher traversal capped at hop-3** (§3.5), and
(iii) does not require a cloud LLM at extraction or query time.

**vs. Self-Refine** (Madaan et al. 2023, NeurIPS) — *Iterative
refinement with self-feedback.* Self-Refine has the same LLM critique
and revise its own output. This work uses the same *single feedback
loop* in the Verifier (§4.3) but replaces self-feedback with
**deterministic entity-presence verification** against retrieved
context (an external signal, not the model's own opinion). The
self-correction round count is bounded to ≤ 1 by default; the
ablation in §11.9 / `agentic_ablation.py` quantifies its marginal
contribution.

**The combined contribution** of this work is: a *parse-grounded,
edge-feasible* instantiation of the IRCoT / HippoRAG / Self-Refine
family in which (a) each agentic step is implementable without a
large LLM, (b) every retrieval modality and every reasoning hop fits
within the edge envelope defined above, and (c) the multi-modal
retrieval is weighted by an empirically-justified RRF schedule
(§3.5, §11.2) rather than uniform fusion. The end-to-end
evaluation in Artifact C tests the hypothesis that this combination
exceeds vector-only and graph-only retrieval on bridge-heavy
multi-hop QA at the edge-device scale.

---

## 2. Repository Structure

```
edge-rag/
│
├── src/                              # Production source
│   ├── data_layer/                   # Artifact A
│   │   ├── __init__.py               # Public exports
│   │   ├── embeddings.py             # BatchedOllamaEmbeddings + SQLite cache
│   │   ├── chunking.py               # SpacySentenceChunker (primary), 4 alt
│   │   ├── coreference.py            # Optional Phase-1 pronoun resolver
│   │   ├── entity_extraction.py      # GLiNER NER + REBEL RE (local fallback)
│   │   ├── entity_types.py           # OntoNotes-5 / GLiNER label map
│   │   ├── svo_extraction.py         # SpaCy dep-parse Subject-Verb-Object triples
│   │   ├── storage.py                # HybridStore, VectorStoreAdapter, KuzuGraphStore
│   │   ├── hybrid_retriever.py       # HybridRetriever, RRFFusion, BM25
│   │   ├── graph_quality.py          # canonical_form, cleanup, cooccurrence,
│   │   │                             # entity-linking, baseline + invariants
│   │   ├── ingestion.py              # DocumentIngestionPipeline (data-layer)
│   │   └── conftest.py
│   │
│   ├── logic_layer/                  # Artifact B
│   │   ├── __init__.py               # Public exports
│   │   ├── _config.py                # ControllerConfig (Navigator config)
│   │   ├── _settings_loader.py       # YAML loader + 37-key reproducibility validator
│   │   ├── planner.py                # S_P: classifier + entity extractor + decomposer
│   │   ├── navigator.py              # S_N: hybrid retrieval orchestration + filters
│   │   ├── verifier.py               # S_V: pre-validation + generation + correction
│   │   ├── controller.py             # Static helpers for bridge entities + query rewriting
│   │   └── conftest.py
│   │
│   ├── pipeline/                     # Orchestration
│   │   ├── __init__.py
│   │   ├── agent_pipeline.py         # AgentPipeline: production S_P → S_N → S_V driver
│   │   ├── ingestion_pipeline.py     # End-to-end ingestion workflow
│   │   └── conftest.py
│   │
│   ├── thesis_evaluations/           # Artifact C (current evaluation suite)
│   │   ├── __init__.py
│   │   ├── benchmark_datasets.py     # CLI: ingest / evaluate / ablation
│   │   ├── agentic_ablation.py       # Component-wise ablation (LLM-only → full)
│   │   ├── chunking_ablation.py      # Chunking-hyperparameter sweep
│   │   ├── quantization_sweep.py     # Cross-model evaluation
│   │   ├── latency_memory_profile.py # Per-stage timing + peak RSS
│   │   ├── verifier_cache_build.py   # Verifier-only harness: freeze retrieval (§11.17)
│   │   ├── verifier_failure_taxonomy.py # Bucket wrong answers by failure mode
│   │   └── thesis_results_aggregator.py
│   │
│   └── utils.py
│
├── config/
│   └── settings.yaml                 # Single source of truth (~860 lines)
│
├── docs/
│   └── figures/                      # Generated paper figures (e.g. graph_preview.png)
│
├── data/                             # Runtime data (gitignored)
│   └── hotpotqa/                     # Per-dataset: vector/, graph/,
│                                     # chunks_export.json, extraction_results.json,
│                                     # questions.json
│
├── cache/                            # SQLite embedding caches (gitignored)
├── evaluation_results/               # Per-question JSONL + summary tables
├── logs/                             # Console/diagnostic log output (gitignored)
│
├── colab_extraction.py               # Phase 2 (GPU): GLiNER + REBEL on Drive-mounted chunks
├── local_importingestion.py          # Phase 3: import Phase-1+2 outputs into stores
├── demo_app.py                       # Single-query live demo (S_P→S_N→S_V + edge-envelope panel)
│
├── tools/                            # Read-only diagnostic entry points (not on the runtime path)
│   ├── diagnose_verbose.py           # Full pipeline trace with gold-tracking hooks
│   ├── diagnose_ingestion.py         # Ingestion-side consistency probe
│   ├── diagnose_graph_baseline.py    # Read-only graph-quality reporter
│   └── diagnose_ablation.py          # Post-hoc analyser of ablation JSONL traces
│
├── test_system/                      # Test suite (739 tests collected)
│   ├── conftest.py
│   ├── fixtures/                     # Gold NER + forbidden-source-terms blocklist
│   ├── test_*.py                     # Module-scoped tests
│   ├── graph_inspect.py              # KuzuDB schema/statistics inspector (dev utility)
│   └── graph_3d.py                   # Knowledge-graph figure generator → docs/figures/
│
├── TECHNICAL_ARCHITECTURE.md         # This document
├── REPRODUCE.md                      # Reproduction protocol
├── pytest.ini                        # CI markers: slow / nightly / llm / integration
├── requirements.txt                  # Library constraints
└── requirements_frozen.txt           # Pinned reproducibility set
```

---

## 3. Data Layer — Artifact A

The data layer is stateless with respect to any individual query. All
persistent state lives on disk: LanceDB directories, KuzuDB column
stores, SQLite cache files. Modules in this layer are independently
testable and do not import the logic layer.

### 3.1 Embeddings

**File:** [`src/data_layer/embeddings.py`](src/data_layer/embeddings.py)

A LangChain-compatible client around Ollama's `/api/embed` endpoint,
augmented with two performance-critical primitives:
content-addressable persistent caching and request batching.

**Cache schema (SQLite, `cache/<dataset>_embeddings.db`):**

```
embeddings (
  text_hash    TEXT PRIMARY KEY,    -- SHA-256(text)
  text_content TEXT NOT NULL,
  embedding    BLOB NOT NULL,       -- JSON-serialised float list
  model_name   TEXT NOT NULL,
  access_count INTEGER DEFAULT 0,
  created_at   TIMESTAMP
)
INDEX idx_model_hash ON (model_name, text_hash)
```

The composite key on `(model_name, text_hash)` automatically invalidates
the cache when the embedding model changes; the SHA-256 key collapses
duplicate inputs to a single embedding.

**Public client:**

```python
class BatchedOllamaEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url:   str = "http://localhost:11434",
        batch_size: int = 64,
        cache_path: Path = Path("./cache/embeddings.db"),
        device:     str = "cpu",
        timeout:    int = 60,
    ): ...

    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...
    def get_metrics(self) -> Dict[str, Any]: ...
    def clear_cache(self) -> None: ...
```

**Batching pipeline** (single `embed_documents(texts)` call):

1. Compute SHA-256 hashes for all inputs; issue **one** SQL batch lookup
   for cache hits.
2. Identify cache misses (delta of input list vs. lookup hits).
3. Issue Ollama API calls in chunks of `batch_size`; on partial failure
   the batch is retried at `batch_size // 2` until single-text.
4. Write back fresh embeddings to the cache.
5. Assemble outputs in original input order.

The `EmbeddingMetrics` dataclass tracks `total_texts`, `cache_hits`,
`cache_misses`, `batch_count`, `total_time_ms`; the derived properties
`cache_hit_rate` (percent) and `avg_time_per_text_ms` are reported in
every evaluation summary (see §7).

The factory `create_embeddings(cfg)` constructs the client from a
`settings.yaml` dict so production and tests share one construction path.

### 3.2 Chunking

**File:** [`src/data_layer/chunking.py`](src/data_layer/chunking.py)

The chunking module exposes five chunker classes; the production
pipeline uses **`SpacySentenceChunker`** with a 3-sentence sliding
window and 1-sentence overlap. The alternative strategies
(`SemanticChunker`, `RecursiveChunker`, `FixedSizeChunker`,
`SentenceChunker`) are retained for the chunking ablation in §7.

```python
class SpacySentenceChunker:
    def __init__(
        self,
        sentences_per_chunk: int = 3,        # settings: ingestion.sentences_per_chunk
        sentence_overlap:    int = 1,        # settings: ingestion.sentence_overlap
        min_chunk_chars:     int = 50,
        max_chunk_chars:     int = 2000,
        spacy_model:         str = "en_core_web_sm",
        entity_aware:        bool = False,    # reserved; not yet implemented
    ): ...
    def chunk_text(self, text: str, source_doc: str = "") -> List[SentenceChunk]: ...
```

Each `SentenceChunk` carries `text`, `sentence_count`, `position`,
`start_char`, `end_char`, and `source_doc`. **Chunk IDs are
deterministic SHA-256 hashes of `source_doc:position:text[:50]`** so
re-ingesting the same source produces byte-identical IDs and is
therefore idempotent at the graph-node level.

A module-level `SpacyModelCache` singleton amortises the SpaCy load
cost across multiple chunker instances.

### 3.3 Entity & Relation Extraction

**File:** [`src/data_layer/entity_extraction.py`](src/data_layer/entity_extraction.py)
**GPU phase:** `colab_extraction.py` (production); local module is the
fallback path.

#### Named Entity Recognition

**Model:** GLiNER (`urchade/gliner_small-v2.1`) — zero-shot span-based
NER (Zaratiana et al. 2023).

The evaluation uses a **fixed OntoNotes-5-aligned prompt set**:
9 GLiNER prompts collapsing to 8 canonical types via the
`GLINER_LABEL_MAP` in [`src/data_layer/entity_types.py`](src/data_layer/entity_types.py).

| Prompt | Canonical type |
|---|---|
| person | PERSON |
| organization | ORGANIZATION |
| location, city, country | LOCATION / GPE |
| date | DATE |
| event | EVENT |
| work of art | WORK_OF_ART |
| product | PRODUCT |

The multi-prompt expansion (city + country alongside location) gives
GLiNER higher recall than a single abstract label (Zaratiana et al.
2023 §4.2); all prompts collapse to OntoNotes-5 canonical types
(Weischedel et al. 2013, LDC2013T19) so the graph schema and downstream
filters operate on a single label set.

#### Relation Extraction

**Model:** REBEL (`Babelscape/rebel-large`) — seq2seq triplet generation
(Cabot & Navigli 2021, EMNLP 2021 Findings).

The Colab phase invokes REBEL with `num_beams=5` and
**log-probability-calibrated per-triplet confidence** (mean softmax of
the triplet's decoded tokens). The prior implementation emitted a
constant 0.5 sentinel; the calibrated variant is what is consumed by
the storage layer's `_triple_frequency_confidence` (see §3.4).

#### Subject-Verb-Object extraction

[`src/data_layer/svo_extraction.py`](src/data_layer/svo_extraction.py)
extracts narrative `(subject, verb, object)` triples from the SpaCy
dependency parse. SVO complements REBEL by recovering narrative
predicates ("X directed Y", "X founded Y") that REBEL's Wikipedia-
infobox training does not cover. Both endpoints must resolve to a known
entity via `canonical_form` (no free-text endpoints).

### 3.4 Storage: Vectors + Graph

**File:** [`src/data_layer/storage.py`](src/data_layer/storage.py)

The storage layer is a facade (`HybridStore`) over two embedded
databases plus an optional embedding cache.

#### LanceDB vector store (`VectorStoreAdapter`)

Exact (brute-force) cosine search on dense embeddings — no ANN index is
built (no `create_index` call anywhere in the ingestion path), so every
query scans the full table. At the edge-corpus scale (≈9 k chunks per
dataset) the exact scan is fast enough and avoids the recall loss and
build-time cost of an approximate index; LanceDB *supports* HNSW/IVF-Flat
indexes (Malkov & Yashunin 2018) should the corpus grow, but the reported
system does not enable one. Distances are converted to similarities in
`[0, 1]` so downstream consumers apply a uniform threshold:

- cosine: `sim = max(0, 1 - dist)`
- L2:     `sim = 1 / (1 + dist)`

`vector_search` over-fetches by `overfetch_factor` (default 3) before
threshold filtering, ensuring threshold-pruning does not silently drop
top-k.

#### KuzuDB property graph (`KuzuGraphStore`)

Native Cypher-based multi-hop traversal (Feng et al. 2023, CIDR 2023).
Columnar / vectorised execution and out-of-core processing via
memory-mapped files keep the graph operable on edge hardware.

**Schema:**

```
NODE TABLE DocumentChunk (chunk_id PK, text, page_number, chunk_index, source_file)
NODE TABLE SourceDocument (doc_id PK, filename, total_pages)
NODE TABLE Entity (entity_id PK, name, type, confidence)

REL TABLE FROM_SOURCE (DocumentChunk → SourceDocument)
REL TABLE NEXT_CHUNK   (DocumentChunk → DocumentChunk)
REL TABLE MENTIONS     (DocumentChunk → Entity)
REL TABLE RELATED_TO   (Entity → Entity)
                       attributes: relation_type, confidence, source_chunks
```

#### Multi-hop entity search

`find_chunks_by_entity_multihop(entity_name, max_results, enable_hop3, max_hops)`
returns chunks reachable from an entity through 0–3 hops of `RELATED_TO`
edges. The hop ladder is gated:

- **Hop 0** — direct `MENTIONS`. Three-stage cascade: exact name match →
  `CONTAINS $name` → `name IN $sub-phrases` (substring-aware alias).
- **Hop 2** — one `RELATED_TO` bridge. Capped fan-out from the matched
  entity (`KuzuGraphStore.HUB_FANOUT_CAP`, default 5) and hub-target
  exclusion via a cached set of high-degree entities (mention-degree >
  `HUB_MENTION_CAP`, default 280 ≈ 3 % of HotpotQA corpus). Both caps are
  class attributes the `create_hybrid_retriever` factory overrides
  per-instance from `settings.yaml` (`graph.hub_mention_cap`;
  `graph.hub_fanout_cap` when present) — absent keys keep the class default.
- **Hop 3** — two `RELATED_TO` bridges. Disabled by default
  (`graph.enable_hop3: false`); when enabled, excludes `cooccurs` edges
  because two-hop noise compounds.

#### Triple-frequency confidence

Each retrieved bridge chunk carries a confidence
`(hop_distance, triple_confidence)` where `triple_confidence` is
**not** REBEL's per-triplet log-prob (which is per-edge, not per-bridge)
but a corpus-support score (DeepDive, Niu et al. 2012; Knowledge Vault,
Dong et al. 2014):

```
triple_confidence = relation_type_weight ×
                    min(1.0, log(1 + n_supporting_chunks) / log(10))
```

`relation_type_weight` distinguishes three relation origins:

| Origin | Detection | Weight |
|---|---|---|
| `cooccurs` (statistical co-mention) | explicit `relation_type == "cooccurs"` | **0.25** |
| SVO (dependency-parse verb lemma) | single-token verb lemma | **0.6** |
| REBEL (Wikidata-style predicate) | multi-token predicate or `_`/space in the string | **1.0** |

Origin classification is performed by `KuzuGraphStore._classified_weight()`.
The result is that any semantic REBEL bridge with even one supporting
chunk (conf ≥ 0.30) outranks any cooccurs bridge regardless of
support (max conf ≤ 0.25 with the 0.25 weight).

**Error contract.** A *missing entity name* returns a neutral `0.5` prior
(a partially-specified bridge is neither boosted nor buried). A *support-
query failure* returns the distinguishable sentinel `_TRIPLE_CONF_ERROR =
-1.0` (outside the valid `(0, 1]` range) and logs a WARNING — so an error
can never masquerade as an honest confidence, the exact failure the
corpus-support score was introduced to remove. `graph_search` floors the
sort key at `0.0`, so an un-scorable bridge sorts to the bottom rather than,
via the `-confidence` negation, to the top.

### 3.5 Hybrid Retriever

**File:** [`src/data_layer/hybrid_retriever.py`](src/data_layer/hybrid_retriever.py)

`HybridRetriever.retrieve(query)` orchestrates three retrieval paths and
fuses them via Reciprocal Rank Fusion (Cormack et al. 2009).

```
        ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
query → │  Dense vector  │    │   BM25 sparse  │    │ Graph multi-hop│
        │  (LanceDB)     │    │  (rank_bm25)   │    │  (KuzuDB)      │
        └───────┬────────┘    └───────┬────────┘    └───────┬────────┘
                │ rank_v                │ rank_b              │ rank_g
                └────────────┬──────────┴──────────┬──────────┘
                             ▼                     ▼
                       ┌─────────────────────────────────┐
                       │  RRFFusion (Cormack 2009)       │
                       │  RRF(d) = Σ w_i / (k + rank_i)  │
                       └──────────────┬──────────────────┘
                                      ▼
                       Top-K fused results to Navigator (S_N)
```

**Per-path weights** (`vector_weight`, `bm25_weight`, `graph_weight`)
default to `1.0` (vanilla RRF). Unequal weights enable per-source
ablation without code changes.

The retriever also exposes:

- An entity-aware hop-2 reranker: per-chunk metadata records which
  sub-query the chunk ranked best for (`_best_sub_query`); the
  cross-encoder reranker (see §4.2) scores against that sub-query,
  not against the surface query, so a hop-2 chunk semantically distant
  from the surface form is not demoted.
- Optional cross-encoder reranker stage (`enable_reranker: true`) using
  `cross-encoder/ms-marco-MiniLM-L-6-v2` (Reimers & Gurevych 2019).

#### Query-side entity extraction and normalization

Query-time entities (used both as graph-search anchors and as
entity-mention filter tokens in §4.2) are produced by
`ImprovedQueryEntityExtractor`, which runs the same GLiNER model as
ingestion (with SpaCy then regex fallbacks). Because the two consumers
trust these spans directly, a deterministic, query-side-only
normalization layer repairs span boundaries and gates non-discriminative
spans **before** they reach retrieval:

- **Span-boundary normalization.** (a) A leading auxiliary/copula verb
  absorbed into a span by the tagger is stripped (`_strip_leading_function_word`,
  closed class `is/are/was/were/do/does/did` only — wh-words and articles
  are deliberately *not* stripped, since legitimate titles begin with
  them, e.g. "Who Framed Roger Rabbit", and article handling is already
  type-aware in the shared normaliser). (b) Overlapping fragments of one
  span are merged to the maximal span using GLiNER character offsets
  (`_dedup_overlapping_spans`), so "Hook-Handed Man" survives intact
  rather than fragmenting. (c) A 4-digit year *interior* to a span
  (alphabetic tokens on both sides) is split off as a separate temporal
  constraint (`_strip_embedded_year`); leading/trailing years are kept,
  as they are often part of a title.
- **Discriminativeness gate** (`_is_junk_entity`). A span is rejected as
  an anchor/filter token if it is a stop-listed token, a generic question
  stem, or a *pure temporal/measure phrase* — all tokens are digits,
  measure nouns, or quantity adjectives with no capitalised proper-noun
  token (e.g. "7 consecutive seasons", "25 laps"). Such phrases match no
  graph node and, used as a filter token, retain noise.

Both passes are query-side only; the ingestion path is untouched, so
query/ingestion entity-ID consistency is preserved.

### 3.6 Graph Quality

**File:** [`src/data_layer/graph_quality.py`](src/data_layer/graph_quality.py)

Operations on the populated KuzuDB graph used during ingestion and on
diagnostic invocations:

| Function | Role |
|---|---|
| `canonical_form(name)` | NFKC normalisation + lowercase + suffix strip ("Jr.", "Sr."); merge key for duplicate detection. |
| `compute_graph_baseline(store)` | Reports `chunks`, `entities`, `mentions`, `relations`, density, isolated-entity rate, duplicate clusters, top hubs. |
| `assert_graph_invariants(metrics, strict)` | Checks `isolated_rate < 5 %`, `duplicate_rate < 2 %`, `relations_per_chunk ≥ 5`. |
| `build_cooccurrence_edges(store, results, name_to_id)` | One `RELATED_TO {relation_type='cooccurs'}` per entity pair sharing a chunk. |
| `cleanup_graph(...)` | Stop-list / orphan / hub / duplicate cleanup. |
| `link_entities_by_embedding(...)` | Embedding-based alias resolution within type buckets. Disabled by default — see §3.6.1. |
| `_redirect_entity_edges_bulk(store, redirect_map)` | Bulk re-pointing of all MENTIONS / RELATED_TO edges from a set of old entities to their merge targets in two batched CREATE queries (one per edge type). Replaces the per-edge MERGE used previously, which scaled with entity degree on KuzuDB. |
| `drop_subsumed_cooccurrence_edges(...)` | Removes cooccurs edges that already have a semantic relation. |
| `drop_isolated_entities(...)` | Drops entities with `MENTIONS` but zero `RELATED_TO` post-linking. |

#### 3.6.1 Embedding-based entity linking — empirical evaluation

`link_entities_by_embedding` clusters entities within a type bucket
(PERSON, ORG, LOCATION, GPE, …) by L2-normalised cosine similarity over
their nomic-embed-text vectors and merges clusters whose similarity
exceeds a threshold (`--linking-threshold`, default `0.92`). The linker
also supports per-bucket resume via `done_buckets` + `on_bucket_done`,
so a mid-phase crash does not lose finished buckets.

A threshold analysis over a 2000-entity sample per type bucket measured
the following merge rates with nomic-embed-text:

| Type | n | threshold 0.98 | threshold 0.99 | Largest cluster (0.98) |
|---|---|---|---|---|
| PERSON | 2 000 | 90.7 % | 90.0 % | 1 395 |
| LOCATION | 2 000 | 73.7 % | 72.5 % | 719 |
| GPE | 2 000 | 93.9 % | 93.3 % | 1 201 |

Sample cluster at threshold 0.98 (PERSON): `Uncle Albert` absorbs
`Julia Calvo`, `Mark Pickerel`, `David Ramsey`, `Vijay Yesudas`, and
1391 others — unrelated individuals. Threshold 0.99 yields nearly
identical merge rates and cluster sizes, indicating the embedding
model has effectively no discriminative range above 0.95 on entity
names. This score-compression behaviour is consistent with the
characterisation of nomic-embed-text in §10.

**Decision.** Embedding-based entity linking is disabled by default
(`--no-entity-linking`) for paper-release ingest. Alias resolution
reduces to exact-match deduplication via `canonical_form`. The
bulk-redirect + per-bucket-checkpoint infrastructure is retained as a
zero-cost engineering scaffold should a discriminative replacement
embedder be substituted in future work.

---

## 4. Logic Layer — Artifact B

The logic layer implements the three-agent reasoning architecture. Each
agent has a single public entry point and minimal cross-agent state.

### 4.1 Planner (S_P)

**File:** [`src/logic_layer/planner.py`](src/logic_layer/planner.py)

The Planner transforms a natural-language question into a structured
`RetrievalPlan` consisting of: a classified query type, an ordered hop
sequence of sub-queries, the named entities relevant to retrieval, and
any temporal or comparative constraints.

#### Three stages

1. **Query classification** — rule-based via SpaCy Matcher and
   compiled regex patterns over closed-class English function words
   (Honnibal & Montani 2017). Output labels: `SINGLE_HOP`, `MULTI_HOP`,
   `COMPARISON`, `TEMPORAL`, `AGGREGATE`, `INTERSECTION`.

   Two **deterministic pre-empts** run before the four-phase scoring
   classifier when closed-class English markers identify the
   construction unambiguously:
   - Distributive predication with floating "both"
     (`<aux> X and Y both <P>`) → `COMPARISON`. Quirk et al. 1985 §10.49.
   - Anaphoric introduction with "another" (`X and another <N> that …`)
     → `MULTI_HOP`. Karttunen 1976.

   Every plan records the pre-empt (if any) on the
   `RetrievalPlan.classifier_preempt` field so per-question
   diagnostics can audit false-positive rates.

2. **Entity extraction** — SpaCy NER restricted to OntoNotes-5 labels
   (Weischedel et al. 2013); per-label confidence estimate via
   `_LABEL_CONFIDENCE` mapping. Bridge-entity detection (for
   `MULTI_HOP` queries) marks entities that appear in
   bridge-connector context (`relcl` dep label or in-between two anchor
   entities). Generic NORP demonyms, DATE values, and other
   class-referring labels are excluded from bridge candidacy because
   they would steer retrieval toward high-degree hub nodes rather than
   specific bridging referents (West & Leskovec 2012).

3. **Plan generation** — Hop-sequence decomposition. Two generalisable
   mechanisms plus a baseline:

   **Mechanism A — Dependency-parse decomposers.** Four English
   constructions are recognised structurally via SpaCy dependency
   labels. Each recogniser is gated by a parse-confidence check
   requiring the relevant anchor to overlap a detected NER span.

   - **Pattern G** — Relative-clause bridge (Quirk et al. 1985 §17.7-15).
     "The [noun] in which [Entity] …" or "the [role] who [verb]
     [Entity]". Keys on the `relcl` dep label; two forms cover the
     relative-pronoun-subject case (form1) and the relative-pronoun-
     object case (form2).
   - **Pattern H** — Chained attribution (Levin 1993). Passive ROOT
     with `agent` by-phrase whose object is an indefinite pronoun,
     plus an `acl` clause on the subject anchored to a named entity.
     The attribution-clause head must come from a small closed class
     of derivation/depiction verbs (`_ATTRIBUTION_ACL_VERBS`).
   - **Pattern E** — Relational-noun + of-PP complement (Partee 1995;
     Barker 1995). A noun whose dependency structure contains a
     `prep("of")` child whose `pobj` is a named entity. **Generalises
     to any noun** for which the parser produces this structure; no
     role enumeration. Implemented by `_find_relational_noun_bridge`.
   - **Pattern F** — Passive-agent voice transformation
     (Bresnan 1982; Quirk et al. 1985 §3.65-71). A verb with
     `auxpass + nsubjpass + agent` children. Past-participle →
     infinitive transformation uses SpaCy's morphological lemmatiser,
     so the recogniser is **vocabulary-independent**. Implemented by
     `_find_passive_agent_bridge`. **Subject guards:** Pattern F is
     skipped when the extracted passive subject is itself an
     interrogative noun phrase (`_PASSIVE_F_INTERROGATIVE_SUBJ_RE`:
     leading what/which/whose) or a bare pronoun
     (`_PASSIVE_F_BARE_PRONOUN_SUBJ`: that/this/it/who/…). The template
     "Who {verb} {subject}?" with such a subject yields a
     self-referential sub-query ("Who hold what government position?")
     or a context-free one ("Who form that?"); both guards fall through
     to the connector-split baseline instead.

   **Mechanism B — Closed-class lexical pre-empts.** As above
   (Patterns I and J), routed before the scoring classifier.

   **Baseline — Connector-split decomposition.** The methodology-
   described baseline (Khattab et al. 2022, DSP). The query is split
   at bridge connectors ("that", "which", "who", "of the") and
   fragments are re-ordered so the bridge sub-query precedes the
   final sub-query. A 2-hop cap collapses spurious 3-part splits
   whose middle parts contain no named entity.

   **Mechanism C — Entity-free definite-description bridge.** When a
   `MULTI_HOP` query names no anchor entity (NER returns nothing
   seedable), the bridge referent is often given by *description*
   rather than by name ("the only player … to have a 0.300 batting
   average for 7 consecutive seasons"). `_find_entity_free_description`
   selects the longest object/complement noun phrase (pobj/dobj/attr
   head) that is entity-free and carries ≥ 2 description-modifier
   dependents (a *definite description*; Russell 1905; Strawson 1950)
   and emits its subtree text as the hop-0 retrieval query. This
   replaces what was previously a silent degrade-to-single-hop.

   **Classifier-abstention override (A4).** The classifier returns
   `SINGLE_HOP` with exactly `classifier_fallback_confidence` (0.5)
   *only* when no pattern scored (its documented no-signal sentinel;
   any real match yields ≥ 0.6). In that abstention case — and only
   then — `_attribute_over_entity_signal` consults the dependency parse:
   if the question asks for a wh-determined attribute of a thing related
   to a named entity (e.g. "what class of instrument does X play?"),
   the type is re-routed to `MULTI_HOP` so the entity-seeded 2-hop
   decomposition runs. The gate cannot override a classification that
   had positive evidence, so it cannot regress confident single-hop
   questions.

   **Never-collapse contract.** `_decompose_multi_hop` enforces a logged
   invariant: a `MULTI_HOP` classification must never silently emit a
   single sub-query; the only permitted single-sub-query output is the
   explicitly-marked degrade path below.

   **Failure modes**, explicit and surfaced in `matched_pattern`:
   - `structural_descriptive_2hop`: Mechanism C fired — entity-free
     definite-description bridge.
   - `fallback_generic_2hop`: classified `MULTI_HOP`, no mechanism
     applied, seed entity available. Emits "Who or what is X?" as
     hop-0 and the original query as hop-1.
   - `fallback_degraded_to_single_hop`: classified `MULTI_HOP`, no
     mechanism applied, no entity *and* no usable description. Logged at
     WARNING — the only sanctioned single-sub-query output.

#### Per-pattern diagnostics

Every `RetrievalPlan` records `matched_pattern` on the plan dataclass
and surfaces it through `to_dict()` into the per-question evaluation
JSONL. This enables hit-rate and SF-F1-conditional analysis per
pattern without parsing logs:

```bash
# Per-pattern hit count
jq -r '.matched_pattern' results.jsonl | sort | uniq -c | sort -rn

# Mean SF-F1 conditional on pattern
jq -r '"\(.matched_pattern)\t\(.sf_f1)"' results.jsonl | \
  awk -F'\t' '{s[$1]+=$2; n[$1]++} END {for (k in s) print k, s[k]/n[k], n[k]}'
```

#### Comparison decomposer

`_decompose_comparison` routes through (in order):

1. Pattern I — boolean conjunction `<aux> X and Y both P` → parallel
   yes/no.
2. Select-between-two — disjunction of two NER entities joined by
   "or"; the disjunction is detected via NER span positions, not
   surface regex.
3. `_ATTR_MAP` rewrites — closed-class English attribute nouns
   (nationality, birthplace, profession, genre, age, country,
   religion) → per-entity factual-lookup templates ("What is the
   nationality of X?").
4. Generic per-entity predicate template — used when no
   attribute-rewrite applies.

### 4.2 Navigator (S_N)

**File:** [`src/logic_layer/navigator.py`](src/logic_layer/navigator.py)

The Navigator executes the `RetrievalPlan` produced by S_P and delivers
a filtered, ranked context list to S_V.

```
                 ┌─────────────────────────────────────────────┐
RetrievalPlan ─→ │           NAVIGATOR (S_N)                    │
sub-queries ───→ │                                              │
                 │  1. Hybrid retrieval per sub-query           │
                 │  2. Reciprocal Rank Fusion (cross-sub-query) │
                 │     + cross-source corroboration boost       │
                 │  3. Pre-generative filter chain:             │
                 │     - Relevance filter                       │
                 │     - Redundancy (Jaccard) filter            │
                 │     - Contradiction filter (numeric heur.)   │
                 │     - Entity-overlap pruning                 │
                 │     - Entity-mention filter (top-K immune)   │
                 │     - Context shrinkage                      │
                 │  4. Cross-encoder reranking (Stage 2.5)      │
                 └──────────────┬───────────────────────────────┘
                                ▼
                  filtered_context, retrieval_methods → S_V
```

**Filter chain ordering rationale.** The six filters are ordered by
*increasing computational cost and decreasing reversibility*, so cheap
deterministic operations narrow the candidate set before expensive or
lossy ones run.

1. **Relevance filter** (lexical overlap) — cheapest, deterministic;
   removes manifest off-topic chunks first so subsequent filters
   operate on a relevance-aligned pool.
2. **Redundancy (Jaccard) filter** — pairwise n-gram comparison;
   removes near-duplicates *before* entity reasoning so duplicate
   chunks do not double-vote downstream.
3. **Contradiction filter** (numeric heuristic, context-aware) —
   removes pairs with mutually-exclusive numeric claims (year vs.
   count classification). Runs after redundancy so a
   single duplicate cluster cannot dominate the contradiction
   evidence count.
4. **Entity-overlap pruning** — drops chunks whose entity set is a
   strict subset of a higher-ranked chunk's entity set; preserves
   broader-coverage chunks.
5. **Entity-mention filter** (tiered, top-K RRF immune, survivor floor)
   — assigns each chunk a tier: tier 0 mentions a *specific* query
   entity (multi-word, or a distinctive single token ≥ 8 chars), tier 1
   mentions only a generic entity or has strong query content-word
   overlap, tier 2 mentions neither. Tier-2 chunks are dropped, with two
   safeguards: the **top-2 RRF chunks are immune** (the implicit-bridge
   carve-out), and a **survivor floor** guarantees that when a full
   candidate set (≥ 5 chunks) was supplied the filter never reduces it
   below 5 — if matching kept fewer, it tops up with the highest-RRF
   dropped chunks. The floor engages only for full sets, so small inputs
   are still filtered normally (it does not force-keep noise in a 2–3
   chunk set). When the query yields *no* usable entities the filter no
   longer silently passes everything: it logs a structural warning and
   sets `_entity_filter_skipped`, then relies on RRF/reranker ranking —
   making the no-gating case observable rather than silent.
6. **Context shrinkage** — last, because it commits to a final budget;
   any filter that runs after this would operate on an already-truncated
   pool.

The cross-encoder reranker (Stage 2.5) runs *after* the chain because
it is the only step with a per-pair neural cost (~2 ms × K on CPU); it
must operate on a small, pre-cleaned pool to stay within the latency
budget defined in §1. The ordering is empirically defended by
`agentic_ablation.py` row-3 / row-4 deltas.

**Cross-encoder reranker (Stage 2.5).** When `enable_reranker: true`,
the top-K candidates after RRF are rescored with
`cross-encoder/ms-marco-MiniLM-L-6-v2` (Reimers & Gurevych 2019). The
key implementation detail: **the reranker scores
`(_best_sub_query, chunk)`, not `(surface_query, chunk)`**, so bridge
chunks semantically distant from the surface form are not demoted.

**Top-K RRF immunity.** The entity-mention filter keeps the top-2 RRF
chunks regardless of entity-mention status. This guards against
over-aggressive entity filtering when the answer-bearing chunk is an
implicit bridge target whose surface form does not contain the planned
entity name.

**Retrieval provenance.** The Navigator emits per-chunk metadata
including which retrieval method(s) produced each chunk
(`vector`/`graph`/`bm25`/`hybrid`). This metadata is used by the
Verifier's credibility score (see §4.3) — graph-retrieved chunks
receive a higher provenance weight than vector-only chunks.

**Iterative multi-hop.** When the plan's `hop_sequence` has dependent
hops (`depends_on` non-empty), the Navigator runs hops **sequentially**
rather than in parallel: bridge entities resolved at hop *i* are
injected into the sub-query of hop *i+1* via
`AgenticController._rewrite_hop_query_with_bridges` (see §4.5). This
implements the IRCoT (Trivedi et al. 2023) and HippoRAG
(Gutiérrez et al. 2024) pattern of feeding step-N retrieved entities
into step-(N+1) queries.

### 4.3 Verifier (S_V)

**File:** [`src/logic_layer/verifier.py`](src/logic_layer/verifier.py)

S_V is the final stage of the pipeline and implements two pre-generation
validation checks, four query-type-specialised prompt templates, a
quantised-SLM generation call, and an optional self-correction loop.

#### Pre-generation validation (default ON)

1. **Entity-Path Validation** — verifies that retrieved chunks cover
   the query entities. When a graph store is available,
   `find_chunks_by_entity_multihop` is used; otherwise falls back to
   substring matching. Coverage below
   `entity_coverage_threshold` (default 0.34) flags the plan as
   `INSUFFICIENT_EVIDENCE`, triggering a specialised prompt.
2. **Source Credibility Scoring** — weighted combination of three
   signals (Dong et al. 2014 KDD multi-source fusion):
   - Cross-reference corroboration (weight 0.4): the chunk shares a
     key phrase or an entity name token with another chunk.
   - Named-entity frequency (weight 0.3): chunk's SpaCy NER density,
     normalised.
   - Retrieval provenance (weight 0.3): graph-retrieved chunks score
     1.0; vector/BM25-only chunks score `provenance_baseline` = 0.5.

   The weights are documented as a deliberate inspection-time choice;
   the paper reports one ablation row (without credibility scoring)
   rather than a weight sweep.

   **Non-destructive by default (`credibility_filter_drop: false`,
   2026-05-29).** Credibility scoring is **informational only** — the
   scores are computed and logged but no chunk is evicted. Set membership
   stays owned by the Navigator RRF order and the downstream `max_docs`
   cap. The legacy behaviour (drop chunks below `min_credibility_score`)
   was a silent gold-eviction path: a top-RRF answer chunk with a low
   credibility score was removed before the LLM saw it (observed; §11.17).
   The drop is retained as an opt-in ablation toggle only.

An **ablation-only** check (default OFF, `enable_contradiction_detection: false`):
Verifier-side NLI contradiction detection. The Navigator's
numeric-divergence filter already runs on the same context and is
enabled by default; the Verifier-side NLI check requires a 270 MB
cross-encoder download and is therefore retained only as a
research-mode toggle for ablation studies.

#### Context ordering and budgeting

Before the prompt is built, the surviving context is re-ordered and
truncated so the answer-bearing chunk reaches the small LLM early and
intact (small models attend poorly to later/long context;
Liu et al. 2024 "lost in the middle").

1. **Relevance re-ordering** (`_reorder_by_question_relevance`). Chunks
   are stable-sorted by a score combining three terms:
   - *IDF-weighted query-term overlap* — a query term occurring in many
     candidate chunks (a generic category word like "magazines") is
     down-weighted; a rare term (the specific entity) is decisive
     (inverse document frequency over the candidate set; Spärck Jones
     1972; Robertson 2004). IDF is applied only with ≥ 4 candidates;
     below that, document frequency is degenerate and the score falls
     back to length-normalised hit count.
   - *sqrt(word-count) length normalisation* — a short direct-answer
     chunk is not penalised against a long topic chunk that accumulates
     hits from sheer length (standard TF length normalisation).
   - *structural-coverage floor* — a chunk naming a distinctive query
     entity (multi-word, or a single token ≥ 8 chars) receives a score
     floor, so a required entity's article (a comparison conjunct, or a
     bridge target) cannot be demoted below the context cap by keyword
     sparsity.
2. **Sentence-aware per-doc truncation** (`_truncate_sentence_aware`).
   When a chunk exceeds `max_chars_per_doc`, the most query-relevant
   *sentences* are kept (in original order, preserving local coherence)
   rather than a blind head-truncation — so an answer-bearing sentence
   in the tail of a chunk survives. Falls back to head-truncation when
   no query is supplied or the chunk has no sentence structure.

#### Prompt selection

S_V selects one of four prompt templates based on `query_type` and
pre-validation status:

| Prompt | When |
|---|---|
| `ANSWER_PROMPT` | Single-hop / temporal / aggregate / default. |
| `BRIDGE_PROMPT` | `query_type in {"multi_hop", "bridge"}` and `hop_sequence` non-empty. The prompt includes a reasoning scaffold built by `_build_bridge_chain`. Non-final steps render as the directive "→ identify the intermediate result" rather than a pre-filled bridge entity: an entity merely *appearing* in context does not mean it *answers* the sub-query, and injecting it was observed to poison the SLM (it copied a wrong-but-grounded value, or over-abstained). The final step renders as "→ derive the final answer" (never a literal placeholder a small model could echo). |
| `COMPARISON_PROMPT` | `query_type == "comparison"`. |
| `INSUFFICIENT_EVIDENCE_PROMPT` | Pre-validation reports insufficient context. |

`CORRECTION_PROMPT` is used in iteration ≥ 2 of the self-correction
loop when prior iterations produced violated claims.

The `ANSWER`/`BRIDGE`/`COMPARISON` prompts (2026-05-29) **discourage
premature abstention**: the "reply I don't know" instruction was softened
to "the answer is almost always present — read carefully; abstain only if
genuinely absent." The aggressive original wording was a measured source
of false-abstention on questions whose gold was in fact present.

**Hard early-return narrowed (2026-05-29).** The verifier short-circuits
the LLM call **only when `working_context` is empty**. The previous
condition also fired on `INSUFFICIENT_EVIDENCE` + zero entity-path
matches, which force-returned "I cannot determine" *without ever calling
the LLM* whenever the planner's entities failed a substring match against
the chunks — a frequent false negative on bridge questions (the answer is
present but the entity is not verbatim). The narrowed gate only adds
generation attempts on context already in hand, so it cannot turn a
correct answer wrong.

#### Self-correction loop

Up to `agent.max_verification_iterations` rounds (default 1 — the loop is
OFF by production; the optional second pass is retained for ablation).
Each round:

1. Call the SLM with the selected prompt.
2. Extract atomic claims via SpaCy sentence splitting + meta-statement
   filtering.
3. Verify each claim against the graph store and/or retrieved context.
4. If all claims pass → return early.
5. If any claim violated → re-prompt with `CORRECTION_PROMPT` listing
   the violated claims.

The implementation follows Self-Refine (Madaan et al. 2023). Claim
verification is a conservative entity-presence proxy, not logical
entailment (Kryscinski et al. 2020 framing). **Note (§11.17): with
`max_iterations=1` (the default), claim verification only sets the
confidence label — it does not change the answer, so it affects
calibration, not EM.**

**Bounded answer-revision retries (all keep-best guarded).** Between
generation and claim extraction, up to one of each of the following may
fire (each at most once per query). Critically, every retry is now
**keep-best guarded** — the revised answer replaces the original only if
it is a real (non-error, non-disclaimer) answer, and for the grounding
gate only if it scores strictly higher entailment. The unconditional
answer-swap was the failure that made an early NLI-gate experiment
net-negative (it replaced 6 of 50 correct answers; §11.17):

- **Bridge-exclusion retry** — fires when the answer equals a resolved
  bridge entity (the SLM returned the intermediate hop result).
- **Format-mismatch retry** — fires on a wh-question answered yes/no.
- **Anti-abstention retry (opt-in, `enable_anti_abstention_retry`)** —
  fires when the answer is a disclaimer *but* pre-validation found query
  entities present in the context; re-prompts to extract the answer.
- **QA-conditioned NLI grounding gate (opt-in,
  `enable_nli_grounding_gate`)** — turns the `(question, answer)` pair
  into a declarative hypothesis (`_qa_to_hypothesis`) and scores its
  entailment against the chunks with the configured NLI cross-encoder
  (`_nli_entailment`, max-pooled over the top chunks). Sub-threshold
  entailment → one bounded grounding retry, accepted only if the retry's
  entailment is higher (keep-best). Conditioning on the question is what
  distinguishes a *wrong-but-grounded* answer (the entity is in the
  context but answers a different question) from a correct one — plain
  answer-grounding cannot. Refs: Bowman et al. 2015 (NLI); QA-to-
  declarative verification.

All four retries share one engine, `Verifier._bounded_retry` (fire-once →
call LLM → shared no-regression guard → per-retry acceptor), so the
keep-best discipline is implemented in exactly one place. Their prompts are
class constants (`BRIDGE_EXCLUSION_RETRY_PROMPT`, `FORMAT_RETRY_PROMPT`,
`ANTI_ABSTENTION_RETRY_PROMPT`, `GROUNDING_RETRY_PROMPT`), so the
reproducibility export (`scripts/export_prompts.py`) captures every prompt
the model actually receives, not only the four primary answer templates.

**Best-answer selection prefers a substantive answer over a disclaimer.**
Across iterations the verifier keeps the answer with the fewest violated
claims, with one override: a substantive (non-disclaimer) answer always
beats a disclaimer. Because a disclaimer is forced to exactly one violated
claim (below), a naive fewest-violations rule could let an "I don't know"
displace a real answer that happened to carry two violations. Abstention is
surfaced through the confidence / `all_verified` signal, never by hijacking
the returned answer string.

**Disclaimer override.** When the SLM returns an epistemic disclaimer
("I don't know", "no information", etc.), the verifier forces a
violated-claim entry and `LOW` confidence so the orchestrator does
not report a non-answer as `all_verified=True`.

**No-entity claim grounding.** Claims with no extractable proper noun
fall through to a token-grounding check whenever the claim is short
(≤ 6 tokens) or contains a numeric token. This catches hallucinated
short factual answers ("9 million inhabitants" vs the context's
"1.5 million inhabitants") that previously auto-verified.

**Short-answer-as-claim (2026-05-29).** HotpotQA gold answers are usually
a bare entity ("Wendell Berry", a year, a place) shorter than
`min_claim_chars`, so claim extraction returned an empty list — which
maps to `all_verified=True` / HIGH confidence via the (0,0) ratio. A
correct terse answer was therefore reported as spuriously HIGH and a
wrong one never flagged. When a real (non-disclaimer) answer produces no
claims, the answer itself is now treated as one claim and grounded like
any other. Calibration-only — never changes the answer string.

#### Iteration history

`VerificationResult.iteration_history` records per-iteration answer,
claims, latency, and error flag. Strings are truncated to 200
characters with a `...[truncated]` marker so 500-question JSONL files
remain `jq`/pandas-tractable.

### 4.4 Pipeline Orchestrator (`AgentPipeline`)

**File:** [`src/pipeline/agent_pipeline.py`](src/pipeline/agent_pipeline.py)

`AgentPipeline` is the **single production orchestrator** of the
three-agent pipeline. All evaluation scripts call `pipeline.process(query)`
and consume `PipelineResult` dataclasses.

```python
class AgentPipeline:
    def __init__(
        self,
        planner:          Optional[Planner]     = None,
        navigator:        Optional[Navigator]   = None,
        verifier:         Optional[Verifier]    = None,
        hybrid_retriever: Optional[Any]         = None,
        graph_store:      Optional[Any]         = None,
        config:           Optional[Dict[str, Any]] = None,
    ): ...
    def process(self, query: str) -> PipelineResult: ...
```

Responsibilities:

- Construct each agent lazily on first `process()` call (or accept
  pre-built instances for tests).
- Chain S_P → S_N → S_V in fixed sequence.
- Forward the planner's `query_type` and `bridge_entities` (resolved
  by iterative-navigate) into the verifier so BRIDGE / COMPARISON
  prompt selection actually fires at evaluation time.
- Forward per-chunk graph-provenance flags
  (`chunk_is_graph_based`) into the verifier so the credibility
  score uses a real signal instead of a constant baseline.
- Maintain a FIFO query-result cache for repeated benchmark queries
  (`enable_caching: true`, `cache_max_size: 1000` by default).
- Track aggregate statistics via Welford's incremental mean
  (Welford 1962).

**Verification-triggered re-retrieval loop (opt-in,
`enable_reretrieval_loop`, 2026-05-29).** After S_V returns, if the
first-pass answer is structurally weak — low confidence, a disclaimer, or
no content token substring-grounded in the chunks (`_should_reretrieve`)
— the pipeline runs **one** additional retrieval pass with a HyDE-style
expanded query (`_expand_query_with_draft`: the failed draft + resolved
entities appended to the question; Gao et al. 2022, arXiv:2212.10496),
re-verifies against the merged context, and keeps the new answer only
when `_reretrieval_better` judges it an improvement (keep-best). It
targets the retrieval-miss failure bucket — the one class of error
post-hoc answer-checking structurally cannot reach. Bounded to one extra
retrieval + verification round (edge-latency cap). Telemetry
(`reretrieval_fired` / `reretrieval_accepted` / `reretrieval_reason`) is
surfaced in `navigator_result.metadata`. Refs: FLARE (Jiang et al. 2023,
EMNLP); Self-RAG (Asai et al. 2024, ICLR); IRCoT (Trivedi et al. 2023,
ACL). **Measured neutral on HotpotQA** (§11.17) — its 10-document
distractor pool gives query expansion no room to surface a missed chunk;
expected to help on open-corpus settings, identified as future work.

The factory `create_full_pipeline(hybrid_retriever, graph_store, config)`
constructs the pipeline used by the evaluation harness. It attaches
the `BatchedOllamaEmbeddings` instance as `pipeline._embeddings` so
the evaluation summary can print cache hit-rate / batch-count /
average per-text latency at the end of every run.

### 4.5 Static Helper Namespace (`AgenticController`)

**File:** [`src/logic_layer/controller.py`](src/logic_layer/controller.py)

`AgenticController` is a stateless container of utility helpers
consumed by `AgentPipeline._iterative_navigate`. It contains:

- `_extract_bridge_entities(chunks, exclude, query)` — bridge-entity
  extraction from retrieved chunk text. **Pass 0** (GPE queries only)
  applies a location-context regex ("in the city of X", "capital of X")
  and returns early. The former surname-anchor (Pass 1) and
  general-proper-noun (Pass 2) passes are **merged into one scored
  candidate pool** rather than a priority-ordered cascade: the old
  early-return on the first surname match let a low-precision
  reconstruction preempt a stronger general candidate (e.g. a spurious
  "Salisbury Gardens" blocking the real "Thomas Mawson"). Candidates from
  both generators now compete on the same scoring function and the
  strongest wins. Chunks arrive in RRF rank order, so the list index is
  the chunk's retrieval rank. A substring-aware exclusion drops contiguous
  sub-phrases of an excluded compound entity.
- `_score_bridge_candidate(candidate, chunk, query, expected_type, chunk_rank)` —
  query-keyword proximity + expected-type bonus − position penalty, with
  type/length features gated on positive proximity, the whole **multiplied
  by a reciprocal chunk-rank prior `1/(1+rank)`** (Cormack et al. 2009;
  the same primitive RRF uses) so an entity from a top-ranked chunk
  outranks one from a low-ranked noise chunk. The local score is clamped
  non-negative before the rank multiply. A returned score of 0 means
  "not found / no query proximity".
  **Abstention floor:** if the best candidate scores ≤ 0,
  `_extract_bridge_entities` returns `[]` — a confidently-wrong bridge
  would misdirect hop-2 retrieval (and reranker hints), which is worse
  than none; hop-2 then falls back to its un-rewritten sub-query.
- `_detect_expected_type(query)` — interrogative-word → expected entity
  type (who → PERSON; where → GPE; when → DATE).
- `_rewrite_hop_query_with_bridges(sub_query, bridges)` — IRCoT-style
  hop-query rewriting that appends resolved bridge entities to the
  next-hop sub-query.

This module has **no `__init__`**, no orchestrator logic, no
runtime state. It is a namespace, not an agent. The production
orchestrator is `AgentPipeline`.

---

## 5. Ingestion: Decoupled Three-Phase Architecture

Phase boundaries are chosen so the GPU-bound entity-extraction step
runs separately from the CPU-only ingestion target.

| Phase | Tool | Hardware | Input | Output |
|---|---|---|---|---|
| **1** | `python -m src.thesis_evaluations.benchmark_datasets ingest --chunks-only` | CPU (local) | Raw dataset corpus | `chunks_export.json` (chunk text + metadata) |
| **2** | `colab_extraction.py` | GPU (Colab T4) | `chunks_export.json` | `extraction_results.json` (GLiNER entities + REBEL relations + per-triplet log-prob confidence) |
| **3** | `python local_importingestion.py` | CPU (local) | Both JSONs | LanceDB + KuzuDB populated |

### Phase 3 sub-phases

| Step | Description |
|---|---|
| 3a | LanceDB ingest (vectors only) |
| 3b | KuzuDB ingest: DocumentChunk + SourceDocument + FROM_SOURCE + NEXT_CHUNK + Entity + MENTIONS + RELATED_TO (REBEL) + SVO narrative relations. Uses `add_entities_bulk` / `add_mentions_relations_bulk` (batched CREATE after Python-side dedup) instead of per-edge MERGE — MERGE on KuzuDB MENTIONS adjacency scaled with entity degree and dominated wall-clock time before this change. |
| 3c | Co-occurrence edges (`RELATED_TO {relation_type='cooccurs'}` between every entity pair sharing a chunk) |
| 3c.5 | Subsumptive cleanup: drop cooccurs edges between pairs that already have a semantic edge |
| 3d | Cleanup: stop-list / orphan / hub / duplicate merge |
| 3d.5 | Embedding-based entity linking (alias resolution beyond canonical_form). **Disabled by default** for paper-release ingest; see §3.6.1. When enabled, uses `_redirect_entity_edges_bulk` (batched CREATE) and persists per-bucket completion to the checkpoint so a mid-phase crash skips already-linked buckets on resume. |
| 3e | Baseline metrics + invariant checks |
| 3f | Post-link isolated-entity drop |

Each phase writes a checkpoint to
`data/<dataset>/graph/.import_checkpoint.json`; `--resume` re-runs only
unfinished phases. Phase 3d.5 additionally stores a `done_buckets`
list so partial progress survives crashes within the phase.

---

## 6. Configuration System

**File:** [`config/settings.yaml`](config/settings.yaml) — single source of truth.

| Top-level block | Owns parameters for |
|---|---|
| `embeddings` | Ollama embedding model, base URL, dimension, cache path. |
| `vector_store` | LanceDB path, distance metric, normalisation, overfetch factor, graph-node text cap. |
| `graph` | Traversal config (`max_hops`, `top_k_entities`, `enable_hop3`, `hub_mention_cap`, `hub_fanout_cap`, `search_budget_seconds`). The KuzuDB location is dataset-scoped and derived at runtime, not configured here. |
| `rag` | Retrieval mode (`vector` / `graph` / `hybrid`), per-source weights, RRF constant `k`, BM25 toggle. |
| `navigator` | Filter thresholds, reranker, contradiction filter, corroboration weights. |
| `planner` | Classifier weights, entity-density thresholds, classification confidence calibration. |
| `verifier` | Pre-validation flags (`enable_entity_path_validation`, `enable_credibility_scoring`, `credibility_filter_drop`), credibility weights, claim-extraction limits, `max_iterations`, fallback confidence, and the opt-in answer-verification toggles (`enable_context_distillation`, `enable_structured_cot`, `enable_nli_grounding_gate`, `nli_grounding_*`, `enable_anti_abstention_retry` — all default OFF; §11.17). |
| `llm` | Active model, base URL, context budget (`max_docs`, `max_chars_per_doc`, `max_context_chars`), timeout. |
| `available_models` | Cross-model ablation roster (Ollama tag → metadata). |
| `agent` | Pipeline-level flags (`enable_planner`, `enable_verifier`, `enable_caching`, `cache_max_size`, `enable_confidence_gate`, `enable_over_decomposition_gate`, `enable_reretrieval_loop` — the last three opt-in). |
| `entity_extraction` | GLiNER prompts, threshold, REBEL config — the values pinned for Phase 2. |
| `ingestion` | Chunker selection, `sentences_per_chunk`, `sentence_overlap`, spaCy model (`spacy_model`). The Planner and Verifier both read `ingestion.spacy_model` and load it via their `_ensure_nlp` accessor, so the configured model — not the import-time default — drives query/claim analysis. |
| `paths`, `performance`, `benchmark` | Operational (data paths, batch size / device, Soft-EM F1 threshold). |

Every configurable threshold referenced in this document is loadable
from `settings.yaml` via the corresponding `from_yaml(cfg)` factory.
Dataclass defaults exist only as documented emergency fallbacks for
unit-test scenarios.

### 6.1 Environment-variable overrides (container / CI surface)

The settings file is the single source of truth for *hyperparameters*; a
small, fixed set of environment variables overrides only the **deployment
bindings** (endpoints and filesystem locations) so the same image runs
unchanged on a laptop, in docker-compose, and in CI. All three are read once
at load time and default to the local-development values, so a run with no
environment set behaves exactly as before.

| Variable | Default | Effect | Read in |
|---|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Rewrites **both** `llm.base_url` and `embeddings.base_url` in the loaded settings dict, so every Ollama client (generation + embeddings) targets the sidecar hostname (`http://ollama:11434` under compose). Trailing slash stripped. | `_settings_loader._apply_env_overrides` |
| `CONFIG_PATH` | `config/settings.yaml` | Selects which settings YAML is loaded when no explicit path is passed (e.g. `config/frozen_paper.yaml` mounted at `/app/config/`). Resolved per-call, not at import. | `_settings_loader._default_settings_path` |
| `INDEX_DIR` / `DATASET_DIR` | `./data` | Overrides the per-dataset store base (`StoreManager` derives `vector/`, `graph/`, `questions.json` from it), so the pre-built LanceDB / KuzuDB / BM25 stores can be mounted read-only at an arbitrary path. `INDEX_DIR` takes precedence; `DATASET_DIR` is an accepted alias. | `benchmark_datasets._resolve_data_root` |

The override applies even when the YAML is missing or unparseable: a bare
`OLLAMA_HOST` still reaches the callers (against an otherwise-empty dict that
falls back to dataclass defaults), so a container can construct clients
against the sidecar before any config is mounted. The validator and all
`from_yaml(cfg)` factories run *after* the override, so the reproducibility
check sees the effective values. See §8.4.

---

## 7. Evaluation Framework — Artifact C

**Directory:** [`src/thesis_evaluations/`](src/thesis_evaluations/)

### 7.1 Benchmark runner

[`benchmark_datasets.py`](src/thesis_evaluations/benchmark_datasets.py)
exposes `ingest`, `evaluate`, `ablation`, `status`, `test`, and `demo`
sub-commands. The `demo` sub-command runs a **single** question end-to-end
and prints the stage breakdown — Planner query-type + sub-queries, the chunks
the Navigator forwarded, the Verifier's answer + confidence, the wall-clock
latency and peak RSS — as the one-command, reviewer-facing trace of the
edge envelope (`demo --question "..." [--dataset hotpotqa]`). It binds to the
same per-dataset stores as `evaluate`, so it exercises the real retrieval
stack, not a stub.

**Question selection (`evaluate`).** The evaluator draws a **random
sample by default** (`--samples N`, default 20) so repeated runs probe
different questions and a gain is shown to be robust rather than tuned to
a fixed prefix. The random seed is auto-generated and logged
(`Question sample seed: N  (re-run with --seed N to reproduce)`), and
`--seed N` fixes the draw for exact reproduction or for comparing two
code versions on the same questions. For deterministic selection,
`--range START-END` takes `questions[START:END]` in index order
(separators `-`, `_`, `:`): `--range 0-20` reproduces the pre-2026-05
"first 20" behaviour; `--range 10-30` runs a defined band or shards a
large run. `--range` overrides `--samples` when both are given.

The evaluation summary block reports:

- Exact Match, F1, average wall-clock per query, coverage.
- Supporting-fact F1, SF-Recall (`all_gold_retrieved` rate).
- Pipeline / LLM failure decomposition.
- Per-question-type breakdown (bridge / comparison / compound).
- **Embedding cache metrics** (cache hit rate, batch count, average
  per-text latency).
- Per-question JSONL with `matched_pattern`, `classifier_preempt`,
  `verifier_iterations`, `all_verified`, `confidence`, and all retrieval
  metrics for downstream `jq`/pandas analysis.

### 7.2 Tier-1 ablation scripts

| Script | Role |
|---|---|
| `agentic_ablation.py` | Five-row decomposition: LLM-only → +Retrieval → +Planner → +Verifier → +SelfCorrect. Each row isolates one contribution. |
| `modality_ablation.py` | Retrieval-only per-lane ablation (dense / BM25 / graph / hybrid) with rank-aware IR metrics (Recall@k, nDCG@k, MRR) + the per-question `graph_rescued` list. Run on HotpotQA, 2WikiMultiHopQA, and MuSiQue. Substantiates the word "hybrid". |
| `graph_rescue_analysis.py` | Post-processes the modality `graph_rescued` + per-lane JSONLs into the headline contribution table (`table_graph_rescue.tex`): the % of questions the graph lane reaches that dense **and** BM25 both miss, and the stronger **graph-necessary** rate (graph closed the last gold gap). 6.2 % / 4.0 % / 8.4 % graph-necessary on HotpotQA / 2Wiki / MuSiQue. |
| `quantization_sweep.py` | Cross-model EM/F1/SF-F1 sweep across Ollama tags. |
| `latency_memory_profile.py` | Per-stage wall-clock + peak RSS distribution. |
| `chunking_ablation.py` | Per-`(sentences_per_chunk, sentence_overlap)` cell: re-chunks, ingests a per-config vector store (graph held constant), runs retrieval-only eval, writes summary.md. |

### 7.3 Diagnostic tooling

| Tool | Use |
|---|---|
| `tools/diagnose_verbose.py` | Full pipeline trace with per-stage gold-paragraph tracking (Planner output, Navigator filter chain, Verifier output). Honours `--skip-llm` for fast non-LLM passes. |
| `tools/diagnose_ingestion.py` | Phase-3 consistency probe (counts vs. checkpoint vs. graph baseline). |
| `tools/diagnose_graph_baseline.py` | Read-only graph-quality reporter; `--strict` exit code gates downstream evaluation. |
| `tools/diagnose_ablation.py` | Post-hoc analyser of `agentic_ablation` JSONL traces; failure-mode classification and self-correction-flip counts. |
| `verifier_cache_build.py` | **Verifier-only harness (§11.17).** Runs Planner → Navigator once per question and dumps the filtered context + plan metadata to JSONL. `--no-planner` builds the bare-RAG variant. Lets the Verifier sweep run against the *same* frozen retrieval — so every EM/F1 delta is attributable to the Verifier alone — without re-paying retrieval cost. The sweep itself is driven by `agentic_ablation.py` (the standalone `verifier_only_ablation.py` / `validate_reretrieval_loop.py` scripts were folded into it on 2026-06-03). |
| `verifier_failure_taxonomy.py` | Buckets wrong answers into retrieval-miss / grounded-hallucination / false-abstention / close-miss by joining a sweep JSONL with the retrieval cache. Localises where the verifier is responsible vs where retrieval is. |

---

## 8. Technology Stack

### 8.1 Core dependencies

| Library | Role |
|---|---|
| **lancedb** | Embedded vector store (Apache Arrow columnar storage; exact cosine scan — no ANN index built). |
| **kuzu** | Embedded property graph (columnar, vectorised Cypher). |
| **sqlite3** | Persistent embedding cache. |
| **spacy** | Tokenisation, sentence splitting, NER (en_core_web_sm). |
| **gliner** | Zero-shot span-based NER (Phase 2 entity extraction; local fallback). |
| **transformers** | REBEL relation extraction (Phase 2 GPU; not used at query time). |
| **sentence-transformers** | Cross-encoder reranker (Stage 2.5 in Navigator). |
| **rank_bm25** | BM25 sparse retrieval. |
| **requests** | Ollama HTTP client. |
| **langchain-core** | `Embeddings` interface compatibility. |
| **coreferee** *(optional)* | Pronoun coreference for Phase-1 chunking. |
| **pytest** | Test runner; markers in `pytest.ini` separate slow/nightly/llm/integration tests. |

### 8.2 External services (local)

| Service | Endpoint | Models in production roster |
|---|---|---|
| Ollama | `http://localhost:11434` (override: `OLLAMA_HOST`) | `qwen2:1.5b` (primary), `gemma2:2b`, `llama3.2:3b`, `phi3`, `qwen3:4b`, `nomic-embed-text` (embeddings) |

The endpoint is the local-development default; `OLLAMA_HOST` redirects both
the generation and the embedding client to a sidecar host (§6.1, §8.4).

### 8.3 Database selection rationale

- **LanceDB** — Embedded, Apache Arrow columnar. Supports HNSW and
  IVF-Flat indexes, though the reported system builds none — retrieval is
  an exact cosine scan over the full table at this corpus scale. No server
  process.
- **KuzuDB** — Native Cypher, memory-mapped, fast for the 1–3 hop
  traversals this paper requires. Out-of-core for graphs > RAM (Feng
  et al. 2023).
- **SQLite** — Universal, zero-configuration cache substrate.

### 8.4 Containerization and continuous integration

The artifact ships a Dockerized **CPU-only inference / retrieval / evaluation**
path. By deliberate scope, the GPU extraction stage (Phase 2 GLiNER + REBEL,
run once offline on Colab) is **not** containerized: it produces the stores,
which are shipped as a *data* artifact (Zenodo) and mounted at runtime, not
baked into the image.

| Asset | Role |
|---|---|
| [`Dockerfile`](Dockerfile) | `python:3.12-slim` (matches the frozen 3.12.3 interpreter), installs `requirements_frozen.txt` (pins `sentence-transformers` + the spaCy model wheels), copies `src/` + `config/`. Entrypoint `python -m src.thesis_evaluations.benchmark_datasets`. The indices/datasets are **not** copied in — they are mounted. |
| [`docker-compose.yml`](docker-compose.yml) | Three services: `ollama` (sidecar holding the weights, with a `list` healthcheck), `model-init` (one-shot pull of `qwen2:1.5b` + `nomic-embed-text`, then exits), and `app`. The app is constrained to the edge envelope via `mem_limit: 2g` / `cpus: "4"`; a run that exceeds 2 GB is OOM-killed by the runtime — that failure *is* the envelope evidence. Stores are bind-mounted read-only; `cache/` is writable for the reranker download + embedding cache. |
| [`scripts/fetch_data.sh`](scripts/fetch_data.sh) / [`.ps1`](scripts/fetch_data.ps1) | Fetch the pre-built stores from the Zenodo artifact into `data/indices/<dataset>/{vector,graph,questions.json}` (the layout `StoreManager` expects). `ZENODO_URL` is a placeholder until the DOI is minted. |
| [`.github/workflows/tests.yml`](.github/workflows/tests.yml) | Two parallel jobs on every push/PR. **`unit`** runs the fast deterministic subset (`pytest -m "not slow and not llm and not nightly and not integration"`) — no Ollama, no GPU, no model weights. **`repro-guard`** enforces the reproducibility contract at push time (see below). |
| [`.github/workflows/docker.yml`](.github/workflows/docker.yml) | Builds the app image and runs a CLI smoke (`--help` and `demo --help`) on `main` push + PR — catches dependency / packaging breakage without needing a model pull. |

#### Reproducibility guard (`repro-guard` job)

The paper pins two frozen artifacts for release reproduction —
`requirements_frozen.txt` (exact `==` library pins) and
`config/frozen_paper.yaml` (the frozen hyperparameter set). The
`_settings_loader._validate_settings` check (§11.16.5) already warns *at
runtime* when a reproducibility-critical key is missing; the `repro-guard`
CI job promotes that contract to a **hard push-time failure**, so a broken
reproduction setup can never be merged silently. It runs four checks, in
parallel with the `unit` job and *without* running the test suite:

| Check | Fails the push when… | Mechanism |
|---|---|---|
| Frozen requirements resolve | a pin in `requirements_frozen.txt` no longer installs (yanked release, incompatible transitive constraint) | `pip install --no-cache-dir -r requirements_frozen.txt` |
| Config schema parity | `frozen_paper.yaml` and `settings.yaml` expose a *different set of keys* (a key added/removed in one but not the other — the silent-default drift class) | [`scripts/check_frozen_config.py`](scripts/check_frozen_config.py) — compares the dotted key-path sets, not the values (the frozen file may hold different *values* by design) |
| Required-settings present | a `_REQUIRED_SETTINGS` key (37 keys) is absent from `settings.yaml` | `python -W error::UserWarning -c "…_load_settings()"` — promotes the runtime WARNING to an error |
| Prompt export in sync | a verifier prompt was edited without re-exporting `docs/prompts/PROMPTS.md` (the B-2 prompt-reproducibility artifact) | `scripts/export_prompts.py --check` |

The parity check is *value-agnostic on purpose*: it asserts the two configs
have the **same surface** (so neither silently grows or loses a knob), while
allowing the frozen file to fix different values than the live default. The
guard is negative-tested — injecting a stray key into either file makes the
check exit non-zero.

A full end-to-end benchmark in CI is **infeasible** on hosted runners (needs
Ollama + a model pull + the multi-GB stores) and is deliberately excluded.
The three CI guards are complementary: `docker.yml` is the **packaging**
guard (the image builds and imports), `tests.yml::unit` is the **logic**
guard (the deterministic test subset passes), and `tests.yml::repro-guard`
is the **reproducibility** guard (the frozen requirements + frozen config +
exported prompts stay self-consistent). The live demo/eval are reproduced
locally with the mounted stores. Runtime endpoint/path indirection is via the
§6.1 environment variables, so no image rebuild is needed to relocate Ollama
or the stores.

---

## 9. Data Flows

### 9.1 Ingestion flow (Phase 1 → 3)

```
   Raw corpus (HotpotQA articles)
              │
              ▼
   Phase 1: SpacySentenceChunker + (optional) coreference
              │
              ▼
   chunks_export.json  ─── upload ───▶  Google Colab (GPU)
                                            │
                                            ▼
              Phase 2: GLiNER NER + REBEL RE
                       (log-prob calibrated confidence)
                                            │
                                            ▼
                                   extraction_results.json
                                            │
              ┌─────────────────────────────┘
              ▼
   Phase 3: local_importingestion.py
              │
              ├─→ 3a Vector ingest (LanceDB)
              └─→ 3b-3f Graph ingest (KuzuDB):
                       chunks + entities + MENTIONS +
                       RELATED_TO (REBEL + SVO) +
                       cooccurrence + cleanup +
                       embedding-based entity linking
```

### 9.2 Query flow

```
   user query
       │
       ▼
   ┌───────────────────────────────────────────────────────────┐
   │           AGENT PIPELINE (process)                         │
   │                                                            │
   │   1. S_P planner.plan(query)                              │
   │        ↳ RetrievalPlan(query_type, sub_queries,           │
   │          entities, hop_sequence, matched_pattern,         │
   │          classifier_preempt)                              │
   │                                                            │
   │   2. S_N navigator.navigate(plan, sub_queries)            │
   │        a. For each sub-query: HybridRetriever.retrieve    │
   │             ├── dense (LanceDB)                            │
   │             ├── BM25 (rank_bm25)                           │
   │             └── graph multi-hop (KuzuDB)                   │
   │        b. RRFFusion(vector, bm25, graph)                   │
   │        c. Cross-encoder rerank (optional)                  │
   │        d. 6-stage pre-generative filter chain              │
   │        ↳ filtered_context + retrieval_methods              │
   │                                                            │
   │   3. (multi-hop only) Iterative navigate:                  │
   │        for hop in plan.hop_sequence:                       │
   │          bridges = AgenticController._extract_bridge      │
   │                       _entities(chunks, exclude, query)    │
   │          next_hop.sub_query = _rewrite_hop_query           │
   │                       _with_bridges(sub_query, bridges)    │
   │                                                            │
   │   4. S_V verifier.generate_and_verify(query, context,     │
   │        entities, hop_sequence, query_type,                │
   │        bridge_entities, chunk_is_graph_based)              │
   │        ↳ pre-validation                                    │
   │        ↳ prompt selection (ANSWER / BRIDGE /              │
   │           COMPARISON / INSUFFICIENT_EVIDENCE)              │
   │        ↳ self-correction loop (up to max_iterations)      │
   │                                                            │
   └────────────────────┬───────────────────────────────────────┘
                        ▼
              PipelineResult(answer, confidence, planner_result,
                             navigator_result, verifier_result,
                             planner_time_ms, navigator_time_ms,
                             verifier_time_ms, total_time_ms,
                             cached_result)
```

---

## 10. Non-Functional Requirements

### 10.1 Reproducibility

- All hyperparameters are sourced from `config/settings.yaml`. No
  hard-coded design-time values remain in production code; dataclass
  defaults are emergency fallbacks only.
- Chunk IDs are deterministic SHA-256 hashes of source + position +
  text prefix, so re-ingestion is idempotent.
- Phase-2 (Colab) writes a checkpoint hashed over
  `(chunks_hash, config_hash)`; resumes are valid only when both hashes
  match.
- Phase-3 writes per-step checkpoints under
  `data/<dataset>/graph/.import_checkpoint.json`; `--resume` re-runs
  only unfinished steps.
- `requirements_frozen.txt` pins library versions; the active Ollama
  model is recorded in every evaluation's summary block.
- A Dockerized artifact (§8.4) reproduces the inference + retrieval path
  on CPU-only hardware, constrained to the 2 GB / 4-core envelope via
  Compose resource limits. Deployment bindings (Ollama endpoint, config
  path, store location) are injected via the §6.1 environment variables,
  so the reproduced image is identical across host, compose, and CI.

### 10.2 Observability

- Every agent logs at `INFO` for stage transitions and `DEBUG` for
  per-pattern decisions.
- The Verifier records `iteration_history` (per-iteration claims,
  verification status, latency).
- `RetrievalPlan` records `matched_pattern` and `classifier_preempt`
  so per-question JSONL supports per-pattern aggregation.
- `EmbeddingMetrics` accumulates cache hit-rate / batch-count /
  per-text latency; reported in every evaluation summary.
- `_install_retriever_title_capture` in the eval harness records, for
  each filtered chunk, the source-document title that produced it —
  enabling supporting-fact tracking against gold labels.

### 10.3 Test coverage

The test suite collects **739 tests** across the layers (721 under the
documented CI default `-m "not nightly"`; the remaining 18 load GLiNER /
REBEL model weights and are run on demand with `-m nightly`). Key modules:

| Module | Test file |
|---|---|
| Storage / retriever | `test_data_layer.py`, `test_graph_inspect.py` (nightly) |
| Embeddings | `test_embeddings.py` |
| Chunking | `test_chunking.py` |
| GLiNER bounds | `test_gliner_boundary.py` (nightly) |
| Planner | `test_planner_semantic.py`, `test_logic_layer.py` |
| Navigator | `test_navigator_semantic.py` |
| Verifier | `test_verifier_semantic.py` |
| Pipeline | `test_pipeline.py` |
| Thesis matrix | `test_thesis_matrix.py`, `test_thesis_matrix_ext.py` — capability-matrix coverage of the documented design decisions |
| Coverage / robustness | `test_missing_coverage.py`, `test_config_robustness.py` — edge-case and config-validation guards |
| Thesis cleanup | `test_thesis_cleanup.py` — regression guards that the paper-cleanup pass is intact (no dataset-revealing strings in source, no removed-pattern markers, etc.) |

Markers in `pytest.ini` separate slow / nightly / llm / integration
tests. The `tests.yml` GitHub Actions workflow (§8.4) runs the fast
deterministic subset on every push/PR with
`-m "not slow and not llm and not nightly and not integration"` — no
Ollama, no GPU, no model weights — and reports status via the README
badge.

### 10.4 Security and data handling

- No outbound API calls at evaluation time: Ollama runs locally; the
  embedding model and SLMs are local.
- No PII processing in the evaluation corpus (HotpotQA articles).
- The Colab phase uses Drive-mounted volumes; checkpoint and
  `extraction_results.json` are written atomically.

---

## 11. Design Decisions and Academic Grounding

### 11.1 Embedded databases over client-server

All three stores (LanceDB, KuzuDB, SQLite) are embedded. The edge
deployment target precludes a separate database server process; the
embedded choice also removes a class of failure modes (network
unavailability, schema migration, version drift).

### 11.2 Reciprocal Rank Fusion over linear interpolation

RRF (Cormack et al. 2009) is robust to mis-calibrated per-source
scores. Dense cosine similarity and BM25 are on incompatible scales;
linear interpolation requires per-corpus tuning. RRF only consumes
ranks, so adding a third source (graph) requires no recalibration.

### 11.3 Graph depth cap at 2 hops by default

Hop-3 retrieval is gated behind `graph.enable_hop3` because two-hop
noise compounds: a `cooccurs` edge followed by another `cooccurs`
edge approximates "anything-to-anything within the corpus". With
`enable_hop3: true`, cooccurs edges are excluded from hop-3 paths.

### 11.4 Hub suppression in graph retrieval

Entities mentioned in > 3 % of the corpus are excluded as bridge
targets. The 3 % threshold parallels TF-IDF's IDF cap (Salton & Buckley
1988): a term in too many documents loses discriminative power. The
result on HotpotQA: ~3 entities qualify as hubs ("United States",
"The Young Ones"-style multi-topic pages, "September 9 2013"-style
dates); their suppression prevents spurious cross-topic bridges
(West & Leskovec 2012 on hub-avoidance in graph IR; GraphRAG, Edge
et al. 2024).

### 11.5 Triple-frequency confidence over REBEL's per-triplet score

REBEL's per-triplet log-prob (now correctly calibrated in Phase 2)
ranks individual *edges*, not *bridges*. Bridge quality is a property
of the entity-pair plus the number of supporting chunks: more chunks
mentioning the pair = stronger corpus support. The
`_triple_frequency_confidence` score follows the corpus-support
inference of DeepDive (Niu et al. 2012) and Knowledge Vault (Dong
et al. 2014).

### 11.6 Cooccurs edges down-weighted, not deleted

Approximately 85 % of entity-pairs in the HotpotQA graph have only
cooccurs edges (no REBEL/SVO semantic relation). Deleting cooccurs
edges eliminates bridge connectivity for those pairs. The system keeps
them but weights them at 0.25 (vs 1.0 for REBEL) so any semantic
bridge with even one supporting chunk outranks any cooccurs bridge.
Following the PMI tradition (Church & Hanks 1990): co-mention is a
weak but non-zero signal — down-weight, do not delete.

### 11.7 Reranker scores against the best sub-query, not the surface query

Multi-hop bridge chunks are often semantically distant from the
surface question (the answer chunk is "Bob Seger" while the surface
is "what is the stage name?"). The cross-encoder reranker therefore
scores `(_best_sub_query, chunk)` rather than `(surface_query, chunk)`,
where `_best_sub_query` is the sub-query for which the chunk had the
highest per-sub-query RRF rank.

### 11.8 Planner pattern classification: dependency parse + surface heuristics

The Planner uses **two complementary mechanisms** for query type
classification.

**Primary mechanism — structural patterns (Patterns E, F, G, H; L is a
relative-clause sub-form of G):** Named patterns use SpaCy dependency-parse
labels (relcl, acl, agent, auxpass, prep("of"), pobj) to fire structural
decomposition — E (relational-noun complement), F (passive-agent voice), G
(relative-clause bridge, with Form-2 "L" handling a relative-pronoun subject),
and H (chained-attribution). Two regex pre-empts run earlier in `classify()`
on closed-class English markers: I (boolean conjunction "Are X and Y both P?")
and J (implicit bridge "another [noun]"). A structural-comparison router
(coordinated NER entities under an interrogative determiner) routes the
remaining comparison forms into the parallel comparison decomposer. (Earlier
Patterns C, D and the string-shape Pattern K were removed in the 2026-05-15
audit because they keyed on surface phrasing rather than parse structure.)
These cite linguistic literature: Quirk et al. 1985; Bresnan 1982;
Karttunen 1976/1977; Partee 1995; Barker 1995; Levin 1993.

**Secondary mechanism — `MULTI_HOP_PATTERNS` (regex pre-screen):**
Before structural patterns fire, a set of surface-level regex patterns
(`MULTI_HOP_PATTERNS`) screens whether the query *may* be multi-hop.
This set includes closed-class syntactic markers ("of a/the X
that/who", possessive chains) as well as a curated list of attribution
verbs (starring, featuring, directed by, written by, composed by) and
relational nouns (father/mother/son/daughter of, creator/founder of)
that consistently signal an unresolved nominal bridge in English (Quirk
et al. 1985 §17.7-15). The pre-screen is conservative: a false
positive escalates to structural parsing, which may still emit a
single-hop plan; a false negative degrades to the generic 2-hop
fallback seeded on detected entities.

The combined approach trades theoretical elegance for empirical
robustness on HotpotQA bridge questions: structural patterns provide
defensible linguistic grounding; the surface pre-screen recovers
questions where the bridge is lexically marked but syntactically opaque
to the dependency parser (e.g., passive nominalizations, fragments).

### 11.9 Single-pass generation by default, self-correction as ablation

`agent.max_verification_iterations` defaults to **1** — the
self-correction loop is **OFF** in the production configuration. The
second pass (Self-Refine, Madaan et al. 2023) is retained as an opt-in
ablation row; its contribution is reported via the `agentic_ablation.py`
row-5 vs row-4 comparison, where on the 1.5 B SLM it does not improve EM
(consistent with the broader negative result in §11.17).

### 11.10 Single execution path (`AgentPipeline`); helpers as a namespace

An earlier iteration carried two orchestrator implementations: a
LangGraph StateGraph and a sequential fallback. Both produced identical
AgentState dicts but maintained twice the surface area. The LangGraph
mode and the sequential fallback have been removed. **`AgentPipeline`
in `src/pipeline/` is the only production orchestrator.**
`AgenticController` in `src/logic_layer/` is a stateless namespace of
bridge-handling helpers consumed by `AgentPipeline._iterative_navigate`.

### 11.11 Query-side NER normalization as a pre-consumer contract

GLiNER spans are consumed directly as graph anchors and filter tokens,
so boundary defects (an absorbed leading auxiliary, a fragmented
hyphenated name, an embedded year) propagate into retrieval. A
deterministic normalization layer repairs boundaries and gates
non-discriminative spans before use (§3.5). The rules are deliberately
narrow and closed-class (auxiliaries only, interior years only,
offset-overlap merges only) so they have no known false positives on
real entity names — wh-words and articles, which legitimately begin
titles, are not touched. This is a query-side-only contract; the
ingestion path is untouched, preserving entity-ID consistency.

### 11.12 Definite descriptions and classifier abstention as decomposition signals

Two decomposition mechanisms key on *general linguistic notions*, not
enumerated surface constructions (consistent with §11.8's structural
philosophy). (i) When a multi-hop query names no entity, the bridge
referent is given by a *definite description* — a heavily-modified noun
phrase denoting an entity by its properties (Russell 1905; Strawson
1950); the longest such entity-free phrase becomes the hop-0 query.
(ii) The classifier's no-signal sentinel (`SINGLE_HOP` at exactly the
fallback confidence) is treated as *abstention*, and only then is the
dependency parse consulted as a tie-breaker. Gating on the exact
sentinel — not a tuned threshold — guarantees the override cannot
regress any classification that had positive evidence.

### 11.13 IDF specificity and structural coverage in context ordering

The Verifier's context re-ordering combines IDF-weighted term overlap
(Spärck Jones 1972; Robertson 2004), sqrt length normalisation, and a
structural-coverage floor. IDF counters the failure where generic
category words ("magazines", "athletes") shared by many candidate
chunks dominate ranking over the rare entity term that actually
identifies the gold chunk; it is guarded to ≥ 4 candidates because
document frequency over fewer is degenerate. The coverage floor is a
*hard guarantee* (not a tuned weight) that a required entity's article
survives the context cap — restricted to distinctive entities so a
common single-word name does not over-fire.

### 11.14 Reciprocal-rank prior and abstention in bridge extraction

The bridge-entity scorer multiplies local relevance by a reciprocal
chunk-rank prior `1/(1+rank)` (Cormack et al. 2009): the retriever's own
rank is the strongest available prior for which chunk holds the bridge
referent, so an entity from a top-ranked chunk is preferred over one
from a low-ranked noise chunk. The three passes are merged into one
confidence-scored pool rather than a priority-ordered cascade, so a
low-precision heuristic can no longer short-circuit a stronger
candidate. An abstention floor (best score ≤ 0 → return none) converts a
confidently-wrong bridge — which would misdirect hop-2 retrieval — into
a neutral no-op, after which hop-2 runs on its un-rewritten sub-query.

### 11.15 Survivor floor over hard entity-mention dropping

The entity-mention filter (§4.2) is lossy by design. A survivor floor
guarantees that, when a full candidate set was retrieved, the filter
never strands the Verifier with too little context to tolerate a single
retrieval error (observed degenerate case: 10 → 2). It generalises the
top-2 RRF immunity into a floor and engages only for full candidate
sets, so it does not force-keep noise in already-small inputs. The
no-usable-entity case is made observable (logged + flagged) rather than
silently passing all chunks unfiltered.

### 11.16 Evaluation-rigor audit

Seven decisions surfaced and validated during an end-to-end audit of the
evaluation path. Each is empirically grounded — the n=100 retrieval-only
and n=40 full-pipeline A/Bs that justify them are reproducible from
`evaluation_results/` and from `src/thesis_evaluations/benchmark_datasets.py`.

**11.16.1 Verifier cap-by-RRF-first contract.** The verifier's
`_reorder_by_question_relevance` (§4.3, §11.13) now operates **inside**
the already-selected `max_docs` window — the cap is applied to the
Navigator's RRF order **before** the reorder runs. Previously the reorder
sorted the entire post-filter context and the `[:max_docs]` slice ran on
the reordered list, which let a lexical-overlap heuristic *evict* a
high-RRF answer chunk that happened to be sparse in question terms
(observed regression: a Navigator-rank-#1 chunk demoted out of the
LLM-visible window). The revised contract separates *selection*
(retrieval-score-based, RRF; Cormack et al. 2009) from *presentation
order* (within-window only, to mitigate small-LLM positional bias; Liu
et al. 2023, *Lost in the Middle*, TACL/arXiv:2307.03172). The reorder
can no longer change set membership.

**11.16.2 Per-anchor fairness merge for parallel decompositions.** When
the planner emits ≥2 parallel sub-queries (the comparison and intersection
paths emit one per anchor with `depends_on=[]`), the Navigator's final
cap interleaves the per-sub-query rankings round-robin (strongest anchor
first each round) rather than taking a purely global top-k. This
guarantees each anchor a fair share of the `max_context_chunks` budget so
a high-degree entity cannot monopolize it and crowd out the second
entity's gold paragraph. Single-hop is unchanged (no-op when only one
sub-query is active). Coverage of every decomposed aspect is the
defining requirement of a HotpotQA comparison question (Yang et al. 2018,
EMNLP); fair list interleaving follows team-draft interleaving (Radlinski
et al. 2008, CIKM).

**11.16.3 Structural-comparison classifier router (Phase 3.6).** The
classifier (§4.1, §11.8) gained a Phase-3.6 pre-empt that routes a query
with **coordinated NER entities under an interrogative determiner**
(SpaCy `conj` dependency between two entity heads + wh-word) into the
`COMPARISON` decomposer, regardless of whether a comparative-morphology
keyword fired. Without it, "Ronald Reagan and George H. W. Bush both held
which position?" scored MULTI_HOP via Phase-3 entity density alone and
was decomposed by `_decompose_multi_hop`, which mangled the coordinate
structure into a subject-less hop. The pre-empt is gated by an
INTERSECTION precedence check (defers when "both X and Y" / "in common"
fired, so joint-property questions still route correctly) and a bridge-cue
precision guard (defers when a relational/attribution cue is present, so
coordinated pairs inside bridge questions stay MULTI_HOP). Refs:
HotpotQA's two official question types are *bridge* and *comparison*
(Yang et al. 2018), and comparison is ≈ 20 % of the distribution;
coordination linguistics: Quirk et al. (1985) §13.

**11.16.4 `create_pipeline` settings-wiring fix.** The evaluation pipeline
(`benchmark_datasets.create_pipeline`) previously hand-built
`RetrievalConfig` with only `mode` and `similarity_threshold` — every
other `rag.*` / `vector_store.*` / `graph.*` key fell back to the
dataclass default. The dataclass defaults for `vector_top_k` and
`bm25_top_k` are both 10, while `settings.yaml` documents 20 (deliberate
2026-05-06 audit). Evaluation runs were therefore using **half the
documented retrieval funnel widths**. The fix calls the canonical
settings-reading factory `create_hybrid_retriever(...)` and then overrides
`mode` only for the per-config ablation. After the fix, every retrieval
knob — `vector_top_k`, `bm25_top_k`, `rrf_k`, `cross_source_boost`,
`enable_bm25`, `graph.max_hops`, `graph.top_k_entities`, `graph.enable_hop3`,
`graph.hub_mention_cap`, `rag.vector_weight` / `graph_weight` /
`bm25_weight` — is honoured in evaluation. Single-source-of-truth (§3.5)
is now enforced end-to-end for the retrieval layer.

**Scope of the fix — read before citing the headline numbers.** This was
a *wiring* defect, not a defect in the documented configuration: with the
fix in place, evaluation reads `vector_top_k=20` / `bm25_top_k=20` from
`settings.yaml` exactly as documented. **All headline EM / Soft-EM / SF
metrics reported in this document and in `evaluation_results/` were
produced with the corrected funnel** (the buggy code path no longer
exists in the tree). The bug therefore does *not* contaminate the
reported results; it is recorded here as a rigor-audit trail, not an open
caveat on the numbers. Two residual notes: (1) the `RetrievalConfig`
dataclass defaults for `vector_top_k` / `bm25_top_k` remain `10` — these
defaults are a *fallback for a config-less construction* and are never
hit on the evaluation path, which always loads `settings.yaml`; the
`_REQUIRED_SETTINGS` guard (§11.16.5) WARNs if either key is missing, so a
silent fallback is observable. (2) The factory `create_hybrid_retriever`
likewise resolves both knobs from `settings.yaml`; the `, 10)` fallbacks
in its `.get(...)` calls only fire on an absent key (same guard applies).

**11.16.4a `cross_source_boost` (β) is ablation-overridable.** The
cross-source RRF boost β (the extra RRF credit a chunk earns when more than
one lane surfaces it; production default 1.2) is exposed as a per-run
override on `benchmark_datasets.create_pipeline(cross_source_boost=...)` so
the β sensitivity sweep (`src/thesis_evaluations/cross_source_boost_sweep.py`)
can vary it **without editing `settings.yaml`**. `None` (the default) reads
the `settings.yaml` value, so the production path is unchanged. This mirrors
the per-source weight overrides (`vector_weight` / `graph_weight` /
`bm25_weight`) already used by the modality ablation, keeping every retrieval
knob ablatable from a single factory rather than via config edits.

**11.16.5 `_REQUIRED_SETTINGS` 7 → 37 (reproducibility guard).**
`src/logic_layer/_settings_loader.py::_validate_settings` validates a tuple of
required keys at load time and emits a `WARNING` (not a fatal error) when
any is missing — a silent dataclass-default fallback is treated as a
reproducibility risk. The list grew from 7 keys to 37, covering every
parameter that meaningfully affects EM or SF metrics: LLM context budget,
embeddings, vector store, graph (incl. `hub_mention_cap`, `hub_fanout_cap`,
`search_budget_seconds`, `enable_hop3`), RAG fusion + BM25, the entire
Navigator filter chain, Verifier validation thresholds, agent pipeline flags,
entity extraction, ingestion, and the benchmark Soft-EM threshold. This is the
guard that would have caught the §11.16.4 bug. The expansion is purely
defensive — the check only WARNs at runtime; the CI `repro-guard` job
(`.github/workflows/tests.yml`) promotes a missing key to a hard push failure
via `-W error::UserWarning`, and `scripts/check_frozen_config.py` enforces
`frozen_paper.yaml` ↔ `settings.yaml` key-schema parity in the same job.

**11.16.6 Delivery-loss instrumentation and Soft-EM verdict.** The
per-question evaluation record now carries two complementary correctness
checkpoints that separate retrieval-stage loss from delivery-stage loss
from the SLM ceiling:

- `all_gold_retrieved` — gold present in the Navigator's filter-chain
  output (≤ `max_context_chunks=8`).
- `gold_in_final_context` — gold present in the post-`max_docs`
  LLM-visible window (top-5). The gap is **delivery loss** — gold the
  pipeline retrieved but discarded before the model saw it.

The aggregator reports `final_context_recall_rate` and
`delivery_loss_rate` alongside the existing SF metrics. Measured on
n=100 retrieval-only at audit time: SF-Recall 65 %, gold-in-LLM-window
29 %, delivery loss 36 %.

**These three figures are an audit-time snapshot, not the current
ceiling.** They were taken *before* the iterative-multi-hop navigation
path (`AgentPipeline._iterative_navigate`, §4.4) and the eval-harness
SF-title-capture fix (`_install_retriever_title_capture`, §10.2) were
wired in. Those changes feed each hop's resolved bridge entity into the
next hop's sub-query and into the reranker's entity hints, raising
retrieval-only SF-Recall on the 20-question bridge slice from ~45 % to
~55 % overall (bridge subset ~29 %→~43 %). Any external analysis that
quotes the 65 %/29 %/36 % triplet as the system's standing performance is
reading a superseded snapshot; the delivery-loss *mechanism* (gold at
navigator rank 6–8 that neither RRF nor the cross-encoder favours, §11.16.7)
is unchanged, but the absolute recall numbers moved.

The headline answer-correctness metric is **Soft-EM**: token-F1 ≥
`benchmark.answer_f1_threshold` (default 0.6, configurable). Strict EM
systematically under-counts correct answers that differ from gold only
by a missing trailing category word (predicted `"<title>"` vs gold
`'"<title>" campaign'` — F1=0.8, Soft-EM=True). Both EM and Soft-EM
are reported. `tools/diagnose_verbose.py` reads the same threshold from
`settings.yaml`, so per-trace verdicts agree with the aggregate
evaluation.

**11.16.7 Negative results preserved (no-op changes that were tested and
rejected).** Four hypotheses for closing the 35 % retrieval-loss /
36 % delivery-loss bottlenecks were implemented, A/B-validated, and
**reverted** when the data did not support them. They are documented
here so the artifact contains the evidence trail and the discussion
chapter can cite them as honest stress-tests:

- **`graph.enable_hop3=true`** (3-hop graph traversal) — flat SF-Recall
  (64 % vs 65 %). The "missing second gold paragraph" failures (the
  bridge pair has no semantic or co-occurrence edge in the graph) are
  not reachable by a third graph hop either; the edges do not exist —
  a graph-completeness limit, not a depth limit.
- **`max_docs` widening 5 → 8** + `max_context_chars` 3500 → 6000 (full
  pipeline, n=40) — **delivery loss fully recovered** (32.5 % → 0 %) but
  Soft-EM *dropped* (35.0 % → 32.5 %) and per-question latency rose
  35× (39 s → 1384 s). The 1.5 B Q4 SLM uses additional context
  *worse*, not better — consistent with Liu et al. (2023). `max_docs=5`
  is therefore the correct setting under the edge-deployment constraint.
- **Rank-fusion reranker** (replace pure cross-encoder sort with RRF
  fusion of CE-rank and RRF-rank; n=100 retrieval-only) — `gold-in-LLM-
  window` unchanged at 29 %. The gold chunks landing at navigator
  rank 6–8 are ranked low by **both** RRF and the cross-encoder; rank
  fusion cannot lift what neither signal favours.
- **Phase 3.7 structural-bridge classifier router** (rescue bridge
  questions from `temporal`/`single_hop` collapse on weak cues; n=100
  retrieval-only) — bridge SF-Recall rose +2.5 pp **but** comparison
  dropped −4.8 pp, `gold-in-LLM-window` dropped 4 pp and delivery loss
  worsened 5 pp. The rescued plans run hop-2 retrieval, which adds
  chunks to the Navigator output, which pushes some gold further down
  the ranking, which the `max_docs=5` cap then cuts.

The common pattern across all four: cheap algorithmic levers for the
retrieval/delivery bottlenecks hit a downstream wall — either the SLM
ceiling, the cap, or both. Together with §11.16.6's measured 50 %
SoftEM | all-gold-retrieved, this establishes the **SLM as the
binding constraint** for further EM improvements on HotpotQA at the
1.5 B Q4 scale; further pipeline-side gains would require a larger
verifier or a tool-augmented verification channel
(CRITIC-style — Gou et al. 2024, ICLR, arXiv:2305.11738) and are
documented as future work.

**Levers that are *implemented*, not future work (anti-confusion note).**
Because §11.16.7 catalogues rejected experiments, an external reader
skimming it has mistaken several *shipped* retrieval levers for missing
opportunities. To be explicit, the following are already in the tree and
on by default unless noted, and are the wrong things to propose as new
work: (a) **per-source RRF weighting** — `rag.vector_weight /
graph_weight / bm25_weight` on `RetrievalConfig`, threaded through
`RRFFusion` (§11.2); equal-weight 1/1/1 is the default, but the knobs
exist for weighted ablations including down-weighting a collapsed lane.
(b) **Bridge-aware / entity-hint reranking** — the cross-encoder scores
`(_best_sub_query, chunk)` and accepts `[ENTITIES: …]` hints from the
Planner (§11.7); this *is* bridge-conditioned reranking. (c) **Iterative
multi-hop retrieval** — `AgentPipeline._iterative_navigate` runs hops
sequentially, extracts the bridge entity after each hop, and rewrites the
next hop's sub-query with it (`_rewrite_hop_query_with_bridges`); this
*is* two-stage bridge-conditioned retrieval. What is *not* enabled and is
genuine future work: knowledge-base / embedding entity linking for
PERSON/ORG (§3.6.1 — embedding-based linking is *disabled by default* with
an empirical justification, not merely unbuilt; the canonical-form
normalizer is always on), a heterogeneous-model verifier (§11.17), and a
tool-augmented verification channel.

### 11.17 Verifier-contribution investigation

§11.16 established the SLM as the binding constraint. This section reports
the dedicated investigation of whether the **agentic verification layer
itself** — independent of retrieval — can improve answer accuracy at the
1.5 B edge scale. The headline outcome is a rigorously-ablated **negative
result**, which the paper treats as a primary contribution (cf. Huang et
al. 2024, *LLMs Cannot Self-Correct Reasoning Yet*, ICLR — a top-venue
negative result on the same family of question).

**11.17.1 Verifier-only test harness (retrieval frozen).** To attribute
EM deltas to the Verifier alone, retrieval is run once and cached
(`verifier_cache_build.py`), then Verifier configurations are swept
against the *identical* chunks (via `agentic_ablation.py`). Because set
membership is constant across variants, any EM/F1 difference is the
Verifier's doing — a cleaner isolation than a full-pipeline ablation
where retrieval co-varies. A `--no-planner` cache provides the bare-RAG
(highest-EM) retrieval substrate for the decisive comparisons.

**11.17.2 Failure-mode taxonomy.** `verifier_failure_taxonomy.py` joins a
sweep with its cache and buckets every wrong answer: on the n=50 no-CoT
sweep, the 31 errors were 35 % **retrieval-miss** (gold absent from the
chunks), 32 % **grounded-hallucination** (the answer entity is in the
chunks but answers a different question), 16 % **false-abstention**, 16 %
**close-miss** (F1≥0.5, EM=0). The retrieval-miss bucket — the largest —
is **structurally unreachable by any answer-side verifier**, which bounds
the achievable verification gain a priori.

**11.17.3 Toggle/prompt mechanisms do not raise EM.** Across two
independent verifier-only sweeps on fixed retrieval:
- **Structured (slot-filling) CoT** (Khot et al. 2023): −4 pp EM on both
  sweeps — the SLM follows the scaffold but still confabulates. Default OFF.
- **RECOMP-style context distillation** (Yu et al. 2024): **retrieval-
  dependent** — +8 pp on planner-decomposed (noisy) context, but −6 pp on
  bare-RAG (clean) context, where it drops information the LLM would have
  used. Net-negative on the best retrieval; default OFF.
- **Few-shot answer exemplars** (generic, non-leaky): +2 pp only.
- **bge-reranker-base** over MiniLM: −8 pp on the bare-RAG row (over-
  reranks the flat-RRF distribution). Reverted to MiniLM.
- **Confidence gate** (Architecture A): no-op — bridge questions have a
  near-flat top-RRF distribution, so the score-gap signal never fires.

**11.17.4 The unconditional-swap failure and the keep-best discipline.**
A QA-conditioned NLI grounding gate plus anti-abstention retry, when first
implemented to **replace** the answer unconditionally, was **net-negative
(−8 pp)**: it fired on 24 % of questions and the retry turned 6 of 50
*correct* answers wrong while fixing 3. The fix is a **keep-best guard**
(§4.3): a retry is accepted only if it is a real answer and (for the
grounding gate) scores strictly higher entailment. With the guard, a
re-run broke 0 and fixed 0 — confirming the guard is correct (no
regressions) but that the gate, on this corpus, has nothing to add. The
generalisable lesson — any answer-revision step must be keep-best guarded,
because the verifier cannot tell which answers are already correct — was
then applied to all retries (bridge / format / anti-abstention).

The guard's necessity is **dataset-dependent in sign but uniform in
direction**, confirmed at n=500 by `scripts/selfcorrect_iteration_probe.py`
(pure re-analysis of the +Verifier vs +SelfCorrect ablation rows). On the
questions where a 2nd verifier iteration fired, the extra pass nets
**−4 on HotpotQA** (3 fixed, 7 broken) but **+6 on 2WikiMultiHopQA** (16
fixed, 10 broken; overall EM 21.0 % → 23.6 %). So a second self-correction
pass is *not* uniformly harmful — but on both datasets it breaks a
non-trivial number of already-correct answers, which is exactly what the
keep-best guard caps. The honest reading: the guard is justified on both
datasets (it bounds the downside regardless of sign), while the *accuracy
benefit* of extra iterations is small and inconsistent — the empirical basis
for the production `max_iterations=1` default (§4.3, self-correction loop)
and the keep-best discipline together.

**11.17.5 Verification-triggered re-retrieval is neutral on HotpotQA.**
The re-retrieval loop (§4.4) was A/B-tested (loop off vs on in one process,
n=50, bare-RAG): EM 40.0 % → 40.0 %, broke 0 / fixed 0, fire-rate 24 %,
**accept-rate 0 %**. It fired mostly on already-correct low-confidence
answers (the keep-best guard protected them) and on genuine misses where
re-search of the curated **10-document** distractor pool surfaced no new
chunk. The loop is sound and safe but idle here; an open-corpus setting
(2WikiMultiHopQA, full-Wiki) is required for it to have room to work and
is identified as future work.

**11.17.6 Why post-hoc verification cannot beat retrieval-only here.**
Two structural causes, each independently sufficient: (i) the dominant
failure is *retrieval-miss*, which answer-checking cannot reach; and
(ii) the verifier and generator are the **same 1.5 B model**, so the
verifier's judgements are correlated with the generator's errors and add
no independent signal — at `max_iterations=1` claim verification only
relabels confidence, and at temperature 0 a second pass is near-identical.
The verifier's defensible contribution is therefore **trust, not
accuracy**: calibrated confidence enables selective prediction (abstain
on low-confidence questions, trading coverage for precision) that
single-pass RAG cannot offer — a property of direct value for edge
deployment, and the subject of the planned risk-coverage evaluation
(§12.2). Production default is the **minimal verifier**; all mechanisms in
§11.17.3–§11.17.5 remain in the codebase as opt-in, tested ablation rows.

---

## 12. Known Limitations and Future Work

### 12.1 Out of scope for this paper's evaluation

- Replacing GLiNER with a fine-tuned NER model.
- Training a learned reranker on the target corpus.
- Streaming generation output to the user during evaluation.
- Multi-GPU inference.

### 12.2 Documented limitations

1. **REBEL relation coverage.** REBEL is trained on Wikipedia
   infoboxes and emits Wikidata-style predicates ("date_of_birth").
   It misses narrative predicates ("X founded Y") that SVO recovers
   from the dependency parse, but neither captures arbitrary
   open-information predicates. The paper discusses this as a recall
   ceiling on the graph-retrieval side.

2. **Pre-validation contradiction detection is disabled by default.**
   The Verifier's NLI-based contradiction check requires a 270 MB
   model download incompatible with the edge-deployment constraint.
   The Navigator's numeric-divergence filter on the same context is
   enabled instead. The Verifier-side check remains as an opt-in
   ablation only.

3. **Pre-empts are not in the four-phase scoring classifier.**
   Patterns I (boolean conjunction) and J (anaphoric "another") run
   as deterministic pre-empts before the SpaCy-Matcher scoring
   pipeline because the function-word markers ("both", "another")
   are unambiguous English structural signals and the scoring
   pipeline would otherwise misclassify them. Every pre-empt firing
   is logged via `RetrievalPlan.classifier_preempt` so its
   false-positive rate is auditable per question.

4. **Chunking ablation is one-dimensional.** The
   `chunking_ablation.py` script varies `(sentences_per_chunk,
   sentence_overlap)` while holding the graph (Phase-2 NER+RE output
   and KuzuDB store) constant. The result therefore measures the
   vector-retrieval component's sensitivity to chunking, not the full
   end-to-end system. This framing is explicit in the script's
   output summary.

5. **Coreference impact is not measured empirically.** Pre-chunking
   coreference resolution is enabled by default (when the resolver is
   installed) on the qualitative argument that pronoun-dropped
   mentions are unrecoverable downstream. The magnitude of the
   effect on graph density is dataset- and resolver-dependent and is
   not part of this paper's quantitative evaluation.

6. **Embedding-based entity linking disabled.** As documented in
   §3.6.1, the empirical probe on the post-ingest entity set showed
   nomic-embed-text producing merge rates of 90–94 % on
   PERSON / LOCATION / GPE at every threshold tested up to 0.99, with
   single clusters absorbing more than one thousand unrelated
   entities. The linker was therefore disabled for the paper-release
   ingest and alias resolution reduces to `canonical_form`
   exact-match deduplication. Aliases that differ in surface form
   beyond canonical normalisation (acronym ↔ expansion; nickname or
   short form ↔ full legal name) remain separate graph nodes. A
   discriminative in-type linker (e.g. SapBERT, BLINK) is identified
   as future work.

7. **Phase 3d.5 latency at scale is engineering-mitigated but
   experimentally unverified on real data.** The bulk-redirect helper
   `_redirect_entity_edges_bulk` and per-bucket checkpointing replace
   the per-edge MERGE pattern that previously made Phase 3d.5 a
   multi-hour bottleneck (measured 14–16 h on a ~46 000-entity
   HotpotQA graph). Because the phase is now disabled by default per
   limitation 6, the expected order-of-magnitude speedup of the new
   path is not reflected in any end-to-end ingest measurement
   reported in this paper. Unit tests pin correctness on synthetic
   graphs.

8. **Post-hoc agentic verification does not improve EM at 1.5 B
   (§11.17).** This is the central, intentional negative result of the
   paper, established by a dedicated verifier-only harness across two
   sweeps and an A/B of the re-retrieval loop. It is *not* a defect to be
   fixed within scope: the dominant failure (retrieval-miss) is
   unreachable by answer-checking, and a same-model verifier shares the
   generator's errors. The production default is the minimal verifier;
   every verification mechanism is retained as an opt-in, tested
   ablation. The paper frames the verifier's contribution as trust
   (calibration / selective prediction), not accuracy.

### 12.3 Future work (beyond this paper)

The paper stands on the architecture, the hybrid-retrieval result, the
verification ablation, and the negative finding. Three extensions are
identified as future work and are explicitly out of this paper's
time/compute budget:

1. **Risk-coverage / selective-prediction evaluation.** Turn the
   verifier's confidence (and calibration: ECE, AUROC vs correctness)
   into accuracy@coverage and abstention-precision curves across all
   three datasets. Substantiates the trust contribution (§11.17.6) from
   data the pipeline already logs; no new model runs.
2. **Open-corpus re-retrieval.** Re-run the §4.4 loop on 2WikiMultiHopQA /
   full-Wikipedia retrieval, where query expansion has room to surface a
   missed document — the honest test of whether verification-triggered
   re-retrieval is a positive EM lever once the corpus is not a curated
   10-document pool.
3. **Model-scale crossover study.** Repeat the §11.17 verification
   ablation at 1.5 B → 3 B → 7 B → 14 B to locate the capability
   threshold at which self-verification flips from net-negative to
   net-positive (generator–verifier error decorrelation). This is the
   route from a single-model negative result to a generalisable law and
   requires GPU/cloud inference for the larger tiers.

---

## References (in-document citations)

- Barker, C. (1995). *Possessive Descriptions.* CSLI Publications.
- Bresnan, J. (1982). "The Passive in Lexical Theory." In Bresnan ed.,
  *The Mental Representation of Grammatical Relations*, MIT Press.
- Cabot, P. L. H., & Navigli, R. (2021). "REBEL: Relation Extraction
  by End-to-end Language generation." EMNLP 2021 Findings.
- Church, K. W., & Hanks, P. (1990). "Word Association Norms, Mutual
  Information and Lexicography." *Computational Linguistics* 16(1).
- Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
  "Reciprocal Rank Fusion outperforms Condorcet and individual rank
  learning methods." SIGIR '09.
- Dong, X. L., et al. (2014). "Knowledge Vault." KDD.
- Edge, D., et al. (2024). "From Local to Global: A Graph RAG
  Approach to Query-Focused Summarization." arXiv:2404.16130.
- Feng, X., et al. (2023). "Kùzu Graph Database Management System."
  CIDR 2023.
- Gutiérrez, B. J., et al. (2024). "HippoRAG: Neurobiologically
  Inspired Long-Term Memory for Large Language Models." NeurIPS;
  arXiv:2405.14831.
- Honnibal, M., & Montani, I. (2017). "spaCy 2." arXiv:1802.04016.
- Karttunen, L. (1976). "Discourse Referents." In McCawley ed.,
  *Syntax and Semantics 7*, Academic Press.
- Karttunen, L. (1977). "Syntax and Semantics of Questions."
  *Linguistics and Philosophy* 1(1).
- Khattab, O., et al. (2022). "Demonstrate-Search-Predict."
  arXiv:2212.14024.
- Kryscinski, W., et al. (2020). "Evaluating the Factual Consistency
  of Abstractive Text Summarization." EMNLP 2020.
- Levin, B. (1993). *English Verb Classes and Alternations.* University
  of Chicago Press.
- Lewis, P., et al. (2020). "Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks." NeurIPS 2020.
- Liu, N. F., et al. (2024). "Lost in the Middle: How Language Models
  Use Long Contexts." *TACL* 12; arXiv:2307.03172.
- Madaan, A., et al. (2023). "Self-Refine." NeurIPS 2023;
  arXiv:2303.17651.
- Malkov, Y. A., & Yashunin, D. A. (2018). "Efficient and robust
  approximate nearest neighbor search using hierarchical navigable
  small world graphs." IEEE TPAMI.
- Niu, F., et al. (2012). "Elementary: Large-scale Knowledge-Base
  Construction." *AI Magazine* 33(3).
- Partee, B. (1995). "Lexical Semantics and Compositionality." In
  Gleitman & Liberman eds., *An Invitation to Cognitive Science:
  Language*, MIT Press.
- Quirk, R., Greenbaum, S., Leech, G., & Svartvik, J. (1985). *A
  Comprehensive Grammar of the English Language.* Longman.
- Reimers, N., & Gurevych, I. (2019). "Sentence-BERT." EMNLP 2019;
  arXiv:1908.10084.
- Robertson, S. (2004). "Understanding inverse document frequency: on
  theoretical arguments for IDF." *Journal of Documentation* 60(5).
- Russell, B. (1905). "On Denoting." *Mind* 14(56).
- Salton, G., & Buckley, C. (1988). "Term-weighting approaches in
  automatic text retrieval." *Information Processing & Management*
  24(5).
- Spärck Jones, K. (1972). "A statistical interpretation of term
  specificity and its application in retrieval." *Journal of
  Documentation* 28(1).
- Strawson, P. F. (1950). "On Referring." *Mind* 59(235).
- Trivedi, H., et al. (2023). "Interleaving Retrieval with
  Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step
  Questions" (IRCoT). ACL; arXiv:2212.10509.
- Weischedel, R., et al. (2013). "OntoNotes Release 5.0." LDC2013T19.
- Welford, B. P. (1962). "Note on a method for calculating corrected
  sums of squares and products." *Technometrics* 4(3).
- West, R., & Leskovec, J. (2012). "Human Wayfinding in Information
  Networks." WWW.
- Yang, Z., et al. (2018). "HotpotQA: A Dataset for Diverse,
  Explainable Multi-hop Question Answering." EMNLP 2018. *Evaluation
  benchmark — no system component is dataset-specific.*
- Zaratiana, U., et al. (2023). "GLiNER: Generalist Model for Named
  Entity Recognition using Bidirectional Transformer."
  arXiv:2311.08526.
