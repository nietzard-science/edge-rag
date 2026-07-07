# Reproduction Guide

**Paper:** Edge-RAG: Empirical Characterization of When Knowledge-Graph Lanes Add Value in CPU-Only Hybrid Retrieval
**System version:** 5.5 (2026-06-01) | **Python:** 3.12.3 | **OS tested:** Windows 11

This document is the reproducibility contract. Following it on a clean
machine recreates the environment and the inputs that produced the paper's
numbers.

> **Artifact-integrity claim — no hidden constants.** Every parameter
> that affects evaluation results lives in `config/settings.yaml`. No
> tuning constant is hardcoded in a production code path: where a value
> is implementation detail (a regex, a closed-class linguistic list with
> a citation), it is documented in `TECHNICAL_ARCHITECTURE.md §11`. The
> startup validator `_settings._validate_settings()` checks 37 required
> keys (`_REQUIRED_SETTINGS`) and emits a WARNING if any is missing —
> silent dataclass-default fallback is treated as a reproducibility risk
> and is exactly the failure mode the validator guards against
> (`vector_store.top_k_vectors` could otherwise be silently 10 instead of
> the documented 20). Reproducing the paper's numbers therefore requires only
> `config/settings.yaml` + `requirements_frozen.txt` + the SHA-verified
> input artifacts; no patching of source is necessary or expected.
>
> **Random seed for the headline run.** The 500-question headline
> evaluation uses **`--range 0-500`** — a *deterministic* slice (first
> 500 questions of `data/hotpotqa/questions.json` in stored order), not
> a random sample. There is no seed to set; the slice is byte-stable for
> any user who passes the SHA-256 verification in Step 3. If a *random*
> sample is required (e.g. for an ablation that re-runs a subset under
> a different config), use **`--samples 500 --seed 42`** — the seed is
> auto-logged by the evaluator and the same `--seed` value reproduces
> the same question set exactly.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.12.3 | Other 3.12.x likely fine; earlier minors untested. |
| Ollama | ≥ 0.4 | <https://ollama.com> — local LLM + embedding server. |
| RAM | ≥ 8 GB | 16 GB recommended; the system targets a < 16 GB budget. |
| Disk | ≥ 10 GB | Databases + HotpotQA corpus + caches. |

---

## Step 1 — Install Python dependencies

```bash
# Exact pinned versions used during the paper's evaluation.
# BOTH commands are required — the second installs lancedb/pylance.
pip install -r requirements_frozen.txt
pip install --no-deps -r requirements_frozen_nodeps.txt

# spaCy English model (required for chunking, NER, dependency parsing).
python -m spacy download en_core_web_sm
```

> **Why two files?** `requirements_frozen.txt` holds the bulk of the pinned
> set; `requirements_frozen_nodeps.txt` holds `lancedb==0.6.13` /
> `pylance==0.10.12`, installed with `--no-deps` because their metadata
> over-pins `pyarrow<15`, which conflicts with the `pyarrow 23.x` the rest
> of the stack needs (lancedb 0.6.13 runs correctly on pyarrow 23 — verified
> against the shipped stores). Skipping the second command leaves the data
> layer unimportable (`ModuleNotFoundError: lancedb`). The Dockerfile and the
> CI repro-guard job run both commands; a manual install must too.
>
> `requirements.txt` (relaxed ranges) is for development only. Installing
> from it may resolve different versions and is **not** guaranteed to
> reproduce the reported numbers.

---

## Step 2 — Start Ollama and pull models

```bash
# Terminal 1 — start the server, keep it open.
ollama serve

# Terminal 2 — pull the models.
ollama pull qwen2:1.5b        # primary SLM for answer generation (S_V)
ollama pull nomic-embed-text  # embedding model (Artifact A)

ollama list                   # verify both are present
```

The system expects Ollama at `http://localhost:11434` (configurable via
`config/settings.yaml → llm.base_url` and `embeddings.base_url`).

---

## Step 3 — Verify the input artifacts

The pipeline depends on three per-dataset input artifacts:

```
data/hotpotqa/chunks_export.json            (Phase-1 chunking output)
data/hotpotqa/graph/extraction_results.json (Phase-2 GLiNER + REBEL output)
data/hotpotqa/questions.json                (benchmark questions)
```

> **Obtaining the artifacts.** These files are **not** in the Git repository
> (the HotpotQA extraction output alone is ~30 MB; the corpus + derived stores
> are larger) — they are distributed separately on Zenodo. The easiest path is
> the fetch script, which downloads and unpacks them into `data/<dataset>/`:
>
> ```bash
> ./scripts/fetch_data.sh          # Windows: .\scripts\fetch_data.ps1
> ```
>
> > **Artifact bundle:** `edge-rag-stores.zip`, archived at
> > **DOI 10.5281/zenodo.20807936**
> > (<https://zenodo.org/records/20807936>). Or download manually and unzip its
> > contents into `data/`.
>
> The bundle's integrity is then verified by the SHA-256 manifest below, so a
> reviewer can confirm the download matches the paper's inputs exactly.

A SHA-256 manifest of the paper's inputs is committed at `data/SHA256.txt`.
Verify your local copies match it before ingesting (any mismatch means
your inputs differ from the paper's inputs and downstream numbers will not
be comparable):

The manifest is three-column (`<sha256>  <path>  <size-bytes>`) with comment
lines, so the verifier must skip comments and read only the first two columns.

```powershell
# Windows PowerShell — recompute and diff against the manifest.
Get-Content data/SHA256.txt | Where-Object { $_.Trim() -and $_ -notmatch '^\s*#' } | ForEach-Object {
  $expected, $path, $size = ($_ -replace '\r','') -split '\s+'
  $actual = (Get-FileHash $path -Algorithm SHA256).Hash.ToLower()
  if ($actual -ne $expected.ToLower()) { Write-Host "MISMATCH $path" }
}
```

```bash
# Unix — strip comments/CRLF, rebuild a 2-column checklist for sha256sum.
grep -vE '^[[:space:]]*(#|$)' data/SHA256.txt | tr -d '\r' \
  | awk '{print $1"  "$2}' | sha256sum -c
```

If you regenerate the artifacts (re-chunk or re-extract), refresh the
manifest by re-hashing every line entry and committing the new
`data/SHA256.txt`.

---

## Step 4 — Ingest the corpus (decoupled three-phase architecture)

Ingestion is split into three phases so the GPU-bound extraction runs
separately from the CPU-only edge target.

| Phase | Command / tool | Hardware |
|---|---|---|
| 1 | `python -m src.thesis_evaluations.benchmark_datasets ingest --chunks-only` | CPU |
| 2 | `colab_extraction.py` (run in Google Colab on a GPU) | GPU |
| 3 | `python local_importingestion.py …` | CPU |

For reproduction, **Phases 1 and 2 are already done** — their outputs are
the artifacts verified in Step 3. You only run **Phase 3**:

```powershell
python -X utf8 local_importingestion.py `
    --chunks data/hotpotqa/chunks_export.json `
    --extractions data/hotpotqa/graph/extraction_results.json `
    --dataset hotpotqa `
    --clear `
    --no-entity-linking `
    --hub-threshold-ratio 0.03 `
    --cooccurrence-min-confidence 0.5
```

> **Note:** `--no-entity-linking` disables embedding-based alias resolution,
> which is the paper-release default. An empirical probe showed
> nomic-embed-text produces 90–94 % merge rates at every tested threshold
> (see §3.6.1 of TECHNICAL_ARCHITECTURE.md); the linker is therefore
> disabled and alias resolution reduces to `canonical_form` exact-match
> deduplication.

This populates `data/hotpotqa/vector/` (LanceDB) and
`data/hotpotqa/graph/` (KuzuDB). Phase 3 takes ~20–40 min on CPU; the
embedding step is accelerated by `cache/hotpotqa_embeddings.db` if a
healthy cache is present.

> **If the embedding cache is corrupted** ("database disk image is
> malformed" repeated on every batch), delete it and re-run — SQLite
> recreates a fresh one and the embeddings are recomputed identically:
> `del cache\embeddings.db cache\hotpotqa_embeddings.db`

After ingestion, confirm graph health:

```bash
python -X utf8 tools/diagnose_graph_baseline.py --dataset hotpotqa
```

Expected: `isolated_rate < 5%`, `duplicate_rate < 2%`,
`relations_per_chunk ≥ 5`.

---

## Step 5 — Run the evaluations

All evaluation entry points are under `src/thesis_evaluations/`.

### Headline benchmark (HotpotQA, 500 questions)

```powershell
python -X utf8 -m src.thesis_evaluations.benchmark_datasets evaluate `
    --dataset hotpotqa --range 0-500 --retrieval-only        # fast: SF metrics only
python -X utf8 -m src.thesis_evaluations.benchmark_datasets evaluate `
    --dataset hotpotqa --range 0-500                          # full: + Soft-EM / EM / F1
```

Writes a per-question JSONL to `evaluation_results/hotpotqa_<model>_<ts>.jsonl`
and prints a summary. The summary reports four complementary
correctness verdicts:

- **`Exact Match`** — HotpotQA-normalised EM **with two documented
  relaxations** (see `compute_exact_match` in
  `src/thesis_evaluations/benchmark_datasets.py`): (a) a multi-word gold
  answer also counts if it appears as a contiguous, word-boundary-anchored
  phrase inside the prediction (handles `"<LANDMARK>, <CITY>"` vs gold
  `"<LANDMARK>"`); (b) a yes/no gold counts if it appears as a standalone
  token within the first tokens of the prediction. It is therefore *not*
  the official strict string-equality EM; both Edge-RAG and the BM25
  baseline are scored with this same function, so all reported deltas are
  internally consistent.
- **`Answer F1`** — token-overlap F1.
- **`Soft-EM (F1 ≥ θ)`** — token-F1 ≥ `benchmark.answer_f1_threshold`
  (default `0.6`, configurable via settings). Headline answer-correctness
  metric — strict EM systematically under-counts cases where the model
  emits a substring of the gold span, e.g. predicted `"<title>"` vs gold
  `'"<title>" campaign'` (F1=0.8, Soft-EM=True). Both EM and Soft-EM are
  reported.
- **`SoftEM | all-gold-retrieved`** — Soft-EM conditional on the
  Navigator having delivered all gold paragraphs. Isolates the SLM
  ceiling from pipeline retrieval quality.

The JSONL also records two **delivery-loss** instrumentation fields
that separate retrieval-stage from delivery-stage gold
loss:

- **`all_gold_retrieved`** — gold present after the Navigator's filter
  chain (≤ `max_context_chunks=8`).
- **`gold_in_final_context`** — gold present after the Verifier's
  `max_docs=5` cap (the LLM-visible window). The gap between the two is
  "delivery loss" — gold retrieved but cut before the model saw it.

### Tier-1 ablation suite

```powershell
# Component ablation: LLM-only -> +Retrieval -> +Planner -> +Verifier -> +SelfCorrect
python -X utf8 -m src.thesis_evaluations.agentic_ablation `
    --dataset hotpotqa --samples 200 --model qwen2:1.5b

# Retrieval-mode ablation (vector / graph / hybrid) — quantifies the
# graph's marginal contribution to recall (the +33pp super-additivity
# result for hybrid is the headline justification for the architecture).
python -X utf8 -m src.thesis_evaluations.benchmark_datasets ablation `
    --dataset hotpotqa --samples 100 --retrieval-only

# Cross-model quantization sweep
python -X utf8 -m src.thesis_evaluations.quantization_sweep `
    --dataset hotpotqa --samples 100 `
    --models "qwen2:1.5b,qwen2.5:3b,llama3.2:3b,phi3"

# Latency / peak-memory profile
python -X utf8 -m src.thesis_evaluations.latency_memory_profile `
    --dataset hotpotqa --samples 50 --budget-seconds 60

# Chunking-hyperparameter ablation (retrieval-only)
python -X utf8 -m src.thesis_evaluations.chunking_ablation `
    --dataset hotpotqa --samples 100 --configs "3:1,5:1,7:1"
```

### Aggregate everything for the paper

```bash
python -X utf8 -m src.thesis_evaluations.thesis_results_aggregator
```

Reads the latest Tier-1 output directories and writes a single bundle
under `evaluation_results/thesis_final_<ts>/`:

- `table_quantization.tex`, `table_ablation.tex`, `table_latency.tex`
- `table_ablation_significance.tex` — paired-bootstrap 95 % CIs and
  p-values on each ablation-component delta.
- `significance_report.md` — plain-text companion (EM / F1 / SF-F1).
- `figure_*.png` — Pareto front, ablation waterfall, stage breakdown.
- `coverage_report.md` — confirms every paper claim has data.

### Statistical significance directly

The paired-bootstrap module can also be invoked standalone on any two
per-question JSONL files:

```powershell
python -X utf8 -m src.thesis_evaluations.bootstrap `
    evaluation_results/.../row4_verifier.jsonl `
    evaluation_results/.../row5_self_correct.jsonl `
    --metric EM
```

It prints the delta, its 95 % CI, the bootstrap p-value, and a
significance verdict.

### Single-query diagnostics

```bash
python -X utf8 tools/diagnose_verbose.py --idx 0            # full per-stage trace
python -X utf8 tools/diagnose_verbose.py --idx 0 --skip-llm # skip the Ollama call
python -X utf8 tools/diagnose_ingestion.py --indices 0      # ingestion-side trace
python -X utf8 tools/diagnose_graph_baseline.py --dataset hotpotqa --strict
```

The verbose diagnostic uses the **same Soft-EM verdict** as the benchmark
(reads `benchmark.answer_f1_threshold` from settings), so per-trace
correctness labels and aggregate numbers stay consistent.

---

## Step 6 — Verify the installation

The test suite runs without Ollama (LLM-dependent tests are marked and
deselected). The 739-test full collection covers the data layer, logic
layer, pipeline, and the agentic-controller stateless helpers. The
documented CI default uses the `not nightly` filter so the GLiNER +
REBEL model-loading tests are not required:

```bash
# CI default — fast, no model weights loaded:
python -X utf8 -m pytest test_system/ -m "not nightly" -q
# Expected: 721 passed, 18 deselected (the nightly subset).

# Full collection including model-loading tests:
python -X utf8 -m pytest test_system/ -q
# Expected: 739 passed.
```

Opt in to `-m nightly` for the long-form model-loading runs, or `-m
llm` for live-Ollama tests. The CI default matches the convention in
`pytest.ini` and `test_system/README.md`.

---

## Hyperparameter provenance & train/dev/test discipline

**Claim: no hyperparameter was tuned on the evaluated test set.** Each dataset
ships exactly 500 questions, which *are* the evaluated test set. To make the
absence of test-set tuning explicit and auditable, two things are recorded:

1. **A held-out dev band.** `data/splits/dev_band.json` reserves the **last 100
   questions** of each retrieval dataset (HotpotQA, 2WikiMultiHop — stored-file
   indices `[400, 500)`) as a dev/sanity region. Any hyperparameter ever chosen
   by inspecting an aggregate metric must be tuned on this band only. The band
   is byte-stable under the `data/SHA256.txt` verification and is regenerated /
   checked with:

   ```bash
   python -X utf8 -m src.thesis_evaluations.make_dev_split --write   # generate
   python -X utf8 -m src.thesis_evaluations.make_dev_split --verify  # CI: disjoint + complete + reproducible
   ```

   The headline numbers are reported on the **full 0–500 set** because the
   shipped settings were *not* obtained by tuning on these questions' aggregate
   metrics (provenance table below). The band therefore currently functions as a
   *reserved* hold-out — available for any future metric-driven tuning — rather
   than one that was consumed. (StrategyQA is LLM-only and has no
   retrieval-tuned knobs, so it carries no dev band.)

2. **The provenance of every non-default-able hyperparameter.** Each value below
   was set from a citation, a measured corpus property, or a single-query
   diagnostic trace (`tools/diagnose_verbose.py --idx N`) — never from the test-500
   aggregate. Every entry is documented in-line in `config/settings.yaml` and,
   where it encodes a design decision, in `TECHNICAL_ARCHITECTURE.md §11/§12`.

   | Hyperparameter | Value | Tuned on | Source |
   |---|---|---|---|
   | `rag.rrf_k` | 60 | — | Literature default (Cormack et al. 2009, SIGIR) |
   | `rag.vector_weight` / `graph_weight` / `bm25_weight` | 1.0 / 1.0 / 1.0 | — | Vanilla equal-weight RRF (no per-source tuning) |
   | `vector_store.top_k_vectors` | 20 | diagnostic traces | Bridge-answer chunks rank 11–15 under nomic score compression (§ settings comment) |
   | `vector_store.similarity_threshold` | 0.3 | — | Inert under nomic compression (all sims > 0.7); documented non-filtering default |
   | `rag.bm25_top_k` | 20 | — | Mirrors `top_k_vectors`; O(N) cost negligible |
   | `graph.top_k_entities` | 10 | — | Mirrors vector funnel; KuzuDB query <30 ms |
   | `graph.max_hops` | 2 | — | Bridge questions are 2-hop by construction (HotpotQA/2Wiki design) |
   | `graph.enable_hop3` | false | — | Off by default; +200–1000 ms, rarely helps (§ settings comment) |
   | `graph.hub_mention_cap` | 280 | corpus measurement | ≈3% of corpus chunks; live-graph degree count (§12.34 P-3) |
   | `verifier.max_docs` | 5 | — | Prompt-budget derivation: 5 × 800 chars (§ settings comment) |
   | `navigator.max_context_chunks` | 8 | — | Leaves headroom above `max_docs=5` (§ settings comment) |
   | `benchmark.answer_f1_threshold` (Soft-EM θ) | 0.6 | — | Principled close-miss threshold; flagged for sensitivity analysis (findings §10.4) |

   > **Soft-EM θ sensitivity.** θ = 0.6 is the one threshold a reviewer is most
   > likely to probe. It was *not* selected to maximise reported Soft-EM; a
   > sensitivity sweep over θ ∈ {0.5, 0.6, 0.7} on the dev band is the
   > recommended robustness check (run the benchmark with
   > `--answer-f1-threshold` and compare on `dev_band.json` ids).

---

## Reproducibility checklist (before submission)

- [ ] `pip install -r requirements_frozen.txt` succeeds on a clean venv.
- [ ] SHA-256 manifest verification (Step 3) reports no mismatches.
- [ ] Settings validator on startup emits **no** `_REQUIRED_SETTINGS`
      WARNING — every guarded key is present in `config/settings.yaml`.
- [ ] Phase-3 ingest completes; `tools/diagnose_graph_baseline.py` reports the
      invariants within threshold.
- [ ] The headline benchmark + Tier-1 ablations have been run with a
      fixed random seed (use `--range 0-500` for deterministic slicing;
      `--samples N --seed X` for reproducible random sampling).
- [ ] `make_dev_split --verify` passes (dev band disjoint, complete,
      reproducible) and the provenance table above is current.
- [ ] `thesis_results_aggregator` reports `ALL CLAIMS COVERED`.
- [ ] `requirements_frozen.txt`, `data/SHA256.txt`, and
      `data/splits/dev_band.json` are committed.

---

## Dataset provenance

**HotpotQA** — fullwiki dev set v1.1 (Yang et al. 2018, EMNLP).
The per-question artifacts in `data/hotpotqa/` (`chunks_export.json`,
`questions.json`) are derived from this set; their SHA-256 checksums are
recorded in `data/SHA256.txt`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: KuzuDB not available` | kuzu not installed | Install from `requirements_frozen.txt`. |
| `ConnectionError: localhost:11434` | Ollama not running | `ollama serve`. |
| `Model 'qwen2:1.5b' not found` | Model not pulled | `ollama pull qwen2:1.5b`. |
| `IO exception: Could not set lock on file ... graph_KuzuDB` | A previous pipeline still holds the KuzuDB exclusive lock | Stop the other process (or wait for it to release); ablation runs auto-close pipelines between configs. |
| `database disk image is malformed` (repeated) | Corrupted embedding cache | `del cache\*.db`, re-run. |
| Unicode errors on Windows | Missing UTF-8 flag | Always run with `python -X utf8`. |
| GLiNER first-run slow | One-time model download (~250 MB) | Wait; cached afterwards. |
| `_validate_settings: required key 'X.Y' absent` WARNING | Key missing from `config/settings.yaml`; system fell back to a dataclass default | Add the key to `settings.yaml`. Reproducibility is not guaranteed otherwise. |
| SHA-256 verification (Step 3) reports MISMATCH | Local inputs differ from the manifest | Obtain the manifest-matching artifacts, or regenerate and re-run all evals. |
