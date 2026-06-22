# Paper Evaluation Suite

This directory contains every evaluation needed to defend the paper:

> **Edge-RAG: Empirical Characterization of When Knowledge-Graph Lanes
> Add Value in CPU-Only Hybrid Retrieval**

The scripts are designed to be run from the project root and produce
LaTeX-ready tables and figures for the final manuscript.

---

## What lives here

| File | Role | What it produces |
|---|---|---|
| `benchmark_datasets.py` | Core evaluation engine — the runner all other scripts call into. Computes EM/F1/SF-F1/SF-Recall/EM\|retr.ok per question. | Per-config aggregates + per-question JSONL. Used standalone for ad-hoc evaluations. |
| `quantization_sweep.py` | **Tier 1A** — runs the pipeline against several Ollama models/quantizations. | Quantization × model table + per-model JSONL + summary.json. |
| `agentic_ablation.py` | **Tier 1B** — the headline paper table. Decomposes the pipeline (LLM-only → +Retrieval → +Planner → +Verifier → +SelfCorrect). | 5-row marginal-Δ table. |
| `latency_memory_profile.py` | **Tier 1C** — instruments every query with per-stage timing and peak RSS. | Per-query CSV + distribution stats + budget-compliance rate. |
| `thesis_results_aggregator.py` | Reads the three Tier-1 outputs and emits LaTeX tables + matplotlib figures. Includes a coverage validator. | `table_*.tex`, `figure_*.png`, `coverage_report.md`. |

---

## How the scripts map to the paper title

| Capability | Supporting script(s) | Specific evidence |
|---|---|---|
| *Quantized Small Language Models* | `quantization_sweep.py` | Multi-model comparison; latency/memory vs. accuracy. |
| *Agentic Verification* | `agentic_ablation.py` | Marginal contribution per agent component (Planner, Verifier, Self-Correction). |
| *Reasoning Fidelity* | `benchmark_datasets.py` (SF-F1 column) | SF-F1 isolates retrieval; EM\|retr.ok isolates LLM reasoning. |
| *Hybrid Retrieval-Augmented Generation* | `benchmark_datasets.py ablation` (vector / graph / hybrid weights) | Existing `ABLATION_CONFIGS` — vector_only / graph_only / hybrid 50-50 / hybrid 70-30. |
| *Resource-Constrained Devices* | `latency_memory_profile.py` | p95 latency, peak RSS, within-60s-budget rate. |

---

## Recommended execution order

These scripts share the ingested HotpotQA store, so **ingest once first**:

```powershell
python -X utf8 local_importingestion.py `
    --chunks data/hotpotqa/chunks_export.json `
    --extractions data/hotpotqa/graph/extraction_results.json `
    --dataset hotpotqa --resume
```

Then run the Tier-1 scripts. Order is independent — they don't share state
beyond the ingested store and `config/settings.yaml`.

### 1. Agentic ablation (the most important table — run this first)

```powershell
python -X utf8 -m src.thesis_evaluations.agentic_ablation `
    --dataset hotpotqa --samples 100 --model qwen2:1.5b
```

Expected runtime: 100 questions × 5 configurations × ~10 s/question = ~85 min.
Output: `evaluation_results/agentic_ablation_<ts>/summary.md`.

### 2. Quantization sweep

Make sure each model is pulled first: `ollama pull qwen2:1.5b`, etc.

```powershell
python -X utf8 -m src.thesis_evaluations.quantization_sweep `
    --dataset hotpotqa --samples 100 `
    --models "qwen2:1.5b,qwen2.5:3b,llama3.2:3b,phi3"
```

Expected runtime: 100 questions × 4 models × ~10–35 s/question.
For a faster pilot, use `--samples 30`.

### 3. Latency / memory profile

```powershell
python -X utf8 -m src.thesis_evaluations.latency_memory_profile `
    --dataset hotpotqa --samples 50 --budget-seconds 60
```

Expected runtime: 50 questions × ~10 s/question = ~10 min.
Use the smallest sample size that gives stable p95 estimates (≥50).

### 4. Aggregate everything for the paper

```powershell
python -X utf8 -m src.thesis_evaluations.thesis_results_aggregator
```

Output: `evaluation_results/thesis_final_<ts>/` with `.tex` files,
`.png` figures, and a `coverage_report.md` that confirms every paper
claim has supporting data.

---

## Per-question JSONL — what's inside

Every script writes one JSONL file with one line per question. Fields:

| Field | Source | Use in paper |
|---|---|---|
| `question_id`, `question`, `gold_answer`, `predicted_answer` | dataset / pipeline | Examples in the appendix. |
| `exact_match`, `f1_score` | gold-comparison | Standard metrics. |
| `retrieval_count`, `retrieved_titles`, `gold_titles` | navigator output + retriever hook | Supporting-fact tracking. |
| `retrieval_recall`, `retrieval_precision`, `sf_f1` | computed from titles | Pipeline retrieval quality. |
| `all_gold_retrieved` | derived | Threshold for "pipeline succeeded". |
| `llm_error`, `llm_error_type` | sentinel detection | Separates timeout/api errors from wrong answers. |
| `pipeline_succeeded_llm_failed` | derived | The paper's key cross-tab cell. |
| `planner_query_type`, `hop_count`, `n_entities` | S_P output | Failure-mode analysis. |
| `verifier_iterations`, `all_verified`, `confidence` | S_V output | Self-correction efficacy. |
| `time_ms` | wall-clock | Latency. |

Filter with `jq` for paper examples:

```bash
# All questions where retrieval succeeded but LLM timed out:
jq 'select(.pipeline_succeeded_llm_failed)' \
    evaluation_results/agentic_ablation_*/row5_self_correct.jsonl

# All bridge-type questions where SF-F1 < 0.5:
jq 'select(.question_type=="bridge" and .sf_f1 < 0.5)' \
    evaluation_results/quantization_sweep_*/qwen2-1.5b.jsonl
```

---

## What's NOT here (Tier 2/3 — optional)

These are valuable but **not required for the paper**. Each
costs roughly a week to implement; do them only if you have spare time
after the Tier-1 scripts.

- **Multi-dataset evaluation** (2WikiMultiHopQA, TriviaQA). The benchmark
  engine already supports 2Wiki — re-run any Tier-1 script with
  `--dataset 2wiki` once you've ingested it.
- **Statistical-significance tests** (paired bootstrap CI). Only useful
  when the ablation deltas are small (< 5 pp EM).
- **Cloud-LLM baseline** (GPT-4 / Claude). Contradicts the edge-device
  claim; better as a "future work" note.
- **AWQ / GPTQ separate quantization pipelines.** Ollama's Q4 / Q8 GGUF
  variants already satisfy the title's "Quantized" claim without inventing
  a second runtime.

---

## Output directory structure (after running everything)

```
evaluation_results/
├── quantization_sweep_20260514_120000/
│   ├── qwen2-1.5b.jsonl
│   ├── qwen2.5-3b.jsonl
│   ├── phi3.jsonl
│   ├── summary.csv
│   ├── summary.md
│   └── summary.json
├── agentic_ablation_20260514_140000/
│   ├── row1_llm_only.jsonl
│   ├── row2_rag_no_agent.jsonl
│   ├── row3_planner.jsonl
│   ├── row4_verifier.jsonl
│   ├── row5_self_correct.jsonl
│   ├── summary.csv
│   ├── summary.md
│   └── summary.json
├── latency_memory_20260514_160000/
│   ├── per_query.jsonl
│   ├── per_stage.csv
│   ├── summary.md
│   └── summary.json
└── thesis_final_20260514_180000/
    ├── table_quantization.tex
    ├── table_ablation.tex
    ├── table_latency.tex
    ├── figure_pareto.png
    ├── figure_ablation_waterfall.png
    ├── figure_stage_breakdown.png
    ├── coverage_report.md
    └── README_HOW_TO_USE.md
```

---

## Common issues

- **`ModuleNotFoundError: src.thesis_evaluations`** — run from project root,
  not from inside `src/`.
- **Ollama timeouts** — `qwen2:1.5b` is safe within 60s; `phi3` and
  `llama3.2:3b` are borderline. Reduce `--samples` or stop other processes.
- **psutil not installed** — peak-RSS columns are blank. Fix with
  `pip install psutil`. The scripts still run otherwise.
- **matplotlib not installed** — aggregator skips figures. Fix with
  `pip install matplotlib`. Tables still generate.
