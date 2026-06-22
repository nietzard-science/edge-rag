"""
Quantization / model-size sweep for the edge-RAG pipeline.

Defends the title's "Quantized Small Language Models on Edge Devices" claim
by running the full pipeline against several Ollama model variants and
producing a side-by-side comparison of answer quality, latency, model
memory, and retrieval quality.

For each model in --models:
    1. Build the pipeline once with that model.
    2. Run `n_samples` questions through it (same retrieval contract as the
       headline run -- per-source RRF weights read from settings.yaml).
    3. Capture EM, F1, SF-F1, SF-recall, LLM-error rate, latency, model RSS.
    4. Write a per-model JSONL (per-question) plus a summary row.
After all models: aggregate into summary.{csv,md,json}.

Metric rationale
----------------
- EM / F1            : final answer quality (user-visible).
- SF-F1 / SF-recall  : retrieval quality, independent of the LLM.
- EM | retrieval-ok  : LLM accuracy CONDITIONED on correct retrieval --
                       isolates model capability from pipeline noise.
- Avg latency        : edge-device feasibility against the latency budget.
- Model RSS          : quantized-model memory footprint (see note below).
- LLM-error rate     : how often the model times out under the budget.

IMPORTANT -- what the memory columns measure
---------------------------------------------
Ollama serves model weights in its OWN process(es); this script talks to it
over HTTP. Two distinct numbers are recorded:
  * peak_rss_mb     -- summed RSS of the ollama server + runner processes,
                       i.e. the quantized MODEL footprint. This is the
                       number the "Quantized ... Edge Devices" claim needs.
  * harness_rss_mb  -- RSS of THIS Python process (LanceDB / KuzuDB /
                       embedding buffers / retriever), reported separately
                       for transparency.
Caveat: model RSS is sampled right after this model's eval. If ollama's
keep-alive retains a previously evaluated model, the figure may include it;
for a fully isolated reading set OLLAMA_KEEP_ALIVE=0 or run one model per
process. Returns None (rendered "-") when no ollama process is reachable
(e.g. a remote ollama host) or psutil is absent.

Exports
-------
- run_one_model(...)        -- run the pipeline with one model, return metrics
- write_summary(rows, dir)  -- emit summary.{csv,md}
- main()                    -- CLI entry point

Dependencies / Requirements
---------------------------
- src.thesis_evaluations.benchmark_datasets   -- shared eval primitives
- ollama server with each --models entry pulled (`ollama pull <name>`)
- psutil (optional)                            -- RSS columns; None if absent

Usage (single line; -X utf8 required on Windows / PowerShell):
    python -X utf8 -m src.thesis_evaluations.quantization_sweep --dataset hotpotqa --samples 100 --models qwen2:1.5b,qwen2.5:3b,phi3

Outputs
-------
evaluation_results/quantization_sweep_<dataset>_<ts>/
    <model>.jsonl     per-question records
    summary.csv / .md / .json

Last reviewed: 2026-06-01 (audit pass, project version 5.5)
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the project root importable when this file is run with `python -m ...`.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.thesis_evaluations.benchmark_datasets import (  # noqa: E402
    StoreManager,
    TestQuestion,
    create_pipeline,
    evaluate_dataset,
    load_config_file,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# Why: per-source RRF weights default to 1.0 (vanilla equal-weight RRF) so the
# sweep uses the SAME retrieval contract as the headline run; otherwise per-
# model EM/F1 would not be comparable to the headline numbers. settings.yaml
# overrides when present.
_DEFAULT_VECTOR_WEIGHT = 1.0
_DEFAULT_GRAPH_WEIGHT = 1.0

# Why: bytes -> MiB conversion for RSS reporting.
_BYTES_PER_MB = 1024 * 1024

# Why: summary fields that render as percentages in the Markdown table.
# Explicit set avoids a numeric-magnitude heuristic.
_PERCENT_FIELDS = frozenset({
    "em", "f1", "sf_f1", "sf_recall", "em_given_retrieval_ok",
    "llm_error_rate", "pipeline_failed_rate", "pipeline_ok_llm_failed_rate",
    "pipeline_ok_llm_wrong_rate", "pipeline_ok_llm_ok_rate",
})

# Why: CLI defaults centralised so a no-flag run is self-documenting and the
# values do not drift between argparse and the help-text examples.
_DEFAULT_SAMPLES = 100
_DEFAULT_MODELS = "qwen2:1.5b,qwen2.5:3b,phi3"

# Bit-width sweep (P0-T5): isolate the effect of QUANTIZATION on a single model
# by iterating precision tags of the SAME base model, instead of comparing
# different model architectures. Ollama exposes precision via tag suffixes,
# e.g. qwen2:1.5b == qwen2:1.5b-q4_K_M (the default), qwen2:1.5b-q8_0, and a
# 16-bit `-f16`/`-fp16` build where available. `--model X --bitwidths a,b,c`
# expands to the variant tags so the sweep produces a Q4/Q8/fp16 table.
_BARE_MODEL_BITWIDTH = "q4_k_m"   # the precision an untagged Ollama pull resolves to

# Edge-memory envelope the paper targets (the "Resource-Constrained Devices"
# claim). Higher-precision variants are checked against this so "Q8/fp16 busts
# the 16 GB budget" is reported as an explicit finding rather than inferred.
_MEMORY_BUDGET_MB = 16 * 1024     # 16 GiB

# Smoke mode: a tiny run to measure real per-query cost BEFORE committing to a
# full n=500 sweep. fp16 on CPU can be far slower per token than Q4 and may
# thrash near the budget, so the smoke run prints a projected full-run wall
# time the user can sanity-check against their compute window.
_SMOKE_SAMPLES = 20

# Why: project-anchored eval-results root so the CLI works regardless of cwd.
# Matches the sibling scripts (benchmark_datasets, agentic_ablation,
# chunking_ablation, latency_memory_profile).
_EVAL_RESULTS_ROOT = _PROJECT_ROOT / "evaluation_results"
_DEFAULT_OUTPUT_DIR = _EVAL_RESULTS_ROOT / "runs" / "quantization_sweep"


# ---------------------------------------------------------------------------
# Memory measurement helpers (best-effort; fall back gracefully)
# ---------------------------------------------------------------------------
try:
    import psutil  # type: ignore
    _PSUTIL_OK = True
except ImportError:
    # Defer logging to main() — importing this script (e.g. from a notebook
    # or another evaluator) must have no logging side effects. Helpers
    # return None when _PSUTIL_OK is False, and the rendered table shows
    # "-" for the affected columns, so the absence is visible downstream.
    _PSUTIL_OK = False


def _current_rss_mb() -> Optional[float]:
    """RSS of THIS Python process in MB (the harness footprint), or None.

    This is the retrieval-stack memory (LanceDB / KuzuDB / embedding
    buffers), NOT the LLM weights -- those live in the ollama process and
    are measured by `_ollama_rss_mb`.
    """
    if not _PSUTIL_OK:
        return None
    try:
        return psutil.Process(os.getpid()).memory_info().rss / _BYTES_PER_MB
    except Exception:  # noqa: BLE001
        # psutil read can fail with platform-specific errors. Memory
        # measurement is non-critical: degrade silently to None rather
        # than fail the sweep.
        return None


def _ollama_rss_mb() -> Optional[float]:
    """Summed RSS (MB) of every ollama process, i.e. the quantized-model
    footprint that the edge-memory claim depends on. None if no ollama
    process is reachable (remote host) or psutil is unavailable.

    Why summed: ollama runs a server process plus one or more runner
    subprocesses (`ollama_llama_server`) that actually hold the GGUF
    weights; the model footprint is their combined RSS.
    """
    if not _PSUTIL_OK:
        return None
    total = 0.0
    found = False
    try:
        for proc in psutil.process_iter(["name"]):
            try:
                name = (proc.info.get("name") or "").lower()
                if "ollama" in name:
                    total += proc.memory_info().rss / _BYTES_PER_MB
                    found = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:  # noqa: BLE001
        # psutil.process_iter can raise platform-specific errors (Windows
        # access violations, Linux /proc churn). The model-RSS column is
        # informational, so degrade silently to None.
        return None
    return total if found else None


# ---------------------------------------------------------------------------
# Bit-width expansion (P0-T5)
# ---------------------------------------------------------------------------

def expand_bitwidths(base_model: str, bitwidths: List[str]) -> List[Dict[str, str]]:
    """Expand a base model + bit-width list into Ollama (tag, label) variants.

    Ollama encodes precision as a tag suffix. The bare model resolves to its
    default quantization (Q4_K_M for the qwen2/qwen2.5 builds), so the bit-width
    matching ``_BARE_MODEL_BITWIDTH`` maps to the bare tag; every other bit-width
    becomes ``<base>-<bitwidth>``. Returns one dict per variant with:
        tag       -- the Ollama model tag to pull/run (e.g. "qwen2:1.5b-q8_0")
        bitwidth  -- the precision label for the summary table (e.g. "q8_0")
        base      -- the shared base model (so the table can group by it)

    The caller is responsible for having pulled each tag; a missing tag fails
    that one variant's eval and is omitted from the summary (per-cell isolation).
    """
    out: List[Dict[str, str]] = []
    for bw in bitwidths:
        bw_norm = bw.strip()
        if not bw_norm:
            continue
        if bw_norm.lower() == _BARE_MODEL_BITWIDTH:
            tag = base_model
        else:
            tag = f"{base_model}-{bw_norm}"
        out.append({"tag": tag, "bitwidth": bw_norm, "base": base_model})
    return out


# ---------------------------------------------------------------------------
# Per-question resume (sleep-survivable long sweeps)
# ---------------------------------------------------------------------------

def _load_done_question_ids(jsonl_path: Path) -> set:
    """Set of question_ids already recorded in jsonl_path (empty if absent)."""
    if not jsonl_path.exists():
        return set()
    ids: set = set()
    try:
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    qid = json.loads(line).get("question_id")
                except json.JSONDecodeError:
                    continue
                if qid is not None:
                    ids.add(qid)
    except OSError:
        return set()
    return ids


def _filter_done_questions(questions, jsonl_path: Path):
    """Return (remaining_questions, n_already_done), preserving input order."""
    done = _load_done_question_ids(jsonl_path)
    if not done:
        return list(questions), 0
    remaining = [q for q in questions if q.id not in done]
    return remaining, len(questions) - len(remaining)


def _aggregate_model_from_jsonl(jsonl_path: Path) -> Optional[Dict[str, Any]]:
    """Recompute a variant's headline metrics from its on-disk JSONL.

    Resume-aware: aggregates every record in the (possibly merged old+new)
    JSONL, so a restarted run reports the full question set. Returns None if the
    file is missing/empty (a variant whose every question errored). Mirrors the
    metric subset the quantization summary/table actually render.
    """
    if not jsonl_path.exists():
        return None
    records: List[Dict[str, Any]] = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        return None
    n = len(records)
    if n == 0:
        return None

    def _rate(pred) -> float:
        return sum(1 for r in records if pred(r)) / n

    retr_ok = [r for r in records if r.get("all_gold_retrieved")]
    return {
        "n_questions": n,
        "em": _rate(lambda r: r.get("exact_match")),
        "f1": sum(r.get("f1_score", 0.0) for r in records) / n,
        "sf_f1": sum(r.get("sf_f1", 0.0) for r in records) / n,
        "sf_recall": _rate(lambda r: r.get("all_gold_retrieved")),
        "em_given_retrieval_ok": (
            sum(1 for r in retr_ok if r.get("exact_match")) / len(retr_ok)
            if retr_ok else 0.0
        ),
        "llm_error_rate": _rate(lambda r: r.get("llm_error")),
        "avg_time_ms": sum(r.get("time_ms", 0.0) for r in records) / n,
    }


# ---------------------------------------------------------------------------
# Single-cell runner
# ---------------------------------------------------------------------------

def run_one_model(
    model_name: str,
    dataset: str,
    questions: List[TestQuestion],
    config: Dict[str, Any],
    store_manager: StoreManager,
    output_dir: Path,
    bitwidth: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Run the pipeline with one model and return summary metrics.

    The pipeline is built fresh for each model. Model weights live in the
    ollama process, so per-model memory is read via `_ollama_rss_mb`
    (sampled after this model's eval, when its weights are warm), while the
    Python harness footprint is read via `_current_rss_mb`.
    """
    logger.info("=" * 70)
    logger.info("MODEL: %s", model_name)
    logger.info("=" * 70)

    # Same retrieval contract as the headline run, read from settings.yaml.
    rag_cfg = config.get("rag", {})
    vector_weight = float(rag_cfg.get("vector_weight", _DEFAULT_VECTOR_WEIGHT))
    graph_weight = float(rag_cfg.get("graph_weight", _DEFAULT_GRAPH_WEIGHT))

    jsonl_path = output_dir / f"{model_name.replace(':', '-').replace('/', '_')}.jsonl"

    # Per-question resume: a long bit-width sweep (fp16 ~45 s/q) is easily
    # interrupted by a laptop sleep. Keep the on-disk JSONL and only run the
    # questions not yet recorded, so a restart continues instead of redoing the
    # whole variant. Metrics are recomputed from the MERGED on-disk JSONL below
    # (not from this call's return value), so they reflect old + new records.
    questions_to_run, n_done = _filter_done_questions(questions, jsonl_path)
    if n_done:
        logger.info("RESUME %s: %d/%d already on disk; running %d new",
                    model_name, n_done, len(questions), len(questions_to_run))

    start = time.time()
    pipeline = None
    if questions_to_run:
        pipeline = create_pipeline(
            dataset, config, store_manager,
            vector_weight=vector_weight, graph_weight=graph_weight,
            model_name=model_name,
        )
        try:
            evaluate_dataset(
                dataset, questions_to_run, pipeline,
                config_name=model_name,
                vector_weight=vector_weight, graph_weight=graph_weight,
                jsonl_out=jsonl_path,
                retrieval_only=False,
            )
        except Exception as exc:  # noqa: BLE001
            # evaluate_dataset can raise on Ollama timeout, missing model
            # tag, malformed plan, or any LangChain-layer error. We catch
            # everything so a single model's failure does not abort the
            # whole sweep — the partial JSONL on disk is preserved and the
            # next --resume picks up the remaining questions.
            logger.error("Pipeline failed on model %s: %s", model_name, exc)
    elapsed = time.time() - start
    # Rebuild this variant's metrics from the merged on-disk JSONL (old + new)
    # so a resumed run reports the full question set, not just this call's slice.
    result = _aggregate_model_from_jsonl(jsonl_path)

    # Sample memory while this model's weights are still warm in ollama.
    # IMPORTANT: only meaningful when THIS attempt actually ran questions — on a
    # pure resume (all questions already on disk, questions_to_run empty) no LLM
    # call was made, so ollama has the model unloaded and the sample catches
    # only the idle server stub (~tens of MB), which is misleading. In that case
    # report None so the table shows "-" rather than a bogus footprint. The
    # authoritative model-memory number is latency_memory_profile (~2 GB peak).
    if questions_to_run:
        model_rss = _ollama_rss_mb()      # quantized-model footprint (warm)
    else:
        model_rss = None
        logger.info("RSS for %s not sampled (pure resume — model not warm; "
                    "use latency_memory_profile for memory)", model_name)
    harness_rss = _current_rss_mb()   # Python retrieval-stack footprint

    # Free the pipeline so the next model's harness measurement isn't polluted.
    del pipeline
    gc.collect()

    if not result:
        return None

    # Memory-budget check (P0-T5): does this variant's model footprint fit the
    # edge envelope? None when RSS is unreadable (remote ollama / no psutil),
    # so a missing measurement is not mistaken for "within budget".
    within_budget: Optional[bool] = (
        (model_rss <= _MEMORY_BUDGET_MB) if model_rss is not None else None
    )

    # `result` is a dict aggregated from the on-disk JSONL (resume-aware), not a
    # ConfigResult; access by key.
    summary = {
        "model": model_name,
        "bitwidth": bitwidth,
        "n_questions": result["n_questions"],
        "em": result["em"],
        "f1": result["f1"],
        "sf_f1": result["sf_f1"],
        "sf_recall": result["sf_recall"],
        "em_given_retrieval_ok": result["em_given_retrieval_ok"],
        "llm_error_rate": result["llm_error_rate"],
        "avg_time_ms": result["avg_time_ms"],
        "total_elapsed_s": elapsed,
        # peak_rss_mb is the OLLAMA model footprint (consumed by the
        # quantization LaTeX table); harness_rss_mb is the Python process.
        "peak_rss_mb": model_rss,
        "harness_rss_mb": harness_rss,
        "memory_budget_mb": _MEMORY_BUDGET_MB,
        "within_memory_budget": within_budget,
        "jsonl_path": str(jsonl_path),
    }
    return summary


# ---------------------------------------------------------------------------
# Aggregation & output
# ---------------------------------------------------------------------------

def write_summary(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    """Emit summary.csv + summary.md from the per-model summary rows."""
    if not rows:
        logger.warning("No rows to write — all models failed.")
        return

    # CSV
    csv_path = output_dir / "summary.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Summary CSV: %s", csv_path)

    # Markdown. A "Bits" column is shown only when at least one row carries a
    # bit-width label (the P0-T5 bit-width sweep); a plain model sweep omits it.
    has_bitwidth = any(r.get("bitwidth") for r in rows)
    md_path = output_dir / "summary.md"
    cols = [("model", "Model")]
    if has_bitwidth:
        cols.append(("bitwidth", "Bits"))
    cols.extend([
        ("em", "EM"),
        ("f1", "F1"),
        ("sf_f1", "SF-F1"),
        ("sf_recall", "SF-Recall"),
        ("em_given_retrieval_ok", "EM|retr.ok"),
        ("llm_error_rate", "LLM-err"),
        ("avg_time_ms", "Latency (ms)"),
        ("peak_rss_mb", "Model RSS (MB)"),
        ("within_memory_budget", "≤16GB?"),
        ("harness_rss_mb", "Harness RSS (MB)"),
    ])
    title = ("# Bit-width Sweep Results" if has_bitwidth
             else "# Quantization Sweep Results")
    lines = [title, ""]
    lines.append("| " + " | ".join(name for _, name in cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        cells: List[str] = []
        for key, _ in cols:
            v = r.get(key)
            if v is None:
                cells.append("-")
            elif key == "within_memory_budget":
                cells.append("yes" if v else "**NO**")
            elif isinstance(v, float):
                if key in _PERCENT_FIELDS:
                    cells.append(f"{v * 100:.1f}%")
                else:
                    cells.append(f"{v:.0f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("**Column legend:**")
    if has_bitwidth:
        lines.append("- Bits: quantization precision of the same base model "
                     "(q4_k_m = Ollama default, q8_0 = 8-bit, f16 = 16-bit).")
    lines.append("- EM/F1: final answer correctness (LLM output).")
    lines.append("- SF-F1: supporting-fact F1 -- did retrieval find the right paragraphs?")
    lines.append("- SF-Recall: % of questions where ALL gold supporting paragraphs were retrieved.")
    lines.append("- EM|retr.ok: EM among questions where retrieval succeeded "
                 "(isolates LLM capability).")
    lines.append("- LLM-err: % of timeouts/API errors (model stability under the latency budget).")
    lines.append("- Model RSS: summed RSS of the ollama server + runner processes "
                 "(quantized-model memory footprint).")
    lines.append(f"- ≤16GB?: whether Model RSS fits the {_MEMORY_BUDGET_MB // 1024} GB "
                 "edge envelope (the Resource-Constrained-Devices claim).")
    lines.append("- Harness RSS: RSS of the Python evaluation process "
                 "(retrieval stack; reported for transparency).")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary Markdown: %s", md_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", "-d", default="hotpotqa",
                        help="Dataset name (default: hotpotqa)")
    parser.add_argument("--samples", "-n", type=int, default=_DEFAULT_SAMPLES,
                        help=f"Number of questions per model "
                             f"(default: {_DEFAULT_SAMPLES})")
    parser.add_argument(
        "--models", "-m", type=str,
        default=_DEFAULT_MODELS,
        help=f"Comma-separated list of Ollama model names (model-size sweep). "
             f"Each must be pulled via `ollama pull <name>`. Default: "
             f"{_DEFAULT_MODELS}. Ignored when --model + --bitwidths are given.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Single base model for a BIT-WIDTH sweep (P0-T5), e.g. qwen2:1.5b. "
             "Combine with --bitwidths to compare quantization levels of the "
             "SAME model instead of different models.",
    )
    parser.add_argument(
        "--bitwidths", type=str, default=None,
        help="Comma-separated precision tags to sweep for --model, e.g. "
             "'q4_k_m,q8_0,f16'. q4_k_m maps to the bare model tag (Ollama "
             "default); others become <model>-<bitwidth>. Each variant tag must "
             "be pulled first (e.g. `ollama pull qwen2:1.5b-q8_0`).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=f"Smoke run: force --samples={_SMOKE_SAMPLES} and print a projected "
             f"full-run (n=500) wall time per variant. Use this BEFORE a full "
             f"fp16 sweep to measure real per-query cost (fp16 on CPU can be "
             f"far slower than Q4 and may approach the memory budget).",
    )
    parser.add_argument("--output", "-o", type=str,
                        default=str(_DEFAULT_OUTPUT_DIR),
                        help="Output directory base (timestamp appended).")
    parser.add_argument(
        "--resume", nargs="?", const="__latest__", default=None, metavar="DIR",
        help="Resume an interrupted sweep into an existing run directory "
             "(survives a laptop sleep that killed the process). Pass a dir to "
             "continue it, or --resume with no value to auto-continue the most "
             "recent <output>_<dataset>_* dir. Variants/questions already on "
             "disk are skipped; only the remaining ones run.",
    )
    args = parser.parse_args()

    if not _PSUTIL_OK:
        # Surface the missing-psutil warning at run time, not at import time.
        logger.warning(
            "psutil not installed -- RSS columns will be None. "
            "Install with: pip install psutil"
        )

    config = load_config_file()
    store_manager = StoreManager()
    if not store_manager.dataset_exists(args.dataset):
        logger.error("Dataset not ingested: %s", args.dataset)
        return

    # --smoke forces the small sample size regardless of --samples so the
    # projection below is measured on a known, cheap run.
    n_samples = _SMOKE_SAMPLES if args.smoke else args.samples
    # Deterministic prefix slice of the dataset's natural order (no shuffle),
    # so a given sample size reproduces the same question set across variants.
    questions = store_manager.load_questions(args.dataset)[: n_samples]
    if not questions:
        logger.error("No questions loaded for %s", args.dataset)
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Resume into an existing dir (sleep-survivable) or start a fresh timestamped
    # one. --resume DIR uses that dir; --resume (bare) finds the newest
    # <output>_<dataset>_* dir; absent → new dir.
    if args.resume:
        if args.resume == "__latest__":
            base = Path(args.output)
            candidates = sorted(
                base.parent.glob(f"{base.name}_{args.dataset}_*"),
                key=lambda p: p.stat().st_mtime if p.exists() else 0,
            )
            if candidates:
                output_dir = candidates[-1]
                logger.info("RESUME: continuing most recent run: %s", output_dir)
            else:
                output_dir = Path(f"{args.output}_{args.dataset}_{ts}")
                logger.info("RESUME: no prior run found; starting fresh: %s", output_dir)
        else:
            output_dir = Path(args.resume)
            if not output_dir.exists():
                logger.error("--resume dir does not exist: %s", output_dir)
                return
            logger.info("RESUME: continuing run: %s", output_dir)
    else:
        output_dir = Path(f"{args.output}_{args.dataset}_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # Two modes: a BIT-WIDTH sweep (--model + --bitwidths → variants of one base
    # model) or the legacy model-size sweep (--models → different models). The
    # bit-width mode takes precedence when both --model and --bitwidths are set.
    if args.model and args.bitwidths:
        bitwidths = [b.strip() for b in args.bitwidths.split(",") if b.strip()]
        variants = expand_bitwidths(args.model, bitwidths)
        logger.info("BIT-WIDTH sweep of %s: %s",
                    args.model, [v["bitwidth"] for v in variants])
        logger.info("  -> tags: %s  (each must be `ollama pull`ed)",
                    [v["tag"] for v in variants])
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        variants = [{"tag": m, "bitwidth": None, "base": m} for m in models]
        logger.info("Model-size sweep: %s", [v["tag"] for v in variants])
    if args.smoke:
        logger.info("SMOKE mode: n=%d (projecting to n=500 after each variant)",
                    n_samples)
    logger.info("Questions per variant: %d", len(questions))

    rows: List[Dict[str, Any]] = []
    for v in variants:
        tag, bw = v["tag"], v["bitwidth"]
        try:
            row = run_one_model(
                tag, args.dataset, questions, config, store_manager, output_dir,
                bitwidth=bw,
            )
            if row:
                rows.append(row)
                budget = row.get("within_memory_budget")
                budget_str = ("" if budget is None
                              else "  [<=16GB]" if budget else "  [OVER 16GB!]")
                logger.info("  -> EM=%.1f%% F1=%.3f SF-F1=%.3f Latency=%.0fms ModelRSS=%s%s",
                            row["em"] * 100, row["f1"], row["sf_f1"],
                            row["avg_time_ms"],
                            f"{row['peak_rss_mb']:.0f}MB" if row["peak_rss_mb"] else "N/A",
                            budget_str)
                if args.smoke:
                    # Project the full n=500 wall time from this variant's
                    # measured per-question latency, so the user can decide
                    # whether the full run fits their compute window.
                    per_q_s = row["avg_time_ms"] / 1000.0
                    proj_500_min = per_q_s * 500 / 60.0
                    logger.info(
                        "     SMOKE projection for %s: %.2fs/question -> "
                        "n=500 ~= %.0f min (%.1f h)%s",
                        tag, per_q_s, proj_500_min, proj_500_min / 60.0,
                        "" if budget is not False
                        else "  -- AND exceeds the 16GB budget",
                    )
        except Exception as exc:  # noqa: BLE001
            # Per-variant crash isolation: a single cell failure (tag not
            # pulled, OOM, KuzuDB lock, …) must not abort the whole sweep.
            # The summary table simply omits the failed variant.
            logger.error("Variant %s crashed: %s", tag, exc)

    write_summary(rows, output_dir)

    # Also dump the raw rows to JSON for thesis_results_aggregator.
    (output_dir / "summary.json").write_text(
        json.dumps({"timestamp": ts, "dataset": args.dataset,
                    "n_samples": len(questions),
                    "smoke": bool(args.smoke),
                    "sweep_kind": ("bitwidth" if (args.model and args.bitwidths)
                                   else "model_size"),
                    "base_model": args.model if (args.model and args.bitwidths) else None,
                    "memory_budget_mb": _MEMORY_BUDGET_MB,
                    "rows": rows},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Done. Inspect: %s/summary.md", output_dir)
    if args.smoke:
        # A 20-question smoke run must NOT overwrite the canonical paper bundle
        # with throwaway numbers — its only purpose is the latency/budget
        # projection above. Skip the auto-refresh entirely.
        logger.info("SMOKE run complete — bundle NOT refreshed (smoke numbers "
                    "are for projection only). Re-run without --smoke for the "
                    "reportable table.")
        return
    # Auto-refresh the canonical paper bundle for THIS dataset. Safe-wrapped:
    # a bundling failure must never break the eval that just produced the data.
    try:
        from .thesis_results_aggregator import update_bundle
        update_bundle(dataset=args.dataset)
    except Exception:  # noqa: BLE001
        # Auto-bundle is a convenience refresh. The per-model JSONLs and
        # summary.md just written are the source of truth and stay valid
        # whether the bundle is re-rendered or not.
        logger.debug("Auto-bundle skipped", exc_info=True)


if __name__ == "__main__":
    main()
