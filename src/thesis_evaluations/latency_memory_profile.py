"""
Per-query latency and memory profile for the edge-RAG pipeline.

Defends the title's "Resource-Constrained Devices" claim by instrumenting
every query with per-stage wall-clock (S_P / S_N / S_V) and peak resident
memory, then producing distributions plus latency-budget compliance.

For each question:
    1. Snapshot harness RSS before the query.
    2. Run pipeline.process(question), pulling per-stage timings from the
       PipelineResult (planner / navigator / verifier time_ms).
    3. Poll harness RSS in a background thread so the peak during the query
       is captured.
    4. Sample the ollama model footprint once after the query (see note).
    5. Compute within-budget rate against the latency budget.

Aggregated outputs (mean / median / p95 / max):
    - per-stage latency
    - peak harness RSS (Python process: retrieval stack)
    - ollama RSS (server + runner: model weights + KV cache)
    - delta harness RSS per query (working-set churn signal)

IMPORTANT -- what the two memory columns measure
-------------------------------------------------
Ollama serves the LLM in its OWN process(es); this script measures TWO
distinct numbers:
  * peak_rss_mb    -- harness peak: the Python evaluation process (LanceDB /
                      KuzuDB / embedding buffers / retriever). The
                      retrieval-stack working set, polled during the query.
  * ollama_rss_mb  -- summed RSS across ollama server + runner processes:
                      the loaded model weights + KV cache. Sampled ONCE
                      after each query (model warm, footprint ~constant) so
                      the heavy process enumeration never pollutes the timed
                      region.
The minimum hardware spec for this configuration is the SUM of the two.
They are reported separately because they have different quantization
sensitivity: harness is invariant; ollama scales with model choice. For a
per-quantization comparison see quantization_sweep.py.

Why these metrics
-----------------
- Per-stage latency  : which stage dominates the cost (usually S_V).
- p95 / max latency  : tail behaviour matters more than mean on edge
                       devices -- a single slow query exceeds budget.
- Peak RSS           : bounds the minimum RAM spec (harness + ollama).
- Within-budget rate : direct edge-feasibility metric.

Exports
-------
- profile_one_query(pipeline, question, budget_seconds) -- single-query profile
- aggregate(records, budget_seconds)                    -- mean/median/p95/max
- write_outputs(records, agg, output_dir)               -- summary.{md,json}, CSV, JSONL
- main()                                                -- CLI entry point

Dependencies / Requirements
---------------------------
- src.thesis_evaluations.benchmark_datasets   -- shared eval primitives
- ollama server reachable                     -- per-query LLM calls
- psutil (optional)                            -- RSS columns; None if absent
- tqdm                                        -- progress bar

Usage (single line; -X utf8 required on Windows / PowerShell):
    python -X utf8 -m src.thesis_evaluations.latency_memory_profile --dataset hotpotqa --samples 50 --model qwen2:1.5b --budget-seconds 60

Outputs
-------
evaluation_results/latency_memory_<dataset>_<ts>/
    per_query.jsonl       one record per question (full breakdown)
    per_stage.csv         pivot: query x stage x ms
    summary.md / .json    aggregate distributions + budget compliance

Last reviewed: 2026-06-01 (audit pass, project version 5.5)
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.thesis_evaluations.benchmark_datasets import (  # noqa: E402
    StoreManager,
    TestQuestion,
    _classify_llm_error,
    _gold_titles_from_supporting_facts,
    _install_retriever_title_capture,
    _retrieved_titles_for_chunks,
    compute_exact_match,
    compute_f1,
    create_pipeline,
    load_config_file,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# Why: bytes -> MiB conversion for RSS reporting.
_BYTES_PER_MB = 1024 * 1024

# Why: per-source RRF weights default to 1.0 so the profile runs under the
# SAME retrieval contract as the headline run; otherwise per-query latency
# is not comparable to the headline. settings.yaml overrides when present.
_DEFAULT_VECTOR_WEIGHT = 1.0
_DEFAULT_GRAPH_WEIGHT = 1.0

# Why: defaults for the CLI. The budget matches the production ollama
# request timeout, so within-budget rate is a true timeout-rate proxy.
_DEFAULT_SAMPLES = 50
_DEFAULT_BUDGET_SECONDS = 60.0

# Why: 50ms poll interval. Fine-grained enough to catch the peak of a
# 5-50s query while cheap (a single cached Process RSS read per tick, so it
# does not steal cycles from the timed query thread).
_RSS_POLL_INTERVAL_S = 0.05

# Why: p95 is the headline tail-latency quantile for the edge-budget claim.
_P95_QUANTILE = 0.95

# Why: emergency LLM-model fallback used only when settings.yaml omits
# llm.model_name AND no --model flag is given. Matches the value in the
# sibling scripts (benchmark_datasets, agentic_ablation, chunking_ablation)
# so the four scripts agree on the default.
_DEFAULT_MODEL_FALLBACK = "qwen2:1.5b"

# Why: project-anchored eval-results root so the CLI works regardless of cwd.
# _PROJECT_ROOT is defined above; pattern matches the sibling scripts.
_EVAL_RESULTS_ROOT = _PROJECT_ROOT / "evaluation_results"
_DEFAULT_OUTPUT_DIR = _EVAL_RESULTS_ROOT / "runs" / "latency_memory"


# ---------------------------------------------------------------------------
# Memory measurement (background poller + ollama snapshot)
# ---------------------------------------------------------------------------

try:
    import psutil  # type: ignore
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False


def _ollama_rss_mb() -> Optional[float]:
    """Summed RSS (MB) of every ollama process -- the loaded-model footprint.

    None if no ollama process is reachable (remote host) or psutil is
    unavailable. Enumerates processes once per call; callers must invoke it
    OUTSIDE the timed region (process enumeration is heavy enough to perturb
    latency if polled).
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
        # psutil.process_iter() can raise platform-specific errors (Windows
        # access violations, Linux /proc churn). Memory measurement is
        # non-critical: degrade silently to None rather than fail the run.
        return None
    return total if found else None


class _RSSPoller:
    """Polls THIS process's RSS in a background thread, recording the max.

    Why a thread? Python-level instrumentation of long-running LLM calls
    cannot sample memory mid-call any other way. The poll reads a single
    cached Process object, so it is cheap and does not perturb the timed
    query. Measures the harness (retrieval stack) only -- the model lives
    in the ollama process (see `_ollama_rss_mb`).
    """

    def __init__(self, interval_s: float = _RSS_POLL_INTERVAL_S) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_mb: float = 0.0

    def _poll(self) -> None:
        proc = psutil.Process(os.getpid()) if _PSUTIL_OK else None
        while not self._stop.is_set():
            if proc is not None:
                try:
                    rss_mb = proc.memory_info().rss / _BYTES_PER_MB
                    if rss_mb > self.peak_mb:
                        self.peak_mb = rss_mb
                except Exception:  # noqa: BLE001
                    # Per-tick read can fail transiently on Windows. Skip
                    # this tick and keep polling — peak_mb stays at its
                    # last successful sample, which is the safe behaviour.
                    pass
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        if not _PSUTIL_OK:
            return
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        if self._thread is None:
            return 0.0
        self._stop.set()
        self._thread.join(timeout=0.5)
        return self.peak_mb


# ---------------------------------------------------------------------------
# Per-query profiler
# ---------------------------------------------------------------------------

def profile_one_query(pipeline, q: TestQuestion, budget_seconds: float) -> Dict[str, Any]:
    """Run one query through the pipeline and capture latency / memory."""
    poller = _RSSPoller()
    poller.start()
    pre_rss = (
        psutil.Process(os.getpid()).memory_info().rss / _BYTES_PER_MB
        if _PSUTIL_OK else None
    )

    start = time.time()
    try:
        result = pipeline.process(q.question)
        elapsed = time.time() - start
        error = None
    except Exception as exc:  # noqa: BLE001
        # pipeline.process() can raise anything from network errors
        # (Ollama) through LangChain wrapping to malformed plans. Record
        # the exception text in the JSONL so the failure is auditable, and
        # let the loop continue to the next question.
        result = None
        elapsed = time.time() - start
        error = str(exc)

    peak_rss = poller.stop()
    # Sampled OUTSIDE the timed region: model footprint is ~constant while
    # warm, so one heavy process enumeration here never perturbs latency.
    ollama_rss = _ollama_rss_mb()

    rec: Dict[str, Any] = {
        "question_id": q.id,
        "question": q.question,
        "gold_answer": q.answer,
        "question_type": q.question_type,
        "total_time_ms": elapsed * 1000.0,
        "within_budget": elapsed <= budget_seconds,
        "pre_rss_mb": pre_rss,
        "peak_rss_mb": peak_rss if peak_rss > 0 else None,
        "delta_rss_mb": (peak_rss - pre_rss) if (pre_rss and peak_rss > 0) else None,
        "ollama_rss_mb": ollama_rss,
        "error": error,
    }

    if result is None:
        rec.update({
            "planner_time_ms": 0.0,
            "navigator_time_ms": 0.0,
            "verifier_time_ms": 0.0,
            "predicted_answer": "",
            "exact_match": False,
            "f1_score": 0.0,
            "llm_error": True,
            "llm_error_type": "exception",
        })
        return rec

    pred = getattr(result, "answer", "") or ""
    em = compute_exact_match(pred, q.answer)
    f1 = compute_f1(pred, q.answer)
    llm_err, llm_err_type = _classify_llm_error(pred)

    # Retrieval quality
    nav = getattr(result, "navigator_result", {}) or {}
    filtered = nav.get("filtered_context", []) if isinstance(nav, dict) else []
    gold_titles = _gold_titles_from_supporting_facts(q.supporting_facts)
    retrieved_titles = _retrieved_titles_for_chunks(filtered)
    all_gold = bool(gold_titles) and set(gold_titles).issubset(set(retrieved_titles))

    rec.update({
        "planner_time_ms": float(getattr(result, "planner_time_ms", 0.0) or 0.0),
        "navigator_time_ms": float(getattr(result, "navigator_time_ms", 0.0) or 0.0),
        "verifier_time_ms": float(getattr(result, "verifier_time_ms", 0.0) or 0.0),
        "predicted_answer": pred,
        "exact_match": em,
        "f1_score": f1,
        "llm_error": llm_err,
        "llm_error_type": llm_err_type,
        "retrieval_count": len(filtered),
        "all_gold_retrieved": all_gold,
    })
    return rec


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _stats(values: List[float]) -> Dict[str, float]:
    """mean / median / p95 / max for a non-empty list."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    vs = sorted(values)
    p95_idx = max(0, int(_P95_QUANTILE * (len(vs) - 1)))
    return {
        "mean": statistics.mean(vs),
        "median": statistics.median(vs),
        "p95": vs[p95_idx],
        "max": vs[-1],
    }


def aggregate(records: List[Dict[str, Any]], budget_seconds: float) -> Dict[str, Any]:
    if not records:
        return {}
    n = len(records)
    total_ms = [r["total_time_ms"] for r in records if r.get("total_time_ms") is not None]
    planner_ms = [r["planner_time_ms"] for r in records if r.get("planner_time_ms") is not None]
    nav_ms = [r["navigator_time_ms"] for r in records if r.get("navigator_time_ms") is not None]
    ver_ms = [r["verifier_time_ms"] for r in records if r.get("verifier_time_ms") is not None]
    peak_rss = [r["peak_rss_mb"] for r in records if r.get("peak_rss_mb")]
    delta_rss = [r["delta_rss_mb"] for r in records if r.get("delta_rss_mb")]
    ollama_rss = [r["ollama_rss_mb"] for r in records if r.get("ollama_rss_mb")]

    within_budget = sum(1 for r in records if r.get("within_budget")) / n
    em_rate = sum(1 for r in records if r.get("exact_match")) / n
    llm_err_rate = sum(1 for r in records if r.get("llm_error")) / n
    all_gold_rate = sum(1 for r in records if r.get("all_gold_retrieved")) / n

    return {
        "n_queries": n,
        "budget_seconds": budget_seconds,
        "within_budget_rate": within_budget,
        "exact_match_rate": em_rate,
        "llm_error_rate": llm_err_rate,
        "all_gold_retrieved_rate": all_gold_rate,
        "total_ms": _stats(total_ms),
        "planner_ms": _stats(planner_ms),
        "navigator_ms": _stats(nav_ms),
        "verifier_ms": _stats(ver_ms),
        "peak_rss_mb": _stats(peak_rss),
        "delta_rss_mb": _stats(delta_rss),
        "ollama_rss_mb": _stats(ollama_rss),
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_outputs(records: List[Dict[str, Any]], agg: Dict[str, Any],
                  output_dir: Path) -> None:
    # Per-query JSONL
    jsonl_path = output_dir / "per_query.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Per-stage CSV (one row per query)
    csv_path = output_dir / "per_stage.csv"
    fieldnames = [
        "question_id", "question_type",
        "planner_time_ms", "navigator_time_ms", "verifier_time_ms",
        "total_time_ms", "within_budget",
        "peak_rss_mb", "delta_rss_mb", "ollama_rss_mb",
        "exact_match", "f1_score", "llm_error", "all_gold_retrieved",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    # Markdown summary
    md = ["# Latency / Memory Profile", ""]
    md.append(f"- Queries: **{agg['n_queries']}**")
    md.append(f"- Budget: **{agg['budget_seconds']:.0f}s**")
    md.append(f"- Within-budget rate: **{agg['within_budget_rate']*100:.1f}%**")
    md.append(f"- EM rate: **{agg['exact_match_rate']*100:.1f}%**")
    md.append(f"- LLM error rate: **{agg['llm_error_rate']*100:.1f}%**")
    md.append(f"- All-gold-retrieved rate: **{agg['all_gold_retrieved_rate']*100:.1f}%**")
    md.append("")
    md.append("## Per-stage latency (ms)")
    md.append("")
    md.append("| Stage | Mean | Median | P95 | Max |")
    md.append("|---|---|---|---|---|")
    for stage_key, stage_name in [
        ("planner_ms", "S_P (Planner)"),
        ("navigator_ms", "S_N (Navigator)"),
        ("verifier_ms", "S_V (Verifier)"),
        ("total_ms", "**Total**"),
    ]:
        s = agg.get(stage_key, {})
        md.append(
            f"| {stage_name} | {s.get('mean', 0):.0f} | {s.get('median', 0):.0f} "
            f"| {s.get('p95', 0):.0f} | {s.get('max', 0):.0f} |"
        )
    md.append("")
    md.append("## Resident memory (MB)")
    md.append("")
    md.append("| Metric | Mean | Median | P95 | Max |")
    md.append("|---|---|---|---|---|")
    s = agg.get("peak_rss_mb", {})
    md.append(
        f"| Harness peak RSS | {s.get('mean', 0):.0f} | {s.get('median', 0):.0f} "
        f"| {s.get('p95', 0):.0f} | {s.get('max', 0):.0f} |"
    )
    s = agg.get("ollama_rss_mb", {})
    md.append(
        f"| Ollama (model) RSS | {s.get('mean', 0):.0f} | {s.get('median', 0):.0f} "
        f"| {s.get('p95', 0):.0f} | {s.get('max', 0):.0f} |"
    )
    s = agg.get("delta_rss_mb", {})
    md.append(
        f"| delta-RSS / query | {s.get('mean', 0):.0f} | {s.get('median', 0):.0f} "
        f"| {s.get('p95', 0):.0f} | {s.get('max', 0):.0f} |"
    )
    md.append("")
    md.append("**Reading the table:**")
    md.append("- p95 latency is the headline edge-feasibility metric -- if "
              "p95 < budget, the system is reliable on edge hardware.")
    md.append("- The dominant stage (almost always S_V) is the optimization target.")
    md.append("- Minimum RAM spec = harness peak RSS + ollama (model) RSS. "
              "delta-RSS / query indicates harness working-set churn.")

    (output_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Per-query records: %s", jsonl_path)
    logger.info("Per-stage CSV:    %s", csv_path)
    logger.info("Summary MD:       %s", output_dir / "summary.md")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", "-d", default="hotpotqa")
    parser.add_argument("--samples", "-n", type=int, default=_DEFAULT_SAMPLES)
    parser.add_argument("--model", "-m", default=None,
                        help="LLM model name (default: from settings.yaml).")
    parser.add_argument("--budget-seconds", type=float, default=_DEFAULT_BUDGET_SECONDS,
                        help="Edge-device latency budget for within-budget %% "
                             "computation. Default: 60s (matches the ollama timeout).")
    parser.add_argument("--output", "-o", type=str,
                        default=str(_DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    if not _PSUTIL_OK:
        logger.warning(
            "psutil not installed — peak-memory columns will be empty. "
            "Install with: pip install psutil"
        )

    config = load_config_file()
    store_manager = StoreManager()
    if not store_manager.dataset_exists(args.dataset):
        logger.error("Dataset not ingested: %s", args.dataset)
        return

    model_name = args.model or config.get("llm", {}).get("model_name", _DEFAULT_MODEL_FALLBACK)
    # Deterministic prefix slice of the dataset's natural order (no shuffle).
    questions = store_manager.load_questions(args.dataset)[: args.samples]
    if not questions:
        logger.error("No questions loaded.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{args.output}_{args.dataset}_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Model: %s | Questions: %d | Budget: %.0fs | Output: %s",
                model_name, len(questions), args.budget_seconds, output_dir)

    # Same retrieval contract as the headline run, read from settings.yaml.
    rag_cfg = config.get("rag", {})
    vector_weight = float(rag_cfg.get("vector_weight", _DEFAULT_VECTOR_WEIGHT))
    graph_weight = float(rag_cfg.get("graph_weight", _DEFAULT_GRAPH_WEIGHT))

    pipeline = create_pipeline(
        args.dataset, config, store_manager,
        vector_weight=vector_weight, graph_weight=graph_weight,
        model_name=model_name,
    )
    # Capture source titles for SF-retrieval tracking.
    original_retrieve = _install_retriever_title_capture(pipeline)

    records: List[Dict[str, Any]] = []
    try:
        for q in tqdm(questions, desc="Profiling", unit="q"):
            try:
                records.append(profile_one_query(pipeline, q, args.budget_seconds))
            except Exception as exc:  # noqa: BLE001
                # Per-question crash isolation: a single query's failure
                # must not abort the whole profile run. Log a warning and
                # move on — partial data is still useful for the summary.
                logger.warning("Q%s crashed: %s", q.id, exc)
    finally:
        # Restore retriever and clean up the pipeline so the process exits
        # cleanly. Attribute order mirrors _install_retriever_title_capture
        # (AgentPipeline exposes it as `hybrid_retriever`).
        if original_retrieve is not None:
            for attr in ("hybrid_retriever", "retriever", "_retriever"):
                cand = getattr(pipeline, attr, None)
                if cand is not None and hasattr(cand, "retrieve"):
                    cand.retrieve = original_retrieve
                    break
        del pipeline
        gc.collect()

    agg = aggregate(records, args.budget_seconds)
    write_outputs(records, agg, output_dir)
    logger.info("Done. Inspect: %s/summary.md", output_dir)
    # Auto-refresh the canonical paper bundle for THIS dataset. Safe-wrapped:
    # a bundling failure must never break the eval that just produced the data.
    try:
        from .thesis_results_aggregator import update_bundle
        update_bundle(dataset=args.dataset)
    except Exception:  # noqa: BLE001
        # Auto-bundle is a convenience refresh. The per_query.jsonl and
        # summary.md just written are the source of truth and stay valid
        # whether the bundle is re-rendered or not.
        logger.debug("Auto-bundle skipped", exc_info=True)


if __name__ == "__main__":
    main()
