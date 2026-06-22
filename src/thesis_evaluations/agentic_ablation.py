"""
Agentic-verification ablation (Tier-1B of the empirical study).

This script measures the marginal contribution of each agentic component
on top of pure RAG. Five rows are produced so the marginal EM/F1/SF-F1
delta of each layer is isolated:

    Row | Planner | Verifier | Iter | Configuration   | Isolates
    ----|---------|----------|------|-----------------|-----------------------
     1  |   off   |   off    |  0   | LLM-only        | parametric baseline
     2  |   off   |   gen    |  1   | RAG (no agent)  | retrieval contribution
     3  |   on    |   gen    |  1   | +Planner        | query decomposition
     4  |   on    |  gen+val |  1   | +Verifier       | pre-validation
     5  |   on    |  gen+val |  2   | +SelfCorrect    | iterative refinement

Row 1 bypasses the pipeline and queries the LLM directly with no context,
so the comparison row 1 -> row 2 isolates retrieval, not orchestration
cost. Rows 2-5 share the same retrieval contract (per-source RRF weights
read from settings.yaml) and differ only in which agentic components are
enabled.

Row 5 implements the Self-Refine loop of Madaan et al. (2023) with
max_iterations=2; whether that second iteration is justified is the
empirical question this script is designed to answer.

Exports
-------
- AblationRow                    -- NamedTuple describing one ablation cell
- ABLATION_ROWS                  -- the five cells in canonical order
- run_llm_only_row(...)          -- runs the parametric baseline (row 1)
- run_pipeline_row(...)          -- runs one of rows 2..5
- write_summary(rows, dir)       -- emits summary.{csv,md} with delta-EM
- main()                         -- CLI entry point

Dependencies / Requirements
---------------------------
- src.thesis_evaluations.benchmark_datasets   -- shared eval primitives
- src.logic_layer.verifier                    -- LLM client for row 1
- ollama server reachable at config.llm.base_url
- tqdm                                        -- progress bar

Outputs
-------
evaluation_results/agentic_ablation_<dataset>_<ts>/
    row{1..5}_*.jsonl     -- per-question records
    summary.csv / .md / .json

References
----------
Madaan et al. (2023) "Self-Refine: Iterative Refinement with Self-Feedback."
    NeurIPS 2023.

Last reviewed: 2026-06-01 (audit pass, project version 5.5)
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.logic_layer.verifier import Verifier, VerifierConfig  # noqa: E402
from src.thesis_evaluations.benchmark_datasets import (  # noqa: E402
    EvalResult,
    StoreManager,
    TestQuestion,
    _classify_llm_error,
    _gold_titles_from_supporting_facts,
    compute_exact_match,
    compute_f1,
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

# Why: the row-4 vs row-5 self-correction decision needs at least n=200 for
# the EM delta to be meaningful given EM-variance on multi-hop QA. 100 is a
# quick-look default for development; production runs should pass
# --samples 200.
_DEFAULT_SAMPLES = 100

# Why: per-source RRF weights default to 1.0 (vanilla RRF). The ablation
# must use the SAME retrieval contract as the headline benchmark; otherwise
# its EM deltas are not comparable. Values are read from settings.yaml at
# runtime; these constants are the fallback when no config key is present.
_DEFAULT_VECTOR_WEIGHT = 1.0
_DEFAULT_GRAPH_WEIGHT = 1.0

# Why: emergency LLM-model fallback used only when settings.yaml omits
# llm.model_name AND no --model flag is given. Matches benchmark_datasets.py
# so the two scripts agree on the default.
_DEFAULT_MODEL_FALLBACK = "qwen2:1.5b"

# Why: project-anchored data / eval-results roots so the CLI works regardless
# of cwd. _PROJECT_ROOT is defined above.
_DATA_ROOT = _PROJECT_ROOT / "data"
_EVAL_RESULTS_ROOT = _PROJECT_ROOT / "evaluation_results"
_DEFAULT_OUTPUT_DIR = _EVAL_RESULTS_ROOT / "runs" / "agentic_ablation"

# Why: explicit set of metric fields that should render as percentages in
# the Markdown summary. Avoids a heuristic on numeric magnitude.
_PERCENT_FIELDS = frozenset({
    "em", "f1", "sf_f1", "sf_recall",
    "em_given_retrieval_ok", "llm_error_rate",
    "pipeline_failed_rate", "pipeline_ok_llm_failed_rate",
    "pipeline_ok_llm_wrong_rate", "pipeline_ok_llm_ok_rate",
})


# ---------------------------------------------------------------------------
# Ablation definitions
# ---------------------------------------------------------------------------

class AblationRow(NamedTuple):
    """One cell of the agentic ablation matrix.

    Promotes the 7-tuple to a typed record so that swapping two flags is a
    type error rather than a silent semantic change.
    """
    name: str
    label: str
    enable_planner: bool
    enable_verifier: bool
    max_iterations: int
    llm_only: bool
    enable_pre_validation: bool


# Why: row 3 disables pre-validation so the row-3 -> row-4 delta isolates
# the contribution of the three pre-generation filters (entity-path,
# contradiction, credibility). Without this split the +Planner and
# +Verifier rows would be identical configurations.
ABLATION_ROWS: Tuple[AblationRow, ...] = (
    AblationRow("row1_llm_only",     "LLM-only (no retrieval)", False, False, 1, True,  False),
    AblationRow("row2_rag_no_agent", "RAG (no agent)",          False, True,  1, False, False),
    AblationRow("row3_planner",      "+Planner (no pre-val)",   True,  True,  1, False, False),
    AblationRow("row4_verifier",     "+Verifier (1 iter)",      True,  True,  1, False, True),
    AblationRow("row5_self_correct", "+SelfCorrect (2 iter)",   True,  True,  2, False, True),
)


# ---------------------------------------------------------------------------
# LLM-only baseline runner (no retrieval at all)
# ---------------------------------------------------------------------------

def _llm_only_prompt(question: str) -> str:
    """Direct factual-QA prompt with no context block.

    Why:    Row 1 must instruct the LLM in the same style as rows 2-5,
            otherwise the row-1 -> row-2 delta would conflate retrieval
            with prompt-engineering differences.
    What:   Same short-answer rules as the Verifier's ANSWER_PROMPT but
            with the context block omitted -- that omission IS the
            ablation.
    Misses: Structural differences in how the LLM weights context-absent
            vs context-present queries are intentionally not modelled
            here; that is a separate question outside the scope of this
            ablation.
    """
    return (
        "You are a factual QA assistant. Answer based on your knowledge.\n\n"
        "Rules:\n"
        "- Give the shortest possible answer: a name, place, date, number, or yes/no.\n"
        "- Do NOT explain or add sentences beyond the direct answer.\n"
        "- If you don't know the answer: reply with \"I don't know.\"\n\n"
        f"Question: {question}\n\n"
        "Answer (as short as possible):"
    )


def run_llm_only_row(
    model_name: str,
    config: Dict[str, Any],
    questions: List[TestQuestion],
    jsonl_out: Path,
) -> Dict[str, Any]:
    """Query the LLM directly with no context -- the parametric baseline.

    Uses the Verifier's LLM-call infrastructure so the comparison is fair
    (same Ollama client, same timeout, same temperature). Bypasses the
    Planner/Navigator/Verifier orchestration entirely.
    """
    llm_cfg = config.get("llm", {})
    verifier_cfg = VerifierConfig(
        model_name=model_name,
        base_url=llm_cfg.get("base_url", "http://localhost:11434"),
        temperature=llm_cfg.get("temperature", 0.0),
        max_tokens=llm_cfg.get("max_tokens", 200),
        timeout=llm_cfg.get("timeout", 60),
    )
    verifier = Verifier(config=verifier_cfg)

    jsonl_out.parent.mkdir(parents=True, exist_ok=True)

    # Per-question resume: skip questions already recorded in the on-disk
    # JSONL (mid-row interruption recovery). The final metrics are
    # recomputed from the JSONL at end, so it does not matter whether a
    # question was processed in this invocation or a prior one.
    questions_to_run, n_already_done = _filter_done_questions(questions, jsonl_out)
    if n_already_done:
        logger.info(
            "RESUME row1_llm_only: %d/%d already on disk; running %d new",
            n_already_done, len(questions), len(questions_to_run),
        )

    # Why: open the JSONL once, not per question -- N=200 questions would
    # otherwise mean 200 open()/close() syscalls + 200 fsyncs. flush()
    # after every line preserves crash-resume behaviour.
    with open(jsonl_out, "a", encoding="utf-8") as fh:
        for q in tqdm(questions_to_run, desc="LLM-only baseline", unit="q"):
            prompt = _llm_only_prompt(q.question)
            try:
                # `_call_llm` is currently a private Verifier method; reused
                # here to keep the LLM client identical to rows 2-5. If the
                # Verifier later exposes a public `generate(prompt)` wrapper,
                # this call site should switch to it.
                answer, latency_ms = verifier._call_llm(prompt)
            except Exception as exc:  # noqa: BLE001
                # `_call_llm` can raise on Ollama timeout, HTTP failure,
                # malformed response, or any LangChain-layer error. We catch
                # everything so a single failed question does not abort the
                # whole row; the failure is captured as a per-question
                # error answer and surfaced via _classify_llm_error later.
                answer, latency_ms = f"[Error: {exc}]", 0.0

            em = compute_exact_match(answer, q.answer)
            f1 = compute_f1(answer, q.answer)
            llm_err, llm_err_type = _classify_llm_error(answer)

            # `supporting_facts` is evaluation-only metadata for SF-F1; the
            # pipeline never reads it, so writing it to the JSONL is not a
            # leak channel.
            gold_titles = _gold_titles_from_supporting_facts(q.supporting_facts)
            rec = asdict(EvalResult(
                question_id=q.id,
                question=q.question,
                gold_answer=q.answer,
                predicted_answer=answer,
                exact_match=em,
                f1_score=f1,
                retrieval_count=0,
                time_ms=latency_ms,
                dataset=q.dataset,
                question_type=q.question_type,
                gold_titles=gold_titles,
                retrieved_titles=[],
                retrieval_recall=0.0,
                retrieval_precision=0.0,
                sf_f1=0.0,
                all_gold_retrieved=False,
                llm_error=llm_err,
                llm_error_type=llm_err_type,
                pipeline_succeeded_llm_failed=False,
                planner_query_type="(skipped)",
                hop_count=0,
                n_entities=0,
                verifier_iterations=0,
                all_verified=False,
                confidence="n/a",
            ))
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

    # Re-aggregate from the on-disk JSONL so the metrics include both new
    # records and any preloaded ones from a prior interrupted run.
    aggregated = _aggregate_row_from_jsonl(
        jsonl_out, "row1_llm_only", "LLM-only (no retrieval)", is_llm_only=True,
    )
    if aggregated is None:
        return {"row_name": "row1_llm_only", "label": "LLM-only", "n_questions": 0}
    return aggregated


# ---------------------------------------------------------------------------
# Standard ablation rows (rows 2–5)
# ---------------------------------------------------------------------------

def run_pipeline_row(
    row: AblationRow,
    dataset: str,
    model_name: str,
    questions: List[TestQuestion],
    config: Dict[str, Any],
    store_manager: StoreManager,
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Run one ablation cell through the full evaluate_dataset() pipeline."""
    logger.info("=" * 70)
    logger.info("ROW: %s -- %s", row.name, row.label)
    logger.info("  planner=%s verifier=%s iter=%d pre_val=%s",
                row.enable_planner, row.enable_verifier, row.max_iterations,
                row.enable_pre_validation)
    logger.info("=" * 70)

    # Read per-source RRF weights from settings.yaml so this ablation uses
    # the same retrieval contract as the headline benchmark. Defaults are
    # 1.0 (vanilla RRF) when settings.yaml does not override them.
    rag_cfg = config.get("rag", {})
    vector_weight = float(rag_cfg.get("vector_weight", _DEFAULT_VECTOR_WEIGHT))
    graph_weight = float(rag_cfg.get("graph_weight", _DEFAULT_GRAPH_WEIGHT))

    pipeline = create_pipeline(
        dataset, config, store_manager,
        vector_weight=vector_weight, graph_weight=graph_weight,
        model_name=model_name,
        enable_planner=row.enable_planner,
        enable_verifier=row.enable_verifier,
        max_iterations=row.max_iterations,
        enable_pre_validation=row.enable_pre_validation,
    )

    jsonl_path = output_dir / f"{row.name}.jsonl"

    # Per-question resume: skip questions already on disk for this row.
    # The JSONL is preserved and appended to; final metrics are recomputed
    # from the merged file so they reflect both prior and new records.
    questions_to_run, n_already_done = _filter_done_questions(questions, jsonl_path)
    if n_already_done:
        logger.info(
            "RESUME %s: %d/%d already on disk; running %d new",
            row.name, n_already_done, len(questions), len(questions_to_run),
        )

    if questions_to_run:
        try:
            evaluate_dataset(
                dataset, questions_to_run, pipeline,
                config_name=row.name,
                vector_weight=vector_weight, graph_weight=graph_weight,
                jsonl_out=jsonl_path,
                retrieval_only=False,
            )
        except Exception as exc:  # noqa: BLE001
            # Catch-all so one row's crash (KuzuDB lock, OOM, malformed
            # plan, …) does not stop the ablation matrix. The partial JSONL
            # on disk is still usable; the next --resume picks up the
            # unfinished questions.
            logger.error("Row %s failed mid-run: %s", row.name, exc)
        finally:
            del pipeline
            gc.collect()
    else:
        del pipeline
        gc.collect()

    return _aggregate_row_from_jsonl(
        jsonl_path, row.name, row.label, is_llm_only=False,
    )


# ---------------------------------------------------------------------------
# Checkpoint / resume helpers
# ---------------------------------------------------------------------------
# Two levels of resume granularity:
#   (1) Row-level: a row whose per-row JSONL already holds `n_questions`
#       lines is fully skipped; metrics are recomputed from the JSONL.
#   (2) Question-level: a row with a PARTIAL JSONL keeps the existing lines
#       and only re-runs questions whose id is not yet recorded. New records
#       are appended; metrics are recomputed from the merged JSONL at end.
# Question-level resume requires the JSONL to be append-only and to contain
# the question_id field — both already true. The row evaluator must NOT
# unlink the existing JSONL on start. After eval, `_aggregate_row_from_jsonl`
# rebuilds metrics from the on-disk truth (old + new records combined).

def _jsonl_line_count(path: Path) -> int:
    """Number of JSONL records in `path` (0 if absent/unreadable)."""
    if not path.exists():
        return 0
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


def _load_done_question_ids(jsonl_path: Path) -> set:
    """Return the set of question_ids already recorded in jsonl_path.

    Empty set if the file is missing, empty, or unreadable. Corrupted lines
    are silently skipped (those questions will be re-run, which is safe).
    """
    if not jsonl_path.exists():
        return set()
    ids = set()
    try:
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = rec.get("question_id")
                if qid is not None:
                    ids.add(qid)
    except OSError:
        return set()
    return ids


def _filter_done_questions(
    questions: List[TestQuestion], jsonl_path: Path,
) -> Tuple[List[TestQuestion], int]:
    """Return (remaining_questions, n_already_done).

    Filters out questions whose id is already recorded in jsonl_path. The
    remaining list preserves the input order (matters for per-seed
    reproducibility — questions are consumed in deterministic order).
    """
    done_ids = _load_done_question_ids(jsonl_path)
    if not done_ids:
        return list(questions), 0
    remaining = [q for q in questions if q.id not in done_ids]
    return remaining, len(questions) - len(remaining)


def _aggregate_row_from_jsonl(
    jsonl_path: Path, row_name: str, label: str, is_llm_only: bool
) -> Optional[Dict[str, Any]]:
    """
    Rebuild a row's summary dict from a completed per-row JSONL, so a
    checkpoint-skipped row still appears in the aggregate summary without
    re-running it. Mirrors the metric set produced by run_pipeline_row /
    run_llm_only_row.
    """
    records: List[Dict[str, Any]] = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Checkpoint read failed for %s: %s", jsonl_path, exc)
        return None
    n = len(records)
    if n == 0:
        return None

    def _rate(pred) -> float:
        return sum(1 for r in records if pred(r)) / n

    em = _rate(lambda r: r.get("exact_match"))
    f1 = sum(r.get("f1_score", 0.0) for r in records) / n
    sf_f1 = sum(r.get("sf_f1", 0.0) for r in records) / n
    sf_recall = _rate(lambda r: r.get("all_gold_retrieved"))
    llm_err = _rate(lambda r: r.get("llm_error"))
    avg_time = sum(r.get("time_ms", 0.0) for r in records) / n
    retr_ok = [r for r in records if r.get("all_gold_retrieved")]
    em_retr_ok = (
        sum(1 for r in retr_ok if r.get("exact_match")) / len(retr_ok)
        if retr_ok else 0.0
    )

    row: Dict[str, Any] = {
        "row_name": row_name,
        "label": label,
        "n_questions": n,
        "em": em,
        "f1": f1,
        "sf_f1": sf_f1,
        "sf_recall": sf_recall,
        "em_given_retrieval_ok": em_retr_ok,
        "llm_error_rate": llm_err,
        "avg_time_ms": avg_time,
    }
    if not is_llm_only:
        # Pipeline-stage breakdown (matches run_pipeline_row's extra fields).
        row["pipeline_failed_rate"] = _rate(
            lambda r: not r.get("all_gold_retrieved")
        )
        row["pipeline_ok_llm_failed_rate"] = _rate(
            lambda r: bool(r.get("pipeline_succeeded_llm_failed"))
        )
        row["pipeline_ok_llm_wrong_rate"] = _rate(
            lambda r: r.get("all_gold_retrieved")
            and not r.get("llm_error")
            and not r.get("exact_match")
        )
        row["pipeline_ok_llm_ok_rate"] = _rate(
            lambda r: r.get("all_gold_retrieved")
            and not r.get("llm_error")
            and r.get("exact_match")
        )
    return row


# ---------------------------------------------------------------------------
# Summary writer with marginal-delta computation
# ---------------------------------------------------------------------------

def write_summary(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    """Emit CSV + Markdown summary with marginal-contribution deltas."""
    if not rows:
        logger.warning("No rows to summarise.")
        return

    csv_path = output_dir / "summary.csv"
    # Union of all row keys in first-seen order. Different row builders
    # (run_llm_only_row vs run_pipeline_row) emit different field sets;
    # the LLM-only baseline has no pipeline_*_rate fields, the pipeline
    # rows do. DictWriter writes empty for any missing key on a per-row
    # basis, so the resulting CSV is sparse but correct.
    fieldnames: List[str] = []
    seen: set = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Summary CSV: %s", csv_path)

    # Markdown table with deltas
    cols = [
        ("label", "Configuration"),
        ("em", "EM"),
        ("f1", "F1"),
        ("sf_f1", "SF-F1"),
        ("sf_recall", "SF-Recall"),
        ("em_given_retrieval_ok", "EM|retr.ok"),
        ("llm_error_rate", "LLM-err"),
        ("avg_time_ms", "Latency (ms)"),
    ]
    lines = ["# Agentic Verification — Ablation Results", ""]
    lines.append("| " + " | ".join(name for _, name in cols) + " | ΔEM |")
    lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    prev_em = None
    for r in rows:
        cells: List[str] = []
        for key, _ in cols:
            v = r.get(key)
            if v is None:
                cells.append("—")
            elif isinstance(v, float):
                if key == "avg_time_ms":
                    cells.append(f"{v:.0f}")
                elif key in _PERCENT_FIELDS:
                    cells.append(f"{v * 100:.1f}%")
                else:
                    cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        # Delta column: EM gain over previous row
        if prev_em is None or r.get("em") is None:
            delta = "—"
        else:
            d = (r["em"] - prev_em) * 100
            delta = f"{d:+.1f}pp"
        prev_em = r.get("em")
        lines.append("| " + " | ".join(cells) + f" | {delta} |")
    lines.append("")
    lines.append("**Reading the table:**")
    lines.append("- Each row adds one component on top of the previous.")
    lines.append("- The **ΔEM** column shows the marginal contribution of that component.")
    lines.append("- Row 2 − Row 1 = retrieval gain.")
    lines.append("- Row 3 − Row 2 = planner gain (query decomposition).")
    lines.append("- Row 4 − Row 3 = verifier pre-validation gain.")
    lines.append("- Row 5 − Row 4 = self-correction loop gain.")

    md_path = output_dir / "summary.md"
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
    parser.add_argument("--dataset", "-d", default="hotpotqa")
    parser.add_argument("--samples", "-n", type=int, default=_DEFAULT_SAMPLES,
                        help=f"Number of evaluation questions (default: "
                             f"{_DEFAULT_SAMPLES}; production self-correction "
                             f"row-4 vs row-5 comparison needs >=200).")
    parser.add_argument("--model", "-m", default=None,
                        help="LLM model name (default: from settings.yaml).")
    parser.add_argument("--output", "-o", type=str,
                        default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument("--skip-llm-only", action="store_true",
                        help="Skip row 1 (parametric baseline). Useful for "
                             "fast re-runs of rows 2-5.")
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for question sampling. Current sampler is a deterministic "
             "prefix slice of the dataset's natural order, so the seed is "
             "logged for reproducibility but does not shuffle.",
    )
    parser.add_argument(
        "--resume", nargs="?", const="__latest__", default=None, metavar="DIR",
        help="Resume an interrupted run. Pass a run directory to continue it, "
             "or use --resume with no value to auto-continue the most recent "
             "{output}_{dataset}_* directory. Rows whose JSONL already has all "
             "N questions are skipped; the first incomplete row is re-run.",
    )
    args = parser.parse_args()

    config = load_config_file()
    store_manager = StoreManager(_DATA_ROOT)
    if not store_manager.dataset_exists(args.dataset):
        logger.error("Dataset not ingested: %s", args.dataset)
        return

    model_name = args.model or config.get("llm", {}).get("model_name", _DEFAULT_MODEL_FALLBACK)
    questions = store_manager.load_questions(args.dataset)[: args.samples]
    if not questions:
        logger.error("No questions loaded for %s", args.dataset)
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Resume logic: reuse an existing run directory instead of starting a fresh
    # timestamped one. --resume DIR uses that dir; --resume (no value) finds the
    # most recent {output}_{dataset}_* dir; absent -> new timestamped dir.
    if args.resume:
        if args.resume == "__latest__":
            base = Path(args.output)
            pattern = f"{base.name}_{args.dataset}_*"
            candidates = sorted(
                base.parent.glob(pattern),
                key=lambda p: p.stat().st_mtime if p.exists() else 0,
            )
            if candidates:
                output_dir = candidates[-1]
                logger.info("Resuming most recent run: %s", output_dir)
            else:
                output_dir = Path(f"{args.output}_{args.dataset}_{ts}")
                logger.info("No prior run to resume; starting fresh: %s", output_dir)
        else:
            output_dir = Path(args.resume)
            if not output_dir.exists():
                logger.error("--resume directory does not exist: %s", output_dir)
                return
            logger.info("Resuming run: %s", output_dir)
    else:
        output_dir = Path(f"{args.output}_{args.dataset}_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Model: %s | Questions: %d | Seed: %d | Output: %s",
                model_name, len(questions), args.seed, output_dir)

    rows: List[Dict[str, Any]] = []
    n_expected = len(questions)
    for ablation_row in ABLATION_ROWS:
        # Checkpoint: skip a row whose JSONL already holds all N questions.
        # Its metrics are recomputed from the JSONL so the summary is complete.
        row_jsonl = output_dir / f"{ablation_row.name}.jsonl"
        done = _jsonl_line_count(row_jsonl)
        if done >= n_expected:
            cached = _aggregate_row_from_jsonl(
                row_jsonl, ablation_row.name, ablation_row.label,
                ablation_row.llm_only,
            )
            if cached:
                rows.append(cached)
                logger.info(
                    "CHECKPOINT: %s already complete (%d/%d) -- skipped, "
                    "metrics loaded from JSONL (EM=%.1f%%)",
                    ablation_row.name, done, n_expected, cached["em"] * 100,
                )
                continue
            logger.warning(
                "CHECKPOINT: %s JSONL complete but unreadable -- re-running",
                ablation_row.name,
            )
        elif done > 0:
            logger.info(
                "CHECKPOINT: %s partial (%d/%d) -- resuming per-question; "
                "existing records preserved, missing question_ids re-run",
                ablation_row.name, done, n_expected,
            )

        if ablation_row.llm_only:
            if args.skip_llm_only:
                logger.info("Skipping %s (--skip-llm-only)", ablation_row.name)
                continue
            try:
                metrics = run_llm_only_row(
                    model_name, config, questions,
                    jsonl_out=row_jsonl,
                )
                if metrics:
                    rows.append(metrics)
            except Exception as exc:  # noqa: BLE001
                # Top-level row-isolation guard: any row crash is logged
                # and the next row still runs (row 1 is the LLM-only path).
                logger.error("Row %s crashed: %s", ablation_row.name, exc)
            continue

        try:
            metrics = run_pipeline_row(
                ablation_row,
                dataset=args.dataset, model_name=model_name,
                questions=questions, config=config,
                store_manager=store_manager, output_dir=output_dir,
            )
            if metrics:
                rows.append(metrics)
                logger.info("  -> EM=%.1f%% F1=%.3f SF-F1=%.3f",
                            metrics["em"] * 100, metrics["f1"], metrics["sf_f1"])
        except Exception as exc:  # noqa: BLE001
            # Same row-isolation guard for the pipeline rows (2..5).
            logger.error("Row %s crashed: %s", ablation_row.name, exc)

    write_summary(rows, output_dir)

    (output_dir / "summary.json").write_text(
        json.dumps({"timestamp": ts, "dataset": args.dataset,
                    "model": model_name,
                    "n_samples": len(questions),
                    "seed": args.seed, "rows": rows},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Done. Inspect: %s/summary.md", output_dir)
    # Auto-refresh the canonical paper bundle for this dataset. Late import:
    # avoids circular dependency with thesis_results_aggregator and keeps the
    # aggregator optional. Safe-wrapped: a bundling failure must never break
    # the eval that just produced the data.
    try:
        from .thesis_results_aggregator import update_bundle
        update_bundle(dataset=args.dataset)
    except Exception:  # noqa: BLE001
        # Auto-bundle is a convenience refresh of the canonical paper
        # bundle. It must NEVER raise back into the caller — the eval that
        # just produced the JSONLs is the source of truth and stays valid
        # whether the bundle is re-rendered or not.
        logger.debug("Auto-bundle skipped", exc_info=True)


if __name__ == "__main__":
    main()
