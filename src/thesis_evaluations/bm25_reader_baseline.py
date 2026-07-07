"""
BM25-only + 1.5B reader baseline — the external-baseline anchor (review #3 / Item 1).

Why this module exists
----------------------
Every reviewer flags the same gap: the ablation is self-referential — it
compares the system to its own LLM-only row, never to an external retrieval
baseline. The cheapest honest answer is a *same-scale, edge-feasible* anchor:
classic BM25 retrieval feeding the identical 1.5B reader. This script runs that
end-to-end (BM25-only context → reader → scored EM/F1/SF) so it is directly
comparable to the `row2_rag_no_agent` (hybrid) row of the agentic ablation.

The configuration is deliberately *exactly* the RAG-no-agent row except for
retrieval: planner OFF, verifier generation-only (no self-correction), and
retrieval restricted to BM25 (`vector_weight=0, graph_weight=0,
enable_bm25=True`). Everything else — prompt, reader model, scorer, max_docs —
is held identical, so the ONLY difference between this baseline and the hybrid
RAG row is the retrieval method. The comparison answers: **does the hybrid stack
beat trivial BM25 at the same scale?**

Honest expectation (from §8.5): BM25 is strong on these datasets — on 2Wiki
BM25-alone already beats hybrid in retrieval IR metrics. So this baseline may
land close to, or above, the hybrid RAG row on EM. That is the point: a fair
anchor that either (a) shows hybrid's end-to-end value, or (b) narrows the
hybrid claim to "value concentrated where lanes are complementary (HotpotQA),
not universal." Either outcome is reported honestly.

settings.yaml is never edited — BM25-only is expressed through
`create_pipeline` overrides (the same mechanism the modality ablation uses).

Per-question resume (`--resume`) mirrors agentic_ablation: laptop sleep / a
KuzuDB lock will not lose completed questions; re-invoking continues.

Outputs
-------
evaluation_results/runs/bm25_reader_<dataset>_<ts>/
    row0_bm25_reader.jsonl   per-question records (same schema as ablation rows)
    summary.json             aggregate EM/F1/SF + the hybrid-row delta if found

Last reviewed: 2026-06-10.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.thesis_evaluations.benchmark_datasets import (  # noqa: E402
    StoreManager,
    create_pipeline,
    evaluate_dataset,
    load_config_file,
    log_frozen_config,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_DATA_ROOT = _PROJECT_ROOT / "data"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "evaluation_results" / "runs" / "bm25_reader"
_DEFAULT_SAMPLES = 500
_DEFAULT_MODEL_FALLBACK = "qwen2:1.5b"
_ROW_NAME = "row0_bm25_reader"


def _load_done_ids(jsonl_path: Path) -> set:
    """question_ids already present in a partial JSONL (for resume)."""
    done: set = set()
    if not jsonl_path.exists():
        return done
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
                done.add(str(qid))
    return done


def _aggregate(jsonl_path: Path) -> Dict[str, Any]:
    """EM / F1 / SF-Recall(strict) / mean-SF from the per-question JSONL."""
    n = em = soft = 0
    f1_sum = sf_strict = sf_mean = 0.0
    th = 0.6
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            if r.get("exact_match"):
                em += 1
            f1 = float(r.get("f1_score", 0.0) or 0.0)
            f1_sum += f1
            if f1 >= th:
                soft += 1
            if r.get("all_gold_retrieved"):
                sf_strict += 1
            sf_mean += float(r.get("retrieval_recall", 0.0) or 0.0)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "em": em / n, "soft_em": soft / n, "f1": f1_sum / n,
        "sf_recall_strict": sf_strict / n, "mean_sf_recall": sf_mean / n,
    }


def run(dataset: str, model_name: str, questions, config: Dict[str, Any],
        store_manager: StoreManager, output_dir: Path,
        resume: bool = True) -> Dict[str, Any]:
    """Run the BM25-only + reader baseline end-to-end on `dataset`."""
    jsonl_path = output_dir / f"{_ROW_NAME}.jsonl"

    to_run = list(questions)
    if resume:
        done = _load_done_ids(jsonl_path)
        if done:
            to_run = [q for q in questions if str(getattr(q, "id", "")) not in done]
            logger.info("RESUME: %d/%d already done; running %d",
                        len(done), len(questions), len(to_run))

    if to_run:
        pipeline = None
        try:
            # BM25-only retrieval + RAG-no-agent reader config (planner off,
            # verifier generation-only). Identical to row2_rag_no_agent EXCEPT
            # vector/graph weights zeroed so only the sparse lane fires.
            pipeline = create_pipeline(
                dataset, config, store_manager,
                vector_weight=0.0, graph_weight=0.0, bm25_weight=1.0,
                enable_bm25=True,
                model_name=model_name,
                enable_planner=False,
                enable_verifier=True,
                max_iterations=1,
                enable_pre_validation=False,
            )
            evaluate_dataset(
                dataset, to_run, pipeline,
                config_name=_ROW_NAME,
                vector_weight=0.0, graph_weight=0.0,
                jsonl_out=jsonl_path,
                retrieval_only=False,
            )
        except Exception as exc:  # noqa: BLE001 — keep the partial JSONL for --resume
            logger.error("BM25-reader run failed mid-way: %s", exc)
        finally:
            if pipeline is not None:
                for attr in ("close", "shutdown"):
                    fn = getattr(pipeline, attr, None)
                    if callable(fn):
                        try:
                            fn()
                            break
                        except Exception:  # noqa: BLE001
                            pass
            del pipeline
            gc.collect()

    return _aggregate(jsonl_path)


def _find_hybrid_em(dataset: str) -> Optional[float]:
    """Best-effort: locate the hybrid RAG-no-agent EM for a side-by-side note.

    Scans evaluation_results/runs for an agentic_ablation_<dataset>_* dir with a
    row2_rag_no_agent.jsonl and returns its EM. Returns None if not found.
    """
    runs = _PROJECT_ROOT / "evaluation_results" / "runs"
    if not runs.exists():
        return None
    best = None
    for d in sorted(runs.glob(f"agentic_ablation_{dataset}_*")):
        p = d / "row2_rag_no_agent.jsonl"
        if p.exists():
            agg = _aggregate(p)
            if agg.get("n"):
                best = agg["em"]  # latest dir wins (sorted ascending)
    return best


def write_summary(agg: Dict[str, Any], output_dir: Path, dataset: str,
                  model_name: str, ts: str) -> None:
    hybrid_em = _find_hybrid_em(dataset)
    summary = {
        "baseline": "bm25_only_reader",
        "dataset": dataset, "model": model_name, "timestamp": ts,
        "config": "planner=off, verifier=gen-only, retrieval=BM25-only "
                  "(vector_weight=0, graph_weight=0, enable_bm25=True)",
        "metrics": agg,
        "hybrid_rag_no_agent_em": hybrid_em,
        "bm25_minus_hybrid_em_pp": (None if hybrid_em is None or not agg.get("n")
                                    else round((agg["em"] - hybrid_em) * 100, 1)),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"# BM25-only + 1.5B reader baseline — {dataset}", ""]
    if agg.get("n"):
        lines += [
            f"- n = **{agg['n']}**, model = `{model_name}`",
            f"- EM = **{agg['em']*100:.1f}%**, Soft-EM = **{agg['soft_em']*100:.1f}%**, "
            f"F1 = **{agg['f1']:.3f}**",
            f"- SF-Recall (strict) = {agg['sf_recall_strict']*100:.1f}%, "
            f"mean SF-Recall = {agg['mean_sf_recall']*100:.1f}%",
        ]
        if hybrid_em is not None:
            delta = (agg["em"] - hybrid_em) * 100
            verdict = ("hybrid wins" if delta < -0.5
                       else "BM25 wins" if delta > 0.5 else "tie")
            lines += [
                "",
                f"**vs hybrid RAG-no-agent:** hybrid EM {hybrid_em*100:.1f}% → "
                f"BM25-reader EM {agg['em']*100:.1f}% (**{delta:+.1f}pp**, {verdict}).",
                "",
                "Reading: if BM25-reader ≈ hybrid, the hybrid stack's end-to-end "
                "value is concentrated where lanes are complementary (HotpotQA), "
                "not universal — the honest, defensible framing.",
            ]
    else:
        lines.append("_(no records — run did not complete)_")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary: %s", output_dir / "summary.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", "-d", default="hotpotqa")
    parser.add_argument("--n", "--samples", dest="samples", type=int,
                        default=_DEFAULT_SAMPLES)
    parser.add_argument("--model", "-m", default=None)
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML (default config/settings.yaml; pass "
                             "config/frozen_paper.yaml to pin the contract).")
    parser.add_argument("--output", "-o", type=str, default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--resume", type=str, default=None, metavar="DIR",
                        help="Resume into an existing output dir (continue a "
                             "partially-completed run).")
    parser.add_argument("--no-resume", action="store_true",
                        help="Force a fresh run even if a JSONL exists.")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    config = load_config_file(config_path)
    log_frozen_config(config_path or (_PROJECT_ROOT / "config" / "settings.yaml"))

    store_manager = StoreManager(_DATA_ROOT)
    if not store_manager.dataset_exists(args.dataset):
        logger.error("Dataset not ingested: %s. Run ingestion first.", args.dataset)
        return
    model_name = (args.model
                  or config.get("llm", {}).get("model_name", _DEFAULT_MODEL_FALLBACK))
    questions = store_manager.load_questions(args.dataset)[: args.samples]
    if not questions:
        logger.error("No questions for %s", args.dataset)
        return

    if args.resume:
        output_dir = Path(args.resume)
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = output_dir.name.split("_")[-1]
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"{args.output}_{args.dataset}_{ts}")
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("BM25+reader baseline | %s | n=%d | model=%s | %s",
                args.dataset, len(questions), model_name, output_dir)
    agg = run(args.dataset, model_name, questions, config, store_manager,
              output_dir, resume=not args.no_resume)
    write_summary(agg, output_dir, args.dataset, model_name, ts)
    logger.info("Done. Inspect: %s/summary.md", output_dir)


if __name__ == "__main__":
    main()
