"""
Retrieval-modality ablation (P0-T3) — isolating the *hybrid* contribution.

Why this script exists
----------------------
The headline result reports +26.2 pp EM for "RAG vs no-retrieval" on HotpotQA.
That number legitimises *retrieval*, but not the word **hybrid** used throughout
the paper: it is retrieval-vs-none, not hybrid-vs-dense. To support the
architecture claim we must show what each retrieval modality contributes on its
own and what fusing them adds. This script runs four retrieval-only
configurations and reports rank-aware IR metrics for each:

    Lane         vector  bm25  graph   Isolates
    ----------   ------  ----  -----   ----------------------------------------
    dense_only     1      0      0      dense ANN (nomic-embed-text) alone
    bm25_only      0      1      0      sparse lexical (BM25) alone
    graph_only     0      0      1      entity-path (KuzuDB) alone
    hybrid_all     1      1      1      RRF fusion of all three

The dense/bm25/graph/all decomposition is the direct evidence for "hybrid":
hybrid_all should dominate each single lane, and the per-query *graph-rescued*
list (gold facts the graph lane surfaced that dense+BM25 both missed) is the
qualitative proof that the graph path is not redundant with dense retrieval.

The pure ``bm25_only`` lane is expressible because HybridRetriever now gates its
dense-vector search on ``vector_weight > 0`` in addition to retrieval mode (see
hybrid_retriever.py §"Vector retrieval"). With ``mode=VECTOR`` derived from
``graph_weight==0`` and ``vector_weight=0``, only the BM25 lane fires.

Metrics
-------
Per lane, over the gold supporting-fact titles (title-level relevance):
  - Recall@{5,10,20}, nDCG@{5,10,20}, MRR        (src.thesis_evaluations.ir_metrics)
  - SF-Recall (strict all_gold) + mean SF-Recall (the existing set-based metrics)
All retrieval-only — the LLM (Verifier) is never invoked, so a 500-question run
over two datasets is CPU-cheap (~3-4 h total).

Outputs
-------
evaluation_results/runs/modality_ablation_<dataset>_<ts>/
    <lane>.jsonl           per-question records (retrieved_titles + gold_titles)
    summary.json           per-lane IR + SF metrics, machine-readable
    summary.md             per-lane table + reading guide
    graph_rescued.md       per-query gold facts graph surfaced that dense+BM25 missed
    graph_rescued.jsonl    same, machine-readable

References
----------
- Järvelin & Kekäläinen (2002) TOIS 20(4) — nDCG.
- Voorhees (1999) TREC-8 — MRR.
- Cormack et al. (2009) SIGIR — Reciprocal Rank Fusion.
- Gutiérrez et al. (2024) NeurIPS — HippoRAG (graph vs dense complementarity).

Last reviewed: 2026-06-05.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.thesis_evaluations.benchmark_datasets import (  # noqa: E402
    StoreManager,
    TestQuestion,
    create_pipeline,
    evaluate_dataset,
    load_config_file,
    log_frozen_config,
)
from src.thesis_evaluations.ir_metrics import (  # noqa: E402
    DEFAULT_KS,
    compute_ir_metrics,
    load_jsonl_records,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_DEFAULT_SAMPLES = 500            # headline retrieval-only n; cheap without LLM
_DEFAULT_MODEL_FALLBACK = "qwen2:1.5b"
_DATA_ROOT = _PROJECT_ROOT / "data"
_EVAL_RESULTS_ROOT = _PROJECT_ROOT / "evaluation_results"
_DEFAULT_OUTPUT_DIR = _EVAL_RESULTS_ROOT / "runs" / "modality_ablation"


class ModalityLane(NamedTuple):
    """One retrieval-modality cell. ``enable_bm25`` is explicit because BM25 is
    orthogonal to the vector/graph retrieval mode (see module docstring)."""
    name: str
    label: str
    vector_weight: float
    graph_weight: float
    bm25_weight: float
    enable_bm25: bool


# dense / bm25 / graph / all — the four lanes the paper "hybrid" claim needs.
MODALITY_LANES: Tuple[ModalityLane, ...] = (
    ModalityLane("dense_only", "Dense (vector) only", 1.0, 0.0, 0.0, False),
    ModalityLane("bm25_only",  "BM25 (sparse) only",  0.0, 0.0, 1.0, True),
    ModalityLane("graph_only", "Graph (entity-path) only", 0.0, 1.0, 0.0, False),
    ModalityLane("hybrid_all", "Hybrid (all three, RRF)",  1.0, 1.0, 1.0, True),
    # Down-weighted-dense hybrid: probe whether shrinking the (collapsed on
    # 2Wiki) dense lane's RRF vote recovers fusion quality vs equal-weight
    # hybrid. vector_weight=0.05 keeps a token dense contribution rather than
    # dropping it entirely, so any residual dense signal still counts. Refs:
    # per-source RRF weighting (Bruch et al. 2023); regime-conditional routing
    # (RegimeRouter, arXiv:2604.09019). NOTE: vector_weight>0 so the dense ANN
    # search still fires (the >0 gate in hybrid_retriever.py); only its fusion
    # weight is reduced.
    ModalityLane("hybrid_dense005", "Hybrid (dense down-weighted 0.05)",
                 0.05, 1.0, 1.0, True),
)

# The three lanes whose union defines "what the non-graph system already finds",
# used by the graph-rescued analysis. Order-independent (set union).
_NON_GRAPH_LANES = ("dense_only", "bm25_only")
_GRAPH_LANE = "graph_only"


# ---------------------------------------------------------------------------
# One lane runner
# ---------------------------------------------------------------------------

def run_lane(
    lane: ModalityLane,
    dataset: str,
    model_name: str,
    questions: List[TestQuestion],
    config: Dict[str, Any],
    store_manager: StoreManager,
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Run one retrieval-modality lane retrieval-only and aggregate its metrics.

    Builds a pipeline with the lane's per-source weights, runs evaluate_dataset
    with ``retrieval_only=True`` (no LLM), then computes IR + SF metrics from the
    per-question JSONL the evaluator writes. Returns the lane's summary dict, or
    None if no records were produced.
    """
    logger.info("=" * 70)
    logger.info("LANE: %s -- %s", lane.name, lane.label)
    logger.info("  vector=%s graph=%s bm25=%s (enable_bm25=%s)",
                lane.vector_weight, lane.graph_weight, lane.bm25_weight,
                lane.enable_bm25)
    logger.info("=" * 70)

    jsonl_path = output_dir / f"{lane.name}.jsonl"
    # Fresh file per run: retrieval is deterministic, so a partial file from a
    # crashed run would otherwise double-count. (Unlike agentic_ablation, the
    # retrieval-only lanes are cheap enough that we re-run rather than resume.)
    if jsonl_path.exists():
        jsonl_path.unlink()

    pipeline = None
    try:
        pipeline = create_pipeline(
            dataset, config, store_manager,
            vector_weight=lane.vector_weight,
            graph_weight=lane.graph_weight,
            bm25_weight=lane.bm25_weight,
            enable_bm25=lane.enable_bm25,
            model_name=model_name,
        )
        evaluate_dataset(
            dataset, questions, pipeline,
            config_name=lane.name,
            vector_weight=lane.vector_weight,
            graph_weight=lane.graph_weight,
            jsonl_out=jsonl_path,
            retrieval_only=True,
        )
    except Exception as exc:  # noqa: BLE001 — one lane crash must not abort the matrix
        logger.error("Lane %s failed: %s", lane.name, exc)
    finally:
        if pipeline is not None:
            _close_pipeline(pipeline)
        del pipeline
        gc.collect()

    return _aggregate_lane(jsonl_path, lane)


def _close_pipeline(pipeline) -> None:
    """Best-effort close so the next lane can acquire the KuzuDB lock."""
    for attr in ("close", "shutdown"):
        fn = getattr(pipeline, attr, None)
        if callable(fn):
            try:
                fn()
                return
            except Exception:  # noqa: BLE001 — close failures are non-fatal
                logger.debug("pipeline.%s() failed", attr, exc_info=True)


def _aggregate_lane(
    jsonl_path: Path,
    lane: ModalityLane,
    ks: Sequence[int] = DEFAULT_KS,
) -> Optional[Dict[str, Any]]:
    """Compute IR + SF metrics for one lane from its per-question JSONL."""
    records = load_jsonl_records(jsonl_path)
    if not records:
        logger.warning("No records for lane %s (%s)", lane.name, jsonl_path)
        return None

    ir = compute_ir_metrics(records, ks=ks)

    judged = [r for r in records if r.get("gold_titles")]
    n_judged = len(judged)
    # Strict SF-Recall: fraction of questions where ALL gold titles retrieved.
    strict = (sum(1 for r in judged if r.get("all_gold_retrieved")) / n_judged
              if n_judged else 0.0)
    # Mean SF-Recall: per-question fraction of gold retrieved, averaged.
    mean_sf_recall = (sum(float(r.get("retrieval_recall", 0.0) or 0.0)
                          for r in judged) / n_judged if n_judged else 0.0)
    mean_sf_f1 = (sum(float(r.get("sf_f1", 0.0) or 0.0) for r in judged)
                  / n_judged if n_judged else 0.0)

    row: Dict[str, Any] = {
        "lane": lane.name,
        "label": lane.label,
        "vector_weight": lane.vector_weight,
        "graph_weight": lane.graph_weight,
        "bm25_weight": lane.bm25_weight,
        "n_questions": ir.n_questions,
        "mrr": ir.mrr,
        "sf_recall_strict": strict,
        "mean_sf_recall": mean_sf_recall,
        "sf_f1": mean_sf_f1,
    }
    for k in ks:
        row[f"recall_at_{k}"] = ir.recall_at_k[k]
        row[f"ndcg_at_{k}"] = ir.ndcg_at_k[k]
    logger.info("  -> %s", ir)
    return row


# ---------------------------------------------------------------------------
# Graph-rescued analysis: gold facts graph found that dense+BM25 both missed
# ---------------------------------------------------------------------------

def _norm(t: str) -> str:
    return " ".join(str(t).strip().lower().split())


def _retrieved_by_qid(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    """Map question_id -> {gold, retrieved, question} from a lane JSONL."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in load_jsonl_records(jsonl_path):
        qid = r.get("question_id")
        if qid is None:
            continue
        out[str(qid)] = {
            "question": r.get("question", ""),
            "gold": [_norm(g) for g in r.get("gold_titles") or []],
            "retrieved": [_norm(t) for t in r.get("retrieved_titles") or []],
        }
    return out


def compute_graph_rescued(output_dir: Path) -> List[Dict[str, Any]]:
    """Per-question gold titles the graph lane retrieved that dense+BM25 missed.

    A 'graph-rescued' fact is a gold supporting-fact title that appears in the
    graph_only lane's retrieved titles for a question but in NEITHER the
    dense_only NOR the bm25_only lane's retrieved titles. These are the cases
    that empirically justify the graph path: the answer-bearing document was
    reachable only via the entity graph.

    Returns a list of {question_id, question, rescued_titles, ...}, one entry
    per question that has at least one rescued title. Lanes whose JSONL is
    missing are skipped gracefully (returns [] if the graph lane is absent).
    """
    graph_path = output_dir / f"{_GRAPH_LANE}.jsonl"
    if not graph_path.exists():
        logger.warning("graph_rescued: %s missing; skipping", graph_path)
        return []
    graph = _retrieved_by_qid(graph_path)
    non_graph = [
        _retrieved_by_qid(output_dir / f"{name}.jsonl")
        for name in _NON_GRAPH_LANES
        if (output_dir / f"{name}.jsonl").exists()
    ]

    rescued: List[Dict[str, Any]] = []
    for qid, g in graph.items():
        gold = set(g["gold"])
        if not gold:
            continue
        graph_hits = gold & set(g["retrieved"])
        if not graph_hits:
            continue
        # Union of what the non-graph lanes retrieved for this question.
        non_graph_hits: set = set()
        for lane_map in non_graph:
            entry = lane_map.get(qid)
            if entry:
                non_graph_hits |= (gold & set(entry["retrieved"]))
        only_graph = graph_hits - non_graph_hits
        if only_graph:
            rescued.append({
                "question_id": qid,
                "question": g["question"],
                "rescued_titles": sorted(only_graph),
                "graph_gold_hits": sorted(graph_hits),
                "non_graph_gold_hits": sorted(non_graph_hits),
                "n_gold": len(gold),
            })
    rescued.sort(key=lambda d: len(d["rescued_titles"]), reverse=True)
    return rescued


def write_graph_rescued(rescued: List[Dict[str, Any]], output_dir: Path) -> None:
    """Write graph_rescued.{jsonl,md}."""
    jsonl_path = output_dir / "graph_rescued.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for r in rescued:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    lines = ["# Graph-rescued supporting facts", ""]
    lines.append(
        "Gold supporting-fact titles the **graph** lane retrieved that the "
        "**dense** and **BM25** lanes both missed. These are the cases where "
        "the answer-bearing document was reachable only via the entity graph — "
        "the direct evidence that the graph path is not redundant with dense "
        "retrieval."
    )
    lines.append("")
    lines.append(f"**{len(rescued)} questions** had at least one graph-rescued fact.")
    lines.append("")
    if rescued:
        lines.append("| Question (truncated) | Graph-rescued title(s) |")
        lines.append("|---|---|")
        for r in rescued[:100]:  # cap the table; full set is in the JSONL
            q = (r["question"][:90] + "…") if len(r["question"]) > 90 else r["question"]
            q = q.replace("|", "\\|")
            titles = ", ".join(t.replace("|", "\\|") for t in r["rescued_titles"])
            lines.append(f"| {q} | {titles} |")
        if len(rescued) > 100:
            lines.append("")
            lines.append(f"_(showing 100 of {len(rescued)}; full list in graph_rescued.jsonl)_")
    else:
        lines.append("_No graph-only rescues in this run._")
    (output_dir / "graph_rescued.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Graph-rescued: %d questions -> %s",
                len(rescued), output_dir / "graph_rescued.md")


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def write_summary(
    rows: List[Dict[str, Any]],
    output_dir: Path,
    dataset: str,
    model_name: str,
    n_samples: int,
    config_hash: Optional[str],
    ks: Sequence[int] = DEFAULT_KS,
    ts: str = "",
) -> None:
    """Emit summary.json + summary.md for the modality ablation."""
    summary = {
        "timestamp": ts,
        "dataset": dataset,
        "model": model_name,
        "n_samples": n_samples,
        "config_sha256": config_hash,
        "retrieval_only": True,
        "ks": list(ks),
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Markdown table. Columns: label, R@k…, nDCG@k…, MRR, strict/mean SF-Recall.
    header = ["Modality"]
    for k in ks:
        header.append(f"R@{k}")
    header.append("MRR")
    for k in ks:
        header.append(f"nDCG@{k}")
    header.append("SF-Rec (strict)")
    header.append("SF-Rec (mean)")

    lines = [f"# Retrieval-Modality Ablation — {dataset} (n={n_samples})", ""]
    if config_hash:
        lines.append(f"_Config sha256: {config_hash} · retrieval-only (no LLM)_")
        lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        cells = [r.get("label", r.get("lane", "—"))]
        for k in ks:
            cells.append(f"{r.get(f'recall_at_{k}', 0.0) * 100:.1f}%")
        cells.append(f"{r.get('mrr', 0.0):.3f}")
        for k in ks:
            cells.append(f"{r.get(f'ndcg_at_{k}', 0.0):.3f}")
        cells.append(f"{r.get('sf_recall_strict', 0.0) * 100:.1f}%")
        cells.append(f"{r.get('mean_sf_recall', 0.0) * 100:.1f}%")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("**Reading the table:**")
    lines.append("- Each row is one retrieval modality run in isolation; "
                 "**Hybrid** fuses all three via RRF.")
    lines.append("- `R@k` / `nDCG@k` are title-level IR metrics over gold "
                 "supporting facts (rank-aware).")
    lines.append("- `SF-Rec (strict)` = fraction of questions with ALL gold "
                 "titles retrieved; `SF-Rec (mean)` = per-question average.")
    lines.append("- The *hybrid* claim holds iff **Hybrid** dominates each "
                 "single lane; see `graph_rescued.md` for the cases the graph "
                 "lane uniquely recovered.")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary: %s", output_dir / "summary.md")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", "-d", default="hotpotqa",
                        help="Dataset name (e.g. hotpotqa, 2wikimultihop).")
    parser.add_argument("--n", "--samples", dest="samples", type=int,
                        default=_DEFAULT_SAMPLES,
                        help=f"Number of questions (default: {_DEFAULT_SAMPLES}).")
    parser.add_argument("--model", "-m", default=None,
                        help="Model name (default: from settings.yaml). Only "
                             "affects pipeline construction; the LLM is never "
                             "called in retrieval-only mode.")
    parser.add_argument("--config", type=str, default=None, metavar="PATH",
                        help="Config YAML (default: config/settings.yaml). Pass "
                             "config/frozen_paper.yaml to pin the frozen "
                             "contract; its SHA-256 is logged + written to "
                             "summary.json.")
    parser.add_argument("--retrieval-only", action="store_true", default=True,
                        help="Retrieval-only (default and only supported mode "
                             "for this script — the LLM adds no IR signal).")
    parser.add_argument("--ks", type=str, default=",".join(str(k) for k in DEFAULT_KS),
                        help="Comma-separated Recall@k / nDCG@k cutoffs.")
    parser.add_argument("--output", "-o", type=str, default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument("--lanes", type=str, default=None,
                        help="Comma-separated subset of lane names to run "
                             "(default: all). Names: "
                             + ", ".join(l.name for l in MODALITY_LANES))
    args = parser.parse_args()

    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())

    config_path = Path(args.config) if args.config else None
    config = load_config_file(config_path)
    config_hash = log_frozen_config(
        config_path or (_PROJECT_ROOT / "config" / "settings.yaml")
    )

    store_manager = StoreManager(_DATA_ROOT)
    if not store_manager.dataset_exists(args.dataset):
        logger.error("Dataset not ingested: %s", args.dataset)
        return

    model_name = (args.model
                  or config.get("llm", {}).get("model_name", _DEFAULT_MODEL_FALLBACK))
    questions = store_manager.load_questions(args.dataset)[: args.samples]
    if not questions:
        logger.error("No questions loaded for %s", args.dataset)
        return

    lanes = MODALITY_LANES
    if args.lanes:
        wanted = {s.strip() for s in args.lanes.split(",") if s.strip()}
        lanes = tuple(l for l in MODALITY_LANES if l.name in wanted)
        if not lanes:
            logger.error("No matching lanes in %r; valid: %s",
                         args.lanes, [l.name for l in MODALITY_LANES])
            return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{args.output}_{args.dataset}_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Dataset: %s | Questions: %d | Lanes: %s | Output: %s",
                args.dataset, len(questions), [l.name for l in lanes], output_dir)

    rows: List[Dict[str, Any]] = []
    for lane in lanes:
        row = run_lane(lane, args.dataset, model_name, questions,
                       config, store_manager, output_dir)
        if row:
            rows.append(row)

    write_summary(rows, output_dir, args.dataset, model_name, len(questions),
                  config_hash, ks=ks, ts=ts)

    # Graph-rescued analysis only meaningful when the graph lane + at least one
    # non-graph lane were both run.
    ran = {l.name for l in lanes}
    if _GRAPH_LANE in ran and ran & set(_NON_GRAPH_LANES):
        rescued = compute_graph_rescued(output_dir)
        write_graph_rescued(rescued, output_dir)
    else:
        logger.info("Graph-rescued analysis skipped (need graph_only + a "
                    "non-graph lane in the run).")

    logger.info("Done. Inspect: %s/summary.md", output_dir)


if __name__ == "__main__":
    main()
