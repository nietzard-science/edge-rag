"""
Cross-source RRF boost (β = cross_source_boost) ablation — review item #8 / B-3.

Why this module exists
----------------------
The hybrid retriever gives a chunk surfaced by ≥2 lanes an additive RRF bonus of
``cross_source_boost / (k + 1)`` (hybrid_retriever.py). β defaults to 1.2 and is
stated in settings.yaml but never isolated. A reviewer's fair question: "is the
hybrid advantage just a tuned β?" This script answers it by sweeping
β ∈ {0.0, 0.8, 1.0, 1.2, 1.5} on the full hybrid lane (all three sources on,
equal weights) and reporting the rank-aware IR metrics at each β — retrieval
only, no generator, so it is fast and CPU-cheap.

β is passed through ``create_pipeline(cross_source_boost=...)`` (already
plumbed), so **settings.yaml is never edited** — the sweep is a pure per-run
override, respecting the single-source-of-truth discipline. β = 0.0 is the
"boost off" control: if the IR metrics are flat across β, the cross-source term
is not what earns the hybrid result (and the claim must not lean on it); if they
peak near 1.2, the shipped default is justified and the curve shows it was not
finely overfit.

Outputs
-------
evaluation_results/runs/beta_ablation_<dataset>_<ts>/
    beta_<value>.jsonl   per-question retrieved-titles for that β
    summary.json         per-β IR + SF metrics
    summary.md           per-β table + reading guide

Tuning-discipline note: any β chosen on the basis of this sweep must be selected
on the dev band (data/splits/dev_band.json), not the full test set — see
REPRODUCE.md. This script reports the full-set curve for transparency; it does
not *select* β.

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
from typing import Any, Dict, List, Sequence

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
from src.thesis_evaluations.ir_metrics import (  # noqa: E402
    DEFAULT_KS,
    compute_ir_metrics,
    load_jsonl_records,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_DATA_ROOT = _PROJECT_ROOT / "data"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "evaluation_results" / "runs" / "beta_ablation"
_DEFAULT_BETAS = (0.0, 0.8, 1.0, 1.2, 1.5)
_DEFAULT_SAMPLES = 500
_DEFAULT_MODEL_FALLBACK = "qwen2:1.5b"


def run_beta(
    beta: float,
    dataset: str,
    model_name: str,
    questions,
    config: Dict[str, Any],
    store_manager: StoreManager,
    output_dir: Path,
) -> Dict[str, Any] | None:
    """Run the full hybrid lane at one β value, retrieval-only, and aggregate."""
    logger.info("=" * 64)
    logger.info("β = %s (full hybrid lane, equal source weights)", beta)
    logger.info("=" * 64)
    jsonl_path = output_dir / f"beta_{beta}.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()  # retrieval is deterministic → re-run, don't resume

    pipeline = None
    try:
        pipeline = create_pipeline(
            dataset, config, store_manager,
            vector_weight=1.0, graph_weight=1.0, bm25_weight=1.0,
            enable_bm25=True,
            cross_source_boost=beta,
            model_name=model_name,
        )
        evaluate_dataset(
            dataset, questions, pipeline,
            config_name=f"beta_{beta}",
            vector_weight=1.0, graph_weight=1.0,
            jsonl_out=jsonl_path,
            retrieval_only=True,
        )
    except Exception as exc:  # noqa: BLE001 — one β must not abort the sweep
        logger.error("β=%s failed: %s", beta, exc)
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

    return _aggregate(jsonl_path, beta)


def _aggregate(jsonl_path: Path, beta: float,
               ks: Sequence[int] = DEFAULT_KS) -> Dict[str, Any] | None:
    records = load_jsonl_records(jsonl_path)
    if not records:
        logger.warning("No records for β=%s", beta)
        return None
    ir = compute_ir_metrics(records, ks=ks)
    judged = [r for r in records if r.get("gold_titles")]
    n = len(judged)
    strict = (sum(1 for r in judged if r.get("all_gold_retrieved")) / n
              if n else 0.0)
    mean_sf = (sum(float(r.get("retrieval_recall", 0.0) or 0.0) for r in judged) / n
               if n else 0.0)
    row: Dict[str, Any] = {
        "beta": beta, "n_questions": ir.n_questions, "mrr": ir.mrr,
        "sf_recall_strict": strict, "mean_sf_recall": mean_sf,
    }
    for k in ks:
        row[f"recall_at_{k}"] = ir.recall_at_k[k]
        row[f"ndcg_at_{k}"] = ir.ndcg_at_k[k]
    logger.info("  β=%s -> %s", beta, ir)
    return row


def write_summary(rows: List[Dict[str, Any]], output_dir: Path, dataset: str,
                  n_samples: int, ks: Sequence[int] = DEFAULT_KS, ts: str = "") -> None:
    (output_dir / "summary.json").write_text(
        json.dumps({"dataset": dataset, "n_samples": n_samples,
                    "timestamp": ts, "rows": rows}, indent=2),
        encoding="utf-8")

    header = ["β"] + [f"R@{k}" for k in ks] + ["MRR"] + \
        [f"nDCG@{k}" for k in ks] + ["SF-Rec (strict)"]
    lines = [f"# Cross-source RRF boost (β) ablation — {dataset} (n={n_samples})",
             "", "Full hybrid lane (vector+BM25+graph, equal weights), "
             "retrieval-only. β = `cross_source_boost`; β=0 is boost-off.", "",
             "| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        cells = [f"{r['beta']}"]
        cells += [f"{r.get(f'recall_at_{k}', 0)*100:.1f}%" for k in ks]
        cells.append(f"{r.get('mrr', 0):.3f}")
        cells += [f"{r.get(f'ndcg_at_{k}', 0):.3f}" for k in ks]
        cells.append(f"{r.get('sf_recall_strict', 0)*100:.1f}%")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "**Reading.** If the metrics are ~flat across β, the "
              "cross-source corroboration term is not the source of the hybrid "
              "advantage (do not lean on it). If they peak near the shipped "
              "β=1.2 and degrade at 0.0/1.5, the default is justified and shown "
              "to be non-overfit. β must be *tuned* (if at all) on the dev band, "
              "not selected from this full-set curve."]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary: %s", output_dir / "summary.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", "-d", default="hotpotqa")
    parser.add_argument("--n", "--samples", dest="samples", type=int,
                        default=_DEFAULT_SAMPLES)
    parser.add_argument("--betas", type=str,
                        default=",".join(str(b) for b in _DEFAULT_BETAS),
                        help="Comma-separated β values (default 0.0,0.8,1.0,1.2,1.5).")
    parser.add_argument("--model", "-m", default=None)
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML (default config/settings.yaml; pass "
                             "config/frozen_paper.yaml to pin the contract).")
    parser.add_argument("--output", "-o", type=str, default=str(_DEFAULT_OUTPUT))
    args = parser.parse_args()

    betas = [float(b) for b in args.betas.split(",") if b.strip()]
    config_path = Path(args.config) if args.config else None
    config = load_config_file(config_path)
    log_frozen_config(config_path or (_PROJECT_ROOT / "config" / "settings.yaml"))

    store_manager = StoreManager(_DATA_ROOT)
    if not store_manager.dataset_exists(args.dataset):
        logger.error("Dataset not ingested: %s", args.dataset)
        return
    model_name = (args.model
                  or config.get("llm", {}).get("model_name", _DEFAULT_MODEL_FALLBACK))
    questions = store_manager.load_questions(args.dataset)[: args.samples]
    if not questions:
        logger.error("No questions for %s", args.dataset)
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{args.output}_{args.dataset}_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("β-ablation | %s | n=%d | β=%s | %s",
                args.dataset, len(questions), betas, output_dir)

    rows = []
    for beta in betas:
        row = run_beta(beta, args.dataset, model_name, questions,
                       config, store_manager, output_dir)
        if row:
            rows.append(row)
    write_summary(rows, output_dir, args.dataset, len(questions), ts=ts)
    logger.info("Done. Inspect: %s/summary.md", output_dir)


if __name__ == "__main__":
    main()
