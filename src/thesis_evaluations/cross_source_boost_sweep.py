"""B-3 — Ablation of the cross-source RRF boost β (cross_source_boost).

β (`rag.cross_source_boost`, default 1.2) gives a chunk extra RRF credit when
more than one retrieval lane surfaces it — an up-weight on cross-lane
consensus (Cormack et al. 2009 give the base RRF; β>1 rewards agreement). The
value is stated in the paper but never isolated. This script sweeps β over a
grid and reports the retrieval-only IR metrics for each, closing the RQ1 gap.

Respects the project rule "no settings.yaml edits": β is passed as a per-run
**override** through `create_pipeline(cross_source_boost=...)` (added for this
ablation), never by editing config/settings.yaml. The hybrid lane weights stay
at the production 1/1/1 so the only thing that varies is β.

Retrieval-only (no LLM) — a 500-question sweep over a handful of β values is
CPU-cheap. Reuses the exact StoreManager / create_pipeline / evaluate_dataset
/ IR-metric machinery as modality_ablation.py for consistency.

Run (user-side; needs Ollama embeddings + the built LanceDB/KuzuDB stores):
    python -X utf8 -m src.thesis_evaluations.cross_source_boost_sweep \
        --dataset hotpotqa --n 500 --betas 1.0,1.1,1.2,1.4,1.6,2.0
"""

from __future__ import annotations

import gc
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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
from src.thesis_evaluations.modality_ablation import _close_pipeline  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

_DATA_ROOT = _PROJECT_ROOT / "data"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "evaluation_results" / "runs" / "beta_sweep"
_DEFAULT_BETAS = (1.0, 1.1, 1.2, 1.4, 1.6, 2.0)
_DEFAULT_MODEL = "qwen2:1.5b"


def run_beta(beta: float, dataset: str, model_name: str, questions: list,
             config: Dict[str, Any], store_manager: StoreManager,
             output_dir: Path) -> Dict[str, Any] | None:
    """One β cell, retrieval-only. Hybrid weights fixed at 1/1/1."""
    name = f"beta_{beta:g}".replace(".", "p")
    logger.info("=" * 70)
    logger.info("BETA = %s  (cross_source_boost)", beta)
    logger.info("=" * 70)
    jsonl_path = output_dir / f"{name}.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    pipeline = None
    try:
        pipeline = create_pipeline(
            dataset, config, store_manager,
            vector_weight=1.0, graph_weight=1.0, bm25_weight=1.0,
            cross_source_boost=beta,          # <- the only thing that varies
            model_name=model_name,
        )
        evaluate_dataset(
            dataset, questions, pipeline,
            config_name=name,
            vector_weight=1.0, graph_weight=1.0,
            jsonl_out=jsonl_path,
            retrieval_only=True,
        )
    except Exception as exc:  # noqa: BLE001 — one cell crash must not abort the sweep
        logger.error("beta=%s failed: %s", beta, exc)
    finally:
        if pipeline is not None:
            _close_pipeline(pipeline)
        del pipeline
        gc.collect()

    records = load_jsonl_records(jsonl_path)
    if not records:
        logger.warning("No records for beta=%s", beta)
        return None
    ir = compute_ir_metrics(records, ks=DEFAULT_KS)
    judged = [r for r in records if r.get("gold_titles")]
    n = len(judged)
    strict_sf = (sum(1 for r in judged if r.get("all_gold_retrieved")) / n) if n else 0.0
    mean_sf = (sum(float(r.get("retrieval_recall", 0.0) or 0.0) for r in judged) / n) if n else 0.0
    row = {"beta": beta, "n": len(records), **ir,
           "sf_recall_strict": strict_sf, "sf_recall_mean": mean_sf}
    return row


def _write_summary(rows: List[Dict[str, Any]], out_dir: Path, dataset: str,
                   n: int, config_hash: str, ts: str) -> None:
    (out_dir / "beta_sweep.json").write_text(
        json.dumps({"dataset": dataset, "n": n, "config_hash": config_hash,
                    "timestamp": ts, "rows": rows}, indent=2),
        encoding="utf-8")

    hdr = "| β | R@5 | R@10 | R@20 | nDCG@10 | MRR | SF-Rec(strict) | SF-Rec(mean) |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [f"# cross_source_boost (β) sweep — {dataset} (n={n})", "",
             "Retrieval-only; hybrid weights fixed 1/1/1; β via per-run override "
             "(no settings.yaml edit). Production default β=1.2.", "", hdr, sep]
    for r in rows:
        lines.append(
            f"| {r['beta']:g} | {r.get('recall@5',0):.3f} | {r.get('recall@10',0):.3f} | "
            f"{r.get('recall@20',0):.3f} | {r.get('ndcg@10',0):.3f} | {r.get('mrr',0):.3f} | "
            f"{r['sf_recall_strict']:.3f} | {r['sf_recall_mean']:.3f} |")
    # LaTeX fragment for the paper.
    tex = ["% Generated by cross_source_boost_sweep.py — do not edit by hand.",
           "\\begin{table}[t]\\centering",
           f"\\caption{{Sensitivity to the cross-source RRF boost $\\beta$ on {dataset} "
           f"($n={n}$, retrieval-only). Production default $\\beta=1.2$.}}",
           "\\label{tab:beta-sweep}",
           "\\begin{tabular}{rrrrr}", "\\toprule",
           "$\\beta$ & R@5 & nDCG@10 & MRR & SF-Rec. \\\\", "\\midrule"]
    for r in rows:
        tex.append(f"{r['beta']:g} & {r.get('recall@5',0)*100:.1f}\\% & "
                   f"{r.get('ndcg@10',0):.3f} & {r.get('mrr',0):.3f} & "
                   f"{r['sf_recall_mean']*100:.1f}\\% \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out_dir / "beta_sweep.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "table_beta_sweep.tex").write_text("\n".join(tex), encoding="utf-8")
    logger.info("Wrote %s", out_dir / "beta_sweep.md")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", "-d", default="hotpotqa")
    ap.add_argument("--n", "--samples", dest="samples", type=int, default=500)
    ap.add_argument("--betas", type=str, default=",".join(str(b) for b in _DEFAULT_BETAS),
                    help="Comma-separated β grid (default 1.0,1.1,1.2,1.4,1.6,2.0).")
    ap.add_argument("--model", "-m", default=None)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--output", "-o", type=str, default=str(_DEFAULT_OUTPUT))
    args = ap.parse_args()

    betas = [float(x) for x in args.betas.split(",") if x.strip()]
    config_path = Path(args.config) if args.config else None
    config = load_config_file(config_path)
    config_hash = log_frozen_config(
        config_path or (_PROJECT_ROOT / "config" / "settings.yaml"))

    store_manager = StoreManager(_DATA_ROOT)
    if not store_manager.dataset_exists(args.dataset):
        logger.error("Dataset not ingested: %s", args.dataset)
        return
    model_name = args.model or config.get("llm", {}).get("model_name", _DEFAULT_MODEL)
    questions = store_manager.load_questions(args.dataset)[: args.samples]
    if not questions:
        logger.error("No questions for %s", args.dataset)
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"{args.output}_{args.dataset}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("β sweep: %s | dataset=%s | n=%d | out=%s",
                betas, args.dataset, len(questions), out_dir)

    rows: List[Dict[str, Any]] = []
    for b in betas:
        row = run_beta(b, args.dataset, model_name, questions, config,
                       store_manager, out_dir)
        if row:
            rows.append(row)

    _write_summary(rows, out_dir, args.dataset, len(questions), config_hash, ts)
    logger.info("Done. Inspect: %s/beta_sweep.md", out_dir)


if __name__ == "__main__":
    main()
