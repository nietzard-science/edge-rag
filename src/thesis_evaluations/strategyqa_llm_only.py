"""
StrategyQA LLM-only baseline (no retrieval).

StrategyQA's open-domain commonsense formulation is incompatible with the
per-question contextual-retrieval design used for HotpotQA / 2WikiMultiHopQA
(the dataset ships no per-question documents in the loaded variant; see the
"0 facts, 500 questions" warning in benchmark_datasets.StrategyQALoader).
Rather than fight the dataset's design, the paper reports StrategyQA only as
a parametric LLM-only reference point — the same parametric baseline the
agentic_ablation row 1 ("LLM-only · no retrieval") produces for the other
datasets. This bounds the deployed SLM's commonsense ability separately from
the retrieval pipeline and substantiates the dataset-scoping decision in the
methodology chapter (see TECHNICAL_ARCHITECTURE.md §11.17.* / §12.3).

This script bypasses the `dataset_exists` ingestion guard in
`benchmark_datasets.main` (which would refuse to run because no vector/graph
stores exist for StrategyQA) and calls `agentic_ablation.run_llm_only_row`
directly with the questions loaded from `questions.json` (saved by the
chunks-only ingest).

Exports
-------
- main()       -- CLI entry point

Dependencies / Requirements
---------------------------
- src.thesis_evaluations.agentic_ablation       -- run_llm_only_row
- src.thesis_evaluations.benchmark_datasets     -- StoreManager + load_config_file
- ollama server reachable                       -- LLM call
- data/strategyqa/questions.json must exist     -- produced by chunks-only ingest

Usage (single line; -X utf8 on Windows / PowerShell):
    python -X utf8 -m src.thesis_evaluations.strategyqa_llm_only --samples 500

Output
------
evaluation_results/strategyqa_llm_only_<ts>/
    row1_llm_only.jsonl       per-question records
    summary.json              aggregate metrics

Last reviewed: 2026-06-01 (audit pass, project version 5.5).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.thesis_evaluations.agentic_ablation import run_llm_only_row  # noqa: E402
from src.thesis_evaluations.benchmark_datasets import (  # noqa: E402
    StoreManager,
    load_config_file,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_DATASET = "strategyqa"

# Why: project-anchored data / eval-results roots + emergency model fallback.
# Matches the convention in the sibling thesis_evaluations scripts.
_DATA_ROOT = _PROJECT_ROOT / "data"
_EVAL_RESULTS_ROOT = _PROJECT_ROOT / "evaluation_results"
_DEFAULT_OUTPUT_DIR = _EVAL_RESULTS_ROOT / "runs" / "strategyqa_llm_only"
_DEFAULT_MODEL_FALLBACK = "qwen2:1.5b"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--samples", "-n", type=int, default=500,
                        help="Number of questions to evaluate (default: 500).")
    parser.add_argument("--model", "-m", default=None,
                        help="LLM model name (default: llm.model_name from settings.yaml).")
    parser.add_argument("--output", "-o", type=str,
                        default=str(_DEFAULT_OUTPUT_DIR),
                        help="Output directory base; a timestamped subdir is created.")
    args = parser.parse_args()

    config = load_config_file()
    store_manager = StoreManager(_DATA_ROOT)

    # Load questions directly. We deliberately skip dataset_exists (StrategyQA
    # has no vector/graph stores; the LLM-only row doesn't need them).
    try:
        questions = store_manager.load_questions(_DATASET)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not load StrategyQA questions: %s", exc)
        logger.error("Run `benchmark_datasets ingest -d strategyqa --chunks-only` first "
                     "to produce data/strategyqa/questions.json.")
        return
    questions = questions[: args.samples]
    if not questions:
        logger.error("No StrategyQA questions loaded.")
        return

    model_name = args.model or config.get("llm", {}).get("model_name", _DEFAULT_MODEL_FALLBACK)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "row1_llm_only.jsonl"

    logger.info("StrategyQA LLM-only baseline:")
    logger.info("  model   = %s", model_name)
    logger.info("  samples = %d", len(questions))
    logger.info("  output  = %s", out_dir)

    row = run_llm_only_row(
        model_name=model_name,
        config=config,
        questions=questions,
        jsonl_out=jsonl_path,
    )
    if row is None or row.get("n_questions", 0) == 0:
        logger.error("LLM-only row produced no results.")
        return

    (out_dir / "summary.json").write_text(
        json.dumps({"timestamp": ts, "dataset": _DATASET,
                    "model": model_name, "n_samples": len(questions),
                    "row": row}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Done. EM=%.1f%% F1=%.3f LLM-err=%.1f%% (n=%d)",
                row["em"] * 100, row["f1"], row["llm_error_rate"] * 100,
                row["n_questions"])
    logger.info("Per-question JSONL: %s", jsonl_path)
    logger.info("Summary:            %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()
