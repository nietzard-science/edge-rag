"""
Chunking hyperparameter ablation for the edge-RAG pipeline.

For each (sentences_per_chunk, sentence_overlap) cell:
    1. Re-chunk the dataset's source articles via SpacySentenceChunker.
    2. Re-ingest into a per-config LanceDB vector store. The KuzuDB graph
       is reused unchanged from the production ingestion -- only the
       chunking-affected layer is rebuilt.
    3. Evaluate in retrieval-only mode (verifier disabled). Headline
       metrics: SF-F1 and SF-Recall against gold supporting-paragraph
       titles.
    4. Emit a per-config JSONL of question-level metrics plus a summary
       table with deltas vs. the production-default baseline (3:1).

The output answers a single methodological question: how sensitive is
retrieval recall to the chunking window size and overlap?

Methodology note
----------------
This ablation varies only the SpacySentenceChunker (window, overlap).
The graph component (Phase-2 NER + RE outputs and the KuzuDB store) is
held CONSTANT across all configurations because re-running Phase 2 per
chunking config would change the entity space and confound the ablation.
The script therefore measures the vector-retrieval contribution of
chunking, not the full hybrid system end-to-end. The paper text must
state this constraint explicitly, and that the configurations probe one
dimension of the chunking hyperparameter space (not exhaustive).

Exports
-------
- DEFAULT_CONFIGS           -- canonical (sentences, overlap) grid
- parse_configs(spec)       -- "3:1,5:1" -> [(3,1),(5,1)] with validation
- run_one_config(...)       -- run one cell, return headline metrics
- write_summary(...)        -- emit summary.{csv,md} with baseline deltas
- main()                    -- CLI entry point

Dependencies / Requirements
---------------------------
- src.thesis_evaluations.benchmark_datasets   -- loaders + chunker + eval
- src.data_layer.chunking.SpacySentenceChunker
- LanceDB (per-config vector store) and the production KuzuDB graph
- ollama server reachable                     -- embeddings during ingest

Usage (single line; -X utf8 required on Windows / PowerShell):
    python -X utf8 -m src.thesis_evaluations.chunking_ablation --dataset hotpotqa --samples 100 --configs "3:1,5:1,7:1,3:0,5:2"

`--configs` is a comma-separated list of "<sentences>:<overlap>" pairs.
Default: "3:1,5:1,7:1" (the three primary cells); pass "all" for the
full canonical grid (DEFAULT_CONFIGS).

Outputs
-------
evaluation_results/chunking_ablation_<ts>/
    config_s<W>_o<O>.jsonl    per-question retrieval-only records
    summary.csv / .md         baseline-relative delta table

Last reviewed: 2026-06-01 (audit pass, project version 5.5)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.thesis_evaluations.benchmark_datasets import (  # noqa: E402
    LOADERS,
    StoreManager,
    TestQuestion,
    create_langchain_documents,
    create_pipeline,
    evaluate_dataset,
    load_config_file,
    run_ingestion,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# Why: per-source RRF weights default to 1.0 so the ablation uses the SAME
# retrieval contract as the headline run. settings.yaml overrides when
# present.
_DEFAULT_VECTOR_WEIGHT = 1.0
_DEFAULT_GRAPH_WEIGHT = 1.0

# Why: emergency fallback when no settings entry is present. The ablation
# runs retrieval-only (verifier disabled) so the model name is unused at
# query time -- but ollama still loads the model when the pipeline is
# constructed, so this must point at a tag the user has pulled.
_DEFAULT_MODEL_FALLBACK = "qwen2:1.5b"

# Why: per-config sample size default for the CLI. 100 is the quick-look
# default; 500 for a final-run with tighter CIs.
_DEFAULT_SAMPLES = 100

# Why: project-anchored eval-results root so the CLI works regardless of cwd.
# Matches the pattern in sibling scripts (benchmark_datasets, agentic_ablation).
_EVAL_RESULTS_ROOT = _PROJECT_ROOT / "evaluation_results"


# ---------------------------------------------------------------------------
# Ablation infrastructure
# ---------------------------------------------------------------------------

class _AblationStoreManager(StoreManager):
    """StoreManager subclass that returns a custom vector-store path while
    keeping all other dataset paths (graph, questions, articles_info)
    pointing at the production locations.

    Used to direct `create_pipeline` at the per-config vector store this
    ablation builds, without disturbing the rest of the dataset layout.
    """

    def __init__(self, vector_override: Path):
        super().__init__()
        self._vector_override = Path(vector_override)

    def get_paths(self, dataset: str) -> Dict[str, Path]:
        paths = super().get_paths(dataset)
        paths["vector"] = self._vector_override
        return paths


# Why: default ablation grid. The three primary cells (3-1, 5-1, 7-1) probe
# window size at constant overlap; the two extra cells (3-0, 5-2) probe
# overlap at constant window. Pass --configs all for the full grid.
DEFAULT_CONFIGS: List[Tuple[int, int]] = [
    (3, 1),  # production default -- the baseline of this ablation
    (5, 1),
    (7, 1),
    (3, 0),
    (5, 2),
]


def parse_configs(spec: str) -> List[Tuple[int, int]]:
    """Parse "3:1,5:1,7:1" → [(3,1), (5,1), (7,1)] with validation."""
    out: List[Tuple[int, int]] = []
    for cell in spec.split(","):
        cell = cell.strip()
        if not cell:
            continue
        try:
            s, o = cell.split(":")
            s_int, o_int = int(s), int(o)
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Bad config cell {cell!r}; expected 'sentences:overlap'"
            ) from exc
        if s_int < 1:
            raise ValueError(f"sentences_per_chunk must be >= 1; got {s_int}")
        if o_int < 0:
            raise ValueError(f"sentence_overlap must be >= 0; got {o_int}")
        if o_int >= s_int:
            raise ValueError(
                f"overlap ({o_int}) must be < window ({s_int}); "
                f"otherwise no progress is made between chunks"
            )
        out.append((s_int, o_int))
    if not out:
        raise ValueError("No valid configs supplied")
    return out


# ────────────────────────────────────────────────────────────────────────────
# PER-CONFIG RUN
# ────────────────────────────────────────────────────────────────────────────

def run_one_config(
    sentences: int,
    overlap: int,
    dataset: str,
    config: Dict,
    store_manager: StoreManager,
    questions: List[TestQuestion],
    jsonl_path: Path,
    ablation_workdir: Path,
    apply_coreference: bool,
) -> Dict[str, float]:
    """Run a single chunking configuration end-to-end.

    Returns a dict of headline metrics so the caller can build the summary
    table without re-reading the JSONL.
    """
    tag = f"s{sentences}_o{overlap}"
    logger.info("=" * 70)
    logger.info("CONFIG: sentences_per_chunk=%d, sentence_overlap=%d",
                sentences, overlap)
    logger.info("=" * 70)

    # Step 1: load source articles via the dataset's loader.
    # LOADERS[dataset].load() returns (articles, questions); discard the
    # latter since the previously-saved questions.json is reused.
    if dataset not in LOADERS:
        raise RuntimeError(f"Unknown dataset: {dataset}")
    loader = LOADERS[dataset]
    articles, _ = loader.load(n_samples=None)  # load all articles
    if not articles:
        raise RuntimeError(
            f"No articles found for {dataset}. Ensure the dataset's source "
            f"corpus is present at the loader's expected location."
        )
    logger.info("Loaded %d source articles", len(articles))

    # Step 2: re-chunk with this config.
    documents = create_langchain_documents(
        articles,
        chunk_sentences=sentences,
        sentence_overlap=overlap,
        apply_coreference=apply_coreference,
    )
    logger.info("Created %d chunks (window=%d, overlap=%d)",
                len(documents), sentences, overlap)

    # Step 3: re-ingest into a dedicated per-config vector store.
    # KuzuDB graph store: reused unchanged from the dataset's main
    # ingestion. Only the vector store is rebuilt per config -- see module
    # docstring.
    vector_path = ablation_workdir / tag / "vector"
    graph_path = store_manager.get_paths(dataset)["graph"]
    if vector_path.exists():
        logger.info("Clearing previous vector store at %s", vector_path)
        shutil.rmtree(vector_path)
    vector_path.parent.mkdir(parents=True, exist_ok=True)

    t_ingest = time.time()
    run_ingestion(documents, vector_path, graph_path, config, dataset)
    ingest_seconds = time.time() - t_ingest
    logger.info("Ingest took %.0fs", ingest_seconds)

    # Step 4: build a pipeline pointing at the per-config vector store.
    # _AblationStoreManager overrides only the vector path; graph store and
    # other dataset files come from the production location, so the graph
    # side of the hybrid retrieval is held constant.
    ablation_store = _AblationStoreManager(vector_override=vector_path)

    # Why: weights match the headline retrieval contract (read from
    # settings.yaml). Model name comes from settings.yaml too -- the
    # verifier is disabled below, so the model is loaded but not queried.
    rag_cfg = config.get("rag", {})
    vector_weight = float(rag_cfg.get("vector_weight", _DEFAULT_VECTOR_WEIGHT))
    graph_weight = float(rag_cfg.get("graph_weight", _DEFAULT_GRAPH_WEIGHT))
    model_name = (
        config.get("llm", {}).get("model_name", _DEFAULT_MODEL_FALLBACK)
    )

    pipeline = create_pipeline(
        dataset, config, ablation_store,
        vector_weight=vector_weight,
        graph_weight=graph_weight,
        model_name=model_name,
        enable_planner=True,
        enable_verifier=False,    # retrieval-only ablation
        max_iterations=1,
    )

    # Step 5: evaluate (retrieval-only).
    t_eval = time.time()
    result = evaluate_dataset(
        dataset=dataset,
        questions=questions,
        pipeline=pipeline,
        config_name=tag,
        vector_weight=vector_weight,
        graph_weight=graph_weight,
        jsonl_out=jsonl_path,
        retrieval_only=True,
    )
    eval_seconds = time.time() - t_eval
    logger.info("Eval took %.0fs", eval_seconds)

    return {
        "sentences": sentences,
        "overlap": overlap,
        "n_chunks": len(documents),
        "n_questions": len(questions),
        "sf_f1": result.avg_sf_f1,
        "sf_recall_rate": result.sf_recall_rate,
        "ingest_seconds": ingest_seconds,
        "eval_seconds": eval_seconds,
    }


# ────────────────────────────────────────────────────────────────────────────
# SUMMARY WRITER
# ────────────────────────────────────────────────────────────────────────────

def write_summary(
    results: List[Dict[str, float]],
    out_dir: Path,
    baseline_tag: str = "s3_o1",
) -> None:
    """Write summary.csv and summary.md.

    summary.md is the table that goes into the paper methodology section.
    All deltas are relative to the baseline (default: s3_o1, the production
    default).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sentences", "overlap", "n_chunks", "n_questions",
            "sf_f1", "sf_recall_rate", "ingest_seconds", "eval_seconds",
        ])
        writer.writeheader()
        writer.writerows(results)
    logger.info("Wrote %s", csv_path)

    # Markdown table -- find baseline.
    baseline = next(
        (r for r in results if f"s{r['sentences']}_o{r['overlap']}" == baseline_tag),
        results[0],
    )

    lines = ["# Chunking Ablation Results", ""]
    lines.append(
        f"Baseline: window={baseline['sentences']} sentences, "
        f"overlap={baseline['overlap']}."
    )
    lines.append("")
    lines.append(
        "| Window | Overlap | Chunks | SF-F1 | SF-Recall | d-SF-F1 | d-SF-Recall |"
    )
    lines.append(
        "|-------:|--------:|-------:|------:|----------:|--------:|------------:|"
    )
    for r in results:
        d_f1 = r["sf_f1"] - baseline["sf_f1"]
        d_rec = r["sf_recall_rate"] - baseline["sf_recall_rate"]
        is_baseline = (r["sentences"] == baseline["sentences"]
                       and r["overlap"] == baseline["overlap"])
        marker = " *(baseline)*" if is_baseline else ""
        lines.append(
            f"| {r['sentences']} | {r['overlap']} | {r['n_chunks']:,} | "
            f"{r['sf_f1']:.3f} | {r['sf_recall_rate']:.2%} | "
            f"{d_f1:+.3f}{marker} | {d_rec:+.2%} |"
        )
    lines.append("")
    lines.append("d-* columns: difference from baseline. Positive = improvement.")
    lines.append("")
    lines.append("## Methodology footnote")
    lines.append("")
    lines.append(
        "This ablation varies one hyperparameter pair (sentences-per-chunk, "
        "sentence-overlap) of the SpacySentenceChunker while holding the rest "
        "of the system constant. The knowledge-graph component (Phase 2 NER+RE "
        "outputs and the KuzuDB store) is reused from the production "
        "ingestion across all configurations; only the vector-retrieval layer "
        "(LanceDB) is rebuilt per config. Evaluation is retrieval-only: the "
        "Verifier (S_V) is disabled and SF-F1 / SF-Recall against gold "
        "supporting-paragraph titles is the headline metric."
    )

    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", md_path)


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunking hyperparameter ablation -- measures retrieval "
                    "recall sensitivity to chunk window / overlap.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default="hotpotqa",
                        help="Dataset name (default: hotpotqa)")
    parser.add_argument("--samples", type=int, default=_DEFAULT_SAMPLES,
                        help=f"Number of questions to evaluate per config "
                             f"(default: {_DEFAULT_SAMPLES}; use 500 for the "
                             f"final run with tighter CIs).")
    parser.add_argument("--configs", default="3:1,5:1,7:1",
                        help="Comma-separated 'sentences:overlap' cells "
                             "(default: '3:1,5:1,7:1'). Use 'all' for the "
                             "full canonical grid (DEFAULT_CONFIGS).")
    parser.add_argument("--no-coreference", action="store_true",
                        help="Disable coreference resolution before chunking. "
                             "Holds coref off across all configs so only the "
                             "chunking dimension varies.")
    parser.add_argument("--workdir", type=Path, default=None,
                        help="Working directory for per-config vector stores. "
                             "Default: evaluation_results/chunking_ablation_<ts>/")
    args = parser.parse_args()

    if args.configs.strip().lower() == "all":
        configs = DEFAULT_CONFIGS
    else:
        configs = parse_configs(args.configs)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.workdir or (_EVAL_RESULTS_ROOT / "runs" / f"chunking_ablation_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    # Setup
    config = load_config_file()
    store_manager = StoreManager()

    if not store_manager.dataset_exists(args.dataset):
        logger.error("Dataset not ingested: %s", args.dataset)
        logger.error(
            "Run `python -m src.thesis_evaluations.benchmark_datasets ingest "
            "--dataset %s` first.", args.dataset,
        )
        sys.exit(1)

    questions = store_manager.load_questions(args.dataset)
    if args.samples:
        questions = questions[: args.samples]
    logger.info("Loaded %d questions for %s", len(questions), args.dataset)

    apply_coref = not args.no_coreference

    # Run each config
    results: List[Dict[str, float]] = []
    for sentences, overlap in configs:
        tag = f"s{sentences}_o{overlap}"
        jsonl_path = out_dir / f"config_{tag}.jsonl"
        try:
            r = run_one_config(
                sentences=sentences,
                overlap=overlap,
                dataset=args.dataset,
                config=config,
                store_manager=store_manager,
                questions=questions,
                jsonl_path=jsonl_path,
                ablation_workdir=out_dir,
                apply_coreference=apply_coref,
            )
            results.append(r)
        except Exception as exc:  # noqa: BLE001
            # Per-config crash isolation: a single cell failure (ingest OOM,
            # KuzuDB lock, malformed chunk, …) must not abort the whole
            # ablation matrix. Record a NaN row so the cell still shows up
            # in summary.md as "did not complete" rather than vanishing
            # silently.
            logger.exception("Config %s failed: %s", tag, exc)
            results.append({
                "sentences": sentences,
                "overlap": overlap,
                "n_chunks": 0,
                "n_questions": len(questions),
                "sf_f1": float("nan"),
                "sf_recall_rate": float("nan"),
                "ingest_seconds": 0.0,
                "eval_seconds": 0.0,
            })

    # Write summary
    write_summary(results, out_dir)
    logger.info("Done.")
    logger.info("See %s/summary.md for the table.", out_dir)


if __name__ == "__main__":
    main()
