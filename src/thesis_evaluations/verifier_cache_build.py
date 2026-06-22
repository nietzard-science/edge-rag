"""
Retrieval cache builder for the verifier-only test suite.

Runs the agentic retrieval stack (Planner -> Navigator -> iterative
multi-hop) once per question and dumps `filtered_context` + plan metadata
to a JSONL file. The output is consumed by `verifier_only_ablation.py`
and `verifier_failure_taxonomy.py`, which sweep Verifier configurations
against the cached retrieval without re-paying the cross-encoder /
iterative-multi-hop cost on every variant.

Why split into two stages
-------------------------
The full pipeline takes ~30-45 s/query (S_P + S_N + S_V); ~70% of that is
the Verifier (distillation LLM call + final answer LLM call). When
iterating on Verifier prompts / flags, re-running the deterministic
retrieval stage (temperature=0) is wasted compute. Cache it once; sweep
verifiers ~10x faster.

The cache is tied to the RETRIEVAL-side configuration in settings.yaml
(reranker model, BM25 toggle, graph hop3, per-source RRF weights). If
those change, rebuild. Verifier-side flags do NOT require a rebuild --
that is the point.

Output JSONL schema (one record per question)
---------------------------------------------
{
    "idx":                int,                # index in the dataset slice
    "question_id":        str,
    "question":           str,
    "gold":               str,                # expected answer
    "question_type":      str,                # bridge / comparison / ...
    "gold_titles":        [str, ...],         # for SF-F1 / SF-Recall
    "supporting_facts":   [...],              # raw dataset supporting_facts
    "query_entities":     [str, ...],         # planner NER output (e.text)
    "plan_query_type":    str,                # MULTI_HOP / SINGLE_HOP / ...
    "matched_pattern":    str | None,         # planner matched_pattern
    "hop_sequence": [
        {"step_id": int, "sub_query": str, "target_entities": [...],
         "depends_on": [...], "is_bridge": bool}, ...
    ],
    "retrieved_chunks": [
        {"text": str, "score": float, "hop_index": int | None,
         "is_graph_based": bool | None}, ...
    ],
    "resolved_bridges":   [str, ...],         # iterative-navigate metadata
    "retrieval_metadata": {...},              # nav.metadata (JSON-safe subset)
    "retrieval_time_ms":  float,
}

Exports
-------
- build_cache(...)             -- run retrieval, dump JSONL, return path
- _serialise_hop / _serialise_plan / _serialise_chunks / _safe_metadata
                                -- internal JSON-safe converters
- main()                       -- CLI entry point

Dependencies / Requirements
---------------------------
- src.thesis_evaluations.benchmark_datasets   -- StoreManager + pipeline factory
- src.pipeline.agent_pipeline                 -- via create_pipeline
- ollama server reachable                     -- planner's query-classify call
- LanceDB / KuzuDB                            -- retrieval stores
- tqdm                                        -- progress bar

Usage (single line; -X utf8 required on Windows / PowerShell):
    python -X utf8 -m src.thesis_evaluations.verifier_cache_build --dataset hotpotqa --samples 50

Outputs
-------
<output-dir>/verifier_cache_<dataset>_n<N>_<planner|noplanner>_<ts>.jsonl

Last reviewed: 2026-06-01 (audit pass, project version 5.5)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.thesis_evaluations.benchmark_datasets import (  # noqa: E402
    StoreManager,
    _close_pipeline,
    _gold_titles_from_supporting_facts,
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

# Why: per-source RRF weights default to 1.0 so the cache is built under the
# SAME retrieval contract as the headline run. Downstream verifier-only
# ablations (verifier_only_ablation, verifier_failure_taxonomy) measure EM/F1
# deltas against the cache, so non-headline weights here would invalidate
# every downstream verifier finding. settings.yaml overrides when present.
_DEFAULT_VECTOR_WEIGHT = 1.0
_DEFAULT_GRAPH_WEIGHT = 1.0

# Why: per-build sample size default. 50 is the quick-look default for
# verifier prompt iteration; 200+ for a publishable verifier ablation.
_DEFAULT_SAMPLES = 50

# Why: project-anchored results root so the CLI works regardless of cwd.
_EVAL_RESULTS_ROOT = _PROJECT_ROOT / "evaluation_results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hop_attrs_to_dict(hop: Any) -> Dict[str, Any]:
    """Pull the five well-known HopStep fields out by attribute access.

    Tolerates HopStep stand-ins (e.g. test doubles) that do not implement
    every dataclass-protocol field.
    """
    return {
        "step_id": getattr(hop, "step_id", None),
        "sub_query": getattr(hop, "sub_query", ""),
        "target_entities": list(getattr(hop, "target_entities", []) or []),
        "depends_on": list(getattr(hop, "depends_on", []) or []),
        "is_bridge": bool(getattr(hop, "is_bridge", False)),
    }


def _serialise_hop(hop: Any) -> Dict[str, Any]:
    """Convert a Planner HopStep into a JSONL-safe dict.

    Prefers `dataclasses.asdict` (captures all fields); falls back to the
    five well-known fields by attribute access when asdict is unavailable
    or fails. Enum fields are coerced to their `.value` by the JSON encoder
    (or by the caller for `query_type`).
    """
    if is_dataclass(hop):
        try:
            return asdict(hop)
        except Exception:  # noqa: BLE001
            return _hop_attrs_to_dict(hop)
    return _hop_attrs_to_dict(hop)


def _serialise_plan(plan: Any) -> Dict[str, Any]:
    """Pull the fields a Verifier-only run needs out of a RetrievalPlan."""
    qt = getattr(plan, "query_type", None)
    qt_str = qt.value if hasattr(qt, "value") else str(qt) if qt is not None else None

    entities = []
    for e in (getattr(plan, "entities", None) or []):
        entities.append(getattr(e, "text", str(e)))

    hop_sequence = [_serialise_hop(h) for h in (getattr(plan, "hop_sequence", None) or [])]

    return {
        "query_type": qt_str,
        "matched_pattern": getattr(plan, "matched_pattern", None),
        "entities": entities,
        "hop_sequence": hop_sequence,
    }


def _serialise_chunks(
    filtered_context: List[str],
    scores: Optional[List[float]],
    metadata: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Serialise the LLM-visible chunks plus per-chunk provenance flags.

    Pulls ``chunk_hop_index`` and ``chunk_is_graph_based`` out of the
    NavigatorResult.metadata when present (set by _iterative_navigate / the
    Navigator respectively). Both are optional -- downstream consumers must
    tolerate None.
    """
    n = len(filtered_context)
    meta = metadata or {}

    hop_index_raw = meta.get("chunk_hop_index")
    hop_index: List[Optional[int]] = (
        [int(h) if isinstance(h, int) else None for h in hop_index_raw]
        if isinstance(hop_index_raw, list) and len(hop_index_raw) == n
        else [None] * n
    )

    graph_raw = meta.get("chunk_is_graph_based")
    graph_flags: List[Optional[bool]] = (
        [bool(g) for g in graph_raw]
        if isinstance(graph_raw, list) and len(graph_raw) == n
        else [None] * n
    )

    score_list: List[float] = (
        [float(s) for s in scores] if scores and len(scores) == n else [0.0] * n
    )

    return [
        {
            "text": text,
            "score": score_list[i],
            "hop_index": hop_index[i],
            "is_graph_based": graph_flags[i],
        }
        for i, text in enumerate(filtered_context)
    ]


def _safe_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Drop non-JSON-serialisable values from NavigatorResult.metadata."""
    if not isinstance(metadata, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in metadata.items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

def build_cache(
    dataset: str,
    n_samples: int,
    config: Dict[str, Any],
    store_manager: StoreManager,
    output_path: Path,
    model_name: Optional[str] = None,
    enable_planner: bool = True,
) -> Path:
    """
    Run Planner -> Navigator over `n_samples` questions and dump JSONL.

    Verifier is disabled (enable_verifier=False); the pipeline still computes
    the full Planner + Navigator chain (including _iterative_navigate for
    bridge-dependent plans), so the cached chunks are exactly the chunks the
    Verifier would have received in the full agentic run.

    enable_planner toggles the retrieval style the verifier is tuned on:
      True  : planner-decomposed retrieval, including iterative multi-hop.
              Higher SF-Recall; lower answer-conversion at small N.
      False : bare-RAG retrieval -- the original query is retrieved once
              with a passthrough plan. Use this to tune the verifier on
              top of the higher-EM retrieval-only path rather than on the
              planner-decomposed one.
    """
    questions = store_manager.load_questions(dataset)[:n_samples]
    if not questions:
        raise RuntimeError(f"No questions loaded for {dataset}")

    # Same retrieval contract as the headline run, read from settings.yaml.
    rag_cfg = config.get("rag", {})
    vector_weight = float(rag_cfg.get("vector_weight", _DEFAULT_VECTOR_WEIGHT))
    graph_weight = float(rag_cfg.get("graph_weight", _DEFAULT_GRAPH_WEIGHT))

    pipeline = create_pipeline(
        dataset=dataset,
        config=config,
        store_manager=store_manager,
        vector_weight=vector_weight,
        graph_weight=graph_weight,
        model_name=model_name,
        enable_planner=enable_planner,
        enable_verifier=False,           # the whole point: skip S_V
        max_iterations=1,
        enable_pre_validation=False,     # pre-val is part of Verifier; skip
    )

    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_err = 0
    total_chunks = 0
    start_wall = time.time()

    # Why: open the JSONL once, not per question -- avoids N open/close
    # syscalls + fsyncs. flush() after every record preserves crash-resume
    # behaviour for long cache builds.
    try:
        with open(output_path, "a", encoding="utf-8") as fh:
            for idx, q in enumerate(tqdm(questions, desc="Building cache", unit="q")):
                t0 = time.time()
                try:
                    result = pipeline.process(q.question)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("idx=%d pipeline.process() raised: %s", idx, exc)
                    n_err += 1
                    continue
                retrieval_ms = (time.time() - t0) * 1000.0

                # navigator_result is asdict()'ed inside process(); pull the
                # bits needed here without depending on which Enum fields
                # exist on the underlying dataclasses.
                nav = result.navigator_result or {}
                filtered_context = list(nav.get("filtered_context") or [])
                scores = list(nav.get("scores") or [])
                nav_metadata = nav.get("metadata") or {}

                # planner_result is also already a dict (plan.to_dict()).
                # When RetrievalPlan.to_dict()'s schema is missing fields,
                # the per-attribute fallback below catches HopStep
                # stand-ins.
                planner = result.planner_result or {}
                entities_list = planner.get("entities") or []
                entities_text: List[str] = []
                for e in entities_list:
                    if isinstance(e, dict):
                        entities_text.append(e.get("text", ""))
                    else:
                        entities_text.append(str(e))

                hop_sequence_dicts: List[Dict[str, Any]] = []
                raw_hops = planner.get("hop_sequence") or []
                for h in raw_hops:
                    if isinstance(h, dict):
                        hop_sequence_dicts.append({
                            "step_id": h.get("step_id"),
                            "sub_query": h.get("sub_query", ""),
                            "target_entities": list(h.get("target_entities") or []),
                            "depends_on": list(h.get("depends_on") or []),
                            "is_bridge": bool(h.get("is_bridge", False)),
                        })
                    else:
                        hop_sequence_dicts.append(_serialise_hop(h))

                query_type_str = planner.get("query_type")
                if hasattr(query_type_str, "value"):
                    query_type_str = query_type_str.value

                chunks_serialised = _serialise_chunks(
                    filtered_context,
                    scores if isinstance(scores, list) else None,
                    nav_metadata if isinstance(nav_metadata, dict) else None,
                )
                total_chunks += len(chunks_serialised)

                gold_titles = _gold_titles_from_supporting_facts(q.supporting_facts)

                record = {
                    "idx": idx,
                    "question_id": q.id,
                    "question": q.question,
                    "gold": q.answer,
                    "question_type": q.question_type,
                    "dataset": q.dataset,
                    "gold_titles": gold_titles,
                    "supporting_facts": q.supporting_facts,
                    "query_entities": entities_text,
                    "plan_query_type": query_type_str,
                    "matched_pattern": planner.get("matched_pattern"),
                    "hop_sequence": hop_sequence_dicts,
                    "retrieved_chunks": chunks_serialised,
                    "resolved_bridges": list(
                        (nav_metadata.get("resolved_bridges") or [])
                        if isinstance(nav_metadata, dict) else []
                    ),
                    "retrieval_metadata": _safe_metadata(
                        nav_metadata if isinstance(nav_metadata, dict) else None
                    ),
                    "retrieval_time_ms": retrieval_ms,
                }

                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                n_ok += 1
    finally:
        _close_pipeline(pipeline)

    elapsed = time.time() - start_wall
    logger.info(
        "Cache built: %s | n_ok=%d n_err=%d | total_chunks=%d | elapsed=%.1fs",
        output_path, n_ok, n_err, total_chunks, elapsed,
    )
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the retrieval cache for the verifier-only test suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", "-d", default="hotpotqa")
    parser.add_argument("--samples", "-n", type=int, default=_DEFAULT_SAMPLES,
                        help=f"Number of questions to cache (default: "
                             f"{_DEFAULT_SAMPLES}; >=200 for a publishable "
                             f"verifier ablation).")
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help=("Output directory. The JSONL is named "
              "verifier_cache_<dataset>_n<N>_<planner|noplanner>_<ts>.jsonl "
              "inside it. Default: <project-root>/evaluation_results/verifier_cache"),
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="LLM model name override (default: settings.yaml). The Verifier "
             "is disabled, so this only affects the Planner's query "
             "classification LLM call (if any).",
    )
    parser.add_argument(
        "--no-planner", action="store_true",
        help="Build the cache on BARE-RAG retrieval (passthrough plan, single "
             "retrieval of the original query) instead of the planner-"
             "decomposed path. Use this to tune the verifier on top of the "
             "higher-EM retrieval-only path. The filename is tagged "
             "'_noplanner' so the two caches don't collide.",
    )
    args = parser.parse_args()

    config = load_config_file()
    store_manager = StoreManager()
    if not store_manager.dataset_exists(args.dataset):
        logger.error("Dataset not ingested: %s", args.dataset)
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else (_EVAL_RESULTS_ROOT / "runs" / "verifier_cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    planner_tag = "noplanner" if args.no_planner else "planner"
    output_path = (
        out_dir
        / f"verifier_cache_{args.dataset}_n{args.samples}_{planner_tag}_{ts}.jsonl"
    )

    logger.info("Building retrieval cache:")
    logger.info("  dataset = %s", args.dataset)
    logger.info("  samples = %d", args.samples)
    logger.info("  planner = %s", not args.no_planner)
    logger.info("  output  = %s", output_path)

    build_cache(
        dataset=args.dataset,
        n_samples=args.samples,
        config=config,
        store_manager=store_manager,
        output_path=output_path,
        model_name=args.model,
        enable_planner=not args.no_planner,
    )

    logger.info("Done. Cache written to: %s", output_path)
    logger.info(
        "Next: python -X utf8 -m src.thesis_evaluations.verifier_only_ablation "
        "--cache %s",
        output_path,
    )


if __name__ == "__main__":
    main()
