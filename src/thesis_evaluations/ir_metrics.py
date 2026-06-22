"""
Information-retrieval metrics over the per-question benchmark JSONL.

Why this module exists
----------------------
The benchmark reports supporting-fact quality as set-based precision / recall /
F1 (`_compute_sf_metrics` in benchmark_datasets.py) and a strict
`all_gold_retrieved` flag. Those answer "did we get the gold documents at all?"
but throw away *rank*: a system that places the gold title first and a system
that places it tenth score identically. IR-standard metrics that a reviewer
expects for a retrieval claim — Recall@k, nDCG@k, MRR — are rank-aware and were
missing. This module adds them as pure post-processing over the JSONL the
benchmark already writes.

Relevance model
---------------
Gold supporting facts are recorded per question as a set of *titles*
(`gold_titles`); the retrieved context is recorded as a rank-ordered list of
*titles* (`retrieved_titles`, deduped, in the order the retriever/Navigator
surfaced them). Relevance is therefore binary at title granularity:
`gain(title) = 1 if title in gold_titles else 0`. This matches the granularity
of the existing SF-F1 / SF-Recall metrics — the codebase resolves supporting
facts to titles, not sentences, because chunks do not carry sentence ids
(see `_compute_sf_metrics`). The metrics here are thus *title-level* IR metrics,
and are labelled as such wherever they surface.

Metrics
-------
- **Recall@k**   — fraction of a question's gold titles appearing in the top-k
                   retrieved titles, averaged over questions. With binary
                   relevance this is |gold ∩ top_k| / |gold|.
- **nDCG@k**     — DCG of the top-k binary-relevance vector normalised by the
                   ideal DCG (all gold ranked first). Uses the standard
                   `1 / log2(rank+1)` discount (Järvelin & Kekäläinen 2002).
- **MRR**        — reciprocal rank of the *first* gold title in the retrieved
                   list (0 if none retrieved), averaged over questions
                   (Voorhees 1999).

Scope
-----
Pure post-processing: stdlib only (math + json), no model calls, no Ollama, no
pipeline. Mirrors bootstrap.py's JSONL-adapter style so the two compose: both
key off `question_id` and read the same per-question records.

References
----------
- Järvelin, K. & Kekäläinen, J. (2002). "Cumulated gain-based evaluation of IR
  techniques." ACM TOIS 20(4). (nDCG.)
- Voorhees, E. (1999). "The TREC-8 Question Answering Track Report." (MRR.)
- Manning, Raghavan & Schütze (2008). "Introduction to Information Retrieval."
  (Recall@k, evaluation conventions.)

Last reviewed: 2026-06-05.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Default cutoffs reported in the paper retrieval tables. k=5/10/20 spans the
# Navigator's max_context_chunks (8) on both sides so the curve is visible.
DEFAULT_KS: tuple[int, ...] = (5, 10, 20)


# ─────────────────────────────────────────────────────────────────────────────
# Single-query primitives (binary relevance over a ranked title list)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(title: str) -> str:
    """Case/space-fold a title for set membership.

    The benchmark already stores normalised titles (``_norm_title``), so this
    is a defensive second pass for callers that pass raw titles directly.
    """
    return " ".join(str(title).strip().lower().split())


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Fraction of gold titles found within the top-k retrieved titles.

    Binary relevance: returns |gold ∩ retrieved[:k]| / |gold|. Returns 0.0 if
    there are no gold titles (an unjudged question contributes nothing).
    """
    gold_set = {_normalize(g) for g in gold or []}
    if not gold_set:
        return 0.0
    top_k = {_normalize(t) for t in list(retrieved or [])[:k]}
    return len(gold_set & top_k) / len(gold_set)


def dcg_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Discounted cumulative gain of the top-k binary-relevance vector.

    gain(rank) = 1 if the title at that rank is gold else 0, discounted by
    1/log2(rank+1) with rank starting at 1 (so the top item has discount 1.0).
    """
    gold_set = {_normalize(g) for g in gold or []}
    if not gold_set:
        return 0.0
    dcg = 0.0
    for i, title in enumerate(list(retrieved or [])[:k]):
        if _normalize(title) in gold_set:
            # rank = i + 1; discount = 1 / log2(rank + 1).
            dcg += 1.0 / math.log2(i + 2)
    return dcg


def ndcg_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """nDCG@k = DCG@k / ideal-DCG@k, in [0, 1].

    The ideal ranking places all gold titles first; with binary relevance the
    ideal DCG is the sum of the first min(|gold|, k) discounts. Returns 0.0 if
    there are no gold titles.
    """
    gold_set = {_normalize(g) for g in gold or []}
    if not gold_set:
        return 0.0
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(retrieved, gold, k) / idcg


def reciprocal_rank(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    """Reciprocal rank of the FIRST gold title in the retrieved list.

    Returns 1/rank (rank starting at 1) for the earliest gold hit, or 0.0 if no
    gold title was retrieved. The per-question component of MRR.
    """
    gold_set = {_normalize(g) for g in gold or []}
    if not gold_set:
        return 0.0
    for i, title in enumerate(retrieved or []):
        if _normalize(title) in gold_set:
            return 1.0 / (i + 1)
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IRMetrics:
    """Corpus-level IR metrics averaged over questions.

    ``recall_at_k`` maps each cutoff k to mean Recall@k; ``ndcg_at_k`` maps each
    cutoff k to mean nDCG@k. ``mrr`` is mean reciprocal rank. ``n_questions`` is
    the number of *judged* questions (those with at least one gold title) the
    averages are taken over — unjudged questions are excluded so an all-zero
    row from a dataset without gold titles cannot silently deflate the mean.
    """
    n_questions: int
    mrr: float
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {"n_questions": self.n_questions, "mrr": self.mrr}
        for k, v in sorted(self.recall_at_k.items()):
            out[f"recall_at_{k}"] = v
        for k, v in sorted(self.ndcg_at_k.items()):
            out[f"ndcg_at_{k}"] = v
        return out

    def __str__(self) -> str:
        rec = "  ".join(f"R@{k}={v:.3f}" for k, v in sorted(self.recall_at_k.items()))
        ndcg = "  ".join(f"nDCG@{k}={v:.3f}" for k, v in sorted(self.ndcg_at_k.items()))
        return f"MRR={self.mrr:.3f}  {rec}  {ndcg}  (n={self.n_questions})"


def compute_ir_metrics(
    records: Sequence[Dict],
    ks: Sequence[int] = DEFAULT_KS,
    retrieved_key: str = "retrieved_titles",
    gold_key: str = "gold_titles",
) -> IRMetrics:
    """Aggregate IR metrics over per-question JSONL records.

    Each record must carry a rank-ordered ``retrieved_titles`` list and a
    ``gold_titles`` list (the fields benchmark_datasets.py already writes).
    Records with no gold titles are skipped (unjudged) so they do not deflate
    the averages; if no record is judged, an all-zero IRMetrics with
    ``n_questions == 0`` is returned.
    """
    judged = [r for r in records if r.get(gold_key)]
    n = len(judged)
    if n == 0:
        return IRMetrics(
            n_questions=0,
            mrr=0.0,
            recall_at_k={k: 0.0 for k in ks},
            ndcg_at_k={k: 0.0 for k in ks},
        )

    recall_sums = {k: 0.0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}
    rr_sum = 0.0
    for r in judged:
        retrieved = r.get(retrieved_key) or []
        gold = r.get(gold_key) or []
        for k in ks:
            recall_sums[k] += recall_at_k(retrieved, gold, k)
            ndcg_sums[k] += ndcg_at_k(retrieved, gold, k)
        rr_sum += reciprocal_rank(retrieved, gold)

    return IRMetrics(
        n_questions=n,
        mrr=rr_sum / n,
        recall_at_k={k: recall_sums[k] / n for k in ks},
        ndcg_at_k={k: ndcg_sums[k] / n for k in ks},
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSONL adapter (mirrors bootstrap.load_jsonl_records)
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl_records(path: Path) -> List[Dict]:
    """Load a per-question JSONL into a list of dicts, skipping bad lines.

    Returns ``[]`` if the file does not exist or cannot be read — a lane whose
    every question errored (e.g. an Ollama timeout) writes no JSONL, and the
    caller treats an empty result as "this lane produced nothing" rather than
    crashing the whole run on a missing file.
    """
    records: List[Dict] = []
    if not Path(path).exists():
        return records
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return records


def compute_ir_metrics_from_jsonl(
    jsonl_path: Path,
    ks: Sequence[int] = DEFAULT_KS,
) -> IRMetrics:
    """Convenience: load a JSONL file and compute IR metrics over it."""
    return compute_ir_metrics(load_jsonl_records(jsonl_path), ks=ks)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _main() -> None:
    """Standalone CLI: print title-level IR metrics for a per-question JSONL.

        python -m src.thesis_evaluations.ir_metrics path/to/results.jsonl
        python -m src.thesis_evaluations.ir_metrics results.jsonl --ks 5,10,20
    """
    import argparse

    parser = argparse.ArgumentParser(description=_main.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jsonl", type=Path, help="Per-question JSONL file.")
    parser.add_argument("--ks", type=str, default="5,10,20",
                        help="Comma-separated Recall@k / nDCG@k cutoffs.")
    parser.add_argument("--json", action="store_true",
                        help="Emit the metrics as a JSON object instead of text.")
    args = parser.parse_args()

    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())
    metrics = compute_ir_metrics_from_jsonl(args.jsonl, ks=ks)
    if args.json:
        print(json.dumps(metrics.as_dict(), indent=2))
    else:
        print(metrics)


if __name__ == "__main__":
    _main()
