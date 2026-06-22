"""
Verifier failure taxonomy -- qualitative diagnostic for the variant sweep.

Joins a verifier-only sweep JSONL (per-question predicted/gold/EM/F1) with
the retrieval cache (per-question chunk texts) and buckets every WRONG
answer (exact_match == False) into a failure mode. The point is to
localise where the verifier loses: is the gold answer absent from the
chunks (retrieval's fault), or present-but-mishandled (the verifier's
fault, and the target for any verification upgrade)?

Buckets
-------
    a_retrieval_miss   gold answer string is NOT in any retrieved chunk.
                       The verifier cannot be blamed -- retrieval failed.
    b_grounded_halluc  gold IS in the chunks; the prediction is a
                       different, low-overlap answer (F1 < 0.5) and not
                       an abstention. The hallucination bucket -- what
                       NLI grounding / answer-entailment targets.
    c_format_mismatch  gold is a name/date/number but the prediction is
                       yes/no (or vice versa). A formatting failure, not
                       a knowledge failure.
    d_abstention       prediction is "I don't know" / disclaimer / empty
                       while the gold was present in the chunks.
    e_close_miss       F1 >= _CLOSE_MISS_F1_THRESHOLD but EM == 0 -- a
                       name/punctuation variant the strict-EM metric
                       under-credits.

"Gold in chunks" is a normalised-substring check (`normalize_answer` on
both sides), a proxy for "the answer span is present in the retrieved
context". It over-counts for very short gold strings (e.g. "yes"); yes/no
golds therefore segregate into `c_format_mismatch` whenever the
prediction polarity disagrees with the gold polarity.

Exports
-------
- _classify(record, chunk_text_norm)  -- bucket one wrong record
- main()                              -- CLI entry point

Dependencies / Requirements
---------------------------
- src.thesis_evaluations.benchmark_datasets   -- normalize_answer

Usage (single line; -X utf8 on Windows / PowerShell):
    python -X utf8 -m src.thesis_evaluations.verifier_failure_taxonomy --sweep <variant>.jsonl --cache <cache>.jsonl

Inputs
------
- --sweep   a per-variant JSONL produced by verifier_only_ablation.py
- --cache   the retrieval cache JSONL produced by verifier_cache_build.py
            (must be the same cache the sweep ran on; join key is
            question_id)

Last reviewed: 2026-06-01 (audit pass, project version 5.5)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.thesis_evaluations.benchmark_datasets import normalize_answer  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# Why: diagnostic threshold for the `e_close_miss` bucket. Intentionally
# below the headline Soft-EM threshold (benchmark.answer_f1_threshold,
# default 0.6) so that name/punctuation variants the strict-EM metric
# under-credits show up here even when they are NOT yet counted as
# Soft-EM correct. Bucket label: "close to right by token overlap, still
# strict-EM wrong".
_CLOSE_MISS_F1_THRESHOLD = 0.5

# Why: print-truncation caps so a per-bucket example takes one line.
_EXAMPLE_QUESTION_TRUNC = 70
_EXAMPLE_ANSWER_TRUNC = 40

# Why: rule width for the printed report header / footer.
_BAR_WIDTH = 64

# Why: default number of example questions printed per bucket.
_DEFAULT_EXAMPLES = 3

# Disclaimer / abstention markers (lowercased substring match on the answer).
_ABSTENTION_MARKERS = (
    "i don't know", "i cannot", "i can not", "cannot determine",
    "not enough information", "insufficient", "no answer", "unknown",
    "the context does not", "the context doesn't",
)
_YESNO = {"yes", "no"}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _gold_in_context(gold: str, chunks_text: str) -> bool:
    """Normalised-substring proxy for 'the gold answer is in the chunks'."""
    g = normalize_answer(gold)
    if not g:
        return False
    return g in chunks_text


def _classify(rec: Dict[str, Any], chunk_text_norm: str) -> str:
    pred = rec.get("predicted_answer", "") or ""
    gold = rec.get("gold_answer", "") or ""
    f1 = float(rec.get("f1_score", 0.0))
    pred_l = pred.strip().lower()
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    gold_present = _gold_in_context(gold, chunk_text_norm)

    if not gold_present:
        return "a_retrieval_miss"

    # gold IS present in the chunks -> the verifier is responsible from here.
    if (not pred_l) or any(m in pred_l for m in _ABSTENTION_MARKERS):
        return "d_abstention"

    if f1 >= _CLOSE_MISS_F1_THRESHOLD:
        return "e_close_miss"

    gold_is_yesno = gold_norm in _YESNO
    pred_is_yesno = pred_norm in _YESNO
    if gold_is_yesno != pred_is_yesno:
        return "c_format_mismatch"

    return "b_grounded_halluc"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bucket wrong verifier answers by failure mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sweep", "-s", required=True,
                        help="A variant JSONL from the verifier-only sweep.")
    parser.add_argument("--cache", "-c", required=True,
                        help="The retrieval cache JSONL the sweep ran on.")
    parser.add_argument("--examples", "-e", type=int, default=_DEFAULT_EXAMPLES,
                        help=f"Examples printed per bucket (default: "
                             f"{_DEFAULT_EXAMPLES}).")
    args = parser.parse_args()

    sweep = _load_jsonl(Path(args.sweep))
    cache = _load_jsonl(Path(args.cache))
    cache_by_id: Dict[str, Dict[str, Any]] = {c["question_id"]: c for c in cache}

    buckets: Counter = Counter()
    examples: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    n_total = len(sweep)
    n_correct = 0
    # Why: a sweep question_id not present in the cache means the join
    # failed (different cache + sweep, or a corrupted file). Such records
    # were silently bucketed as `a_retrieval_miss` previously; now counted
    # separately and reported, so misattribution is impossible.
    n_cache_miss = 0

    for rec in sweep:
        if rec.get("exact_match"):
            n_correct += 1
            continue
        qid = rec.get("question_id", "")
        crec = cache_by_id.get(qid)
        if crec is None:
            n_cache_miss += 1
            continue
        chunk_text_norm = normalize_answer(
            " ".join(ch.get("text", "") for ch in (crec.get("retrieved_chunks") or []))
        )
        b = _classify(rec, chunk_text_norm)
        buckets[b] += 1
        if len(examples[b]) < args.examples:
            examples[b].append((
                rec.get("question", "")[:_EXAMPLE_QUESTION_TRUNC],
                rec.get("gold_answer", "")[:_EXAMPLE_ANSWER_TRUNC],
                (rec.get("predicted_answer", "") or "")[:_EXAMPLE_ANSWER_TRUNC],
            ))

    n_wrong = n_total - n_correct
    n_bucketed = n_wrong - n_cache_miss
    order = [
        "a_retrieval_miss", "b_grounded_halluc", "c_format_mismatch",
        "d_abstention", "e_close_miss",
    ]
    print("=" * _BAR_WIDTH)
    print(f"Verifier failure taxonomy  |  sweep={Path(args.sweep).name}")
    print("=" * _BAR_WIDTH)
    print(f"Total: {n_total}   Correct(EM): {n_correct}   Wrong: {n_wrong}")
    if n_cache_miss:
        logger.warning(
            "%d wrong records had no matching question_id in the cache "
            "(excluded from bucket counts). Verify the --sweep was run on "
            "the --cache passed in.", n_cache_miss,
        )
        print(f"Excluded (cache miss): {n_cache_miss}   "
              f"Bucketed: {n_bucketed}")
    print("-" * _BAR_WIDTH)
    for b in order:
        cnt = buckets.get(b, 0)
        pct = (100.0 * cnt / n_bucketed) if n_bucketed else 0.0
        print(f"  {b:<20} {cnt:>3}   ({pct:4.0f}% of bucketed)")
    print("-" * _BAR_WIDTH)
    verifier_fault = sum(buckets.get(b, 0) for b in order if b != "a_retrieval_miss")
    print(f"  retrieval fault : {buckets.get('a_retrieval_miss', 0)}")
    print(f"  verifier fault  : {verifier_fault}  <- addressable by a verification upgrade")
    print("=" * _BAR_WIDTH)
    for b in order:
        if not examples[b]:
            continue
        print(f"\n[{b}] examples:")
        for q, g, p in examples[b]:
            print(f"   Q: {q}")
            print(f"     gold={g!r}  pred={p!r}")


if __name__ == "__main__":
    main()
