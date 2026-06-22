"""
Failure-mode taxonomy across agentic-ablation rows (B-5).

Why this module exists
----------------------
`verifier_failure_taxonomy.py` buckets wrong answers for the *verifier-only*
sweep, joining a sweep JSONL to a separate retrieval-cache JSONL for the
"is the gold in the retrieved context?" signal. The agentic-ablation rows
(row1_llm_only … row5_self_correct) use a different, *richer* schema: each
per-question record already carries a harness-computed gold-presence flag
(`all_gold_retrieved` / `gold_in_final_context`, derived from gold
supporting-fact titles) — a more reliable signal than the normalised-substring
proxy, and no separate cache join is needed.

This script applies the *same five-bucket taxonomy* to those rows so the
failure distribution can be compared **across** ablation configurations. The
examiner's question is the motivation: Table 17 in the paper is computed only
for the agentic (+Verifier) row, which leaves open whether the agent actually
*changes the shape* of the failures versus plain RAG. Running the identical
bucketer on `row2_rag_no_agent.jsonl` and `row4_verifier.jsonl` answers that
directly — pure re-analysis of existing logs, no new model runs.

Buckets (identical semantics to verifier_failure_taxonomy.py)
-------------------------------------------------------------
    a_retrieval_miss   gold supporting facts NOT all retrieved
                       (retrieval's fault; unreachable by any answer-checker).
    b_grounded_halluc  gold retrieved; prediction is a different, low-overlap
                       (F1 < close-miss threshold) non-abstention answer.
    c_format_mismatch  gold retrieved; yes/no-vs-name polarity disagreement.
    d_abstention       gold retrieved; prediction is a disclaimer / empty.
    e_close_miss       gold retrieved; F1 >= close-miss threshold but EM == 0.

Gold-presence signal
--------------------
Uses `all_gold_retrieved` when present (the strict, HotpotQA-joint flag: every
gold supporting-fact title retrieved). Falls back to `gold_in_final_context`,
then to a normalised-substring check of `gold_answer` against the concatenated
`retrieved_titles` — so the script still runs on a row that lacks the flags,
with a logged note about which signal was used.

Exports
-------
- classify_row(record)                       -- bucket one wrong record
- bucket_file(path)                           -- {bucket: count} for one row
- build_comparison(paths)                     -- cross-row table (md + dict)
- main()                                      -- CLI

Last reviewed: 2026-06-09.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.thesis_evaluations.benchmark_datasets import normalize_answer  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Same diagnostic close-miss threshold as verifier_failure_taxonomy.py, kept in
# sync deliberately so the two taxonomies are directly comparable.
_CLOSE_MISS_F1_THRESHOLD = 0.5

_ABSTENTION_MARKERS = (
    "i don't know", "i cannot", "i can not", "cannot determine",
    "not enough information", "insufficient", "no answer", "unknown",
    "the context does not", "the context doesn't",
)
_YESNO = {"yes", "no"}

_BUCKET_ORDER = (
    "a_retrieval_miss", "b_grounded_halluc", "c_format_mismatch",
    "d_abstention", "e_close_miss",
)

# B-6 sub-buckets of b_grounded_halluc (opt-in via --split-halluc).
_HALLUC_SUBBUCKETS = ("b1_wrong_entity_in_titles", "b2_freegen_or_other")


def halluc_subbucket(rec: Dict[str, Any]) -> str:
    """Split a grounded hallucination by whether the wrong answer is traceable
    to a *retrieved* entity (B-6).

    `b1_wrong_entity_in_titles` — the prediction (normalised) appears inside one
    of the row's `retrieved_titles`: the model picked the wrong *retrieved*
    document/entity (a bridge / wrong-hop error a better reranker targets).
    `b2_freegen_or_other` — matches no retrieved title: a free-generation /
    chunk-body-copy grounding error.

    Title-level proxy: the ablation rows store `retrieved_titles`, not chunk
    bodies, so b2 over-counts answers copied from a body whose title differs.
    """
    pred_norm = normalize_answer(rec.get("predicted_answer", "") or "")
    if not pred_norm:
        return "b2_freegen_or_other"
    for t in rec.get("retrieved_titles", []) or []:
        if pred_norm in normalize_answer(t or ""):
            return "b1_wrong_entity_in_titles"
    return "b2_freegen_or_other"

# Canonical ablation rows in build order → display label.
_ROW_FILES: Tuple[Tuple[str, str], ...] = (
    ("row1_llm_only.jsonl",     "LLM-only"),
    ("row2_rag_no_agent.jsonl", "RAG (no agent)"),
    ("row3_planner.jsonl",      "+Planner"),
    ("row4_verifier.jsonl",     "+Verifier"),
    ("row5_self_correct.jsonl", "+SelfCorrect"),
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _gold_present(rec: Dict[str, Any]) -> Tuple[bool, str]:
    """Best available 'gold reached the model' signal + which one was used.

    Preference order (strongest first):
      1. all_gold_retrieved   (strict: every gold supporting-fact title found)
      2. gold_in_final_context (gold in the LLM-visible window)
      3. substring of gold_answer in concatenated retrieved_titles (fallback)
    """
    if "all_gold_retrieved" in rec and rec["all_gold_retrieved"] is not None:
        return bool(rec["all_gold_retrieved"]), "all_gold_retrieved"
    if "gold_in_final_context" in rec and rec["gold_in_final_context"] is not None:
        return bool(rec["gold_in_final_context"]), "gold_in_final_context"
    gold = normalize_answer(rec.get("gold_answer", "") or "")
    titles = normalize_answer(" ".join(rec.get("retrieved_titles", []) or []))
    return (bool(gold) and gold in titles), "retrieved_titles_substring"


def classify_row(rec: Dict[str, Any]) -> str:
    """Bucket one WRONG ablation record (caller guarantees exact_match False)."""
    gold_present, _ = _gold_present(rec)
    if not gold_present:
        return "a_retrieval_miss"

    pred = (rec.get("predicted_answer", "") or "")
    pred_l = pred.strip().lower()
    if (not pred_l) or any(m in pred_l for m in _ABSTENTION_MARKERS):
        return "d_abstention"

    f1 = float(rec.get("f1_score", 0.0) or 0.0)
    if f1 >= _CLOSE_MISS_F1_THRESHOLD:
        return "e_close_miss"

    gold_norm = normalize_answer(rec.get("gold_answer", "") or "")
    pred_norm = normalize_answer(pred)
    if (gold_norm in _YESNO) != (pred_norm in _YESNO):
        return "c_format_mismatch"

    return "b_grounded_halluc"


def bucket_file(path: Path, split_halluc: bool = False) -> Dict[str, Any]:
    """Bucket every wrong record in one ablation-row JSONL.

    When ``split_halluc`` is set, also sub-splits the ``b_grounded_halluc``
    bucket into wrong-retrieved-entity vs free-generation (B-6).
    """
    recs = _load_jsonl(path)
    n_total = len(recs)
    n_correct = sum(1 for r in recs if r.get("exact_match"))
    buckets: Counter = Counter()
    signals: Counter = Counter()
    halluc_split: Counter = Counter()
    for r in recs:
        if r.get("exact_match"):
            continue
        b = classify_row(r)
        buckets[b] += 1
        _, sig = _gold_present(r)
        signals[sig] += 1
        if split_halluc and b == "b_grounded_halluc":
            halluc_split[halluc_subbucket(r)] += 1
    n_wrong = n_total - n_correct
    out = {
        "path": str(path),
        "n_total": n_total,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "buckets": dict(buckets),
        "gold_signal_used": dict(signals),
        "retrieval_fault": buckets.get("a_retrieval_miss", 0),
        "answer_fault": sum(buckets.get(b, 0) for b in _BUCKET_ORDER
                            if b != "a_retrieval_miss"),
    }
    if split_halluc:
        out["halluc_split"] = dict(halluc_split)
    return out


def build_comparison(
    inputs: Sequence[Tuple[Path, str]],
    split_halluc: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Cross-row failure-distribution comparison (Markdown + machine dict)."""
    rows: Dict[str, Dict[str, Any]] = {}
    for path, label in inputs:
        if not path.exists():
            continue
        rows[label] = bucket_file(path, split_halluc=split_halluc)

    # Markdown: one column per row, % of wrong per bucket.
    md: List[str] = [
        "# Failure-mode taxonomy across ablation rows",
        "",
        "Each wrong answer (`exact_match == False`) bucketed by the same "
        "five-mode taxonomy used for the verifier row, applied to every "
        "ablation configuration so the *shift* in failure shape is visible. "
        "Percentages are of that row's wrong answers. Pure re-analysis of "
        "existing logs (no new model runs).",
        "",
    ]
    labels = list(rows.keys())
    if not labels:
        return "_(no ablation rows found)_", {"rows": {}}

    header = ["Bucket"] + [f"{lab}" for lab in labels]
    md.append("| " + " | ".join(header) + " |")
    md.append("|" + "|".join(["---"] * len(header)) + "|")

    def pct(label: str, bucket: str) -> str:
        r = rows[label]
        nw = r["n_wrong"]
        c = r["buckets"].get(bucket, 0)
        return f"{c} ({100*c/nw:.0f}%)" if nw else "—"

    def sub_pct(label: str, sub: str) -> str:
        r = rows[label]
        hs = r.get("halluc_split", {})
        b = r["buckets"].get("b_grounded_halluc", 0)
        c = hs.get(sub, 0)
        return f"{c} ({100*c/b:.0f}% of b)" if b else "—"

    for b in _BUCKET_ORDER:
        md.append("| " + b + " | " + " | ".join(pct(l, b) for l in labels) + " |")
        if split_halluc and b == "b_grounded_halluc" and any(
                "halluc_split" in rows[l] for l in labels):
            for sub in _HALLUC_SUBBUCKETS:
                md.append("| &nbsp;&nbsp;↳ " + sub + " | "
                          + " | ".join(sub_pct(l, sub) for l in labels) + " |")
    # Fault split rows.
    md.append("| **retrieval fault** | "
              + " | ".join(
                  f"{rows[l]['retrieval_fault']} "
                  f"({100*rows[l]['retrieval_fault']/rows[l]['n_wrong']:.0f}%)"
                  if rows[l]['n_wrong'] else "—" for l in labels) + " |")
    md.append("| **answer fault** | "
              + " | ".join(
                  f"{rows[l]['answer_fault']} "
                  f"({100*rows[l]['answer_fault']/rows[l]['n_wrong']:.0f}%)"
                  if rows[l]['n_wrong'] else "—" for l in labels) + " |")
    md.append("| _n wrong_ | "
              + " | ".join(str(rows[l]["n_wrong"]) for l in labels) + " |")
    md.append("")

    # A short, automatic read of the RAG → +Verifier shift if both present.
    if "RAG (no agent)" in rows and "+Verifier" in rows:
        rag, ver = rows["RAG (no agent)"], rows["+Verifier"]
        def share(r, b):
            return 100 * r["buckets"].get(b, 0) / r["n_wrong"] if r["n_wrong"] else 0.0
        md.append("**RAG → +Verifier shift (does the agent change the failure "
                  "shape?).**")
        md.append("")
        for b in _BUCKET_ORDER:
            d = share(ver, b) - share(rag, b)
            md.append(f"- `{b}`: {share(rag,b):.0f}% → {share(ver,b):.0f}% "
                      f"({d:+.0f} pp)")
        md.append("")
        md.append("Gold-presence signal used per row: "
                  + "; ".join(f"{l}={max(rows[l]['gold_signal_used'], key=rows[l]['gold_signal_used'].get)}"
                              for l in labels) + ".")

    summary = {"rows": rows}
    return "\n".join(md), summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path,
                        help="An agentic-ablation directory (rowN_*.jsonl) or a "
                             "single row JSONL.")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output dir (default: alongside input).")
    parser.add_argument("--rows", type=str, default=None,
                        help="Comma-separated subset of row labels/filenames.")
    parser.add_argument("--split-halluc", action="store_true",
                        help="B-6: sub-split b_grounded_halluc into "
                             "wrong-retrieved-entity vs free-generation.")
    args = parser.parse_args()

    if args.input.is_file():
        inputs = [(args.input, args.input.stem)]
    else:
        inputs = [(args.input / f, lab) for f, lab in _ROW_FILES
                  if (args.input / f).exists()]
        if args.rows:
            wanted = {s.strip() for s in args.rows.split(",")}
            inputs = [(p, l) for p, l in inputs if l in wanted or p.name in wanted]
    if not inputs:
        logger.error("No ablation row JSONLs found at %s", args.input)
        sys.exit(1)

    md, summary = build_comparison(inputs, split_halluc=args.split_halluc)
    out_dir = args.output or (args.input.parent if args.input.is_file()
                              else args.input)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "failure_taxonomy_by_row.md").write_text(md, encoding="utf-8")
    (out_dir / "failure_taxonomy_by_row.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(md)
    logger.info("Wrote %s", out_dir / "failure_taxonomy_by_row.md")


if __name__ == "__main__":
    main()
