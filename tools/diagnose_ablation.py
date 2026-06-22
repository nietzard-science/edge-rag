"""
Diagnostic harness for agentic_ablation runs.

Reads the per-row JSONL traces produced by
`src.thesis_evaluations.agentic_ablation` and emits a precise breakdown:

- per-row metric drift (EM, SF-Recall, EM|retr.ok)
- per-question transitions between adjacent rows (gained / lost /
  unchanged)
- planner routing distribution (query_type, matched_pattern, hop_count)
- failure-mode classification for lost cases (bridge confusion,
  context dilution, format perturbation, abstention, retrieval miss)
- verifier action breakdown (did pre-validation fire? did self-
  correction iterate? did the answer change between iterations?)

Writes `diagnostic_report.md` into the supplied run directory.

Used post-hoc to understand WHY a marginal-contribution row in the
ablation produced no measurable gain (or a loss) -- the paper
discussion chapter's evidence base.

Exports
-------
- ROW_ORDER                       -- canonical agentic_ablation row order
- load_row(run_dir, name)         -- read one row's JSONL by question_id
- summarise_row(name, recs)       -- per-row headline metrics
- transition_table(prev, curr)    -- per-question gained/lost/unchanged
- classify_failure(rec, baseline) -- bucket a wrong-answer record
- planner_routing_breakdown(recs) -- query_type / matched_pattern / hop_count
- render_markdown(report)         -- assemble the report text
- main(run_dir)                   -- CLI entry point

Dependencies / Requirements
---------------------------
- per-row JSONL files produced by agentic_ablation.py
  (row1_llm_only.jsonl ... row5_self_correct.jsonl)
- no live LLM / store access (pure post-hoc analysis of JSONL traces)

Usage (single line; -X utf8 required on Windows / PowerShell):
    python -X utf8 diagnose_ablation.py evaluation_results/agentic_ablation_hotpotqa_<ts>

Last reviewed: 2026-05-30 (audit pass, project version 5.4)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

ROW_ORDER = [
    "row1_llm_only",
    "row2_rag_no_agent",
    "row3_planner",
    "row4_verifier",
    "row5_self_correct",
]

# Why: closed set of yes/no answer phrasings the model emits when it
# defaults to a boolean for an open-ended question. Used by the
# `format_yn` failure-mode classifier; surface as a constant so a new
# phrasing variant (e.g. "Yes, indeed.") is one edit, not a hunt.
_YESNO_ANSWER_VARIANTS = frozenset({
    "yes", "no", "yes.", "no.",
    "yes, they are.", "no, they are not.",
    "yes, they were.", "no, they were not.",
})

# Why: lower-cased substring markers used to detect an abstention answer
# ("I don't know", "cannot determine", ...). Mirrors the same constant in
# `verifier_failure_taxonomy.py`; the two are intentionally separate
# leaf-script copies rather than a shared util import so each diagnostic
# can be edited in isolation.
_ABSTENTION_MARKERS = (
    "i don't know", "i do not know", "cannot determine",
    "not enough information", "unknown",
)

# Why: gold/yes-no string set used to decide whether the gold answer
# itself is a boolean (only then a yes/no prediction is allowed).
_YESNO_GOLD = frozenset({"yes", "no"})

# Why: metric fields in `summarise_row` output that render as percentages
# in the Markdown table. Explicit set avoids the magnitude-based "<=1.0"
# heuristic.
_PERCENT_FIELDS = frozenset({
    "em", "sf_recall", "em_given_retr_ok",
    "pre_val_preempt_rate", "all_verified_rate",
})

# Why: print-truncation caps so each per-case row stays scannable.
_TRUNC_QUESTION = 60
_TRUNC_GOLD = 40
_TRUNC_PREDICTED = 60

# Why: rule width for the printed CLI header / footer.
_BAR_WIDTH = 70


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_row(run_dir: Path, name: str) -> Dict[str, Dict[str, Any]]:
    """Load one row's JSONL into a dict keyed by question_id."""
    path = run_dir / f"{name}.jsonl"
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["question_id"]] = rec
    return out


# ---------------------------------------------------------------------------
# Per-record failure classification
# ---------------------------------------------------------------------------

def classify_failure(
    rec: Dict[str, Any],
    baseline_rec: Optional[Dict[str, Any]] = None,
) -> str:
    """Pick the dominant failure category for a wrong-answer record.

    Categories (mutually exclusive, evaluated top-down):
        retrieval_miss      gold paragraph not retrieved at all
        gold_cut_by_cap     retrieved but gold not in the final LLM context
        llm_error           LLM timeout / API error
        format_yn           open-ended question received yes/no answer
        format_abstention   "I don't know" / disclaimer answer
        bridge_confusion    answer is a substring of the question
                            (heuristic for "picked the bridge entity")
        context_dilution    baseline got it right; this row's added
                            decomposition / context changed the LLM's pick
        unclassified        none of the above
    """
    pred = (rec.get("predicted_answer") or "").strip().lower()
    gold = (rec.get("gold_answer") or "").strip().lower()
    if rec.get("llm_error"):
        return "llm_error"
    if not rec.get("all_gold_retrieved"):
        return "retrieval_miss"
    # Why: default True so records from older runs (before
    # `gold_in_final_context` was emitted) silently bypass the delivery
    # check instead of being misattributed to `gold_cut_by_cap`.
    if not rec.get("gold_in_final_context", True):
        return "gold_cut_by_cap"
    if pred in _YESNO_ANSWER_VARIANTS and gold not in _YESNO_GOLD:
        return "format_yn"
    if any(m in pred for m in _ABSTENTION_MARKERS):
        return "format_abstention"
    # Why:    bridge-confusion proxy. When the planner decomposes a
    #         multi-hop query, the model can answer with the BRIDGE entity
    #         (intermediate hop) instead of the final answer; that bridge
    #         entity is typically a substring of the original question.
    # What:   `pred` is a non-empty lowercased substring of the question
    #         text AND differs from gold -> bucket as bridge_confusion.
    # Misses: short predictions that happen to appear in the question by
    #         coincidence (e.g. "year"); legitimate final answers that
    #         also appear in the question (rare but possible);
    #         predictions that are a paraphrase of the bridge entity
    #         rather than a verbatim substring.
    qtxt = (rec.get("question") or "").lower()
    if pred and pred in qtxt and pred != gold:
        return "bridge_confusion"
    # If baseline got it right, classify as context_dilution: the added
    # decomposition or context in the current row changed the LLM's pick
    # despite gold being available in both.
    if baseline_rec is not None and baseline_rec.get("exact_match"):
        return "context_dilution"
    return "unclassified"


# ---------------------------------------------------------------------------
# Per-question transitions between adjacent rows
# ---------------------------------------------------------------------------

def transition_table(
    prev: Dict[str, Dict[str, Any]],
    curr: Dict[str, Dict[str, Any]],
) -> Dict[str, List[str]]:
    """Categorise per-question transitions: gained / lost / stayed-*.

    Returns a dict with four lists of question_ids, mutually exclusive.
    """
    out: Dict[str, List[str]] = {
        "stayed_correct": [],
        "stayed_wrong": [],
        "gained": [],
        "lost": [],
    }
    qids = set(prev) & set(curr)
    for q in qids:
        a = prev[q].get("exact_match")
        b = curr[q].get("exact_match")
        if a and b:
            out["stayed_correct"].append(q)
        elif not a and not b:
            out["stayed_wrong"].append(q)
        elif not a and b:
            out["gained"].append(q)
        else:
            out["lost"].append(q)
    return out


# ---------------------------------------------------------------------------
# Per-row summarisation
# ---------------------------------------------------------------------------

def safe_mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarise_row(name: str, recs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not recs:
        return {"row": name, "n": 0}
    n = len(recs)
    n_retr_ok = sum(1 for r in recs.values() if r.get("all_gold_retrieved"))
    return {
        "row": name,
        "n": n,
        "em": sum(1 for r in recs.values() if r.get("exact_match")) / n,
        "sf_recall": n_retr_ok / n,
        "em_given_retr_ok": (
            sum(
                1 for r in recs.values()
                if r.get("all_gold_retrieved") and r.get("exact_match")
            ) / max(1, n_retr_ok)
        ),
        "avg_retrieval_count": safe_mean(
            [r.get("retrieval_count", 0) for r in recs.values()]
        ),
        "max_retrieval_count": max(
            (r.get("retrieval_count", 0) for r in recs.values()), default=0
        ),
        "avg_verifier_iters": safe_mean(
            [r.get("verifier_iterations", 0) for r in recs.values()]
        ),
        "max_verifier_iters": max(
            (r.get("verifier_iterations", 0) for r in recs.values()), default=0
        ),
        "pre_val_preempt_rate": sum(
            1 for r in recs.values() if r.get("classifier_preempt") is not None
        ) / n,
        "all_verified_rate": sum(
            1 for r in recs.values() if r.get("all_verified")
        ) / n,
    }


def planner_routing_breakdown(recs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    qtype = Counter(r.get("planner_query_type", "unknown") for r in recs.values())
    pattern = Counter((r.get("matched_pattern") or "none") for r in recs.values())
    hop = Counter(r.get("hop_count", 0) for r in recs.values())
    return {
        "query_type": dict(qtype),
        "matched_pattern": dict(pattern),
        "hop_count": dict(hop),
    }


# ---------------------------------------------------------------------------
# Cross-row analysis helpers (factored out of main)
# ---------------------------------------------------------------------------

def _classify_lost_failures(
    prev: Dict[str, Dict[str, Any]],
    curr: Dict[str, Dict[str, Any]],
    lost_qids: List[str],
) -> List[Dict[str, Any]]:
    """Bucket each LOST question_id with its failure category and the
    fields needed for the per-case detail table."""
    out: List[Dict[str, Any]] = []
    for q in lost_qids:
        cat = classify_failure(curr[q], baseline_rec=prev.get(q))
        out.append({
            "qid": q,
            "category": cat,
            "question": curr[q].get("question"),
            "gold": curr[q].get("gold_answer"),
            "predicted": curr[q].get("predicted_answer"),
        })
    return out


def _compute_self_correction_flips(
    r4: Dict[str, Dict[str, Any]],
    r5: Dict[str, Dict[str, Any]],
) -> Dict[str, int]:
    """Count answer-text changes between row4 and row5, split by their
    correctness effect (fixed / broke / churn)."""
    common = set(r4) & set(r5)
    answer_changed = 0
    fixed = 0
    broke = 0
    churn = 0
    for q in common:
        p4 = (r4[q].get("predicted_answer") or "").strip().lower()
        p5 = (r5[q].get("predicted_answer") or "").strip().lower()
        if p4 == p5:
            continue
        answer_changed += 1
        a4 = r4[q].get("exact_match")
        a5 = r5[q].get("exact_match")
        if not a4 and a5:
            fixed += 1
        elif a4 and not a5:
            broke += 1
        else:
            churn += 1
    return {
        "answer_changed": answer_changed,
        "fixed": fixed,
        "broke": broke,
        "churn": churn,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def render_markdown(report: Dict[str, Any]) -> str:
    lines = ["# Agentic-Ablation Diagnostic Report", ""]
    lines.append(f"Source: `{report['run_dir']}`")
    lines.append("")

    # 1. Per-row summary
    lines.append("## 1. Per-row summary")
    lines.append("")
    cols = ["row", "n", "em", "sf_recall", "em_given_retr_ok",
            "avg_retrieval_count", "max_retrieval_count",
            "avg_verifier_iters", "max_verifier_iters",
            "pre_val_preempt_rate", "all_verified_rate"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for s in report["summary"]:
        cells: List[str] = []
        for k in cols:
            v = s.get(k)
            if v is None:
                cells.append("-")
            elif isinstance(v, float):
                if k in _PERCENT_FIELDS:
                    cells.append(f"{v * 100:.1f}%")
                else:
                    cells.append(f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # 2. Adjacent-row transitions
    lines.append("## 2. Per-question transitions (gained / lost / unchanged)")
    lines.append("")
    for t in report["transitions"]:
        lines.append(f"### {t['from']}  ->  {t['to']}")
        lines.append("")
        lines.append(f"- stayed correct:   **{len(t['stayed_correct'])}**")
        lines.append(f"- stayed wrong:     **{len(t['stayed_wrong'])}**")
        lines.append(f"- GAINED (wrong->right):  **{len(t['gained'])}**")
        lines.append(f"- LOST (right->wrong):    **{len(t['lost'])}**")
        net = len(t['gained']) - len(t['lost'])
        lines.append(f"- net delta:        **{net:+d}** ({net * 2:+d}pp)")
        lines.append("")

    # 3. Planner routing distribution (where Planner is enabled, i.e. row3+)
    if report.get("planner_routing"):
        lines.append("## 3. Planner routing distribution")
        lines.append("")
        for row_name, info in report["planner_routing"].items():
            lines.append(f"### {row_name}")
            lines.append("")
            lines.append(f"- query_type breakdown: `{info['query_type']}`")
            lines.append(f"- matched_pattern:      `{info['matched_pattern']}`")
            lines.append(f"- hop_count:            `{info['hop_count']}`")
            lines.append("")

    # 4. Failure-mode classification for LOST cases per transition
    lines.append("## 4. Failure modes for LOST cases (right -> wrong)")
    lines.append("")
    for t in report["transitions"]:
        if not t["lost_failures"]:
            continue
        lines.append(f"### {t['from']}  ->  {t['to']}  ({len(t['lost'])} lost)")
        lines.append("")
        cat_counts = Counter(f["category"] for f in t["lost_failures"])
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- **{cat}**: {count}")
        lines.append("")
        lines.append("#### Per-case detail")
        lines.append("")
        lines.append("| Category | Question (truncated) | Gold | Predicted |")
        lines.append("|---|---|---|---|")
        for f in t["lost_failures"]:
            q = (f["question"] or "")[:_TRUNC_QUESTION].replace("|", "\\|")
            g = (f["gold"] or "")[:_TRUNC_GOLD].replace("|", "\\|")
            p = (f["predicted"] or "")[:_TRUNC_PREDICTED].replace("|", "\\|")
            lines.append(f"| {f['category']} | {q} | {g} | {p} |")
        lines.append("")

    # 5. Verifier-action breakdown
    lines.append("## 5. Verifier action breakdown")
    lines.append("")
    lines.append("Did pre-validation actually fire? Did self-correction iterate?")
    lines.append("If pre_val_preempt_rate is 0.0 in row4, pre-validation is dormant.")
    lines.append(
        "If avg_verifier_iters is 1.0 in row5, self-correction never iterated "
        "past round 1."
    )
    lines.append("")
    for s in report["summary"]:
        if s["row"] in ("row4_verifier", "row5_self_correct"):
            lines.append(
                f"- **{s['row']}**: "
                f"pre_val_preempt={s['pre_val_preempt_rate'] * 100:.1f}% "
                f"| all_verified={s['all_verified_rate'] * 100:.1f}% "
                f"| verifier_iters avg={s['avg_verifier_iters']:.2f} "
                f"max={s['max_verifier_iters']}"
            )
    lines.append("")

    # 6. Self-correction effect (row4 -> row5)
    if report.get("self_correction_flips"):
        flips = report["self_correction_flips"]
        lines.append("## 6. Self-correction effect (row4 -> row5)")
        lines.append("")
        lines.append(f"- questions where answer changed AT ALL: **{flips['answer_changed']}**")
        lines.append(f"- of those: wrong->right (fix):  **{flips['fixed']}**")
        lines.append(f"- of those: right->wrong (break): **{flips['broke']}**")
        lines.append(f"- of those: wrong->wrong (churn): **{flips['churn']}**")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(run_dir: Path) -> None:
    rows = {name: load_row(run_dir, name) for name in ROW_ORDER}
    present_rows = [n for n in ROW_ORDER if rows[n]]
    if not present_rows:
        print(f"No row JSONLs found in {run_dir}")
        return

    summary = [summarise_row(name, rows[name]) for name in present_rows]

    transitions: List[Dict[str, Any]] = []
    for prev_name, curr_name in zip(present_rows, present_rows[1:]):
        t = transition_table(rows[prev_name], rows[curr_name])
        lost_failures = _classify_lost_failures(
            rows[prev_name], rows[curr_name], t["lost"],
        )
        transitions.append({
            "from": prev_name,
            "to": curr_name,
            **t,
            "lost_failures": lost_failures,
        })

    planner_routing = {
        n: planner_routing_breakdown(rows[n])
        for n in present_rows
        if n in {"row3_planner", "row4_verifier", "row5_self_correct"}
    }

    sc_flips: Optional[Dict[str, int]] = None
    if rows.get("row4_verifier") and rows.get("row5_self_correct"):
        sc_flips = _compute_self_correction_flips(
            rows["row4_verifier"], rows["row5_self_correct"],
        )

    report = {
        "run_dir": str(run_dir),
        "summary": summary,
        "transitions": transitions,
        "planner_routing": planner_routing,
        "self_correction_flips": sc_flips,
    }

    out_path = run_dir / "diagnostic_report.md"
    out_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Diagnostic report written: {out_path}")

    # Print key headlines to stdout
    print()
    print("=" * _BAR_WIDTH)
    print("KEY HEADLINES")
    print("=" * _BAR_WIDTH)
    for s in summary:
        print(
            f"  {s['row']:<20} n={s['n']:>3}  "
            f"EM={s.get('em', 0) * 100:>5.1f}%  "
            f"SF-R={s.get('sf_recall', 0) * 100:>5.1f}%  "
            f"avg_chunks={s.get('avg_retrieval_count', 0):>5.1f}  "
            f"max_chunks={s.get('max_retrieval_count', 0):>3}  "
            f"v_iters_avg={s.get('avg_verifier_iters', 0):.2f}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: diagnose_ablation.py <run_dir>")
        sys.exit(1)
    target = Path(sys.argv[1])
    if not target.is_dir():
        print(f"Not a directory: {target}")
        sys.exit(1)
    main(target)
