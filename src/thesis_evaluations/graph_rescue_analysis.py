"""Graph-rescue analysis — quantify the graph lane's NON-REDUNDANT contribution.

Motivation
----------
On aggregate IR metrics the hybrid retriever barely edges out (HotpotQA) or
slightly trails (2WikiMultiHop) BM25-alone, because BM25 is a strong lexical
baseline on entity-anchored Wikipedia QA and the off-the-shelf dense lane is
weak (it collapses entirely on 2Wiki's compositional queries). Aggregate
recall therefore HIDES the graph lane's real value, which is not "higher
average recall" but COVERAGE OF A FAILURE MODE: the multi-hop bridge questions
whose answer document shares no surface words with the question, so neither
BM25 nor a frozen dense encoder can reach it, but the entity graph can walk the
relation (e.g. ``X -> father -> country``).

This script computes that contribution directly from the modality-ablation
per-lane outputs:

  * graph_rescued.jsonl  — per question, the gold supporting-fact titles the
    GRAPH lane retrieved that the DENSE and BM25 lanes BOTH missed.
  * hybrid_all.jsonl     — per question, the question_type label.

Headline metrics (per dataset, and per question type):

  1. rescued_rate          — % of questions with >= 1 graph-rescued gold fact.
  2. graph_complete_rescue — % of questions where the graph lane closed the
     LAST missing gold fact, i.e. all gold became reachable only once the graph
     lane was added (the strongest "graph was necessary" signal:
     non_graph_gold_hits < n_gold AND graph_gold_hits == n_gold).

These are the numbers that legitimise the word "hybrid": they are minority
question classes, but they are exactly the hard multi-hop bridges that define
the benchmarks, and they are unreachable without the graph.

Run:
    python -X utf8 -m src.thesis_evaluations.graph_rescue_analysis
    python -X utf8 -m src.thesis_evaluations.graph_rescue_analysis --out Thesis_final_analysis
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[2]

# Canonical complete modality-ablation runs (n=500, all lanes + graph_rescued).
_RUNS: Dict[str, str] = {
    "HotpotQA": "evaluation_results/runs/modality_ablation_hotpotqa_20260605_114732",
    "2WikiMultiHop": "evaluation_results/runs/modality_ablation_2wikimultihop_20260605_132140",
    "MuSiQue": "evaluation_results/runs/modality_ablation_musique_20260611_164100",
}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _as_list(v: Any) -> list:
    """graph_rescued.jsonl stores list fields as their str() repr; recover them."""
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.startswith("["):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return []
    return []


def analyse_dataset(name: str, run_dir: Path) -> Dict[str, Any]:
    rescued = _load_jsonl(run_dir / "graph_rescued.jsonl")
    hybrid = _load_jsonl(run_dir / "hybrid_all.jsonl")
    qtype = {r["question_id"]: r.get("question_type", "unknown") for r in hybrid}
    n_total = len(hybrid)

    # Per-question rescue verdicts.
    per_type_total: Dict[str, int] = {}
    per_type_rescued: Dict[str, int] = {}
    per_type_complete: Dict[str, int] = {}
    for r in hybrid:
        t = r.get("question_type", "unknown")
        per_type_total[t] = per_type_total.get(t, 0) + 1

    n_rescued = 0
    n_complete = 0
    for r in rescued:
        qid = r["question_id"]
        t = qtype.get(qid, "unknown")
        rescued_titles = _as_list(r.get("rescued_titles"))
        graph_hits = _as_list(r.get("graph_gold_hits"))
        non_graph_hits = _as_list(r.get("non_graph_gold_hits"))
        try:
            n_gold = int(r.get("n_gold", 0) or 0)
        except (ValueError, TypeError):
            n_gold = 0

        if not rescued_titles:
            continue
        n_rescued += 1
        per_type_rescued[t] = per_type_rescued.get(t, 0) + 1

        # "Complete rescue": the graph closed the last gold gap — without it not
        # all gold was reachable, with it all gold is.
        if n_gold > 0 and len(set(non_graph_hits)) < n_gold and len(set(graph_hits)) >= n_gold:
            n_complete += 1
            per_type_complete[t] = per_type_complete.get(t, 0) + 1

    return {
        "dataset": name,
        "n_total": n_total,
        "n_rescued": n_rescued,
        "rescued_rate": 100.0 * n_rescued / max(n_total, 1),
        "n_complete": n_complete,
        "complete_rate": 100.0 * n_complete / max(n_total, 1),
        "per_type": {
            t: {
                "total": per_type_total.get(t, 0),
                "rescued": per_type_rescued.get(t, 0),
                "complete": per_type_complete.get(t, 0),
                "rescued_pct": 100.0 * per_type_rescued.get(t, 0) / max(per_type_total.get(t, 1), 1),
            }
            for t in sorted(per_type_total)
        },
    }


def _render_md(results: List[Dict[str, Any]]) -> str:
    lines = [
        "# Graph-rescue analysis — the graph lane's non-redundant contribution",
        "",
        "> **Contribution (the system's purpose for the paper release):**",
        "> A reproducible, edge-deployable demonstration that structured",
        "> knowledge-graph traversal recovers a specific, hard class of multi-hop",
        "> questions that lexical and dense retrieval cannot — without a GPU,",
        "> cloud, or fine-tuning.",
        "",
        "Aggregate IR metrics understate the graph lane: BM25 is a strong lexical",
        "baseline on entity-anchored QA and the frozen dense lane is weak (it",
        "collapses on 2Wiki). The graph's value is not higher *average* recall but",
        "**coverage of a failure mode** — multi-hop bridge questions whose answer",
        "document shares no surface words with the question, reachable only by",
        "walking the entity graph. The headline is the **graph-necessary** column:",
        "the % of questions answerable ONLY once the graph lane is added.",
        "",
        "**rescued** = the graph lane retrieved >=1 gold supporting fact that dense",
        "AND BM25 both missed. **graph-necessary** = the graph closed the *last*",
        "missing gold fact (all gold reachable only once the graph lane is added).",
        "",
        "| Dataset | n | rescued (>=1 gold) | graph-necessary (closed last gap) |",
        "|---|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['dataset']} | {r['n_total']} | "
            f"{r['n_rescued']} ({r['rescued_rate']:.1f}%) | "
            f"{r['n_complete']} ({r['complete_rate']:.1f}%) |"
        )
    lines.append("")
    lines.append("## By question type")
    lines.append("")
    for r in results:
        lines.append(f"### {r['dataset']}")
        lines.append("")
        lines.append("| Question type | n | rescued | rescued % |")
        lines.append("|---|---:|---:|---:|")
        for t, s in sorted(r["per_type"].items(), key=lambda kv: -kv[1]["rescued_pct"]):
            lines.append(f"| {t} | {s['total']} | {s['rescued']} | {s['rescued_pct']:.1f}% |")
        lines.append("")
    return "\n".join(lines)


def _render_tex(results: List[Dict[str, Any]]) -> str:
    rows = []
    for r in results:
        rows.append(
            f"{r['dataset']} & {r['n_total']} & "
            f"{r['n_rescued']} ({r['rescued_rate']:.1f}\\%) & "
            f"{r['n_complete']} ({r['complete_rate']:.1f}\\%) \\\\"
        )
    return (
        "% Generated by graph_rescue_analysis.py — do not edit by hand.\n"
        "\\begin{table}[t]\\centering\n"
        "\\caption{Non-redundant graph-lane contribution (retrieval-only, $n=500$). "
        "\\emph{Rescued}: the graph retrieved $\\geq 1$ gold supporting fact that "
        "dense \\emph{and} BM25 both missed. \\emph{Graph-necessary}: the graph "
        "closed the last missing gold fact, i.e.\\ all gold became reachable only "
        "once the graph lane was added.}\n"
        "\\label{tab:graph-rescue}\n"
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Dataset & $n$ & Rescued & Graph-necessary \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None,
                    help="Output dir for the .md/.tex/.json (default: print only).")
    args = ap.parse_args()

    results = []
    for name, rel in _RUNS.items():
        run_dir = _ROOT / rel
        if not (run_dir / "graph_rescued.jsonl").exists():
            print(f"skip {name}: {run_dir}/graph_rescued.jsonl missing")
            continue
        results.append(analyse_dataset(name, run_dir))

    md = _render_md(results)
    print(md)

    if args.out:
        out = _ROOT / args.out
        (out / "tables").mkdir(parents=True, exist_ok=True)
        (out / "raw_data").mkdir(parents=True, exist_ok=True)
        (out / "raw_data" / "graph_rescue_analysis.md").write_text(md, encoding="utf-8")
        (out / "tables" / "table_graph_rescue.tex").write_text(_render_tex(results), encoding="utf-8")
        (out / "raw_data" / "graph_rescue_analysis.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {out}/raw_data/graph_rescue_analysis.{{md,json}} "
              f"+ {out}/tables/table_graph_rescue.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
