"""
Paper results aggregator -- combines the tier-1 evaluator outputs into
LaTeX tables, matplotlib figures, and a one-page overview for the paper.

Discovers the most recent quantization_sweep_*, agentic_ablation_*, and
latency_memory_* result directories (optionally scoped to one --dataset),
reads each summary.json + per-row JSONL, and emits a single bundle:
    table_quantization.tex          model x accuracy x latency x memory
    table_ablation.tex              marginal-delta per agentic component
    table_ablation_significance.tex paired-bootstrap CIs on the deltas
    table_latency.tex               per-stage timing + budget compliance
    significance_report.md          plain-text companion to the CIs
    figure_pareto.png               latency vs. EM (one point per model)
    figure_ablation_waterfall.png   cumulative EM gain per component
    figure_stage_breakdown.png      mean vs. p95 per stage
    overview.md                     single-page human-readable summary
    coverage_report.md              claim-by-claim coverage map
    README_HOW_TO_USE.md            integration instructions

This is the single source of truth for the chapter-to-script mapping:
"rerun this, paste the .tex files, done."

Exports
-------
- build_quantization_table / build_ablation_table / build_latency_table
                                -- LaTeX table builders
- build_significance_table / build_significance_report
                                -- paired-bootstrap CIs on ablation deltas
- build_overview_md             -- one-page Markdown summary
- plot_pareto / plot_ablation_waterfall / plot_stage_breakdown
                                -- matplotlib figures (lazy-imported)
- validate_coverage             -- claim-by-claim coverage check
- main()                        -- CLI entry point
- update_bundle(output, dataset) -- in-place refresh hook for tier-1 scripts

Dependencies / Requirements
---------------------------
- reads summary.json from quantization_sweep / agentic_ablation /
  latency_memory_profile output directories
- src.thesis_evaluations.bootstrap   -- paired_bootstrap_from_jsonl (lazy;
  significance table skipped if unavailable)
- matplotlib (optional, lazy)         -- figures skipped if absent
- runs without either optional dep, emitting the tables it can build

Usage (single line; -X utf8 on Windows / PowerShell):
    python -X utf8 -m src.thesis_evaluations.thesis_results_aggregator --dataset hotpotqa --latest

Last reviewed: 2026-06-01 (audit pass, project version 5.5)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Why: anchor evaluation paths on the project root so the CLI works
# regardless of cwd, matching the rest of the package.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# Why: project-anchored results roots so the CLI works regardless of cwd.
EVAL_ROOT_DEFAULT = _PROJECT_ROOT / "evaluation_results"
# Single canonical bundle location. Both the manual CLI (`main`, via
# --output) and the auto-refresh hook (`update_bundle`) now target the SAME
# directory, so there is exactly one "these are the results" folder instead
# of the two historical ones (thesis_final / thesis_bundle_latest) that drifted
# apart depending on how the aggregator was invoked.
_DEFAULT_BUNDLE_DIR = EVAL_ROOT_DEFAULT / "bundle"
_DEFAULT_LATEST_BUNDLE_DIR = EVAL_ROOT_DEFAULT / "bundle"

# Raw tier-1 run directories now live under evaluation_results/runs/ to keep
# the top level scannable. Discovery (`_latest_dir`) searches this subdir
# first, then falls back to the eval-root top level so runs produced by an
# older code path (or still in flight when the layout changed) are still found.
_RUNS_SUBDIR = "runs"


# ---------------------------------------------------------------------------
# Discovery helpers -- find most recent run of each script
# ---------------------------------------------------------------------------


def _latest_dir(root: Path, prefix: str, dataset: Optional[str] = None) -> Optional[Path]:
    """Return the most recently modified directory matching the prefix.

    When ``dataset`` is given, match ``<prefix>_<dataset>_*`` exactly so a
    stale or cross-dataset run cannot pollute the per-dataset bundle. Old
    legacy dirs of the form ``<prefix>_<ts>`` (no dataset segment) are
    excluded in dataset mode — they belong to the pre-multidataset era.

    When ``dataset`` is None, falls back to ``<prefix>_*`` (legacy behaviour,
    used by the historical timestamped-bundle mode).

    Search order: ``<root>/runs/`` (current layout) first, then ``<root>/``
    itself (legacy / in-flight runs that landed at the top level). The newest
    match across both locations wins, so the transition to the runs/ layout
    needs no manual move of an already-running evaluation.
    """
    if not root.exists():
        return None
    full_prefix = f"{prefix}_{dataset}_" if dataset else f"{prefix}_"
    candidates: List[Path] = []
    for search_root in (root / _RUNS_SUBDIR, root):
        if search_root.exists():
            candidates.extend(
                p for p in search_root.iterdir()
                if p.is_dir() and p.name.startswith(full_prefix)
            )
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_summary_json(directory: Path) -> Optional[Dict[str, Any]]:
    p = directory / "summary.json"
    if not p.exists():
        logger.warning("No summary.json in %s", directory)
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        # Either an I/O failure (file truncated mid-write) or a JSON parse
        # error (older summary schema). Either way, surface as a warning
        # and skip this directory rather than aborting the whole bundle.
        logger.error("Failed to read %s: %s", p, exc)
        return None


# ---------------------------------------------------------------------------
# LaTeX table builders
# ---------------------------------------------------------------------------

def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "--"
    return f"{v * 100:.1f}\\%" if v <= 1.0 else f"{v:.3f}"


def _fmt_num(v: Optional[float], digits: int = 0) -> str:
    if v is None:
        return "--"
    return f"{v:.{digits}f}"


def build_quantization_table(summary: Dict[str, Any]) -> str:
    rows = summary.get("rows", [])
    if not rows:
        return "% (no quantization-sweep rows)"
    lines = [
        "% Quantization sweep — model × accuracy × latency × memory.",
        "% Generated by thesis_results_aggregator.py — do not edit by hand.",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Quantization × Model-size sweep on " +
        summary.get("dataset", "HotpotQA") +
        f" ($n={summary.get('n_samples', '?')}$). "
        "EM and F1 measure final answer quality; SF-F1 isolates pipeline "
        "retrieval quality; EM$|$retr.ok\\ shows model accuracy conditioned "
        "on correct retrieval.}",
        "\\label{tab:quantization_sweep}",
        "\\begin{tabular}{l rrrrr rr}",
        "\\toprule",
        "Model & EM & F1 & SF-F1 & SF-Rec. & EM$|$retr.ok & Lat.\\ (ms) & RSS (MB) \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r.get('model','--')} & "
            f"{_fmt_pct(r.get('em'))} & "
            f"{_fmt_num(r.get('f1'), 3)} & "
            f"{_fmt_num(r.get('sf_f1'), 3)} & "
            f"{_fmt_pct(r.get('sf_recall'))} & "
            f"{_fmt_pct(r.get('em_given_retrieval_ok'))} & "
            f"{_fmt_num(r.get('avg_time_ms'), 0)} & "
            f"{_fmt_num(r.get('peak_rss_mb'), 0)} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def build_ablation_table(summary: Dict[str, Any]) -> str:
    rows = summary.get("rows", [])
    if not rows:
        return "% (no ablation rows)"
    lines = [
        "% Agentic verification ablation — marginal contribution per component.",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Agentic verification ablation on " +
        summary.get("dataset", "HotpotQA") +
        f" ($n={summary.get('n_samples', '?')}$, "
        f"model: {summary.get('model', '?')}). "
        "Each row adds one component; $\\Delta$EM gives the marginal "
        "contribution.}",
        "\\label{tab:agentic_ablation}",
        "\\begin{tabular}{l rrrr r}",
        "\\toprule",
        "Configuration & EM & F1 & SF-F1 & EM$|$retr.ok & $\\Delta$EM \\\\",
        "\\midrule",
    ]
    prev_em = None
    for r in rows:
        em = r.get("em")
        delta = "--"
        if prev_em is not None and em is not None:
            delta = f"{(em - prev_em) * 100:+.1f}pp"
        prev_em = em
        lines.append(
            f"{r.get('label','--')} & "
            f"{_fmt_pct(em)} & "
            f"{_fmt_num(r.get('f1'), 3)} & "
            f"{_fmt_num(r.get('sf_f1'), 3)} & "
            f"{_fmt_pct(r.get('em_given_retrieval_ok'))} & "
            f"{delta} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def build_modality_table(summary: Dict[str, Any]) -> str:
    """LaTeX table of the retrieval-modality ablation (dense/BM25/graph/hybrid).

    Reads a ``modality_ablation`` summary.json (per-lane IR + SF metrics) and
    renders Recall@k / nDCG@10 / MRR / SF-Recall per modality. This is the
    rank-aware evidence for the *hybrid* claim — hybrid should dominate every
    single lane. Returns a `%`-comment placeholder if no rows are present.
    """
    rows = summary.get("rows", [])
    if not rows:
        return "% (no modality-ablation rows)"
    ks = summary.get("ks", [5, 10, 20])
    # Column spec: label + one Recall@k per cutoff + nDCG@10 + MRR + SF-Recall.
    ndcg_k = 10 if 10 in ks else ks[len(ks) // 2]
    recall_cols = " ".join("r" for _ in ks)
    header_recall = " & ".join(f"R@{k}" for k in ks)
    lines = [
        "% Retrieval-modality ablation — dense / BM25 / graph / hybrid.",
        "% Generated by thesis_results_aggregator.py — do not edit by hand.",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Retrieval-modality ablation on " +
        summary.get("dataset", "HotpotQA") +
        f" ($n={summary.get('n_samples', '?')}$, retrieval-only). "
        "Title-level IR metrics over gold supporting facts isolate each "
        "modality's contribution; the hybrid row fuses all three via RRF.}",
        "\\label{tab:modality_ablation}",
        f"\\begin{{tabular}}{{l {recall_cols} r r r}}",
        "\\toprule",
        f"Modality & {header_recall} & nDCG@{ndcg_k} & MRR & SF-Rec. \\\\",
        "\\midrule",
    ]
    for r in rows:
        cells = [str(r.get("label", r.get("lane", "--")))]
        for k in ks:
            cells.append(_fmt_pct(r.get(f"recall_at_{k}")))
        cells.append(_fmt_num(r.get(f"ndcg_at_{ndcg_k}"), 3))
        cells.append(_fmt_num(r.get("mrr"), 3))
        cells.append(_fmt_pct(r.get("sf_recall_strict")))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Statistical significance — paired bootstrap CIs on the ablation deltas
# ---------------------------------------------------------------------------

# Agentic ablation row JSONL filenames, in pipeline-build order. Each
# consecutive pair (row_i, row_{i+1}) isolates one component's marginal
# contribution; the paired bootstrap tests whether that contribution is
# statistically distinguishable from zero on the shared question set.
_ABLATION_ROW_FILES: List[Tuple[str, str]] = [
    ("row1_llm_only.jsonl",      "LLM-only"),
    ("row2_rag_no_agent.jsonl",  "+Retrieval"),
    ("row3_planner.jsonl",       "+Planner"),
    ("row4_verifier.jsonl",      "+Verifier"),
    ("row5_self_correct.jsonl",  "+SelfCorrect"),
]


def build_significance_table(
    ablation_dir: Optional[Path],
    metric: str = "EM",
) -> str:
    """Build a LaTeX table of paired-bootstrap significance tests on the
    consecutive-row deltas of the agentic ablation.

    For each adjacent pair of ablation rows the table reports the marginal
    delta, its 95 % CI, the bootstrap p-value, and a significance verdict.
    A delta whose CI excludes zero is the empirical justification for the
    corresponding component's inclusion in the system; a delta whose CI
    crosses zero is reported honestly as not statistically distinguishable
    at the evaluated sample size.

    Returns a LaTeX-table string, or a `%`-comment placeholder when the
    per-row JSONL files are not present (e.g. the ablation has not been run).
    """
    if ablation_dir is None or not ablation_dir.exists():
        return "% (no agentic_ablation directory — significance table skipped)"

    # Lazy import: the aggregator must still run (skipping this table) on a
    # machine where the bootstrap module is unavailable for any reason.
    try:
        from src.thesis_evaluations.bootstrap import (
            holm_bonferroni,
            paired_bootstrap_from_jsonl,
        )
    except ImportError:
        return "% (bootstrap module unavailable — significance table skipped)"

    # Collect the row JSONL files that actually exist, in build order.
    present: List[Tuple[Path, str]] = []
    for fname, label in _ABLATION_ROW_FILES:
        p = ablation_dir / fname
        if p.exists():
            present.append((p, label))

    if len(present) < 2:
        return ("% (fewer than 2 ablation row JSONL files found in "
                f"{ablation_dir} — significance table skipped)")

    lines = [
        "% Paired-bootstrap significance of the agentic-ablation deltas.",
        f"% Metric: {metric}. 10,000 resamples, paired on the shared",
        "% question set. Generated by thesis_results_aggregator.py.",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Statistical significance of the marginal component "
        f"contributions ({metric}). Each row compares two adjacent ablation "
        "configurations on the identical question set via a paired "
        "bootstrap (10{,}000 resamples). $\\Delta$ is the point estimate of "
        "the difference; the 95\\,\\% CI and bootstrap $p$-value test the "
        "null hypothesis $\\Delta = 0$. The consecutive-row deltas form one "
        "comparison family, so $p_{\\text{Holm}}$ reports the "
        "Holm--Bonferroni family-wise-corrected $p$-value (Holm 1979); the "
        "final column is the corrected verdict at family-wise $\\alpha=0.05$.}",
        "\\label{tab:ablation_significance}",
        "\\begin{tabular}{l r c r r c}",
        "\\toprule",
        "Component & $\\Delta$" + metric +
        " & 95\\,\\% CI & $p$ & $p_{\\text{Holm}}$ & Significant \\\\",
        "\\midrule",
    ]

    # First pass: compute every consecutive-pair delta. The pairs are ONE
    # comparison family (the marginal contributions of one ablation ladder on
    # one metric), so they are Holm-corrected together — reporting only the
    # raw p-values would inflate the family-wise error rate over m pairs.
    pair_labels: List[str] = []
    results = []
    for (path_a, label_a), (path_b, label_b) in zip(present, present[1:]):
        pair_labels.append(f"{label_a} $\\rightarrow$ {label_b}")
        try:
            results.append(paired_bootstrap_from_jsonl(
                jsonl_a=path_a, jsonl_b=path_b, metric=metric,
                name_a=label_a, name_b=label_b,
            ))
        except Exception as exc:  # noqa: BLE001 — one bad pair must not abort the table
            logger.warning("Significance test %s vs %s failed: %s",
                            label_a, label_b, exc)
            results.append(None)

    # Holm correction over the pairs that succeeded; map back to display order.
    ok = [(i, r) for i, r in enumerate(results) if r is not None]
    holm_adj: Dict[int, Tuple[float, bool]] = {}
    if ok:
        corrected = holm_bonferroni([r for _, r in ok], alpha=0.05)
        # holm_bonferroni returns ascending-p order; re-key by identity.
        by_id = {id(c.result): c for c in corrected}
        for i, r in ok:
            c = by_id[id(r)]
            holm_adj[i] = (c.p_adjusted, c.significant_corrected)

    # Display deltas as percentage points for EM/SF-Recall, raw for F1.
    scale = 100.0 if metric in ("EM", "SF-Recall") else 1.0
    unit = "pp" if scale == 100.0 else ""
    digits = 1 if scale == 100.0 else 3
    for i, res in enumerate(results):
        if res is None:
            lines.append(f"{pair_labels[i]} & -- & -- & -- & -- & -- \\\\")
            continue
        p_holm, sig_holm = holm_adj[i]
        sig = "yes" if sig_holm else "no"
        lines.append(
            f"{pair_labels[i]} & "
            f"{res.delta * scale:+.{digits}f}{unit} & "
            f"[{res.delta_ci_low * scale:+.{digits}f}, "
            f"{res.delta_ci_high * scale:+.{digits}f}]{unit} & "
            f"{res.p_value:.4f} & "
            f"{p_holm:.4f} & "
            f"{sig} \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def build_significance_report(
    ablation_dir: Optional[Path],
    metrics: Tuple[str, ...] = ("EM", "F1", "SF-F1"),
) -> str:
    """Plain-text significance report across multiple metrics.

    Written alongside the LaTeX table as a human-readable companion so the
    numbers can be sanity-checked without compiling LaTeX.
    """
    if ablation_dir is None or not ablation_dir.exists():
        return "(no agentic_ablation directory — significance report skipped)"
    try:
        from src.thesis_evaluations.bootstrap import (
            holm_bonferroni,
            paired_bootstrap_from_jsonl,
        )
    except ImportError:
        return "(bootstrap module unavailable)"

    present: List[Tuple[Path, str]] = []
    for fname, label in _ABLATION_ROW_FILES:
        p = ablation_dir / fname
        if p.exists():
            present.append((p, label))
    if len(present) < 2:
        return f"(fewer than 2 ablation row JSONL files in {ablation_dir})"

    out = ["# Agentic ablation — paired-bootstrap significance", "",
           f"Source: {ablation_dir}", ""]
    for metric in metrics:
        out.append(f"## {metric}")
        out.append("")
        # All consecutive-pair deltas for this metric are ONE comparison
        # family → Holm-correct them together (matches the LaTeX table).
        results = []
        for (pa, la), (pb, lb) in zip(present, present[1:]):
            try:
                results.append(paired_bootstrap_from_jsonl(
                    jsonl_a=pa, jsonl_b=pb, metric=metric,
                    name_a=la, name_b=lb,
                ))
            except Exception as exc:  # noqa: BLE001
                out.append(f"- {la} -> {lb}: FAILED ({exc})")
                results.append(None)
        ok = [r for r in results if r is not None]
        corrected = {id(c.result): c for c in holm_bonferroni(ok, alpha=0.05)}
        for res in results:
            if res is None:
                continue
            c = corrected[id(res)]
            verdict = ("significant after Holm" if c.significant_corrected
                       else "NOT significant after Holm")
            out.append(f"- {res}  | Holm p={c.p_adjusted:.4f} ({verdict})")
        out.append("")
    out.append(
        "A component's raw contribution is 'significant' when the 95% CI on "
        "the metric difference excludes zero. Because several deltas are "
        "tested per metric, the family-wise error rate is controlled with "
        "Holm-Bonferroni (Holm 1979): the corrected verdict is the one to "
        "trust. A non-significant delta is reported honestly: it means the "
        "component's marginal effect could not be distinguished from sampling "
        "noise at the evaluated n."
    )
    return "\n".join(out)


def build_latency_table(summary: Dict[str, Any]) -> str:
    if not summary:
        return "% (no latency summary)"
    lines = [
        "% Per-stage latency + peak memory.",
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{Per-stage latency and peak memory "
        f"($n={summary.get('n_queries', '?')}$, "
        f"budget {summary.get('budget_seconds', 60):.0f}\\,s). "
        f"Within-budget rate: {_fmt_pct(summary.get('within_budget_rate'))}.}}",
        "\\label{tab:latency_profile}",
        "\\begin{tabular}{l rrrr}",
        "\\toprule",
        "Stage & Mean (ms) & Median (ms) & P95 (ms) & Max (ms) \\\\",
        "\\midrule",
    ]
    for key, name in [("planner_ms", "$S_P$ (Planner)"),
                       ("navigator_ms", "$S_N$ (Navigator)"),
                       ("verifier_ms", "$S_V$ (Verifier)"),
                       ("total_ms", "\\textbf{Total}")]:
        s = summary.get(key, {})
        lines.append(
            f"{name} & {_fmt_num(s.get('mean'), 0)} & "
            f"{_fmt_num(s.get('median'), 0)} & "
            f"{_fmt_num(s.get('p95'), 0)} & "
            f"{_fmt_num(s.get('max'), 0)} \\\\"
        )
    lines.append("\\midrule")
    s = summary.get("peak_rss_mb", {})
    lines.append(
        f"Peak RSS (MB) & {_fmt_num(s.get('mean'), 0)} & "
        f"{_fmt_num(s.get('median'), 0)} & "
        f"{_fmt_num(s.get('p95'), 0)} & "
        f"{_fmt_num(s.get('max'), 0)} \\\\"
    )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# One-page human-readable overview — the single file a reviewer scans first
# ---------------------------------------------------------------------------

def build_overview_md(
    qsweep: Optional[Dict[str, Any]],
    ablation: Optional[Dict[str, Any]],
    latency: Optional[Dict[str, Any]],
    coverage_ok: bool,
    coverage_missing: List[str],
    dataset: Optional[str],
    ts: str,
) -> str:
    """Single-page Markdown overview consolidating every metric in the bundle.

    Designed for at-a-glance scanning by the author + the reviewer. Each
    section guards against missing data so a partial bundle still renders.
    """
    def pct(v: Optional[float]) -> str:
        if v is None:
            return "-"
        return f"{v * 100:.1f}%" if isinstance(v, float) and 0 <= v <= 1 else f"{v}"
    def num(v: Optional[float], digits: int = 0) -> str:
        if v is None:
            return "-"
        try:
            return f"{float(v):.{digits}f}"
        except Exception:  # noqa: BLE001
            # Defensive type-coercion fallback: a summary row may carry a
            # string sentinel like "N/A" in unusual paths. Render as "-"
            # rather than letting overview.md fail to write.
            return "-"

    L: List[str] = []
    L.append(f"# Paper Results Overview -- {dataset or 'all datasets'}")
    L.append("")
    L.append(f"**Refreshed:** {ts}    **Coverage:** "
             f"{'ALL CLAIMS COVERED [OK]' if coverage_ok else 'INCOMPLETE [--]'}")
    if not coverage_ok and coverage_missing:
        L.append("")
        L.append(f"**Missing:** {', '.join(coverage_missing)} -- run the "
                 f"corresponding tier-1 script to fill.")
    L.append("")

    # Headline (from ablation full-pipeline row, if present)
    L.append("## Headline (full pipeline)")
    L.append("")
    if ablation and ablation.get("rows"):
        rows = ablation["rows"]
        # The most-complete row is the one with the most components enabled
        # -- use the last row of the ablation table (row5 / row4 depending
        # on what the user ran).
        headline = rows[-1]
        L.append(f"_Source: agentic_ablation row **{headline.get('label', '?')}** "
                 f"(model: {ablation.get('model', '?')}, n={ablation.get('n_samples', '?')})_")
        L.append("")
        L.append("| Metric | Value |")
        L.append("|---|---|")
        L.append(f"| EM | **{pct(headline.get('em'))}** |")
        L.append(f"| F1 | {num(headline.get('f1'), 3)} |")
        L.append(f"| SF-F1 | {num(headline.get('sf_f1'), 3)} |")
        L.append(f"| SF-Recall | {pct(headline.get('sf_recall'))} |")
        L.append(f"| EM \\| retrieval-ok | {pct(headline.get('em_given_retrieval_ok'))} <- SLM ceiling |")
    else:
        L.append("_(no ablation summary yet -- run `agentic_ablation.py`)_")
    L.append("")

    # Cross-model (quantization sweep)
    L.append("## Cross-model comparison (quantization sweep)")
    L.append("")
    if qsweep and qsweep.get("rows"):
        L.append(f"_Source: quantization_sweep, n={qsweep.get('n_samples', '?')}_")
        L.append("")
        L.append("| Model | EM | F1 | SF-F1 | SF-Recall | EM\\|retr.ok | Latency (ms) | Peak RSS (MB) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in qsweep["rows"]:
            L.append(
                f"| `{r.get('model','-')}` | "
                f"{pct(r.get('em'))} | "
                f"{num(r.get('f1'), 3)} | "
                f"{num(r.get('sf_f1'), 3)} | "
                f"{pct(r.get('sf_recall'))} | "
                f"{pct(r.get('em_given_retrieval_ok'))} | "
                f"{num(r.get('avg_time_ms'), 0)} | "
                f"{num(r.get('peak_rss_mb'), 0)} |"
            )
    else:
        L.append("_(no quantization sweep yet -- run `quantization_sweep.py`)_")
    L.append("")

    # Component contribution (ablation)
    L.append("## Component contribution (agentic ablation)")
    L.append("")
    if ablation and ablation.get("rows"):
        L.append("| Configuration | EM | d-EM | F1 | SF-F1 | EM\\|retr.ok |")
        L.append("|---|---|---|---|---|---|")
        prev_em = None
        for r in ablation["rows"]:
            em = r.get("em")
            delta = "-"
            if prev_em is not None and em is not None:
                delta = f"{(em - prev_em) * 100:+.1f} pp"
            prev_em = em
            L.append(
                f"| {r.get('label', '-')} | "
                f"{pct(em)} | "
                f"{delta} | "
                f"{num(r.get('f1'), 3)} | "
                f"{num(r.get('sf_f1'), 3)} | "
                f"{pct(r.get('em_given_retrieval_ok'))} |"
            )
    else:
        L.append("_(no ablation summary yet)_")
    L.append("")

    # Resource profile (latency / memory)
    L.append("## Resource profile (edge feasibility)")
    L.append("")
    if latency:
        L.append(f"_Source: latency_memory_profile, n={latency.get('n_queries', '?')}, "
                 f"budget {latency.get('budget_seconds', 60):.0f}s_")
        L.append("")
        L.append("| Stage | Mean (ms) | Median (ms) | P95 (ms) | Max (ms) |")
        L.append("|---|---|---|---|---|")
        for key, name in [("planner_ms", "S_P (Planner)"),
                          ("navigator_ms", "S_N (Navigator)"),
                          ("verifier_ms", "S_V (Verifier)"),
                          ("total_ms", "**Total**")]:
            s = latency.get(key, {}) or {}
            L.append(
                f"| {name} | {num(s.get('mean'), 0)} | "
                f"{num(s.get('median'), 0)} | "
                f"{num(s.get('p95'), 0)} | "
                f"{num(s.get('max'), 0)} |"
            )
        rss = latency.get("peak_rss_mb", {}) or {}
        wb = latency.get("within_budget_rate")
        L.append("")
        L.append(f"- **Within {latency.get('budget_seconds', 60):.0f}s budget**: "
                 f"{pct(wb) if wb is not None else '-'}")
        L.append(f"- **Peak RSS**: mean {num(rss.get('mean'), 0)} MB / "
                 f"max {num(rss.get('max'), 0)} MB")
    else:
        L.append("_(no latency profile yet -- run `latency_memory_profile.py`)_")
    L.append("")

    # Pointers
    L.append("## Files in this bundle")
    L.append("")
    L.append("- `table_quantization.tex` / `table_ablation.tex` / `table_latency.tex` -- LaTeX, paste into paper.")
    L.append("- `table_ablation_significance.tex` + `significance_report.md` -- paired-bootstrap CIs.")
    L.append("- `figure_pareto.png` / `figure_ablation_waterfall.png` / `figure_stage_breakdown.png` -- plots.")
    L.append("- `coverage_report.md` -- claim-by-claim coverage map.")
    L.append("- `README_HOW_TO_USE.md` -- paste-into-paper instructions.")
    L.append("")
    L.append("_Generated by `thesis_results_aggregator.py`. Re-run any tier-1 "
             "script to refresh this overview._")

    return "\n".join(L)


# ---------------------------------------------------------------------------
# Plots (matplotlib, lazy import)
# ---------------------------------------------------------------------------

# Module-level seen-flag so a matplotlib-less run logs the warning once,
# not once per plot function (three calls in main()).
_MATPLOTLIB_WARNED: bool = False


def _try_import_matplotlib():
    global _MATPLOTLIB_WARNED
    try:
        import matplotlib
        matplotlib.use("Agg")  # no display required
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        if not _MATPLOTLIB_WARNED:
            logger.warning("matplotlib not installed — figures will be skipped.")
            _MATPLOTLIB_WARNED = True
        return None


def plot_pareto(qsweep: Optional[Dict[str, Any]], output_dir: Path) -> None:
    plt = _try_import_matplotlib()
    if plt is None or not qsweep:
        return
    rows = qsweep.get("rows", [])
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        x = r.get("avg_time_ms", 0)
        y = (r.get("em") or 0) * 100
        ax.scatter([x], [y], s=90)
        ax.annotate(r.get("model", "?"), (x, y),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Avg. end-to-end latency (ms)")
    ax.set_ylabel("Exact-match (%)")
    ax.set_title("Pareto front: latency vs. accuracy across models")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    path = output_dir / "figure_pareto.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Pareto figure: %s", path)


def plot_ablation_waterfall(ablation: Optional[Dict[str, Any]],
                            output_dir: Path) -> None:
    plt = _try_import_matplotlib()
    if plt is None or not ablation:
        return
    rows = ablation.get("rows", [])
    if not rows:
        return
    labels = [r.get("label", "?") for r in rows]
    em_vals = [(r.get("em") or 0) * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(em_vals)), em_vals)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Exact-match (%)")
    ax.set_title("Agentic verification — cumulative gain per component")
    for i, v in enumerate(em_vals):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = output_dir / "figure_ablation_waterfall.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Ablation waterfall figure: %s", path)


def plot_stage_breakdown(latency: Optional[Dict[str, Any]],
                          output_dir: Path) -> None:
    plt = _try_import_matplotlib()
    if plt is None or not latency:
        return
    stages = [
        ("planner_ms", "$S_P$"),
        ("navigator_ms", "$S_N$"),
        ("verifier_ms", "$S_V$"),
    ]
    means = [latency.get(k, {}).get("mean", 0) for k, _ in stages]
    p95s = [latency.get(k, {}).get("p95", 0) for k, _ in stages]
    labels = [name for _, name in stages]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = list(range(len(labels)))
    width = 0.35
    ax.bar([i - width / 2 for i in x], means, width, label="mean")
    ax.bar([i + width / 2 for i in x], p95s, width, label="p95")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-stage latency: mean vs. p95")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = output_dir / "figure_stage_breakdown.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Stage-breakdown figure: %s", path)


# ---------------------------------------------------------------------------
# Validation — does the suite collectively answer the paper questions?
# (The canonical claim list is built inline in ``validate_coverage`` below.)
# ---------------------------------------------------------------------------

def _has_sf_f1(rows: Optional[List[Dict[str, Any]]]) -> bool:
    """Return True iff at least one row carries a non-null sf_f1 value."""
    if not rows:
        return False
    return any(r.get("sf_f1") is not None for r in rows)


def validate_coverage(qsweep: Optional[Dict],
                       ablation: Optional[Dict],
                       latency: Optional[Dict],
                       modality: Optional[Dict] = None) -> List[Tuple[str, str, bool]]:
    """For each paper claim, return (claim, supporting_script, present).

    Honest reporting: the Hybrid-RAG claim is marked fully present when a
    ``modality_ablation`` run (dense/BM25/graph/hybrid with rank-aware IR
    metrics) is loaded — that is the direct lane decomposition. If only SF-F1
    columns are available (no modality run), the claim falls back to the older
    SF-F1-proxy wording so the bundle still reports honestly.
    """
    out: List[Tuple[str, str, bool]] = []
    out.append(("Quantized Small Language Models",
                "quantization_sweep",
                qsweep is not None and bool(qsweep.get("rows"))))
    out.append(("Agentic Verification (contribution)",
                "agentic_ablation",
                ablation is not None and bool(ablation.get("rows"))))
    out.append(("Resource-Constrained Devices",
                "latency_memory_profile",
                latency is not None and bool(latency.get("total_ms"))))
    # Preferred evidence: the modality ablation's lane decomposition. Fall back
    # to the SF-F1 proxy only when no modality run is present.
    if modality is not None and bool(modality.get("rows")):
        out.append(("Hybrid Retrieval-Augmented Generation "
                    "(dense/BM25/graph/hybrid IR ablation)",
                    "modality_ablation -- Recall@k / nDCG / MRR per lane",
                    True))
    else:
        sf_present = (
            (qsweep is not None and _has_sf_f1(qsweep.get("rows")))
            or (ablation is not None and _has_sf_f1(ablation.get("rows")))
        )
        out.append(("Hybrid Retrieval-Augmented Generation (SF-F1 proxy)",
                    "SF-F1 columns present in quantization_sweep or "
                    "agentic_ablation summary; run modality_ablation for the "
                    "dense/BM25/graph/hybrid lane decomposition",
                    sf_present))
    out.append(("Reasoning Fidelity",
                "EM/F1/EM|retr.ok across all scripts",
                qsweep is not None and ablation is not None))
    return out


# ---------------------------------------------------------------------------
# Top-level glue
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--quantization-dir", type=str, default=None)
    parser.add_argument("--ablation-dir", type=str, default=None)
    parser.add_argument("--latency-dir", type=str, default=None)
    parser.add_argument("--eval-root", type=str,
                        default=str(EVAL_ROOT_DEFAULT))
    parser.add_argument("--output", "-o", type=str,
                        default=str(_DEFAULT_BUNDLE_DIR))
    parser.add_argument(
        "--latest", action="store_true",
        help="Write to `<output>` (no timestamp suffix), overwriting any "
             "prior bundle. Used by the auto-bundle hook on tier-1 scripts "
             "so the canonical bundle always reflects the latest results.",
    )
    parser.add_argument(
        "--dataset", "-d", type=str, default=None,
        help="Restrict aggregation to one dataset. When given, only run dirs "
             "matching `<script>_<dataset>_*` are considered (so multi-dataset "
             "runs do not pollute each other) and in --latest mode the output "
             "is `<output>/<dataset>/`. When omitted, behaviour is legacy: "
             "newest dir of each script regardless of dataset.",
    )
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    q_dir = Path(args.quantization_dir) if args.quantization_dir \
        else _latest_dir(eval_root, "quantization_sweep", args.dataset)
    a_dir = Path(args.ablation_dir) if args.ablation_dir \
        else _latest_dir(eval_root, "agentic_ablation", args.dataset)
    l_dir = Path(args.latency_dir) if args.latency_dir \
        else _latest_dir(eval_root, "latency_memory", args.dataset)
    # Modality ablation (dense/BM25/graph/hybrid) — discovered the same way as
    # the other tier-1 runs. Optional: absent on bundles built before P0-T3.
    m_dir = _latest_dir(eval_root, "modality_ablation", args.dataset)

    logger.info("Dataset filter         : %s", args.dataset or "(none -- legacy)")
    logger.info("Quantization sweep dir : %s", q_dir or "(missing)")
    logger.info("Agentic ablation dir   : %s", a_dir or "(missing)")
    logger.info("Latency profile dir    : %s", l_dir or "(missing)")
    logger.info("Modality ablation dir  : %s", m_dir or "(missing)")

    qsweep = _read_summary_json(q_dir) if q_dir else None
    ablation = _read_summary_json(a_dir) if a_dir else None
    latency = _read_summary_json(l_dir) if l_dir else None
    modality = _read_summary_json(m_dir) if m_dir else None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # --latest mode: overwrite a fixed canonical bundle directory (no
    # timestamp suffix). When --dataset is also given, the bundle becomes
    # `<output>/<dataset>/` so each dataset has its own subdirectory and
    # parallel datasets cannot overwrite each other. Default (legacy) mode
    # timestamps each run so historical bundles are kept.
    if args.latest:
        output_dir = Path(args.output) / args.dataset if args.dataset else Path(args.output)
    else:
        output_dir = Path(f"{args.output}_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Tables ──────────────────────────────────────────────────────────
    if qsweep:
        (output_dir / "table_quantization.tex").write_text(
            build_quantization_table(qsweep), encoding="utf-8"
        )
    if ablation:
        (output_dir / "table_ablation.tex").write_text(
            build_ablation_table(ablation), encoding="utf-8"
        )

    # Statistical-significance table + report — paired bootstrap on the
    # consecutive-row deltas of the agentic ablation. Always attempted when
    # an ablation directory was found; the builders return a `%`-comment
    # placeholder if the per-row JSONL files are absent.
    if a_dir is not None:
        sig_table = build_significance_table(a_dir, metric="EM")
        (output_dir / "table_ablation_significance.tex").write_text(
            sig_table, encoding="utf-8"
        )
        sig_report = build_significance_report(a_dir)
        (output_dir / "significance_report.md").write_text(
            sig_report, encoding="utf-8"
        )
        logger.info("Significance table + report written.")

    if latency:
        (output_dir / "table_latency.tex").write_text(
            build_latency_table(latency), encoding="utf-8"
        )

    # Retrieval-modality ablation (dense/BM25/graph/hybrid) — the rank-aware
    # IR evidence for the "hybrid" claim. Written when a modality_ablation run
    # is present; silently skipped otherwise (older bundles).
    if modality:
        (output_dir / "table_modality.tex").write_text(
            build_modality_table(modality), encoding="utf-8"
        )
        logger.info("Modality-ablation table written.")

    # ── Plots ───────────────────────────────────────────────────────────
    plot_pareto(qsweep, output_dir)
    plot_ablation_waterfall(ablation, output_dir)
    plot_stage_breakdown(latency, output_dir)

    # ── Validation ──────────────────────────────────────────────────────
    coverage = validate_coverage(qsweep, ablation, latency, modality)
    cov_lines = ["# Paper claim coverage", ""]
    cov_lines.append("| Claim | Supporting script | Present? |")
    cov_lines.append("|---|---|---|")
    all_present = True
    for claim, script, present in coverage:
        mark = "[OK]" if present else "[--]"
        cov_lines.append(f"| {claim} | {script} | {mark} |")
        if not present:
            all_present = False
    cov_lines.append("")
    cov_lines.append("**Status:** " +
                     ("All paper claims have empirical support." if all_present
                      else "**Some paper claims lack supporting data.** Run the "
                           "missing scripts before finalising the manuscript."))

    # ── Single-page overview — the file you scan first ──────────────────
    # Consolidates headline + cross-model + ablation + latency in one
    # markdown view so the author and a reviewer can see the whole picture
    # without opening every .tex / .json. Re-rendered on every aggregation,
    # so opening it always shows the current state.
    missing_sources: List[str] = []
    if not qsweep: missing_sources.append("quantization_sweep")
    if not ablation: missing_sources.append("agentic_ablation")
    if not latency: missing_sources.append("latency_memory_profile")
    overview_md = build_overview_md(
        qsweep, ablation, latency,
        coverage_ok=all_present,
        coverage_missing=missing_sources,
        dataset=args.dataset,
        ts=ts,
    )
    (output_dir / "overview.md").write_text(overview_md, encoding="utf-8")

    # How-to-use README
    readme_lines = [
        "# Paper results bundle",
        "",
        f"Generated: {ts}",
        "",
        "## Files in this directory",
        "",
        "- **`overview.md` -- single-page human-readable summary of every metric in this bundle. Read this first.**",
        "- `table_quantization.tex` -- paste into the Quantization section.",
        "- `table_ablation.tex` -- paste into the Agentic-Verification section.",
        "- `table_ablation_significance.tex` -- paste alongside the ablation table; "
        "paired-bootstrap 95% CIs + p-values on each component delta.",
        "- `significance_report.md` -- plain-text companion to the significance "
        "table (EM / F1 / SF-F1); sanity-check the numbers without LaTeX.",
        "- `table_latency.tex` -- paste into the Resource-Profile section.",
        "- `figure_pareto.png` -- latency vs. EM Pareto front.",
        "- `figure_ablation_waterfall.png` -- cumulative-gain plot per component.",
        "- `figure_stage_breakdown.png` -- per-stage mean vs. p95 timing.",
        "- `coverage_report.md` -- checks whether each paper claim has data.",
        "",
        "## How to integrate into the paper",
        "",
        "1. Copy each `.tex` file's contents verbatim into the corresponding ",
        "   chapter. They use only `tabular`, `booktabs`, and standard math; ",
        "   no special packages required.",
        "2. Copy the `.png` figures into your `figures/` directory.",
        "3. Run `\\input{table_X.tex}` or paste the contents directly.",
        "4. Cross-reference labels: `tab:quantization_sweep`, ",
        "   `tab:agentic_ablation`, `tab:latency_profile`.",
        "",
        "## Re-running",
        "",
        "If you rerun any tier-1 script, this aggregator picks up the newest",
        "output directory automatically. To pin a specific run, pass",
        "`--quantization-dir <path>` etc.",
    ]
    (output_dir / "coverage_report.md").write_text("\n".join(cov_lines), encoding="utf-8")
    (output_dir / "README_HOW_TO_USE.md").write_text("\n".join(readme_lines), encoding="utf-8")

    logger.info("=" * 70)
    logger.info("Paper bundle written to: %s", output_dir)
    logger.info("Coverage: %s",
                "ALL CLAIMS COVERED" if all_present else "INCOMPLETE -- see coverage_report.md")
    logger.info("=" * 70)


def update_bundle(
    output: Optional[str] = None,
    dataset: Optional[str] = None,
) -> Optional[Path]:
    """Refresh the canonical paper bundle in-place.

    Convenience entry-point called at the end of each tier-1 evaluation
    script's main() so the bundle at ``output`` always reflects the latest
    results. Idempotent (auto-discovers the latest of each tier-1 run dir
    under ``./evaluation_results/``) and safe to call repeatedly.

    When ``dataset`` is given, the bundle is written to ``<output>/<dataset>/``
    and only run dirs matching ``<script>_<dataset>_*`` are considered —
    parallel datasets get parallel subdirectories and cannot overwrite each
    other. When ``dataset`` is None, legacy single-bundle behaviour applies.

    Failures are caught and logged at WARNING level -- an aggregation
    failure must never break the eval that just produced the data.

    Returns:
        The bundle path on success, ``None`` on failure.
    """
    resolved_output = output if output is not None else str(_DEFAULT_LATEST_BUNDLE_DIR)
    old_argv = sys.argv
    argv = [
        "thesis_results_aggregator",
        "--output", resolved_output,
        "--latest",
    ]
    if dataset:
        argv.extend(["--dataset", dataset])
    sys.argv = argv
    try:
        main()
        return Path(resolved_output) / dataset if dataset else Path(resolved_output)
    except Exception as exc:  # noqa: BLE001
        # Aggregation failure (missing dir, schema drift, matplotlib stack
        # error, …) must NEVER break the eval that just produced the data.
        # The per-script JSONLs and summary.md are the source of truth.
        logger.warning("update_bundle: aggregation failed (%s)", exc)
        return None
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
