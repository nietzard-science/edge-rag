"""
Significance + 95% CIs for the cross-model and bit-width sweeps (roadmap #2/#9).

Why this module exists
----------------------
The cross-model sweep (paper Table 14 / findings §6) and bit-width sweep
(Table 15 / findings §6.5) report point-estimate EM/F1 per model but no
uncertainty and no significance test. The "Pareto-competitive" and "quantization
nearly free" claims are therefore stated without showing whether the EM
differences are distinguishable from noise at n=100. This script attaches, to
every model in a sweep:

  * a bootstrap 95% CI on its own EM (and F1), and
  * a *paired* bootstrap delta vs. a chosen baseline model (same question set),
    with 95% CI and two-sided p-value.

It reuses `bootstrap.py` (paired design: each model ran on the identical
questions, joined by question_id) — pure post-processing of the per-question
JSONLs the sweep already wrote, no new model runs.

The honest reading these numbers enable
---------------------------------------
At n=100 the per-model EM CIs are wide (~±10 pp), so small cross-model /
cross-precision EM gaps are typically NOT significant. That is itself the
finding: the "smallest model wins on EM" and "higher precision helps EM" claims
should be softened to "statistically indistinguishable on EM; the separation is
on latency/memory" — which is exactly the Pareto argument.

Exports
-------
- summarize_sweep(dir, baseline, labels)  -- {model: {em, ci, delta, p, ...}}
- build_markdown(summary, ...)            -- augmented table + reading note
- main()                                  -- CLI

Last reviewed: 2026-06-09.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.thesis_evaluations.bootstrap import (  # noqa: E402
    bootstrap_ci_from_jsonl,
    load_jsonl_records,
    paired_bootstrap_from_jsonl,
    _records_to_metric,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _em_pct(path: Path) -> Tuple[float, int]:
    recs = load_jsonl_records(path)
    m = _records_to_metric(recs, "EM")
    if not m:
        return 0.0, 0
    return 100.0 * sum(m.values()) / len(m), len(m)


def summarize_sweep(
    sweep_dir: Path,
    baseline_file: str,
    files: Sequence[Tuple[str, str]],
    metric: str = "EM",
) -> Dict[str, Dict]:
    """Per-model CI + paired delta-vs-baseline for one sweep directory.

    Args:
        sweep_dir:     directory of per-model JSONLs.
        baseline_file: the JSONL filename used as the paired-comparison baseline.
        files:         ordered (filename, display_label) pairs to report.
        metric:        "EM" or "F1".
    """
    base_path = sweep_dir / baseline_file
    out: Dict[str, Dict] = {}
    for fname, label in files:
        p = sweep_dir / fname
        if not p.exists():
            logger.warning("missing %s", p)
            continue
        em, n = _em_pct(p)
        ci = bootstrap_ci_from_jsonl(p, metric)
        row = {
            "label": label,
            "file": fname,
            "n": n,
            "point": ci.point_estimate,
            "ci_low": ci.ci_low,
            "ci_high": ci.ci_high,
            "is_baseline": fname == baseline_file,
        }
        if fname != baseline_file and base_path.exists():
            r = paired_bootstrap_from_jsonl(base_path, p, metric,
                                            name_a="baseline", name_b=label)
            row.update({
                "delta": r.delta,
                "delta_ci_low": r.delta_ci_low,
                "delta_ci_high": r.delta_ci_high,
                "p_value": r.p_value,
                "significant": r.significant,
            })
        out[label] = row
    return out


def build_markdown(
    summary: Dict[str, Dict],
    metric: str,
    baseline_label: str,
    title: str,
) -> str:
    """Augmented table: point + 95% CI + paired Δ vs baseline + p-value."""
    scale = 100.0
    lines = [
        f"### {title}",
        "",
        f"| Model | {metric} | 95% CI | Δ vs {baseline_label} | Δ 95% CI | p | Sig. |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, r in summary.items():
        pt = f"{r['point']*scale:.1f}%"
        ci = f"[{r['ci_low']*scale:.1f}, {r['ci_high']*scale:.1f}]"
        if r.get("is_baseline"):
            lines.append(f"| {label} | {pt} | {ci} | — (baseline) | — | — | — |")
        else:
            d = f"{r['delta']*scale:+.1f}pp"
            dci = f"[{r['delta_ci_low']*scale:+.1f}, {r['delta_ci_high']*scale:+.1f}]"
            p = f"{r['p_value']:.3f}"
            sig = "★" if r.get("significant") else "ns"
            lines.append(f"| {label} | {pt} | {ci} | {d} | {dci} | {p} | {sig} |")
    lines.append("")
    # Auto reading note: are ANY deltas significant?
    deltas = [r for r in summary.values() if not r.get("is_baseline")]
    any_sig = any(r.get("significant") for r in deltas)
    if deltas and not any_sig:
        lines.append(
            f"**Reading:** no pairwise {metric} difference vs {baseline_label} "
            f"is significant at the evaluated n (every Δ 95% CI crosses zero). "
            f"The models are statistically indistinguishable on {metric}; any "
            f"separation is on latency/memory, not accuracy. CIs are wide "
            f"(~±10 pp at n=100), so these comparisons are underpowered for "
            f"small effects — reported honestly rather than over-claimed."
        )
    elif deltas:
        sigs = [r["label"] for r in deltas if r.get("significant")]
        lines.append(
            f"**Reading:** significant {metric} difference(s) vs "
            f"{baseline_label}: {', '.join(sigs)}. Remaining comparisons cross "
            f"zero and are reported as not distinguishable at this n."
        )
    return "\n".join(lines)


# Default sweep layouts (the two runs behind §6 / §6.5).
_CROSS_MODEL = (
    ("qwen2-1.5b.jsonl", "qwen2:1.5b"),
    ("qwen2.5-3b.jsonl", "qwen2.5:3b"),
    ("phi3.jsonl",       "phi3 (~3.8B)"),
)
_BITWIDTH = (
    ("qwen2-1.5b-instruct.jsonl",      "Q4_K_M"),
    ("qwen2-1.5b-instruct-q8_0.jsonl", "Q8_0"),
    ("qwen2-1.5b-instruct-fp16.jsonl", "fp16"),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("sweep_dir", type=Path,
                        help="A quantization_sweep_* directory of per-model JSONLs.")
    parser.add_argument("--kind", choices=("cross_model", "bitwidth", "auto"),
                        default="auto",
                        help="Which default layout to use (auto: detect by "
                             "filenames present).")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Baseline JSONL filename (default: kind-specific).")
    parser.add_argument("--metric", choices=("EM", "F1"), default="EM")
    parser.add_argument("--output", "-o", type=Path, default=None)
    args = parser.parse_args()

    present = {p.name for p in args.sweep_dir.glob("*.jsonl")}
    kind = args.kind
    if kind == "auto":
        kind = "bitwidth" if any("fp16" in n or "q8_0" in n for n in present) \
            else "cross_model"
    files = _BITWIDTH if kind == "bitwidth" else _CROSS_MODEL
    baseline = args.baseline or files[0][0]
    title = ("Bit-width sweep — EM with 95% CI and paired significance vs Q4"
             if kind == "bitwidth"
             else "Cross-model sweep — EM with 95% CI and paired significance "
                  "vs qwen2:1.5b")

    summary = summarize_sweep(args.sweep_dir, baseline, files, args.metric)
    md = build_markdown(summary, args.metric, files[0][1], title)
    print(md)

    out_dir = args.output or args.sweep_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"significance_{kind}.md").write_text(md, encoding="utf-8")
    (out_dir / f"significance_{kind}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", out_dir / f"significance_{kind}.md")


if __name__ == "__main__":
    main()
