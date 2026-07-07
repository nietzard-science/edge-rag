"""
Trust / selective-prediction evaluation over existing per-question JSONLs.

Post-processes any JSONL produced by `agentic_ablation.py` or
`verifier_only_ablation.py` and reports the metrics that substantiate the
verifier's *trust* contribution — the dimension on which agentic verification
beats plain RAG even when EM ties (cf. TECHNICAL_ARCHITECTURE.md §11.17.6 and
§12.3 point 1). No new LLM calls; pure post-processing.

What it measures
----------------
- **Risk-coverage curve.** At each confidence threshold, drop the lower-
  confidence answers (treat as abstentions) and report accuracy on the rest.
  The accuracy-vs-coverage trade-off is the headline plot for the "trustworthy
  edge RAG" claim. Plain RAG sits at the (1.0 coverage, EM) point only — it
  cannot trade coverage for precision (Geifman & El-Yaniv 2017/2019).
- **Abstention precision.** Of the answers the system declines (abstains), what
  fraction would it have got *wrong* if forced to answer. High abstention
  precision = "the system refuses where it would hallucinate."
- **Confidence × correctness contingency.** The raw counts behind ECE / AUROC.
- **Expected Calibration Error (proxy).** Binned by categorical confidence
  using centred proxy probabilities (HIGH=0.9, MEDIUM=0.7, LOW=0.3). A coarse
  but standard summary for a 3-level confidence signal (Guo et al. 2017 ICML).
- **AUROC of confidence vs correctness.** How well the confidence ranks
  correct/wrong (1.0 = perfect ranker, 0.5 = random).

Both EM and Soft-EM (F1 >= benchmark.answer_f1_threshold) versions of every
metric are reported, because Soft-EM is the headline correctness verdict
elsewhere in the paper.

Inputs accepted (each record must have at minimum `confidence` (str) and
`exact_match` (bool) and `f1_score` (float)):
- `agentic_ablation.py`'s per-row JSONLs (row1_llm_only.jsonl, etc.)
- `verifier_only_ablation.py`'s per-variant JSONLs (no_cot.jsonl, etc.)

Exports
-------
- compute_trust_metrics(records)   -- pure function, returns a dict
- write_summary(metrics, out_dir)  -- writes summary.md / summary.json
- main()                           -- CLI

Last reviewed: 2026-06-01 (audit pass, project version 5.5).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.thesis_evaluations.benchmark_datasets import (  # noqa: E402
    ANSWER_F1_THRESHOLD,
    load_config_file,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Categorical-confidence levels, ranked. Used for both abstention thresholds
# and the proxy-probability ECE bin centres.
_LEVELS: Tuple[str, ...] = ("high", "medium", "low", "error")

# Proxy probabilities for the 3 substantive confidence levels (Guo 2017 §3).
# 'error' is excluded from ECE (no meaningful predicted probability for a
# pipeline-error sentinel).
_PROXY_PROB: Dict[str, float] = {"high": 0.9, "medium": 0.7, "low": 0.3}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _normalise_conf(c: Any) -> str:
    """Map any confidence string to one of {high, medium, low, error}."""
    if not c:
        return "error"
    s = str(c).strip().lower()
    if s in _LEVELS:
        return s
    if s in {"n/a", "na", "none", ""}:
        return "error"
    return s if s in _LEVELS else "error"


def _is_correct(rec: Dict[str, Any], threshold: float, mode: str) -> bool:
    """EM mode → exact_match; soft mode → f1_score >= threshold."""
    if mode == "em":
        return bool(rec.get("exact_match"))
    return float(rec.get("f1_score", 0.0)) >= threshold


def _risk_coverage(
    records: List[Dict[str, Any]], threshold: float,
) -> List[Dict[str, Any]]:
    """For each cumulative-abstention level, compute coverage + accuracy."""
    # Abstention rules (cumulative): abstain on {error}, {error,low},
    # {error,low,medium}, {error,low,medium,high}=abstain on everything.
    rules = [
        ("answer all",                  set()),
        ("abstain on error",            {"error"}),
        ("abstain on error+low",        {"error", "low"}),
        ("abstain on error+low+medium", {"error", "low", "medium"}),
    ]
    n = len(records)
    out: List[Dict[str, Any]] = []
    for label, drop in rules:
        kept = [r for r in records if _normalise_conf(r.get("confidence")) not in drop]
        n_kept = len(kept)
        coverage = n_kept / n if n else 0.0
        em = sum(1 for r in kept if _is_correct(r, threshold, "em")) / n_kept if n_kept else 0.0
        soft = sum(1 for r in kept if _is_correct(r, threshold, "soft")) / n_kept if n_kept else 0.0
        # Of the abstained, how many would have been wrong? (precision of refusal)
        abstained = [r for r in records if _normalise_conf(r.get("confidence")) in drop]
        if abstained:
            wrong_em   = sum(1 for r in abstained if not _is_correct(r, threshold, "em"))
            wrong_soft = sum(1 for r in abstained if not _is_correct(r, threshold, "soft"))
            abst_prec_em   = wrong_em   / len(abstained)
            abst_prec_soft = wrong_soft / len(abstained)
        else:
            abst_prec_em = abst_prec_soft = float("nan")
        out.append({
            "rule": label,
            "coverage": coverage,
            "n_answered": n_kept,
            "em_at_coverage": em,
            "soft_em_at_coverage": soft,
            "n_abstained": len(abstained),
            "abstention_precision_em": abst_prec_em,
            "abstention_precision_soft": abst_prec_soft,
        })
    return out


def _contingency(records: List[Dict[str, Any]], threshold: float) -> Dict[str, Dict[str, int]]:
    """confidence × {correct, wrong} counts, both EM and Soft-EM."""
    table = {lvl: {"correct_em": 0, "wrong_em": 0, "correct_soft": 0, "wrong_soft": 0}
             for lvl in _LEVELS}
    for r in records:
        lvl = _normalise_conf(r.get("confidence"))
        em = _is_correct(r, threshold, "em")
        soft = _is_correct(r, threshold, "soft")
        table[lvl]["correct_em"   if em   else "wrong_em"]   += 1
        table[lvl]["correct_soft" if soft else "wrong_soft"] += 1
    return table


def _ece(records: List[Dict[str, Any]], threshold: float, mode: str) -> float:
    """Expected Calibration Error using categorical-confidence proxy bins."""
    bins = {lvl: [] for lvl in _PROXY_PROB}
    for r in records:
        lvl = _normalise_conf(r.get("confidence"))
        if lvl in _PROXY_PROB:
            bins[lvl].append(1 if _is_correct(r, threshold, mode) else 0)
    n = sum(len(v) for v in bins.values())
    if n == 0:
        return float("nan")
    ece = 0.0
    for lvl, hits in bins.items():
        if not hits:
            continue
        acc = sum(hits) / len(hits)
        weight = len(hits) / n
        ece += weight * abs(acc - _PROXY_PROB[lvl])
    return ece


def _auroc(records: List[Dict[str, Any]], threshold: float, mode: str) -> float:
    """AUROC using categorical confidence as the ranker. Mann-Whitney form."""
    rank_map = {"high": 3.0, "medium": 2.0, "low": 1.0, "error": 0.0}
    pos = [rank_map[_normalise_conf(r.get("confidence"))]
           for r in records if _is_correct(r, threshold, mode)]
    neg = [rank_map[_normalise_conf(r.get("confidence"))]
           for r in records if not _is_correct(r, threshold, mode)]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for p in pos:
        for n_ in neg:
            if p > n_:
                wins += 1
            elif p == n_:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def compute_trust_metrics(
    records: List[Dict[str, Any]], soft_em_threshold: float,
) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"n": 0}
    em = sum(1 for r in records if _is_correct(r, soft_em_threshold, "em")) / n
    soft = sum(1 for r in records if _is_correct(r, soft_em_threshold, "soft")) / n
    conf_hist = Counter(_normalise_conf(r.get("confidence")) for r in records)
    return {
        "n": n,
        "em": em,
        "soft_em": soft,
        "soft_em_threshold": soft_em_threshold,
        "confidence_distribution": {lvl: conf_hist.get(lvl, 0) for lvl in _LEVELS},
        "risk_coverage": _risk_coverage(records, soft_em_threshold),
        "contingency": _contingency(records, soft_em_threshold),
        "ece_em":   _ece(records, soft_em_threshold, "em"),
        "ece_soft": _ece(records, soft_em_threshold, "soft"),
        "auroc_em":   _auroc(records, soft_em_threshold, "em"),
        "auroc_soft": _auroc(records, soft_em_threshold, "soft"),
    }


def _fmt_pct(x: float) -> str:
    return "—" if x != x else f"{x * 100:.1f}%"  # nan check (nan != nan)


def write_summary(per_file_metrics: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(per_file_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines: List[str] = ["# Trust / Selective-Prediction Evaluation", ""]
    lines.append("Risk-coverage, abstention precision, and calibration for the agentic verifier.")
    lines.append("Plain-RAG baselines (with no confidence signal) sit at the `answer all` row only —")
    lines.append("they cannot trade coverage for precision.")
    lines.append("")
    for path, m in per_file_metrics.items():
        lines.append(f"## {path}")
        if m.get("n", 0) == 0:
            lines.append("(empty)")
            continue
        lines.append(f"- n = **{m['n']}**, EM = **{_fmt_pct(m['em'])}**, "
                     f"Soft-EM (F1≥{m['soft_em_threshold']}) = **{_fmt_pct(m['soft_em'])}**")
        cd = m["confidence_distribution"]
        lines.append(f"- Confidence distribution: high {cd['high']} · medium {cd['medium']} "
                     f"· low {cd['low']} · error {cd['error']}")
        lines.append(f"- ECE  EM/Soft = {_fmt_pct(m['ece_em'])} / {_fmt_pct(m['ece_soft'])}  "
                     f"(lower = better calibrated)")
        lines.append(f"- AUROC EM/Soft = {m['auroc_em']:.3f} / {m['auroc_soft']:.3f}  "
                     f"(0.5 = random ranker)")
        lines.append("")
        lines.append("### Risk-coverage")
        lines.append("| Rule | Coverage | Acc@cov (EM) | Acc@cov (Soft) | Abstained | Abst.-precision (Soft) |")
        lines.append("|---|---|---|---|---|---|")
        for rc in m["risk_coverage"]:
            lines.append(
                f"| {rc['rule']} | {_fmt_pct(rc['coverage'])} | "
                f"{_fmt_pct(rc['em_at_coverage'])} | "
                f"{_fmt_pct(rc['soft_em_at_coverage'])} | "
                f"{rc['n_abstained']} | "
                f"{_fmt_pct(rc['abstention_precision_soft'])} |"
            )
        lines.append("")
        lines.append("### Confidence × correctness contingency")
        lines.append("| Confidence | Correct (Soft) | Wrong (Soft) | Correct (EM) | Wrong (EM) |")
        lines.append("|---|---|---|---|---|")
        for lvl in _LEVELS:
            row = m["contingency"][lvl]
            lines.append(f"| {lvl} | {row['correct_soft']} | {row['wrong_soft']} | "
                         f"{row['correct_em']} | {row['wrong_em']} |")
        lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Trust eval summary written: %s", output_dir / "summary.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trust / selective-prediction post-processor for per-question JSONLs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--jsonl", "-j", type=str, help="Single per-question JSONL.")
    g.add_argument("--dir", "-d", type=str, help="Directory: process every *.jsonl in it.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory (default: alongside the input).")
    parser.add_argument("--threshold", "-t", type=float, default=None,
                        help="Soft-EM F1 threshold (default: from settings.yaml or 0.6).")
    args = parser.parse_args()

    cfg = load_config_file()
    threshold = args.threshold if args.threshold is not None else (
        (cfg.get("benchmark", {}) or {}).get("answer_f1_threshold", ANSWER_F1_THRESHOLD)
    )

    jsonl_paths: List[Path] = []
    if args.jsonl:
        jsonl_paths = [Path(args.jsonl)]
        default_out = Path(args.jsonl).parent / "trust_eval"
    else:
        d = Path(args.dir)
        jsonl_paths = sorted(d.glob("*.jsonl"))
        default_out = d / "trust_eval"
    out_dir = Path(args.output) if args.output else default_out

    per_file: Dict[str, Dict[str, Any]] = {}
    for p in jsonl_paths:
        try:
            recs = _load_jsonl(p)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s (%s)", p, exc)
            continue
        per_file[p.name] = compute_trust_metrics(recs, soft_em_threshold=float(threshold))
        logger.info("%s : n=%d EM=%.1f%% Soft-EM=%.1f%%",
                    p.name, per_file[p.name].get("n", 0),
                    per_file[p.name].get("em", 0) * 100,
                    per_file[p.name].get("soft_em", 0) * 100)

    write_summary(per_file, out_dir)


if __name__ == "__main__":
    main()
