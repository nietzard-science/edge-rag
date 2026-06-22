"""
Edge-cloud deferral economics over existing per-question JSONLs.

Why this module exists
----------------------
`trust_eval.py` establishes that the verifier's confidence is a usable
*abstention* signal (high precision on the answers it declines). That is the
descriptive half of the trust contribution. This module supplies the
*prescriptive, deployable* half that the selective-prediction prior art
(UniCR, arXiv:2509.01455; HCMA, arXiv:2410.02173) does NOT cover at the edge:
a concrete **edge-cloud deferral policy** with a **real cost curve**.

The deployment scenario
-----------------------
A privacy-preserving / offline-first edge device answers what it can with the
local quantized SLM (free: no API call, no data leaves the device) and *defers*
the questions it is least confident about to a strong cloud model (a real
$/query API call, and the question text leaves the device). The operating
question for a practitioner is therefore not "how good is abstention precision"
but:

    "If I defer the bottom X% of questions by confidence, what accuracy do I
     buy, at what $ cost per 1000 questions, and at what added latency — and
     where is the knee of that curve?"

This module answers exactly that, sweeping the deferral fraction from 0%
(pure-edge, free, the local Soft-EM) to 100% (pure-cloud, the cloud ceiling)
and reporting the accuracy/cost/latency Pareto frontier in between. The knee of
that curve is the headline figure for the "calibrated abstention for edge RAG"
framing.

Cost model (real, not abstract)
-------------------------------
Each deferred question costs one cloud API call, priced from measured token
counts:

    cost_per_q = (prompt_tokens/1e6)*cloud_price_in_per_mtok
               + (gen_tokens   /1e6)*cloud_price_out_per_mtok

Defaults are grounded in published GPT-4-class API pricing (USD per 1M tokens)
and a measured-from-the-corpus token estimate; **every number is a CLI flag**,
so the reader can re-price for any cloud model or token regime. The cost curve
is reported per 1000 questions (the unit a practitioner budgets in).

Accuracy model for the deferred slice
-------------------------------------
The local correctness of every question is known (from the JSONL). The cloud's
correctness on the deferred slice is NOT known without actually calling the
cloud, so this module exposes it as an explicit, documented assumption with
three modes (``--cloud-accuracy-mode``):

  * ``oracle``  — deferred questions are answered correctly (cloud accuracy
                  = 1.0 on the slice). Yields the *upper bound* on what
                  deferral can buy; honestly labelled as a ceiling.
  * ``fixed``   — cloud answers the deferred slice at a fixed accuracy
                  ``--cloud-accuracy`` (default 0.90, a conservative
                  GPT-4-class multi-hop QA estimate). The realistic curve.
  * ``lift``    — cloud accuracy on the slice = local-slice accuracy +
                  ``--cloud-lift`` (capped at 1.0). Models "cloud is better by
                  a margin" without assuming perfection.

The mode and parameters are written into every output so the assumption behind
a given curve is never implicit. The *honest headline* uses ``fixed`` at a
stated cloud accuracy; ``oracle`` is reported alongside as the ceiling.

Ranking signal for deferral
----------------------------
Questions are deferred worst-confidence-first. The categorical verifier label
(error > low > medium > high) is the primary key; within a tier, ties are
broken by ascending ``f1_score`` proxy if present (so the within-tier order is
deterministic and the worst answers go first). This makes the deferral fraction
a continuous knob even though the raw confidence signal is only 4-valued.

No new model calls — pure post-processing of the per-question JSONLs, stdlib
only. Mirrors the discipline of `bootstrap.py` and `trust_eval.py`.

References
----------
- Geifman & El-Yaniv (2017) NeurIPS — selective prediction / risk-coverage.
- Guo et al. (2017) ICML — calibration of modern neural nets.
- Conti et al. / UniCR (2025) arXiv:2509.01455 — calibrated refusal for RAG
  (no edge/cost dimension; this module adds it).
- Zellinger & Thomson / HCMA (2024) arXiv:2410.02173 — error-budgeted
  hierarchical abstention (no deployment cost curve).

Last reviewed: 2026-06-09.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Worst-first confidence ranking. A larger number = defer earlier.
_CONF_RANK: Dict[str, int] = {"error": 3, "low": 2, "medium": 1, "high": 0}


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

@dataclass
class CloudCostModel:
    """Per-query cloud cost from token counts and published $/1M-token prices.

    Defaults are a GPT-4-class operating point (USD/1M tokens) and a
    token-count estimate measured to be representative of a multi-hop QA
    prompt (retrieved context + question) plus a short answer. Override any
    field from the CLI to re-price for a different cloud model / regime.
    """
    price_in_per_mtok: float = 2.50      # USD / 1M prompt tokens
    price_out_per_mtok: float = 10.00    # USD / 1M generated tokens
    prompt_tokens: int = 1200            # retrieved context + question (est.)
    gen_tokens: int = 40                 # short extractive answer (est.)
    cloud_latency_ms: float = 1500.0     # added round-trip per deferred query

    def cost_per_query_usd(self) -> float:
        return (self.prompt_tokens / 1e6) * self.price_in_per_mtok + \
               (self.gen_tokens / 1e6) * self.price_out_per_mtok


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class _QRecord:
    """The minimal per-question state the deferral sweep needs."""
    qid: str
    conf: str                 # normalised {high, medium, low, error}
    correct_local: bool       # local SLM correctness (EM or Soft-EM)
    f1: float                 # tie-break proxy within a confidence tier
    local_latency_ms: float


def _normalise_conf(c: Any) -> str:
    if not c:
        return "error"
    s = str(c).strip().lower()
    return s if s in _CONF_RANK else "error"


def _load_records(path: Path, threshold: float, mode: str) -> List[_QRecord]:
    """Read a per-question JSONL into _QRecord list.

    ``mode`` selects the correctness definition: 'em' uses exact_match,
    'soft' uses f1_score >= threshold (the paper headline correctness).
    """
    out: List[_QRecord] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = r.get("question_id")
            if qid is None:
                continue
            f1 = float(r.get("f1_score", 0.0) or 0.0)
            correct = (bool(r.get("exact_match")) if mode == "em"
                       else f1 >= threshold)
            out.append(_QRecord(
                qid=str(qid),
                conf=_normalise_conf(r.get("confidence")),
                correct_local=correct,
                f1=f1,
                local_latency_ms=float(r.get("time_ms", 0.0) or 0.0),
            ))
    return out


# ---------------------------------------------------------------------------
# Deferral sweep
# ---------------------------------------------------------------------------

@dataclass
class DeferralPoint:
    """One operating point on the deferral curve."""
    defer_fraction_target: float   # the requested bottom-X fraction
    defer_fraction_actual: float   # achieved (categorical ties → granular steps)
    n_total: int
    n_deferred: int
    n_local: int
    accuracy: float                # blended local+cloud accuracy
    accuracy_local_only: float     # accuracy of the kept-local slice
    cost_usd_per_1000q: float      # $ to serve 1000 questions at this policy
    mean_latency_ms: float         # blended mean latency per question
    cloud_accuracy_used: float     # the cloud-slice accuracy assumed here


def _cloud_slice_accuracy(
    deferred: Sequence[_QRecord],
    mode: str,
    cloud_accuracy: float,
    cloud_lift: float,
) -> float:
    """Assumed cloud accuracy on the deferred slice under the chosen mode."""
    if not deferred:
        return 0.0
    if mode == "oracle":
        return 1.0
    if mode == "fixed":
        return cloud_accuracy
    if mode == "lift":
        local_acc = sum(1 for r in deferred if r.correct_local) / len(deferred)
        return min(1.0, local_acc + cloud_lift)
    raise ValueError(f"unknown cloud-accuracy mode: {mode!r}")


def _deferral_order(records: Sequence[_QRecord]) -> List[_QRecord]:
    """Worst-confidence-first, ascending-f1 tie-break (deterministic)."""
    return sorted(
        records,
        key=lambda r: (-_CONF_RANK[r.conf], r.f1, r.qid),
    )


def sweep_deferral(
    records: Sequence[_QRecord],
    cost: CloudCostModel,
    fractions: Sequence[float],
    cloud_mode: str = "fixed",
    cloud_accuracy: float = 0.90,
    cloud_lift: float = 0.30,
) -> List[DeferralPoint]:
    """Compute the accuracy / cost / latency curve over deferral fractions.

    For each target fraction f, defer the worst ``round(f*n)`` questions by
    confidence to the cloud and keep the rest local. Blended accuracy is

        (correct_local_kept + cloud_accuracy * n_deferred) / n_total

    cost is ``n_deferred`` cloud calls (everything local is free), reported per
    1000 questions, and latency blends local SLM latency (kept) with
    local+cloud round-trip (deferred).
    """
    n = len(records)
    if n == 0:
        return []
    ordered = _deferral_order(records)
    cost_per_q = cost.cost_per_query_usd()

    points: List[DeferralPoint] = []
    for f in fractions:
        k = max(0, min(n, round(f * n)))
        deferred = ordered[:k]
        kept = ordered[k:]
        cloud_acc = _cloud_slice_accuracy(deferred, cloud_mode,
                                          cloud_accuracy, cloud_lift)
        correct_kept = sum(1 for r in kept if r.correct_local)
        blended_correct = correct_kept + cloud_acc * len(deferred)
        accuracy = blended_correct / n

        local_only_acc = (correct_kept / len(kept)) if kept else float("nan")
        cost_per_1000 = cost_per_q * (len(deferred) / n) * 1000.0

        # Latency: kept pay local only; deferred pay local + cloud round-trip
        # (the device still runs the local pass before deciding to defer).
        total_latency = 0.0
        for r in kept:
            total_latency += r.local_latency_ms
        for r in deferred:
            total_latency += r.local_latency_ms + cost.cloud_latency_ms
        mean_latency = total_latency / n

        points.append(DeferralPoint(
            defer_fraction_target=f,
            defer_fraction_actual=len(deferred) / n,
            n_total=n,
            n_deferred=len(deferred),
            n_local=len(kept),
            accuracy=accuracy,
            accuracy_local_only=local_only_acc,
            cost_usd_per_1000q=cost_per_1000,
            mean_latency_ms=mean_latency,
            cloud_accuracy_used=cloud_acc,
        ))
    return points


def find_knee(points: Sequence[DeferralPoint]) -> Optional[DeferralPoint]:
    """The deferral point of maximum marginal accuracy-per-dollar drop-off.

    The 'knee' is the operating point past which additional spend buys
    sharply less accuracy. We approximate it as the point maximising
    (accuracy gain over pure-local) / (cost per 1000q), among points that
    actually defer something. This is the practitioner's recommended setting:
    the most accuracy per dollar.
    """
    if not points:
        return None
    base = points[0].accuracy  # pure-local accuracy (fraction 0)
    best: Optional[DeferralPoint] = None
    best_ratio = -1.0
    for p in points:
        if p.n_deferred == 0 or p.cost_usd_per_1000q <= 0:
            continue
        ratio = (p.accuracy - base) / p.cost_usd_per_1000q
        if ratio > best_ratio:
            best_ratio = ratio
            best = p
    return best


def cost_to_hit_targets(
    records: Sequence[_QRecord],
    cost: CloudCostModel,
    targets: Sequence[float],
    cloud_mode: str = "fixed",
    cloud_accuracy: float = 0.90,
    cloud_lift: float = 0.30,
) -> List[Dict[str, Any]]:
    """Minimum deferral (and its $ cost) to reach each target blended accuracy.

    This answers the deployment question the error-budget literature poses
    (HCMA, Zellinger & Thomson 2024): "I need accuracy >= T on answered+deferred
    traffic — what is the cheapest deferral policy that achieves it?" We scan the
    deferral fraction at single-question granularity (worst-confidence-first) and
    return the first fraction whose blended accuracy meets the target. A target
    above the cloud ceiling is reported as unreachable.
    """
    n = len(records)
    if n == 0:
        return []
    ordered = _deferral_order(records)
    cost_per_q = cost.cost_per_query_usd()
    # Precompute cumulative local-correct over the *kept* tail for each k.
    # kept = ordered[k:]; correct_kept(k) = total_correct - correct(ordered[:k]).
    total_correct = sum(1 for r in ordered if r.correct_local)

    out: List[Dict[str, Any]] = []
    for t in targets:
        hit: Optional[Dict[str, Any]] = None
        correct_prefix = 0  # local-correct among the deferred prefix ordered[:k]
        for k in range(0, n + 1):
            if k > 0 and ordered[k - 1].correct_local:
                correct_prefix += 1
            deferred = ordered[:k]
            kept_correct = total_correct - correct_prefix
            cloud_acc = _cloud_slice_accuracy(deferred, cloud_mode,
                                              cloud_accuracy, cloud_lift)
            blended = (kept_correct + cloud_acc * k) / n
            if blended >= t:
                hit = {
                    "target_accuracy": t,
                    "reachable": True,
                    "defer_fraction": k / n,
                    "n_deferred": k,
                    "achieved_accuracy": blended,
                    "cost_usd_per_1000q": cost_per_q * (k / n) * 1000.0,
                }
                break
        if hit is None:
            out.append({
                "target_accuracy": t,
                "reachable": False,
                "defer_fraction": 1.0,
                "n_deferred": n,
                "achieved_accuracy": _cloud_slice_accuracy(
                    ordered, cloud_mode, cloud_accuracy, cloud_lift),
                "cost_usd_per_1000q": cost_per_q * 1000.0,
            })
        else:
            out.append(hit)
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt_pct(x: float) -> str:
    return "—" if x != x else f"{x * 100:.1f}%"   # x!=x → NaN


def build_summary(
    rows_by_config: Dict[str, List[DeferralPoint]],
    cost: CloudCostModel,
    cloud_mode: str,
    cloud_accuracy: float,
    cloud_lift: float,
    correctness_mode: str,
    threshold: float,
    targets_by_config: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Render Markdown + machine-readable summary for the deferral sweep."""
    cost_per_q = cost.cost_per_query_usd()
    md: List[str] = [
        "# Edge-Cloud Deferral Economics",
        "",
        "Accuracy / cost / latency of deferring the lowest-confidence questions "
        "from the local edge SLM to a cloud model. Pure post-processing of the "
        "per-question logs (no new model calls).",
        "",
        "## Cost & accuracy assumptions (all overridable)",
        f"- Correctness metric: **{'Soft-EM (F1≥%.2f)' % threshold if correctness_mode=='soft' else 'EM'}**",
        f"- Cloud price: **${cost.price_in_per_mtok:.2f}/1M in**, "
        f"**${cost.price_out_per_mtok:.2f}/1M out**",
        f"- Tokens/query: **{cost.prompt_tokens} prompt + {cost.gen_tokens} gen** "
        f"→ **${cost_per_q:.5f}/deferred query** (${cost_per_q*1000:.2f}/1000 deferred)",
        f"- Cloud-slice accuracy mode: **{cloud_mode}**"
        + (f" (accuracy={cloud_accuracy:.2f})" if cloud_mode == "fixed"
           else f" (lift=+{cloud_lift:.2f})" if cloud_mode == "lift"
           else " (=1.0 ceiling)"),
        f"- Added cloud latency: **{cost.cloud_latency_ms:.0f} ms/deferred query**",
        "",
    ]
    summary: Dict[str, Any] = {
        "cost_model": asdict(cost),
        "cost_per_deferred_query_usd": cost_per_q,
        "cloud_accuracy_mode": cloud_mode,
        "cloud_accuracy": cloud_accuracy,
        "cloud_lift": cloud_lift,
        "correctness_mode": correctness_mode,
        "soft_em_threshold": threshold,
        "configs": {},
    }

    for cfg, points in rows_by_config.items():
        md.append(f"## {cfg}")
        md.append("")
        md.append("| Defer % | Accuracy | Local-only acc | $/1000q | "
                  "Mean latency (ms) | # deferred |")
        md.append("|---|---|---|---|---|---|")
        for p in points:
            md.append(
                f"| {p.defer_fraction_actual*100:.0f}% | "
                f"{_fmt_pct(p.accuracy)} | {_fmt_pct(p.accuracy_local_only)} | "
                f"${p.cost_usd_per_1000q:.2f} | {p.mean_latency_ms:.0f} | "
                f"{p.n_deferred} |"
            )
        knee = find_knee(points)
        if knee is not None:
            md.append("")
            md.append(
                f"**Knee (best accuracy-per-$):** defer "
                f"{knee.defer_fraction_actual*100:.0f}% → "
                f"accuracy {_fmt_pct(knee.accuracy)} at "
                f"${knee.cost_usd_per_1000q:.2f}/1000q "
                f"(+{(knee.accuracy - points[0].accuracy)*100:.1f}pp over pure-local "
                f"for ${knee.cost_usd_per_1000q:.2f})."
            )

        targets = (targets_by_config or {}).get(cfg)
        if targets:
            md.append("")
            md.append("**Cost to hit an accuracy target** (cheapest "
                      "worst-confidence-first deferral):")
            md.append("")
            md.append("| Target acc | Reachable | Defer % | $/1000q | Achieved |")
            md.append("|---|---|---|---|---|")
            for t in targets:
                reach = "yes" if t["reachable"] else "**no (>ceiling)**"
                md.append(
                    f"| {t['target_accuracy']*100:.0f}% | {reach} | "
                    f"{t['defer_fraction']*100:.0f}% | "
                    f"${t['cost_usd_per_1000q']:.2f} | "
                    f"{_fmt_pct(t['achieved_accuracy'])} |"
                )
        md.append("")
        summary["configs"][cfg] = {
            "points": [asdict(p) for p in points],
            "knee": asdict(knee) if knee is not None else None,
            "targets": targets,
            "pure_local_accuracy": points[0].accuracy if points else None,
            "pure_cloud_accuracy": points[-1].accuracy if points else None,
        }

    md.append("**Reading the curve.** Defer 0% = pure-edge (free, the local "
              "Soft-EM); defer 100% = pure-cloud (the cloud ceiling, full API "
              "cost). The knee is the recommended operating point: the most "
              "accuracy bought per dollar. The `oracle` mode (if run) reports "
              "the upper bound; the `fixed` mode at a stated cloud accuracy is "
              "the honest headline.")
    return "\n".join(md), summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEFAULT_FRACTIONS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0)
# The agentic-ablation row files, in build order, so a whole ablation dir can
# be swept in one call and the configs line up with the rest of the paper.
_ROW_FILES: Tuple[Tuple[str, str], ...] = (
    ("row1_llm_only.jsonl",     "LLM-only"),
    ("row2_rag_no_agent.jsonl", "RAG (no agent)"),
    ("row3_planner.jsonl",      "+Planner"),
    ("row4_verifier.jsonl",     "+Verifier"),
    ("row5_self_correct.jsonl", "+SelfCorrect"),
)


def _resolve_inputs(path: Path) -> List[Tuple[Path, str]]:
    """A single JSONL → one config; a directory → its known row files."""
    if path.is_file():
        return [(path, path.stem)]
    found: List[Tuple[Path, str]] = []
    for fname, label in _ROW_FILES:
        p = path / fname
        if p.exists():
            found.append((p, label))
    # Fall back to any *.jsonl if the canonical row files are absent.
    if not found:
        found = [(p, p.stem) for p in sorted(path.glob("*.jsonl"))]
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path,
                        help="A per-question JSONL, or an ablation directory "
                             "containing rowN_*.jsonl files.")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output dir (default: alongside input).")
    parser.add_argument("--metric", choices=("soft", "em"), default="soft",
                        help="Correctness metric (default: soft = headline).")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Soft-EM F1 threshold (default: from settings.yaml).")
    # Cloud cost model
    parser.add_argument("--price-in", type=float, default=2.50,
                        help="Cloud $/1M prompt tokens (default 2.50).")
    parser.add_argument("--price-out", type=float, default=10.00,
                        help="Cloud $/1M generated tokens (default 10.00).")
    parser.add_argument("--prompt-tokens", type=int, default=1200,
                        help="Est. prompt tokens per deferred query (default 1200).")
    parser.add_argument("--gen-tokens", type=int, default=40,
                        help="Est. generated tokens per deferred query (default 40).")
    parser.add_argument("--cloud-latency-ms", type=float, default=1500.0,
                        help="Added cloud round-trip ms per deferred query.")
    # Cloud accuracy assumption
    parser.add_argument("--cloud-accuracy-mode", choices=("fixed", "oracle", "lift"),
                        default="fixed",
                        help="How to model cloud accuracy on the deferred slice "
                             "(default fixed).")
    parser.add_argument("--cloud-accuracy", type=float, default=0.90,
                        help="Cloud accuracy for 'fixed' mode (default 0.90).")
    parser.add_argument("--cloud-lift", type=float, default=0.30,
                        help="Accuracy lift for 'lift' mode (default +0.30).")
    parser.add_argument("--fractions", type=str, default=None,
                        help="Comma-separated deferral fractions "
                             "(default: 0,0.05,0.1,0.15,0.2,0.3,0.5,1.0).")
    parser.add_argument("--targets", type=str, default="0.6,0.7,0.8,0.9",
                        help="Comma-separated accuracy targets for the "
                             "cost-to-hit-target table (default 0.6,0.7,0.8,0.9).")
    args = parser.parse_args()

    if args.threshold is not None:
        threshold = args.threshold
    else:
        try:
            from src.thesis_evaluations.benchmark_datasets import ANSWER_F1_THRESHOLD
            threshold = ANSWER_F1_THRESHOLD
        except Exception:  # noqa: BLE001 — fall back to the documented default
            threshold = 0.6

    fractions = (tuple(float(x) for x in args.fractions.split(",") if x.strip())
                 if args.fractions else _DEFAULT_FRACTIONS)

    cost = CloudCostModel(
        price_in_per_mtok=args.price_in,
        price_out_per_mtok=args.price_out,
        prompt_tokens=args.prompt_tokens,
        gen_tokens=args.gen_tokens,
        cloud_latency_ms=args.cloud_latency_ms,
    )

    inputs = _resolve_inputs(args.input)
    if not inputs:
        logger.error("No JSONL inputs found at %s", args.input)
        sys.exit(1)

    targets = tuple(float(x) for x in args.targets.split(",") if x.strip())

    rows_by_config: Dict[str, List[DeferralPoint]] = {}
    targets_by_config: Dict[str, List[Dict[str, Any]]] = {}
    for path, label in inputs:
        records = _load_records(path, threshold, args.metric)
        if not records:
            logger.warning("No records in %s — skipping", path)
            continue
        rows_by_config[label] = sweep_deferral(
            records, cost, fractions,
            cloud_mode=args.cloud_accuracy_mode,
            cloud_accuracy=args.cloud_accuracy,
            cloud_lift=args.cloud_lift,
        )
        targets_by_config[label] = cost_to_hit_targets(
            records, cost, targets,
            cloud_mode=args.cloud_accuracy_mode,
            cloud_accuracy=args.cloud_accuracy,
            cloud_lift=args.cloud_lift,
        )
        logger.info("Swept %s (n=%d)", label, len(records))

    md, summary = build_summary(
        rows_by_config, cost, args.cloud_accuracy_mode,
        args.cloud_accuracy, args.cloud_lift, args.metric, threshold,
        targets_by_config=targets_by_config,
    )

    out_dir = args.output or (args.input.parent if args.input.is_file()
                              else args.input)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "deferral_economics.md").write_text(md, encoding="utf-8")
    (out_dir / "deferral_economics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", out_dir / "deferral_economics.md")


if __name__ == "__main__":
    main()
