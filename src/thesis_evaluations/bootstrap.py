"""
Paired bootstrap confidence intervals for benchmark metrics.

Why this module exists
----------------------
The benchmark runner (benchmark_datasets.py) reports point estimates:
"EM = 0.44". A point estimate alone cannot answer the question a paper
reviewer will ask — "is 0.44 actually better than the baseline 0.42, or
is that sampling noise?". With n = 500 questions a 2-point EM difference
is ~10 questions flipping, which can easily be variance.

This module computes:
  1. A bootstrap confidence interval for a single config's metric.
  2. A *paired* bootstrap CI for the DIFFERENCE between two configs
     evaluated on the same question set, plus a two-sided p-value for
     the null hypothesis "the difference is zero".

Method
------
Bootstrap resampling (Efron 1979) makes no distributional assumption:
the sampling distribution of the metric is estimated by repeatedly
resampling the per-question records with replacement and recomputing
the metric on each resample.

The PAIRED variant is the correct design here because every evaluation
configuration is run on the *identical* question set. For each bootstrap
resample we draw one set of question indices and apply it to BOTH
configs, then take the difference. This cancels per-question difficulty
out of the variance of the difference and yields tighter, more honest
intervals than resampling each config independently.

References
----------
- Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife."
  The Annals of Statistics 7(1).
- Koehn, P. (2004). "Statistical Significance Tests for Machine
  Translation Evaluation." EMNLP 2004. (Introduced bootstrap resampling
  for paired system comparison in NLP; the standard reference for
  benchmark-difference significance.)

Scope
-----
Pure post-processing: operates on the per-question JSONL files the
benchmark already writes. No model calls, no Ollama, no re-running the
pipeline. Stdlib only (random + statistics) — no numpy/scipy dependency.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ── Reproducibility ──────────────────────────────────────────────────────────
# A fixed default seed makes every CI in the paper reproducible. Callers can
# override per call; the paper evaluation should use the default so re-running
# the aggregator yields byte-identical intervals.
_DEFAULT_SEED = 20260518
_DEFAULT_RESAMPLES = 10_000


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BootstrapCI:
    """Confidence interval for a single metric on a single config."""
    metric: str
    point_estimate: float
    ci_low: float
    ci_high: float
    n_questions: int
    n_resamples: int
    confidence: float          # e.g. 0.95

    def __str__(self) -> str:
        return (
            f"{self.metric}={self.point_estimate:.3f} "
            f"[{self.ci_low:.3f}, {self.ci_high:.3f}] "
            f"({int(self.confidence * 100)}% CI, n={self.n_questions})"
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "metric": self.metric,
            "point_estimate": self.point_estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_questions": self.n_questions,
            "n_resamples": self.n_resamples,
            "confidence": self.confidence,
        }


@dataclass
class PairedBootstrapResult:
    """Paired comparison of two configs (B minus A) on the same question set."""
    metric: str
    name_a: str
    name_b: str
    mean_a: float
    mean_b: float
    delta: float                # mean_b - mean_a (point estimate of the difference)
    delta_ci_low: float
    delta_ci_high: float
    p_value: float              # two-sided; H0: delta == 0
    n_questions: int
    n_resamples: int
    confidence: float

    @property
    def significant(self) -> bool:
        """True iff the (1-confidence)-level CI on the difference excludes 0."""
        return self.delta_ci_low > 0.0 or self.delta_ci_high < 0.0

    def __str__(self) -> str:
        sig = "significant" if self.significant else "not significant"
        return (
            f"{self.metric}: {self.name_b} - {self.name_a} = "
            f"{self.delta:+.3f} [{self.delta_ci_low:+.3f}, {self.delta_ci_high:+.3f}] "
            f"(p={self.p_value:.4f}, {sig}, n={self.n_questions})"
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "metric": self.metric,
            "name_a": self.name_a,
            "name_b": self.name_b,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "delta": self.delta,
            "delta_ci_low": self.delta_ci_low,
            "delta_ci_high": self.delta_ci_high,
            "p_value": self.p_value,
            "significant": self.significant,
            "n_questions": self.n_questions,
            "n_resamples": self.n_resamples,
            "confidence": self.confidence,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CORE BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────

def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile of an already-sorted sequence.

    q in [0, 1]. Matches numpy.percentile's default 'linear' method so the
    intervals are comparable to any numpy-based re-implementation.
    """
    if not sorted_values:
        raise ValueError("percentile of empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def bootstrap_ci(
    values: Sequence[float],
    metric_name: str = "metric",
    n_resamples: int = _DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = _DEFAULT_SEED,
) -> BootstrapCI:
    """Bootstrap CI for the MEAN of a per-question metric.

    Args:
        values:      Per-question metric values (e.g. 1.0/0.0 for EM, or
                     F1 floats). One entry per question.
        metric_name: Label for the result (e.g. "EM", "F1", "SF-F1").
        n_resamples: Number of bootstrap resamples.
        confidence:  CI confidence level (0.95 -> 95% CI).
        seed:        RNG seed for reproducibility.

    Returns:
        BootstrapCI with point estimate (the observed mean) and the
        [ci_low, ci_high] percentile interval.
    """
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        raise ValueError("bootstrap_ci: empty values")

    point = sum(vals) / n
    rng = random.Random(seed)

    means: List[float] = []
    for _ in range(n_resamples):
        # Resample n indices with replacement, recompute the mean.
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        means.append(s / n)
    means.sort()

    alpha = 1.0 - confidence
    return BootstrapCI(
        metric=metric_name,
        point_estimate=point,
        ci_low=_percentile(means, alpha / 2.0),
        ci_high=_percentile(means, 1.0 - alpha / 2.0),
        n_questions=n,
        n_resamples=n_resamples,
        confidence=confidence,
    )


def paired_bootstrap(
    values_a: Sequence[float],
    values_b: Sequence[float],
    metric_name: str = "metric",
    name_a: str = "A",
    name_b: str = "B",
    n_resamples: int = _DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = _DEFAULT_SEED,
) -> PairedBootstrapResult:
    """Paired bootstrap comparison of two configs on the SAME question set.

    `values_a[i]` and `values_b[i]` MUST be the metric for the same
    question i under config A and config B respectively. The caller is
    responsible for aligning the two lists by question_id before calling
    this function (see `paired_bootstrap_from_jsonl`).

    For each resample, ONE set of question indices is drawn and applied
    to both configs — this is what makes the test 'paired'.

    The two-sided p-value is the bootstrap estimate of
    P(|resampled delta| >= |observed delta| | H0: true delta = 0). It is
    computed by re-centring the resampled-delta distribution on zero and
    counting how often a re-centred resample is at least as extreme as
    the observed delta.

    Returns:
        PairedBootstrapResult.
    """
    a = [float(v) for v in values_a]
    b = [float(v) for v in values_b]
    if len(a) != len(b):
        raise ValueError(
            f"paired_bootstrap: length mismatch ({len(a)} vs {len(b)}); "
            f"the two configs must be evaluated on the same question set"
        )
    n = len(a)
    if n == 0:
        raise ValueError("paired_bootstrap: empty values")

    mean_a = sum(a) / n
    mean_b = sum(b) / n
    observed_delta = mean_b - mean_a

    rng = random.Random(seed)

    deltas: List[float] = []
    for _ in range(n_resamples):
        # Draw ONE index set, apply to both configs (the paired step).
        sa = 0.0
        sb = 0.0
        for _ in range(n):
            idx = rng.randrange(n)
            sa += a[idx]
            sb += b[idx]
        deltas.append((sb / n) - (sa / n))
    deltas.sort()

    alpha = 1.0 - confidence
    delta_ci_low = _percentile(deltas, alpha / 2.0)
    delta_ci_high = _percentile(deltas, 1.0 - alpha / 2.0)

    # Two-sided bootstrap p-value: re-centre the resampled-delta
    # distribution on zero (the null), count how often a re-centred
    # resample is at least as extreme (in absolute value) as the
    # observed delta. +1 smoothing avoids p == 0 (Davison & Hinkley 1997).
    abs_obs = abs(observed_delta)
    extreme = sum(1 for d in deltas if abs(d - observed_delta) >= abs_obs)
    p_value = (extreme + 1) / (n_resamples + 1)
    p_value = min(1.0, p_value)

    return PairedBootstrapResult(
        metric=metric_name,
        name_a=name_a,
        name_b=name_b,
        mean_a=mean_a,
        mean_b=mean_b,
        delta=observed_delta,
        delta_ci_low=delta_ci_low,
        delta_ci_high=delta_ci_high,
        p_value=p_value,
        n_questions=n,
        n_resamples=n_resamples,
        confidence=confidence,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MULTIPLE-COMPARISON CORRECTION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CorrectedComparison:
    """One paired comparison annotated with its family-wise-corrected verdict."""
    result: PairedBootstrapResult
    rank: int                    # 1 = smallest raw p in the family
    p_raw: float
    p_adjusted: float            # Holm-adjusted p (monotone, capped at 1.0)
    alpha: float                 # family-wise error rate the family is held at
    significant_corrected: bool  # p_adjusted <= alpha

    def as_dict(self) -> Dict[str, object]:
        d = self.result.as_dict()
        d.update({
            "rank": self.rank,
            "p_raw": self.p_raw,
            "p_adjusted": self.p_adjusted,
            "alpha": self.alpha,
            "significant_raw": self.result.significant,
            "significant_corrected": self.significant_corrected,
        })
        return d


def holm_bonferroni(
    results: Sequence[PairedBootstrapResult],
    alpha: float = 0.05,
) -> List[CorrectedComparison]:
    """Holm-Bonferroni correction over a *family* of paired comparisons.

    Why this exists
    ---------------
    The paper reports several component-delta p-values per dataset (retrieval,
    planner, verifier, self-correction). Each `paired_bootstrap` p-value controls
    only its OWN false-positive rate at alpha; testing m hypotheses and reporting
    the ones below 0.05 inflates the family-wise error rate to ~1-(1-alpha)^m. A
    reviewer will apply this correction mentally, so the paper applies it
    explicitly. Holm (1979) is the standard uniformly-more-powerful replacement
    for plain Bonferroni: it makes no independence assumption (valid for the
    correlated component deltas here) yet rejects at least as many hypotheses.

    Procedure (Holm 1979, step-down)
    --------------------------------
    Sort the m raw p-values ascending: p_(1) <= ... <= p_(m). The i-th
    (1-indexed) adjusted p is ``max over j<=i of ((m - j + 1) * p_(j))``, capped
    at 1.0; the running max enforces monotonicity so a later hypothesis is never
    reported as more significant than an earlier one. A hypothesis is rejected
    iff its adjusted p <= alpha.

    Args:
        results: The family of paired comparisons to correct *together*. The
                 caller decides the family boundary (typically: all component
                 deltas for one dataset+metric). Correcting across an
                 incoherent family over-penalises; this function does not guess.
        alpha:   Family-wise error rate (default 0.05).

    Returns:
        One CorrectedComparison per input, in ASCENDING raw-p order (most
        significant first), each carrying its Holm-adjusted p and corrected
        verdict. The input order is recoverable via ``result.name_a/name_b``.
    """
    if not results:
        return []

    m = len(results)
    # Stable sort by raw p so ties keep input order (deterministic output).
    order = sorted(range(m), key=lambda i: results[i].p_value)

    corrected: List[CorrectedComparison] = []
    running_max = 0.0
    for rank, idx in enumerate(order, start=1):
        p_raw = results[idx].p_value
        # Holm step-down multiplier: (m - rank + 1).
        adj = min(1.0, (m - rank + 1) * p_raw)
        running_max = max(running_max, adj)   # enforce monotone non-decreasing
        corrected.append(CorrectedComparison(
            result=results[idx],
            rank=rank,
            p_raw=p_raw,
            p_adjusted=running_max,
            alpha=alpha,
            significant_corrected=running_max <= alpha,
        ))
    return corrected


def format_holm_table(corrected: Sequence[CorrectedComparison]) -> str:
    """Render a Holm-corrected family as a Markdown table for the paper."""
    if not corrected:
        return "_(no comparisons in family)_"
    lines = [
        "| Delta | ΔEM | raw p | Holm p | Significant (corrected) |",
        "|---|---|---|---|---|",
    ]
    for c in corrected:
        r = c.result
        name = f"{r.name_b} − {r.name_a}"
        verdict = "★ yes" if c.significant_corrected else "no"
        lines.append(
            f"| {name} | {r.delta:+.3f} | {c.p_raw:.4f} | "
            f"{c.p_adjusted:.4f} | {verdict} |"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# JSONL ADAPTERS
# ─────────────────────────────────────────────────────────────────────────────

# Metric key in the per-question JSONL -> a function that maps the raw
# JSONL value to a float. EM is stored as a bool; F1/SF-F1 as floats.
_METRIC_EXTRACTORS: Dict[str, Callable[[Dict], float]] = {
    "EM":    lambda r: 1.0 if r.get("exact_match") else 0.0,
    "F1":    lambda r: float(r.get("f1_score", 0.0) or 0.0),
    "SF-F1": lambda r: float(r.get("sf_f1", 0.0) or 0.0),
    "SF-Recall": lambda r: 1.0 if r.get("all_gold_retrieved") else 0.0,
}


def load_jsonl_records(path: Path) -> List[Dict]:
    """Load a per-question JSONL file into a list of dicts.

    Lines that fail to parse are skipped with no exception — a partially
    written JSONL (e.g. an interrupted run) still yields its valid lines.
    """
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _records_to_metric(
    records: Sequence[Dict],
    metric: str,
) -> Dict[str, float]:
    """Map a list of JSONL records to {question_id: metric_value}."""
    if metric not in _METRIC_EXTRACTORS:
        raise ValueError(
            f"Unknown metric {metric!r}; available: "
            f"{sorted(_METRIC_EXTRACTORS)}"
        )
    extractor = _METRIC_EXTRACTORS[metric]
    out: Dict[str, float] = {}
    for r in records:
        qid = r.get("question_id")
        if qid is None:
            continue
        out[str(qid)] = extractor(r)
    return out


def bootstrap_ci_from_jsonl(
    jsonl_path: Path,
    metric: str = "EM",
    n_resamples: int = _DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = _DEFAULT_SEED,
) -> BootstrapCI:
    """Bootstrap CI for one metric on one config's JSONL output."""
    records = load_jsonl_records(jsonl_path)
    by_qid = _records_to_metric(records, metric)
    return bootstrap_ci(
        values=list(by_qid.values()),
        metric_name=metric,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )


def paired_bootstrap_from_jsonl(
    jsonl_a: Path,
    jsonl_b: Path,
    metric: str = "EM",
    name_a: Optional[str] = None,
    name_b: Optional[str] = None,
    n_resamples: int = _DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = _DEFAULT_SEED,
) -> PairedBootstrapResult:
    """Paired bootstrap comparison of two configs' JSONL outputs.

    The two files are aligned by `question_id`: only questions present in
    BOTH files are compared (the intersection). If the intersection is
    smaller than either file, a note is implicit in the returned
    `n_questions`. This guards against comparing configs that were run on
    slightly different question subsets.
    """
    name_a = name_a or jsonl_a.stem
    name_b = name_b or jsonl_b.stem

    by_qid_a = _records_to_metric(load_jsonl_records(jsonl_a), metric)
    by_qid_b = _records_to_metric(load_jsonl_records(jsonl_b), metric)

    # Align by question_id — only the intersection is comparable.
    common_qids = sorted(set(by_qid_a) & set(by_qid_b))
    if not common_qids:
        raise ValueError(
            f"No shared question_ids between {jsonl_a.name} and "
            f"{jsonl_b.name}; cannot run a paired comparison."
        )

    values_a = [by_qid_a[q] for q in common_qids]
    values_b = [by_qid_b[q] for q in common_qids]

    return paired_bootstrap(
        values_a=values_a,
        values_b=values_b,
        metric_name=metric,
        name_a=name_a,
        name_b=name_b,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _main() -> None:
    """Standalone CLI: compute a CI for one JSONL, or a paired comparison.

        # single-config CI
        python -m src.thesis_evaluations.bootstrap A.jsonl --metric EM

        # paired comparison
        python -m src.thesis_evaluations.bootstrap A.jsonl B.jsonl --metric EM
    """
    import argparse

    parser = argparse.ArgumentParser(description=_main.__doc__)
    parser.add_argument("jsonl_a", type=Path, help="First config's JSONL.")
    parser.add_argument("jsonl_b", type=Path, nargs="?", default=None,
                        help="Second config's JSONL (enables paired mode).")
    parser.add_argument("--metric", default="EM",
                        choices=sorted(_METRIC_EXTRACTORS),
                        help="Metric to analyse (default: EM).")
    parser.add_argument("--resamples", type=int, default=_DEFAULT_RESAMPLES)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    args = parser.parse_args()

    if args.jsonl_b is None:
        ci = bootstrap_ci_from_jsonl(
            args.jsonl_a, metric=args.metric,
            n_resamples=args.resamples, confidence=args.confidence,
            seed=args.seed,
        )
        print(ci)
    else:
        res = paired_bootstrap_from_jsonl(
            args.jsonl_a, args.jsonl_b, metric=args.metric,
            n_resamples=args.resamples, confidence=args.confidence,
            seed=args.seed,
        )
        print(res)


if __name__ == "__main__":
    _main()
