"""
Unit tests for the paired-bootstrap significance module (paper §7,
Evaluation Layer — significance testing for the ablation tables).

These tests are deterministic (fixed RNG seed) and have no external
dependencies — no Ollama, no model, no eval run.

Last reviewed: 2026-05-30 (audit pass, project version 5.4).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.thesis_evaluations.bootstrap import (
    BootstrapCI,
    CorrectedComparison,
    PairedBootstrapResult,
    bootstrap_ci,
    bootstrap_ci_from_jsonl,
    format_holm_table,
    holm_bonferroni,
    paired_bootstrap,
    paired_bootstrap_from_jsonl,
    load_jsonl_records,
    _percentile,
)


def _mk_result(name_b: str, delta: float, p: float) -> PairedBootstrapResult:
    """Construct a PairedBootstrapResult with a given delta/p for Holm tests.

    The CI bounds are irrelevant to Holm correction (which keys on p_value),
    so they are set to bracket the delta trivially.
    """
    return PairedBootstrapResult(
        metric="EM", name_a="prev", name_b=name_b,
        mean_a=0.0, mean_b=delta, delta=delta,
        delta_ci_low=delta - 0.01, delta_ci_high=delta + 0.01,
        p_value=p, n_questions=500, n_resamples=10_000, confidence=0.95,
    )


# ─────────────────────────────────────────────────────────────────────────────
# _percentile
# ─────────────────────────────────────────────────────────────────────────────

class TestPercentile:

    def test_median_of_odd(self):
        assert _percentile([1.0, 2.0, 3.0], 0.5) == 2.0

    def test_endpoints(self):
        vals = [10.0, 20.0, 30.0, 40.0]
        assert _percentile(vals, 0.0) == 10.0
        assert _percentile(vals, 1.0) == 40.0

    def test_linear_interpolation(self):
        # 25th percentile of [0,1,2,3]: pos = 0.25*3 = 0.75 -> 0*0.25 + 1*0.75
        assert _percentile([0.0, 1.0, 2.0, 3.0], 0.25) == pytest.approx(0.75)

    def test_single_element(self):
        assert _percentile([7.0], 0.3) == 7.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _percentile([], 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# bootstrap_ci — single-config CI
# ─────────────────────────────────────────────────────────────────────────────

class TestBootstrapCI:

    def test_point_estimate_is_the_mean(self):
        ci = bootstrap_ci([1.0] * 42 + [0.0] * 58, "EM", n_resamples=2000)
        assert ci.point_estimate == pytest.approx(0.42)

    def test_ci_brackets_the_point_estimate(self):
        ci = bootstrap_ci([1.0] * 42 + [0.0] * 58, "EM", n_resamples=2000)
        assert ci.ci_low <= ci.point_estimate <= ci.ci_high

    def test_all_ones_gives_degenerate_ci(self):
        """If every question is correct, the CI collapses to [1.0, 1.0]."""
        ci = bootstrap_ci([1.0] * 50, "EM", n_resamples=1000)
        assert ci.point_estimate == 1.0
        assert ci.ci_low == 1.0
        assert ci.ci_high == 1.0

    def test_larger_n_gives_tighter_ci(self):
        """The CI width must shrink as the sample size grows."""
        small = bootstrap_ci([1.0, 0.0] * 25, "EM", n_resamples=3000)   # n=50
        large = bootstrap_ci([1.0, 0.0] * 250, "EM", n_resamples=3000)  # n=500
        width_small = small.ci_high - small.ci_low
        width_large = large.ci_high - large.ci_low
        assert width_large < width_small, (
            f"Expected tighter CI at n=500 ({width_large:.3f}) than "
            f"n=50 ({width_small:.3f})"
        )

    def test_deterministic_under_fixed_seed(self):
        """Same seed -> byte-identical CI (paper reproducibility)."""
        a = bootstrap_ci([1.0, 0.0, 1.0] * 30, "EM", n_resamples=1500, seed=123)
        b = bootstrap_ci([1.0, 0.0, 1.0] * 30, "EM", n_resamples=1500, seed=123)
        assert a.ci_low == b.ci_low
        assert a.ci_high == b.ci_high

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci([], "EM")


# ─────────────────────────────────────────────────────────────────────────────
# paired_bootstrap — two-config comparison
# ─────────────────────────────────────────────────────────────────────────────

class TestPairedBootstrap:

    def test_identical_configs_delta_is_zero_not_significant(self):
        """Two configs with identical per-question results: delta == 0,
        CI brackets 0, not significant."""
        same = [1.0, 0.0, 1.0, 1.0, 0.0] * 20
        res = paired_bootstrap(same, same, "EM", "A", "B", n_resamples=2000)
        assert res.delta == pytest.approx(0.0, abs=1e-9)
        assert res.delta_ci_low <= 0.0 <= res.delta_ci_high
        assert not res.significant

    def test_strict_improvement_is_significant(self):
        """B correct on every question, A wrong on every question:
        delta == +1.0, highly significant."""
        res = paired_bootstrap([0.0] * 100, [1.0] * 100, "EM",
                               "base", "better", n_resamples=2000)
        assert res.delta == pytest.approx(1.0)
        assert res.delta_ci_low > 0.0
        assert res.significant
        assert res.p_value < 0.05

    def test_strict_regression_is_significant_negative(self):
        """B worse than A everywhere: delta == -1.0, significant."""
        res = paired_bootstrap([1.0] * 100, [0.0] * 100, "EM",
                               "good", "bad", n_resamples=2000)
        assert res.delta == pytest.approx(-1.0)
        assert res.delta_ci_high < 0.0
        assert res.significant

    def test_tiny_difference_not_significant(self):
        """A 1-question difference out of 200 is sampling noise — the CI
        should bracket zero."""
        a = [1.0] * 100 + [0.0] * 100
        b = [1.0] * 101 + [0.0] * 99       # exactly one question flipped
        res = paired_bootstrap(a, b, "EM", "A", "B", n_resamples=4000)
        assert res.delta == pytest.approx(0.005)
        assert not res.significant, (
            "A single flipped question out of 200 must not register as "
            "significant"
        )

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            paired_bootstrap([1.0, 0.0], [1.0], "EM")

    def test_paired_design_uses_one_index_set(self):
        """Paired bootstrap must cancel shared difficulty. If A and B differ
        by a constant per-question offset, the delta CI must be tight even
        though each config's marginal variance is large."""
        # A: alternating 1/0. B: same but every question +0 (identical).
        # Add a constant: B = A everywhere except a fixed +0.1 is impossible
        # for 0/1 metrics, so use F1-style floats.
        a = [0.3, 0.7, 0.5, 0.9, 0.1] * 40
        b = [x + 0.05 for x in a]   # B uniformly 0.05 better
        res = paired_bootstrap(a, b, "F1", "A", "B", n_resamples=3000)
        assert res.delta == pytest.approx(0.05, abs=1e-9)
        # Because the difference is constant, the paired delta has ZERO
        # variance — the CI collapses onto the point estimate.
        assert res.delta_ci_low == pytest.approx(0.05, abs=1e-9)
        assert res.delta_ci_high == pytest.approx(0.05, abs=1e-9)

    def test_deterministic_under_fixed_seed(self):
        a = [1.0, 0.0] * 50
        b = [1.0, 1.0] * 50
        r1 = paired_bootstrap(a, b, "EM", seed=777, n_resamples=1500)
        r2 = paired_bootstrap(a, b, "EM", seed=777, n_resamples=1500)
        assert r1.delta_ci_low == r2.delta_ci_low
        assert r1.p_value == r2.p_value


# ─────────────────────────────────────────────────────────────────────────────
# JSONL adapters
# ─────────────────────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


class TestJsonlAdapters:

    def test_load_jsonl_skips_malformed_lines(self, tmp_path):
        p = tmp_path / "partial.jsonl"
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"question_id": "q1", "exact_match": True}) + "\n")
            fh.write("{ this is not valid json\n")        # malformed — skipped
            fh.write(json.dumps({"question_id": "q2", "exact_match": False}) + "\n")
            fh.write("\n")                                  # blank — skipped
        records = load_jsonl_records(p)
        assert len(records) == 2

    def test_bootstrap_ci_from_jsonl_EM(self, tmp_path):
        p = tmp_path / "config.jsonl"
        recs = [{"question_id": f"q{i}", "exact_match": i < 30, "f1_score": 0.5}
                for i in range(100)]
        _write_jsonl(p, recs)
        ci = bootstrap_ci_from_jsonl(p, metric="EM", n_resamples=2000)
        assert ci.point_estimate == pytest.approx(0.30)
        assert ci.n_questions == 100

    def test_paired_bootstrap_from_jsonl_aligns_by_question_id(self, tmp_path):
        """The two files share question_ids; the paired test must align on
        them, not on line order."""
        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        # A: q0..q49, B: SAME ids but shuffled order + B is strictly better.
        recs_a = [{"question_id": f"q{i}", "exact_match": False,
                   "f1_score": 0.0} for i in range(50)]
        recs_b = [{"question_id": f"q{i}", "exact_match": True,
                   "f1_score": 1.0} for i in reversed(range(50))]
        _write_jsonl(a, recs_a)
        _write_jsonl(b, recs_b)
        res = paired_bootstrap_from_jsonl(a, b, metric="EM", n_resamples=2000)
        assert res.n_questions == 50
        assert res.delta == pytest.approx(1.0)
        assert res.significant

    def test_paired_bootstrap_from_jsonl_uses_intersection(self, tmp_path):
        """When the two files cover overlapping-but-not-identical question
        sets, only the shared question_ids are compared."""
        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        _write_jsonl(a, [{"question_id": f"q{i}", "exact_match": True,
                          "f1_score": 1.0} for i in range(60)])   # q0..q59
        _write_jsonl(b, [{"question_id": f"q{i}", "exact_match": True,
                          "f1_score": 1.0} for i in range(40, 100)])  # q40..q99
        res = paired_bootstrap_from_jsonl(a, b, metric="EM", n_resamples=500)
        # Intersection q40..q59 = 20 questions.
        assert res.n_questions == 20

    def test_paired_bootstrap_from_jsonl_no_overlap_raises(self, tmp_path):
        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        _write_jsonl(a, [{"question_id": "q1", "exact_match": True}])
        _write_jsonl(b, [{"question_id": "q2", "exact_match": True}])
        with pytest.raises(ValueError, match="No shared question_ids"):
            paired_bootstrap_from_jsonl(a, b, metric="EM")

    def test_unknown_metric_raises(self, tmp_path):
        p = tmp_path / "c.jsonl"
        _write_jsonl(p, [{"question_id": "q1", "exact_match": True}])
        with pytest.raises(ValueError, match="Unknown metric"):
            bootstrap_ci_from_jsonl(p, metric="NONSENSE")


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass formatting
# ─────────────────────────────────────────────────────────────────────────────

class TestResultFormatting:

    def test_bootstrap_ci_str(self):
        ci = BootstrapCI("EM", 0.44, 0.41, 0.47, 500, 10000, 0.95)
        s = str(ci)
        assert "EM=0.440" in s
        assert "[0.410, 0.470]" in s
        assert "n=500" in s

    def test_paired_result_significant_flag(self):
        # CI strictly above zero -> significant.
        r = PairedBootstrapResult(
            "EM", "A", "B", 0.42, 0.46, 0.04, 0.01, 0.07,
            0.012, 500, 10000, 0.95,
        )
        assert r.significant
        # CI crossing zero -> not significant.
        r2 = PairedBootstrapResult(
            "EM", "A", "B", 0.42, 0.44, 0.02, -0.01, 0.05,
            0.21, 500, 10000, 0.95,
        )
        assert not r2.significant

    def test_as_dict_roundtrip_keys(self):
        ci = BootstrapCI("F1", 0.5, 0.45, 0.55, 100, 5000, 0.95)
        d = ci.as_dict()
        for k in ("metric", "point_estimate", "ci_low", "ci_high",
                  "n_questions", "n_resamples", "confidence"):
            assert k in d


# ─────────────────────────────────────────────────────────────────────────────
# Aggregator integration — significance table builder
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregatorSignificanceTable:

    def test_missing_dir_returns_placeholder(self):
        from src.thesis_evaluations.thesis_results_aggregator import (
            build_significance_table,
        )
        out = build_significance_table(None)
        assert out.startswith("%")

    def test_table_built_from_row_jsonl(self, tmp_path):
        """A minimal 3-row ablation directory must yield a LaTeX table with
        one significance row per consecutive pair."""
        from src.thesis_evaluations.thesis_results_aggregator import (
            build_significance_table,
        )
        # row1: all wrong. row2: half right. row3: all right.
        _write_jsonl(tmp_path / "row1_llm_only.jsonl",
                     [{"question_id": f"q{i}", "exact_match": False,
                       "f1_score": 0.0} for i in range(40)])
        _write_jsonl(tmp_path / "row2_rag_no_agent.jsonl",
                     [{"question_id": f"q{i}", "exact_match": i < 20,
                       "f1_score": 0.5} for i in range(40)])
        _write_jsonl(tmp_path / "row3_planner.jsonl",
                     [{"question_id": f"q{i}", "exact_match": True,
                       "f1_score": 1.0} for i in range(40)])
        table = build_significance_table(tmp_path, metric="EM")
        assert "\\begin{tabular}" in table
        assert "\\bottomrule" in table
        # Two consecutive pairs -> two data rows with the arrow marker.
        assert table.count("$\\rightarrow$") == 2
        # Both transitions are strict improvements -> both significant.
        assert table.count("& yes \\\\") == 2

    def test_fewer_than_two_rows_returns_placeholder(self, tmp_path):
        from src.thesis_evaluations.thesis_results_aggregator import (
            build_significance_table,
        )
        _write_jsonl(tmp_path / "row1_llm_only.jsonl",
                     [{"question_id": "q1", "exact_match": True}])
        out = build_significance_table(tmp_path, metric="EM")
        assert out.startswith("%")


# ─────────────────────────────────────────────────────────────────────────────
# Holm–Bonferroni multiple-comparison correction
# ─────────────────────────────────────────────────────────────────────────────

class TestHolmBonferroni:

    def test_empty_family_returns_empty(self):
        assert holm_bonferroni([]) == []

    def test_single_comparison_is_uncorrected(self):
        """With m=1 the multiplier is (1-1+1)=1, so adjusted p == raw p."""
        res = _mk_result("+Retrieval", 0.262, 0.0001)
        [c] = holm_bonferroni([res])
        assert c.p_adjusted == pytest.approx(0.0001)
        assert c.significant_corrected is True

    def test_known_four_family_values(self):
        """The HotpotQA family (m=4): only the retrieval lift survives.

        Hand-computed Holm step-down with multipliers (4,3,2,1) over the
        sorted raw p-values [0.0001, 0.033, 0.44, 0.81]:
            0.0001*4 = 0.0004   -> significant
            0.033 *3 = 0.099    -> not significant
            0.44  *2 = 0.88     -> not significant
            0.81  *1 = 0.81 -> running-max 0.88 -> not significant
        """
        fam = [
            _mk_result("+Retrieval", 0.262, 0.0001),
            _mk_result("+Planner", -0.038, 0.033),
            _mk_result("+Verifier", -0.004, 0.44),
            _mk_result("+SelfCorrect", -0.004, 0.81),
        ]
        corr = holm_bonferroni(fam, alpha=0.05)
        # Output is in ascending raw-p order.
        assert [c.result.name_b for c in corr] == [
            "+Retrieval", "+Planner", "+Verifier", "+SelfCorrect"]
        assert corr[0].p_adjusted == pytest.approx(0.0004)
        assert corr[1].p_adjusted == pytest.approx(0.099)
        assert corr[2].p_adjusted == pytest.approx(0.88)
        assert corr[3].p_adjusted == pytest.approx(0.88)  # running max, not 0.81
        assert [c.significant_corrected for c in corr] == [True, False, False, False]

    def test_adjusted_p_is_monotone_nondecreasing(self):
        """Holm's running max guarantees a later (larger raw-p) hypothesis is
        never reported as more significant than an earlier one."""
        fam = [
            _mk_result("a", 0.1, 0.001),
            _mk_result("b", 0.1, 0.20),
            _mk_result("c", 0.1, 0.04),
            _mk_result("d", 0.1, 0.50),
        ]
        corr = holm_bonferroni(fam)
        adj = [c.p_adjusted for c in corr]
        assert adj == sorted(adj)

    def test_adjusted_p_capped_at_one(self):
        fam = [_mk_result(f"r{i}", 0.0, 0.9) for i in range(5)]
        corr = holm_bonferroni(fam)
        assert all(c.p_adjusted <= 1.0 for c in corr)

    def test_corrected_can_differ_from_raw_verdict(self):
        """A raw-significant delta (p=0.045) becomes non-significant under
        correction in a family of four — the §3.3 self-correction case."""
        fam = [
            _mk_result("+SelfCorrect", 0.026, 0.045),
            _mk_result("+Retrieval", -0.010, 0.57),
            _mk_result("+Verifier", -0.004, 0.83),
            _mk_result("+Planner", 0.004, 0.84),
        ]
        corr = holm_bonferroni(fam)
        sc = next(c for c in corr if c.result.name_b == "+SelfCorrect")
        assert sc.result.significant is True          # raw CI excludes 0…
        assert sc.significant_corrected is False        # …but Holm rejects it

    def test_as_dict_carries_both_verdicts(self):
        [c] = holm_bonferroni([_mk_result("+Retrieval", 0.262, 0.0001)])
        d = c.as_dict()
        for k in ("p_raw", "p_adjusted", "significant_raw",
                  "significant_corrected", "rank", "alpha"):
            assert k in d

    def test_format_table_has_one_row_per_comparison(self):
        fam = [
            _mk_result("+Retrieval", 0.262, 0.0001),
            _mk_result("+Planner", -0.038, 0.033),
        ]
        table = format_holm_table(holm_bonferroni(fam))
        assert "Holm p" in table
        assert table.count("−") >= 2  # one "name_b − name_a" per data row

    def test_format_empty_family(self):
        assert "no comparisons" in format_holm_table([])
