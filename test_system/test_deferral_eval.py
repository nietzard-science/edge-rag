"""
Unit tests for the edge-cloud deferral economics module
(src.thesis_evaluations.deferral_eval).

Deterministic, no model calls, stdlib + the module only.

Last reviewed: 2026-06-09.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.thesis_evaluations.deferral_eval import (
    CloudCostModel,
    _QRecord,
    _deferral_order,
    cost_to_hit_targets,
    find_knee,
    sweep_deferral,
    _load_records,
)


def _rec(qid, conf, correct, f1=0.0, lat=1000.0) -> _QRecord:
    return _QRecord(qid=qid, conf=conf, correct_local=correct, f1=f1,
                    local_latency_ms=lat)


# ─────────────────────────────────────────────────────────────────────────────
# Cost model
# ─────────────────────────────────────────────────────────────────────────────

class TestCloudCostModel:

    def test_cost_per_query_matches_hand_calc(self):
        # 1200 prompt @ $2.50/M + 40 gen @ $10/M
        c = CloudCostModel(price_in_per_mtok=2.5, price_out_per_mtok=10.0,
                           prompt_tokens=1200, gen_tokens=40)
        expected = (1200 / 1e6) * 2.5 + (40 / 1e6) * 10.0
        assert c.cost_per_query_usd() == pytest.approx(expected)

    def test_zero_tokens_zero_cost(self):
        c = CloudCostModel(prompt_tokens=0, gen_tokens=0)
        assert c.cost_per_query_usd() == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Deferral order (worst-confidence-first, deterministic)
# ─────────────────────────────────────────────────────────────────────────────

class TestDeferralOrder:

    def test_error_before_low_before_high(self):
        recs = [_rec("a", "high", True), _rec("b", "error", False),
                _rec("c", "low", False)]
        order = [r.qid for r in _deferral_order(recs)]
        assert order == ["b", "c", "a"]  # error, low, high

    def test_within_tier_ascending_f1(self):
        recs = [_rec("a", "high", True, f1=0.9),
                _rec("b", "high", False, f1=0.1)]
        order = [r.qid for r in _deferral_order(recs)]
        assert order == ["b", "a"]  # lower f1 deferred first


# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────

class TestSweep:

    def _mixed(self):
        # 6 high (4 correct), 4 low (0 correct) → local acc 4/10 = 0.4
        recs = [_rec(f"h{i}", "high", i < 4, f1=0.8) for i in range(6)]
        recs += [_rec(f"l{i}", "low", False, f1=0.1) for i in range(4)]
        return recs

    def test_zero_deferral_is_pure_local(self):
        recs = self._mixed()
        pts = sweep_deferral(recs, CloudCostModel(), [0.0])
        assert pts[0].accuracy == pytest.approx(0.4)
        assert pts[0].cost_usd_per_1000q == 0.0
        assert pts[0].n_deferred == 0

    def test_full_deferral_is_cloud_ceiling(self):
        recs = self._mixed()
        pts = sweep_deferral(recs, CloudCostModel(), [1.0],
                             cloud_mode="fixed", cloud_accuracy=0.9)
        assert pts[0].accuracy == pytest.approx(0.9)
        assert pts[0].n_deferred == 10

    def test_oracle_full_deferral_is_one(self):
        recs = self._mixed()
        pts = sweep_deferral(recs, CloudCostModel(), [1.0], cloud_mode="oracle")
        assert pts[0].accuracy == pytest.approx(1.0)

    def test_deferring_low_conf_raises_local_only_acc(self):
        """Because the low-confidence questions are all wrong, deferring them
        must raise the accuracy of the kept-local slice — the core evidence
        that the confidence signal targets the right questions."""
        recs = self._mixed()
        pts = sweep_deferral(recs, CloudCostModel(), [0.0, 0.4])
        # defer 40% (the 4 low-conf, all wrong) → kept 6 high, 4 correct → 4/6
        assert pts[1].accuracy_local_only == pytest.approx(4 / 6)
        assert pts[1].accuracy_local_only > pts[0].accuracy_local_only

    def test_cost_scales_with_deferred_count(self):
        recs = self._mixed()
        c = CloudCostModel(prompt_tokens=1000, gen_tokens=0, price_in_per_mtok=1.0)
        # cost_per_q = 1000/1e6 * 1.0 = 0.001 USD
        pts = sweep_deferral(recs, c, [0.5])  # defer 5 of 10
        # $/1000q = 0.001 * 0.5 * 1000 = 0.5
        assert pts[0].cost_usd_per_1000q == pytest.approx(0.5)

    def test_latency_blends_local_and_cloud(self):
        recs = [_rec("a", "low", False, lat=2000.0),
                _rec("b", "high", True, lat=2000.0)]
        c = CloudCostModel(cloud_latency_ms=1500.0)
        pts = sweep_deferral(recs, c, [0.5])  # defer 1 (the low one)
        # one deferred: 2000+1500, one kept: 2000 → mean (3500+2000)/2 = 2750
        assert pts[0].mean_latency_ms == pytest.approx(2750.0)

    def test_empty_records(self):
        assert sweep_deferral([], CloudCostModel(), [0.0, 0.5]) == []


# ─────────────────────────────────────────────────────────────────────────────
# Knee
# ─────────────────────────────────────────────────────────────────────────────

class TestKnee:

    def test_knee_none_on_empty(self):
        assert find_knee([]) is None

    def test_knee_prefers_accuracy_per_dollar(self):
        recs = [_rec(f"l{i}", "low", False) for i in range(5)]
        recs += [_rec(f"h{i}", "high", True) for i in range(5)]
        pts = sweep_deferral(recs, CloudCostModel(), [0.0, 0.1, 0.5])
        knee = find_knee(pts)
        assert knee is not None
        assert knee.n_deferred > 0


# ─────────────────────────────────────────────────────────────────────────────
# Cost-to-hit-target
# ─────────────────────────────────────────────────────────────────────────────

class TestCostToHitTargets:

    def _mixed(self):
        recs = [_rec(f"h{i}", "high", i < 4, f1=0.8) for i in range(6)]
        recs += [_rec(f"l{i}", "low", False, f1=0.1) for i in range(4)]
        return recs

    def test_target_below_local_needs_no_deferral(self):
        recs = self._mixed()  # local acc 0.4
        [t] = cost_to_hit_targets(recs, CloudCostModel(), [0.3])
        assert t["reachable"] is True
        assert t["n_deferred"] == 0
        assert t["cost_usd_per_1000q"] == 0.0

    def test_target_above_ceiling_unreachable(self):
        recs = self._mixed()
        [t] = cost_to_hit_targets(recs, CloudCostModel(), [0.95],
                                  cloud_mode="fixed", cloud_accuracy=0.9)
        assert t["reachable"] is False

    def test_monotone_targets_need_monotone_deferral(self):
        recs = self._mixed()
        ts = cost_to_hit_targets(recs, CloudCostModel(), [0.5, 0.7, 0.85],
                                 cloud_mode="fixed", cloud_accuracy=0.9)
        fracs = [t["defer_fraction"] for t in ts]
        assert fracs == sorted(fracs)  # higher target → at least as much defer

    def test_achieved_meets_target_when_reachable(self):
        recs = self._mixed()
        for t in cost_to_hit_targets(recs, CloudCostModel(), [0.6, 0.8],
                                     cloud_mode="fixed", cloud_accuracy=0.9):
            if t["reachable"]:
                assert t["achieved_accuracy"] >= t["target_accuracy"] - 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# JSONL loading
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadRecords:

    def test_soft_vs_em_mode(self, tmp_path):
        p = tmp_path / "r.jsonl"
        p.write_text(
            json.dumps({"question_id": "q1", "confidence": "high",
                        "exact_match": False, "f1_score": 0.7}) + "\n"
            + json.dumps({"question_id": "q2", "confidence": "low",
                          "exact_match": True, "f1_score": 1.0}) + "\n",
            encoding="utf-8",
        )
        soft = _load_records(p, threshold=0.6, mode="soft")
        em = _load_records(p, threshold=0.6, mode="em")
        # q1: f1=0.7 → soft-correct but EM-wrong
        assert soft[0].correct_local is True
        assert em[0].correct_local is False

    def test_skips_records_without_qid_and_bad_lines(self, tmp_path):
        p = tmp_path / "r.jsonl"
        p.write_text(
            json.dumps({"confidence": "high", "exact_match": True}) + "\n"   # no qid
            + "not json\n"
            + json.dumps({"question_id": "q1", "confidence": "high",
                          "exact_match": True, "f1_score": 1.0}) + "\n",
            encoding="utf-8",
        )
        recs = _load_records(p, threshold=0.6, mode="soft")
        assert [r.qid for r in recs] == ["q1"]

    def test_missing_confidence_normalised_to_error(self, tmp_path):
        p = tmp_path / "r.jsonl"
        p.write_text(
            json.dumps({"question_id": "q1", "exact_match": False,
                        "f1_score": 0.0}) + "\n", encoding="utf-8")
        recs = _load_records(p, threshold=0.6, mode="soft")
        assert recs[0].conf == "error"
