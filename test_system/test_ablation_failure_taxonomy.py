"""
Unit tests for the cross-row ablation failure taxonomy (B-5).

Deterministic, no model calls, no eval run.

Last reviewed: 2026-06-09.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.thesis_evaluations.ablation_failure_taxonomy import (
    _gold_present,
    bucket_file,
    build_comparison,
    classify_row,
)


def _wrong(**kw):
    base = {"exact_match": False, "f1_score": 0.0, "gold_answer": "paris",
            "predicted_answer": "london", "all_gold_retrieved": True}
    base.update(kw)
    return base


class TestGoldPresentSignal:

    def test_prefers_all_gold_retrieved(self):
        present, sig = _gold_present({"all_gold_retrieved": True,
                                      "gold_in_final_context": False})
        assert present is True and sig == "all_gold_retrieved"

    def test_falls_back_to_final_context(self):
        present, sig = _gold_present({"gold_in_final_context": True})
        assert present is True and sig == "gold_in_final_context"

    def test_falls_back_to_title_substring(self):
        present, sig = _gold_present({"gold_answer": "Ed Wood",
                                      "retrieved_titles": ["ed wood", "x"]})
        assert present is True and sig == "retrieved_titles_substring"


class TestClassify:

    def test_retrieval_miss_when_gold_absent(self):
        assert classify_row(_wrong(all_gold_retrieved=False)) == "a_retrieval_miss"

    def test_abstention(self):
        assert classify_row(
            _wrong(predicted_answer="I cannot determine the answer")
        ) == "d_abstention"

    def test_empty_prediction_is_abstention(self):
        assert classify_row(_wrong(predicted_answer="")) == "d_abstention"

    def test_close_miss(self):
        assert classify_row(_wrong(f1_score=0.6)) == "e_close_miss"

    def test_format_mismatch_yesno_vs_name(self):
        assert classify_row(
            _wrong(gold_answer="yes", predicted_answer="Barack Obama", f1_score=0.0)
        ) == "c_format_mismatch"

    def test_grounded_halluc_default(self):
        # gold present, real non-abstention answer, low F1, both non-yes/no
        assert classify_row(
            _wrong(gold_answer="paris", predicted_answer="london", f1_score=0.0)
        ) == "b_grounded_halluc"


class TestBucketFile:

    def test_counts_and_fault_split(self, tmp_path):
        p = tmp_path / "row.jsonl"
        recs = [
            {"exact_match": True, "f1_score": 1.0},                       # correct
            _wrong(all_gold_retrieved=False),                            # retrieval miss
            _wrong(gold_answer="paris", predicted_answer="london"),       # halluc
            _wrong(predicted_answer="I don't know"),                      # abstention
        ]
        p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
        out = bucket_file(p)
        assert out["n_total"] == 4
        assert out["n_correct"] == 1
        assert out["n_wrong"] == 3
        assert out["retrieval_fault"] == 1
        assert out["answer_fault"] == 2  # halluc + abstention

    def test_skips_bad_lines(self, tmp_path):
        p = tmp_path / "row.jsonl"
        p.write_text(
            json.dumps(_wrong()) + "\nnot json\n"
            + json.dumps({"exact_match": True}) + "\n", encoding="utf-8")
        out = bucket_file(p)
        assert out["n_total"] == 2  # bad line skipped


class TestBuildComparison:

    def _row(self, tmp_path, name, n_miss, n_halluc):
        recs = [_wrong(all_gold_retrieved=False) for _ in range(n_miss)]
        recs += [_wrong(gold_answer="paris", predicted_answer="london")
                 for _ in range(n_halluc)]
        p = tmp_path / name
        p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
        return p

    def test_comparison_md_and_shift(self, tmp_path):
        rag = self._row(tmp_path, "row2_rag_no_agent.jsonl", 5, 5)
        ver = self._row(tmp_path, "row4_verifier.jsonl", 4, 6)
        md, summary = build_comparison([(rag, "RAG (no agent)"),
                                        (ver, "+Verifier")])
        assert "RAG (no agent)" in summary["rows"]
        assert "+Verifier" in summary["rows"]
        # Shift section present
        assert "does the agent change the failure" in md
        # RAG retrieval-fault 50%, Verifier 40%
        assert summary["rows"]["RAG (no agent)"]["retrieval_fault"] == 5
        assert summary["rows"]["+Verifier"]["retrieval_fault"] == 4

    def test_empty_inputs(self, tmp_path):
        md, summary = build_comparison([(tmp_path / "missing.jsonl", "X")])
        assert summary["rows"] == {}
