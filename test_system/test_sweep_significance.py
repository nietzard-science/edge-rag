"""
Unit tests for the sweep significance/CI module (roadmap #2/#9).

Deterministic, no model calls.

Last reviewed: 2026-06-09.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.thesis_evaluations.sweep_significance import (
    build_markdown,
    summarize_sweep,
)


def _write(path: Path, em_flags):
    """Write a per-question JSONL with given exact_match booleans."""
    with open(path, "w", encoding="utf-8") as fh:
        for i, em in enumerate(em_flags):
            fh.write(json.dumps({"question_id": f"q{i}", "exact_match": bool(em),
                                 "f1_score": 1.0 if em else 0.0}) + "\n")


class TestSummarizeSweep:

    def test_baseline_flagged_and_delta_computed(self, tmp_path):
        # baseline 50% (5/10), variant 80% (8/10), same questions
        _write(tmp_path / "a.jsonl", [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        _write(tmp_path / "b.jsonl", [1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        summary = summarize_sweep(
            tmp_path, "a.jsonl",
            [("a.jsonl", "A"), ("b.jsonl", "B")], metric="EM")
        assert summary["A"]["is_baseline"] is True
        assert "delta" not in summary["A"]
        assert summary["B"]["is_baseline"] is False
        assert summary["B"]["point"] == pytest.approx(0.8)
        assert summary["B"]["delta"] == pytest.approx(0.3)  # +30pp
        assert "p_value" in summary["B"]

    def test_missing_file_skipped(self, tmp_path):
        _write(tmp_path / "a.jsonl", [1, 0])
        summary = summarize_sweep(
            tmp_path, "a.jsonl",
            [("a.jsonl", "A"), ("missing.jsonl", "B")], metric="EM")
        assert "A" in summary and "B" not in summary

    def test_ci_brackets_point(self, tmp_path):
        _write(tmp_path / "a.jsonl", [1, 0, 1, 0, 1, 0, 1, 0])
        s = summarize_sweep(tmp_path, "a.jsonl", [("a.jsonl", "A")], "EM")
        r = s["A"]
        assert r["ci_low"] <= r["point"] <= r["ci_high"]


class TestBuildMarkdown:

    def test_table_has_ci_and_p_columns(self, tmp_path):
        _write(tmp_path / "a.jsonl", [1, 1, 0, 0])
        _write(tmp_path / "b.jsonl", [1, 1, 1, 0])
        s = summarize_sweep(tmp_path, "a.jsonl",
                            [("a.jsonl", "A"), ("b.jsonl", "B")], "EM")
        md = build_markdown(s, "EM", "A", "Test")
        assert "95% CI" in md
        assert "| p |" in md
        assert "baseline" in md

    def test_nonsignificant_reading_note(self, tmp_path):
        # Identical files → delta 0 → never significant → "no ... significant" note
        _write(tmp_path / "a.jsonl", [1, 0, 1, 0])
        _write(tmp_path / "b.jsonl", [1, 0, 1, 0])
        s = summarize_sweep(tmp_path, "a.jsonl",
                            [("a.jsonl", "A"), ("b.jsonl", "B")], "EM")
        md = build_markdown(s, "EM", "A", "Test")
        assert "no pairwise EM difference" in md

    def test_significant_reading_note(self, tmp_path):
        # Strongly separated, larger n → significant → "significant ... difference" note
        _write(tmp_path / "a.jsonl", [0] * 50)
        _write(tmp_path / "b.jsonl", [1] * 50)
        s = summarize_sweep(tmp_path, "a.jsonl",
                            [("a.jsonl", "A"), ("b.jsonl", "B")], "EM")
        md = build_markdown(s, "EM", "A", "Test")
        assert "significant EM difference" in md
