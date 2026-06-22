"""
Semantic-correctness tests for src/logic_layer/controller.py (AgenticController).

These lock the *contract* of the bridge-entity helpers used by
AgentPipeline._iterative_navigate, not exact entity sets (which depend on the
proper-noun regex). They run as pure functions — no LLM, no Ollama, no stores —
so they belong in the fast CI subset.

Contract under test:
  - _extract_bridge_entities never returns a Hop-1 entity (the `exclude` set)
  - it abstains (returns []) when no candidate has query proximity (C4 floor)
  - it never returns a role/title token as a PERSON bridge
  - it caps at TOP_K_BRIDGES
  - _rewrite_hop_query_with_bridges injects a bridge, and never double-injects
  - the query-keyword hoist preserves scoring (precomputed == auto-derived)

Last reviewed: 2026-06-13.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logic_layer.controller import AgenticController as AC


class TestBridgeExtraction:

    def test_excludes_hop1_entity(self):
        # Hop-1 resolved "David Lynch"; a Hop-2 bridge must not be him (FM-1).
        chunks = ["David Lynch directed the film Blue Velvet. "
                  "It starred Kyle MacLachlan and Isabella Rossellini."]
        out = AC._extract_bridge_entities(
            chunks, exclude=["David Lynch"], query="Who starred in the film?")
        assert all("lynch" not in o.lower() for o in out)
        assert "David Lynch" not in out

    def test_abstains_when_no_query_proximity(self):
        # No candidate near the query keywords → return [] (C4 floor), not a
        # confidently-wrong bridge that would misdirect Hop-N retrieval.
        chunks = ["Unrelated text about geology, sediment, and mineral strata."]
        out = AC._extract_bridge_entities(
            chunks, exclude=[], query="Who directed Blue Velvet?")
        assert out == []

    def test_role_token_not_returned_as_person(self):
        # FM-2: a role/title noun must not surface as a PERSON bridge.
        chunks = ["The director was acclaimed. The producer and the "
                  "investigator disagreed about the film."]
        out = AC._extract_bridge_entities(
            chunks, exclude=[], query="Who is the director who won the award?")
        flat = {tok for o in out for tok in o.lower().split()}
        assert not (flat & AC._ROLE_TOKENS)

    def test_topk_cap(self):
        chunks = ["Alice Brown met Bob Carter near Carol Davis, Dan Evans, "
                  "and Frank Green at the venue."]
        out = AC._extract_bridge_entities(
            chunks, exclude=[], query="Who met whom at the venue?")
        assert len(out) <= AC.TOP_K_BRIDGES

    def test_empty_inputs_safe(self):
        assert AC._extract_bridge_entities([], exclude=[], query="") == []
        assert AC._extract_bridge_entities(["text"], exclude=[], query="") == [] \
            or isinstance(AC._extract_bridge_entities(["text"], [], ""), list)

    def test_deterministic(self):
        chunks = ["David Lynch directed Blue Velvet starring Kyle MacLachlan."]
        a = AC._extract_bridge_entities(chunks, ["David Lynch"], "Who starred?")
        b = AC._extract_bridge_entities(chunks, ["David Lynch"], "Who starred?")
        assert a == b


class TestQueryKeywordHoist:
    """The perf refactor (hoisting _query_keywords out of the per-candidate
    loop) must not change scores."""

    def test_precomputed_equals_auto_derived(self):
        chunk = ("David Lynch directed Blue Velvet. It starred "
                 "Kyle MacLachlan and Isabella Rossellini.")
        query = "Who starred in the film?"
        kw = AC._query_keywords(query)
        s_pre = AC._score_bridge_candidate(
            "Kyle MacLachlan", chunk, query, "PERSON", 0, query_keywords=kw)
        s_auto = AC._score_bridge_candidate(
            "Kyle MacLachlan", chunk, query, "PERSON", 0)
        assert abs(s_pre - s_auto) < 1e-12

    def test_query_keywords_drops_stopwords(self):
        kw = AC._query_keywords("Who is the director of the film?")
        assert "the" not in kw and "of" not in kw and "is" not in kw
        assert "director" in kw and "film" in kw


class TestRewriteHopQuery:

    def test_injects_bridge(self):
        sq = "What year was the director born?"
        out = AC._rewrite_hop_query_with_bridges(sq, ["David Lynch"])
        assert "David Lynch" in out
        assert out != sq

    def test_no_double_injection(self):
        sq = "What year was David Lynch born?"
        assert AC._rewrite_hop_query_with_bridges(sq, ["David Lynch"]) == sq

    def test_empty_bridges_noop(self):
        sq = "What year was the director born?"
        assert AC._rewrite_hop_query_with_bridges(sq, []) == sq

    def test_caps_injected_bridges(self):
        sq = "Where were they born?"
        out = AC._rewrite_hop_query_with_bridges(
            sq, ["A One", "B Two", "C Three", "D Four", "E Five"])
        # only TOP_K_BRIDGES names injected
        assert out.count(",") <= AC.TOP_K_BRIDGES - 1 + sq.count(",")


class TestExpectedTypeDetection:

    def test_who_is_person(self):
        assert AC._detect_expected_type("Who directed the film?") == "PERSON"

    def test_where_is_gpe(self):
        assert AC._detect_expected_type("Where was he born?") == "GPE"

    def test_the_role_who_is_person(self):
        assert AC._detect_expected_type(
            "In which year was the actress who played Bobbi born?") == "PERSON"

    def test_no_interrogative_returns_none(self):
        assert AC._detect_expected_type("The film was released in 1986.") is None
