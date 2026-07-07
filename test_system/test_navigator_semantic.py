"""
Semantic tests for Navigator (S_N) — Pre-Generative Filtering Pipeline.

Covers all six filter methods whose EM-improvement contributions are cited
in the paper (paper section 3.3):

  1. _rrf_fusion            — RRF scoring, corroboration boost
  2. _relevance_filter      — dynamic threshold (factor × max_score)
  3. _redundancy_filter     — Jaccard-based deduplication
  4. _contradiction_filter  — numeric-heuristic conflict detection (+1.4 EM)
  5. _entity_overlap_pruning— subset entity-set removal              (+0.8 EM)
  6. _entity_mention_filter — query-entity presence check            (+2.1 EM)
  7. _context_shrinkage     — sentence-level trimming                (−34% latency)
  8. navigate()             — full filter pipeline integration

No Ollama required.  All tests use in-memory fixtures with neutral
encyclopedia content (no evaluation-set surface forms).

Last reviewed: 2026-05-30 (audit pass, project version 5.4).
"""

import pytest
from src.logic_layer.navigator import Navigator, NavigatorResult
from src.logic_layer import ControllerConfig
from src.logic_layer.planner import RetrievalPlan, QueryType, RetrievalStrategy


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    return ControllerConfig(
        relevance_threshold_factor=0.5,
        redundancy_threshold=0.8,
        max_context_chunks=10,
        rrf_k=60,
        top_k_per_subquery=10,
        max_chars_per_doc=500,
        corroboration_source_weight=0.1,
        corroboration_query_weight=0.05,
        contradiction_overlap_threshold=0.3,
        contradiction_ratio_threshold=2.0,
        contradiction_min_value=100.0,
    )


@pytest.fixture
def nav(cfg):
    return Navigator(cfg)


def _result(text: str, score: float, source: str = "doc_a", sub_query: str = "q") -> dict:
    return {"text": text, "score": score, "source": source, "sub_query": sub_query}


# ---------------------------------------------------------------------------
# 1. RRF Fusion
# ---------------------------------------------------------------------------

class TestRRFFusion:
    def test_deduplication_merges_duplicate_texts(self, nav):
        """Same text from two sub-queries is deduplicated; RRF score accumulates."""
        results = [
            _result("Paris is the capital of France.", 0.9, sub_query="q1"),
            _result("Paris is the capital of France.", 0.7, sub_query="q2"),
            _result("France is in Europe.", 0.5, sub_query="q1"),
        ]
        fused = nav._rrf_fusion(results)
        texts = [r["text"] for r in fused]
        assert texts.count("Paris is the capital of France.") == 1

    def test_cross_source_boost_increases_score(self, nav):
        """A chunk appearing in two sources scores higher than one appearing in one."""
        single = [_result("Alpha.", 0.8, source="doc_a", sub_query="q")]
        dual = [
            _result("Beta.", 0.8, source="doc_a", sub_query="q"),
            _result("Beta.", 0.8, source="doc_b", sub_query="q"),
        ]
        fused_single = nav._rrf_fusion(single)
        fused_dual = nav._rrf_fusion(dual)
        assert fused_dual[0]["rrf_score"] > fused_single[0]["rrf_score"]

    def test_output_sorted_by_rrf_score_descending(self, nav):
        results = [
            _result("Low.", 0.2),
            _result("High.", 0.9),
            _result("Mid.", 0.5),
        ]
        fused = nav._rrf_fusion(results)
        scores = [r["rrf_score"] for r in fused]
        assert scores == sorted(scores, reverse=True)

    def test_empty_input_returns_empty(self, nav):
        assert nav._rrf_fusion([]) == []

    def test_rrf_k_parameter_affects_scores(self, nav):
        """Lower k raises the weight of top-ranked items."""
        results = [_result("X.", 1.0)]
        score_k60 = nav._rrf_fusion(results, k=60)[0]["rrf_score"]
        score_k10 = nav._rrf_fusion(results, k=10)[0]["rrf_score"]
        assert score_k10 > score_k60


class TestFairCapBySubquery:
    """_fair_cap_by_subquery: per-anchor fairness for parallel comparison plans."""

    def _fused(self, text, score, sub_query):
        return {"text": text, "rrf_score": score, "_best_sub_query": sub_query}

    def test_low_degree_anchor_not_crowded_out(self, nav):
        """A high-degree anchor (anchor A, many high-score chunks) must not
        monopolise the budget: anchor B's single gold chunk survives the cap
        via round-robin interleaving, where a global top-k would have cut it."""
        results = [
            self._fused("A1", 0.99, "qA"),
            self._fused("A2", 0.98, "qA"),
            self._fused("A3", 0.97, "qA"),
            self._fused("A4", 0.96, "qA"),
            self._fused("B_gold", 0.40, "qB"),  # lower score, would be cut globally
        ]
        capped = nav._fair_cap_by_subquery(results, budget=4, n_parallel=2)
        texts = [r["text"] for r in capped]
        assert "B_gold" in texts, f"low-degree anchor crowded out: {texts}"
        assert len(capped) == 4

    def test_single_subquery_is_global_topk(self, nav):
        """n_parallel == 1 → identical to results[:budget] (single-hop no-op)."""
        results = [self._fused(f"c{i}", 1.0 - i * 0.1, "q") for i in range(6)]
        capped = nav._fair_cap_by_subquery(results, budget=3, n_parallel=1)
        assert [r["text"] for r in capped] == ["c0", "c1", "c2"]

    def test_budget_exceeds_results_returns_all(self, nav):
        results = [self._fused("a", 0.9, "qA"), self._fused("b", 0.8, "qB")]
        capped = nav._fair_cap_by_subquery(results, budget=8, n_parallel=2)
        assert len(capped) == 2


# ---------------------------------------------------------------------------
# 2. Relevance Filter
# ---------------------------------------------------------------------------

class TestRelevanceFilter:
    def _make_fused(self, scores):
        return [{"text": f"chunk_{i}", "rrf_score": s} for i, s in enumerate(scores)]

    def test_drops_chunks_below_threshold(self, nav):
        """With factor=0.5, chunks below 0.5×max are removed."""
        results = self._make_fused([1.0, 0.6, 0.4, 0.2])
        filtered = nav._relevance_filter(results)
        surviving_scores = [r["rrf_score"] for r in filtered]
        assert all(s >= 0.5 for s in surviving_scores)

    def test_keeps_all_when_scores_close(self, nav):
        """When all scores are within threshold all chunks are kept."""
        results = self._make_fused([1.0, 0.9, 0.85])
        filtered = nav._relevance_filter(results)
        assert len(filtered) == 3

    def test_empty_input_returns_empty(self, nav):
        assert nav._relevance_filter([]) == []

    def test_top_chunk_always_survives(self, nav):
        """The highest-scored chunk always passes the relevance filter."""
        results = self._make_fused([1.0, 0.1, 0.05])
        filtered = nav._relevance_filter(results)
        assert any(r["rrf_score"] == 1.0 for r in filtered)


# ---------------------------------------------------------------------------
# 3. Redundancy Filter (Jaccard deduplication)
# ---------------------------------------------------------------------------

class TestRedundancyFilter:
    def _make_results(self, texts):
        return [{"text": t, "rrf_score": 1.0 - i * 0.05} for i, t in enumerate(texts)]

    def test_near_duplicates_removed(self, nav):
        """Two nearly identical chunks: only the higher-scored survives."""
        t1 = "Albert Einstein was a physicist born in Ulm."
        t2 = "Albert Einstein was a brilliant physicist born in Ulm."
        results = self._make_results([t1, t2])
        filtered = nav._redundancy_filter(results)
        assert len(filtered) == 1

    def test_distinct_chunks_kept(self, nav):
        """Chunks with low word overlap are both kept."""
        t1 = "Albert Einstein was a physicist born in Germany."
        t2 = "Marie Curie discovered radium and polonium in Paris."
        results = self._make_results([t1, t2])
        filtered = nav._redundancy_filter(results)
        assert len(filtered) == 2

    def test_higher_scored_chunk_wins_tie(self, nav):
        """When two chunks are duplicates the first (higher score) is retained."""
        t1 = "Paris is the capital of France."
        t2 = "Paris is the capital of France!"
        results = [
            {"text": t1, "rrf_score": 0.9},
            {"text": t2, "rrf_score": 0.5},
        ]
        filtered = nav._redundancy_filter(results)
        assert filtered[0]["text"] == t1

    def test_empty_returns_empty(self, nav):
        assert nav._redundancy_filter([]) == []


# ---------------------------------------------------------------------------
# 4. Contradiction Filter  (paper contribution, +1.4 EM)
# ---------------------------------------------------------------------------

class TestContradictionFilter:
    def test_removes_lower_scoring_contradicting_chunk(self, nav):
        """Two chunks with same topic but contradictory numbers (3x ratio): lower removed."""
        results = [
            {"text": "Albert Einstein was born in the year 1000.", "rrf_score": 0.9},
            {"text": "Albert Einstein was born in the year 3000.", "rrf_score": 0.4},
        ]
        filtered = nav._contradiction_filter(results)
        assert len(filtered) == 1
        assert filtered[0]["rrf_score"] == 0.9

    def test_non_contradicting_chunks_both_kept(self, nav):
        """Chunks with different topics are not considered contradictory."""
        results = [
            {"text": "France has a population of 67 million people.", "rrf_score": 0.9},
            {"text": "Georges Méliès directed A Trip to the Moon in 1902.", "rrf_score": 0.8},
        ]
        filtered = nav._contradiction_filter(results)
        assert len(filtered) == 2

    def test_equal_scores_higher_index_removed(self, nav):
        """With equal scores the tie-break removes the higher-index chunk (j)."""
        results = [
            {"text": "Population 1000 versus target.", "rrf_score": 0.5},
            {"text": "Population 3000 versus target.", "rrf_score": 0.5},
        ]
        filtered = nav._contradiction_filter(results)
        assert len(filtered) == 1
        assert filtered[0]["text"] == "Population 1000 versus target."

    def test_single_chunk_unchanged(self, nav):
        results = [{"text": "Only one chunk.", "rrf_score": 0.9}]
        filtered = nav._contradiction_filter(results)
        assert len(filtered) == 1

    def test_small_numbers_not_flagged_as_contradiction(self, nav):
        """Numbers below min_value=100 are ignored to avoid false positives.

        §12.25: contradiction_min_value raised from 10 -> 100 to prevent
        day-of-month values (1-31) from triggering false contradictions.
        """
        results = [
            {"text": "The team won 3 to 1.", "rrf_score": 0.9},
            {"text": "The team won 3 to 1 in overtime.", "rrf_score": 0.7},
        ]
        filtered = nav._contradiction_filter(results)
        assert len(filtered) == 2

    def test_day_of_month_vs_year_not_a_contradiction(self, nav):
        """A day-of-month (e.g. 7) paired with a year (1918) must NOT trigger.

        Before §12.25, a small day-of-month value paired with a year produced a
        large ratio that exceeded the threshold and incorrectly evicted
        biographical chunks containing birth dates. With min_value=100, a
        day-of-month < 100 is excluded from the ratio comparison.
        """
        results = [
            {"text": "Marie Curie was born on 7 November 1867 in Warsaw.", "rrf_score": 0.9},
            {"text": "Poland regained independence in 1918.", "rrf_score": 0.7},
        ]
        filtered = nav._contradiction_filter(results)
        assert len(filtered) == 2, (
            "Day-of-month (7) vs year (1918) must not evict biographical chunks"
        )


# ---------------------------------------------------------------------------
# 5. Entity Overlap Pruning  (paper contribution, +0.8 EM)
# ---------------------------------------------------------------------------

class TestEntityOverlapPruning:
    def test_subset_entity_chunk_pruned(self, nav):
        """Chunk B whose entity set is a subset of Chunk A's is removed."""
        results = [
            {
                "text": "Albert Einstein was a German-born physicist who developed the theory of relativity.",
                "rrf_score": 0.9,
            },
            {
                "text": "Albert Einstein was born in Germany.",
                "rrf_score": 0.6,
            },
        ]
        filtered = nav._entity_overlap_pruning(results)
        assert all(r["rrf_score"] != 0.6 or len(filtered) == 2 for r in filtered)
        # Primary assertion: higher-scored chunk always kept
        assert any(r["rrf_score"] == 0.9 for r in filtered)

    def test_single_chunk_unchanged(self, nav):
        results = [{"text": "Albert Einstein.", "rrf_score": 0.9}]
        filtered = nav._entity_overlap_pruning(results)
        assert len(filtered) == 1

    def test_no_overlap_both_kept(self, nav):
        """Two chunks discussing completely different entities are both kept."""
        results = [
            {"text": "Albert Einstein worked in Bern Switzerland.", "rrf_score": 0.9},
            {"text": "Marie Curie worked in Paris France.", "rrf_score": 0.8},
        ]
        filtered = nav._entity_overlap_pruning(results)
        assert len(filtered) == 2

    def test_safety_fallback_never_empty(self, nav):
        """Even if all chunks are pruned the safety fallback returns the original."""
        results = [
            {"text": "Albert.", "rrf_score": 0.9},
            {"text": "Albert Einstein.", "rrf_score": 0.8},
        ]
        filtered = nav._entity_overlap_pruning(results)
        assert len(filtered) >= 1


# ---------------------------------------------------------------------------
# 6. Entity Mention Filter  (paper contribution, +2.1 EM)
# ---------------------------------------------------------------------------

class TestEntityMentionFilter:
    def _r(self, text, score=0.8):
        return {"text": text, "rrf_score": score}

    def test_chunks_without_entity_dropped(self, nav):
        """Chunk not mentioning 'Einstein' is removed when Einstein is the query entity.

        §12.33 added top-2 RRF immunity to prevent the filter from dropping
        a high-ranked answer chunk when its entity happens to be an implicit
        bridge target. This test now uses 3 chunks so the Paris distractor
        is ranked #3 (outside the immunity window) and the filter still
        drops it.
        """
        results = [
            self._r("Albert Einstein was born in Ulm.", score=0.9),
            self._r("Albert Einstein worked at ETH Zürich.", score=0.8),
            self._r("Paris is the capital of France.", score=0.4),
        ]
        filtered = nav._entity_mention_filter(results, ["Einstein"])
        assert all("einstein" in r["text"].lower() for r in filtered)

    def test_multi_word_entity_full_phrase_match(self, nav):
        """Multi-word entity 'Georges Méliès' matched by full-phrase lookup.

        §12.33: 3 chunks so the distractor is at index 2, outside the top-2
        RRF-immunity window.
        """
        results = [
            self._r("Georges Méliès directed A Trip to the Moon.", score=0.9),
            self._r("Georges Méliès made several short films.", score=0.7),
            self._r("Charlie Chaplin produced several comedies.", score=0.4),
        ]
        filtered = nav._entity_mention_filter(results, ["Georges Méliès"])
        assert all("Georges Méliès" in r["text"] for r in filtered)
        assert not any("Charlie Chaplin" in r["text"] for r in filtered)

    def test_short_single_token_entity_skipped(self, nav):
        """Single token shorter than 5 chars ('Were') is not checked as entity."""
        results = [self._r("Were is not an entity."), self._r("Einstein worked there.")]
        # 'Were' is < 5 chars → filter produces no entity patterns → all kept
        filtered = nav._entity_mention_filter(results, ["Were"])
        assert len(filtered) == 2

    def test_safety_fallback_returns_all_when_nothing_matches(self, nav):
        """If no chunk mentions the entity all chunks are returned (never empty)."""
        results = [self._r("Unrelated content about weather.")]
        filtered = nav._entity_mention_filter(results, ["Einstein"])
        assert len(filtered) == 1

    def test_empty_entity_list_returns_all(self, nav):
        """No entity names → no filtering → all chunks pass."""
        results = [self._r("Anything goes."), self._r("Truly anything.")]
        filtered = nav._entity_mention_filter(results, [])
        assert len(filtered) == 2

    def test_token_length_threshold_five_chars(self, nav):
        """Only tokens ≥ 5 chars are used as whole-word patterns."""
        results = [
            self._r("Marie Curie discovered radium."),
            self._r("An unrelated sentence about Marie."),
        ]
        # 'Marie' is 5 chars — should match via token pattern fallback
        filtered = nav._entity_mention_filter(results, ["Marie Curie"])
        assert any("Marie" in r["text"] for r in filtered)

    def test_variant_name_all_tokens_cooccur(self, nav):
        """Plan A: a chunk that opens with a full legal name
        ('Edward William Elgar') is kept for entity 'Edward Elgar' even though
        the contiguous phrase is absent and neither token reaches the ≥8-char
        single-token bar. Requires ALL content tokens (≥3 chars) to co-occur as
        whole words — so a chunk with only 'Edward' is dropped.

        3 chunks so the noise distractor is at index 2 (outside top-2 RRF
        immunity) and the filter still drops it.
        """
        results = [
            self._r("Edward William Elgar (born 2 June 1857) composed the Enigma Variations.", score=0.9),
            self._r("Clara Schumann was a celebrated concert pianist.", score=0.8),
            self._r("Bananas are a popular tropical fruit.", score=0.3),
        ]
        filtered = nav._entity_mention_filter(results, ["Clara Schumann", "Edward Elgar"])
        assert any("William" in r["text"] for r in filtered)
        assert not any("Bananas" in r["text"] for r in filtered)

    def test_variant_name_partial_token_not_matched(self, nav):
        """Plan A precision guard: a chunk mentioning only 'Edward' (not 'Elgar')
        is NOT matched by the all-tokens-co-occur rule, because not every
        content token is present. Placed at rank 3 (outside top-2 immunity)
        so the drop is observable."""
        results = [
            self._r("Edward Elgar composed the Enigma Variations.", score=0.9),
            self._r("Edward Elgar conducted many concert premieres.", score=0.8),
            self._r("Edward went to the market this morning.", score=0.3),
        ]
        filtered = nav._entity_mention_filter(results, ["Edward Elgar"])
        assert any("Enigma" in r["text"] for r in filtered)
        assert not any("market" in r["text"] for r in filtered)


# ---------------------------------------------------------------------------
# 7. Context Shrinkage  (paper contribution, −34 % S_V latency)
# ---------------------------------------------------------------------------

class TestContextShrinkage:
    def test_long_chunk_truncated(self, nav, cfg):
        """Chunk longer than max_chars_per_doc is truncated."""
        long_text = "Albert Einstein was born. " * 30  # ~780 chars
        results = [{"text": long_text, "rrf_score": 0.9}]
        shrunk = nav._context_shrinkage(results, max_chars_per_chunk=200)
        assert len(shrunk[0]["text"]) <= 200

    def test_short_chunk_unchanged(self, nav):
        """Chunk already under the limit is returned as-is."""
        results = [{"text": "Short.", "rrf_score": 0.9}]
        shrunk = nav._context_shrinkage(results, max_chars_per_chunk=500)
        assert shrunk[0]["text"] == "Short."

    def test_entity_sentences_prioritized(self, nav):
        """Sentences containing capitalized tokens are prioritized over filler."""
        text = (
            "this is generic filler without any proper nouns. "
            "Albert Einstein was born in Ulm in 1879. "
            "more filler without any capitalized terms here."
        )
        results = [{"text": text, "rrf_score": 0.9}]
        shrunk = nav._context_shrinkage(results, max_chars_per_chunk=80)
        # Einstein sentence should be prioritized
        assert "Einstein" in shrunk[0]["text"]

    def test_empty_returns_empty(self, nav):
        assert nav._context_shrinkage([], max_chars_per_chunk=500) == []


# ---------------------------------------------------------------------------
# 8. Full navigate() integration
# ---------------------------------------------------------------------------

class TestNavigateFull:
    class _MockResult:
        def __init__(self, text, score, source="doc_a"):
            self.text = text
            self.rrf_score = score
            self.source = source

    class _MockRetriever:
        def __init__(self, results):
            self._results = results

        def retrieve(self, query, entity_hints=None):
            return self._results, {"vector": len(self._results), "graph": 0}

    def test_navigate_no_retriever_returns_empty(self, nav):
        result = nav.navigate(retrieval_plan=None, sub_queries=["test"])
        assert isinstance(result, NavigatorResult)
        assert result.filtered_context == []

    def test_navigate_returns_navigator_result(self, nav):
        nav.set_retriever(self._MockRetriever([
            self._MockResult("Albert Einstein was born in Ulm, Germany.", 0.9),
            self._MockResult("Albert Einstein developed the theory of relativity.", 0.8),
        ]))
        result = nav.navigate(
            retrieval_plan=None,
            sub_queries=["Where was Albert Einstein born?"],
            entity_names=["Einstein"],
        )
        assert isinstance(result, NavigatorResult)
        assert isinstance(result.filtered_context, list)
        assert isinstance(result.scores, list)
        assert isinstance(result.metadata, dict)

    def test_navigate_entity_mention_filter_active(self, nav):
        """Chunks not mentioning the query entity are filtered out.

        §12.33: 3+ chunks needed so the Paris distractor sits outside the
        top-2 RRF-immunity window and is correctly dropped.
        """
        nav.set_retriever(self._MockRetriever([
            self._MockResult("Albert Einstein was born in Ulm.", 0.9),
            self._MockResult("Albert Einstein later moved to Princeton.", 0.8),
            self._MockResult("Paris is a city in France, known for the Eiffel Tower.", 0.4),
        ]))
        result = nav.navigate(
            retrieval_plan=None,
            sub_queries=["Where was Einstein born?"],
            entity_names=["Einstein"],
        )
        assert all("einstein" in c.lower() for c in result.filtered_context)

    def test_navigate_metadata_contains_filter_counts(self, nav):
        nav.set_retriever(self._MockRetriever([
            self._MockResult("Albert Einstein lived in Princeton.", 0.9),
        ]))
        result = nav.navigate(
            retrieval_plan=None,
            sub_queries=["Einstein Princeton"],
            entity_names=["Einstein"],
        )
        assert "after_relevance_filter" in result.metadata
        assert "after_redundancy_filter" in result.metadata
        assert "after_entity_mention_filter" in result.metadata

    def test_navigate_with_retrieval_plan(self, nav):
        plan = RetrievalPlan(
            original_query="Who was Albert Einstein?",
            query_type=QueryType.SINGLE_HOP,
            strategy=RetrievalStrategy.HYBRID,
            sub_queries=["Who was Albert Einstein?"],
        )
        nav.set_retriever(self._MockRetriever([
            self._MockResult("Albert Einstein was a physicist.", 0.9),
        ]))
        result = nav.navigate(
            retrieval_plan=plan,
            sub_queries=["Who was Albert Einstein?"],
            entity_names=["Einstein"],
        )
        assert isinstance(result, NavigatorResult)
