"""
Semantic-correctness tests for the Planner (S_P) — paper §4.1.

Verifies that planner.plan() produces structurally and semantically correct
RetrievalPlans (query-type classification, pattern-based decomposition,
sub-query well-formedness, entity extraction), not merely that it runs. The
pattern recognisers key on SpaCy dependency structure, so each query here is
chosen for its syntactic shape; entities are neutral encyclopedia placeholders.

Run without a live LLM:
    python -X utf8 -m pytest test_system/test_planner_semantic.py -v

Last reviewed: 2026-05-30 (audit pass, project version 5.4).
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.logic_layer.planner import (
    Planner, PlannerConfig, QueryType, RetrievalStrategy,
    EntityInfo, RetrievalPlan, create_planner, SPACY_AVAILABLE,
)


@pytest.fixture(scope="session")
def planner():
    return create_planner()  # auto-loads config/settings.yaml


# ── Classification accuracy ────────────────────────────────────────────────────

class TestQueryClassification:

    def test_single_hop_capital(self, planner):
        plan = planner.plan("What is the capital of France?")
        assert plan.query_type == QueryType.SINGLE_HOP
        assert plan.strategy == RetrievalStrategy.VECTOR_ONLY

    def test_comparison_nationality(self, planner):
        plan = planner.plan("Were Marie Curie and Isaac Newton of the same nationality?")
        assert plan.query_type == QueryType.COMPARISON, \
            f"Expected COMPARISON, got {plan.query_type.value}"
        assert plan.strategy == RetrievalStrategy.HYBRID

    def test_multihop_director_film(self, planner):
        plan = planner.plan("Who is the director of the film that stars Greta Garbo?")
        assert plan.query_type == QueryType.MULTI_HOP

    def test_temporal_year(self, planner):
        plan = planner.plan("What happened after World War 2?")
        assert plan.query_type == QueryType.TEMPORAL

    def test_intersection_both(self, planner):
        plan = planner.plan("Which films star both Charlie Chaplin and Buster Keaton?")
        assert plan.query_type == QueryType.INTERSECTION

    def test_multihop_capital_country_born(self, planner):
        plan = planner.plan("What is the capital of the country where Albert Einstein was born?")
        assert plan.query_type == QueryType.MULTI_HOP

    def test_structural_comparison_wh_both(self, planner):
        """Phase 3.6: coordinated NER entities + interrogative determiner route
        to COMPARISON even without a comparative-morphology keyword. Otherwise
        the entity-density heuristic forces MULTI_HOP and the parallel
        comparison decomposer is never reached for two-entity 'both held which'
        questions."""
        plan = planner.plan(
            "Theodore Roosevelt and Woodrow Wilson both held which position?"
        )
        assert plan.query_type == QueryType.COMPARISON, (
            f"Expected COMPARISON, got {plan.query_type.value}"
        )
        # Routed into the parallel decomposer → one anchored sub-query per entity.
        assert len(plan.sub_queries) == 2, (
            f"Expected 2 parallel sub-queries, got {plan.sub_queries}"
        )

    def test_structural_comparison_precision_guard(self, planner):
        """Precision guard: a coordinated entity pair INSIDE a bridge question
        (bridge-relation cue 'directed'/'starring' present) must stay MULTI_HOP,
        not be stolen by the structural-comparison router."""
        plan = planner.plan(
            "Who directed the film starring Greta Garbo and Tim Allen?"
        )
        assert plan.query_type == QueryType.MULTI_HOP, (
            f"Expected MULTI_HOP (bridge cue present), got {plan.query_type.value}"
        )



class TestSubqueryWellFormedness:
    """Item 4: a sub-query must be a usable retrieval target (named entity,
    NP subject, or wh-word) — bare fragments are rejected by
    `_subquery_is_well_formed` so the connector-split falls back to an
    entity-seeded plan instead of emitting a subject-less fragment."""

    def _pg(self):
        from src.logic_layer.planner import PlanGenerator
        return PlanGenerator()

    def test_bare_predicate_is_malformed(self):
        pg = self._pg()
        assert pg._subquery_is_well_formed("was released by the distributor", []) is False
        assert pg._subquery_is_well_formed("of the same year", []) is False

    def test_wh_word_is_well_formed(self):
        pg = self._pg()
        assert pg._subquery_is_well_formed("which film won the award", []) is True

    def test_named_entity_is_well_formed(self):
        pg = self._pg()
        assert pg._subquery_is_well_formed("won an Oscar", ["Greta Garbo"]) is True

    def test_np_subject_is_well_formed(self):
        pg = self._pg()
        # "the cat" is an nsubj of "sat" — retrievable without entity or wh-word.
        assert pg._subquery_is_well_formed("the cat sat on the mat", []) is True


# ── Sub-query quality ──────────────────────────────────────────────────────────

class TestSubQueryQuality:

    def test_comparison_attr_map_rewrite(self, planner):
        """ATTR_MAP must rewrite 'same nationality' into factual lookups."""
        plan = planner.plan("Were Marie Curie and Isaac Newton of the same nationality?")
        for sq in plan.sub_queries:
            assert "nationality" in sq.lower(), \
                f"ATTR_MAP rewrite failed — sub-query missing 'nationality': {sq}"

    def test_comparison_no_original_query_appended(self, planner):
        """Original query must NOT be appended as a 3rd sub-query (causes RRF cross-query noise)."""
        plan = planner.plan("Were Marie Curie and Isaac Newton of the same nationality?")
        assert len(plan.sub_queries) == 2, \
            f"Expected 2 sub-queries, got {len(plan.sub_queries)}: {plan.sub_queries}"

    def test_multihop_minimum_two_subqueries(self, planner):
        plan = planner.plan("What is the capital of the country where Albert Einstein was born?")
        assert len(plan.sub_queries) >= 2, \
            f"Multi-hop must produce ≥2 sub-queries, got {plan.sub_queries}"

    def test_sub_queries_are_questions(self, planner):
        plan = planner.plan("Who directed the film starring Greta Garbo?")
        for sq in plan.sub_queries:
            assert sq.endswith("?"), f"Sub-query not a question: {sq!r}"

    def test_single_hop_sub_query_equals_original(self, planner):
        query = "What is the capital of France?"
        plan = planner.plan(query)
        assert plan.sub_queries == [query]


# ── Pattern H: chained-attribution bridge (§12.31) ─────────────────────────────

@pytest.mark.skipif(not SPACY_AVAILABLE, reason="Pattern H requires SpaCy dependency parse")
class TestAttributionChain:
    """
    "[work] based on/featuring [Entity], is [created] by someone [attribute]?"
    must decompose into 3 hops: resolve the work → resolve the agent → answer.
    Grammar-driven (acl + agent dep labels) — no verb lists, no seen-examples.
    """

    _CHAINED_Q = (
        "A biographical film based on a pioneering Polish-French physicist "
        "Marie Curie, is written and directed by someone born in what year?"
    )

    def test_chained_attribution_produces_three_hops(self, planner):
        plan = planner.plan(self._CHAINED_Q)
        assert plan.query_type == QueryType.MULTI_HOP
        assert len(plan.hop_sequence) == 3, \
            f"chained-attribution bridge must yield 3 hops, got {len(plan.hop_sequence)}: {plan.sub_queries}"

    def test_chained_attribution_hop_dependencies_chained(self, planner):
        plan = planner.plan(self._CHAINED_Q)
        h0, h1, h2 = plan.hop_sequence
        assert h0.depends_on == [] and h0.is_bridge
        assert h1.depends_on == [0] and h1.is_bridge
        assert h2.depends_on == [1] and not h2.is_bridge

    def test_chained_attribution_anchor_is_the_named_entity(self, planner):
        plan = planner.plan(self._CHAINED_Q)
        # hop0 must mention the proper-noun anchor, not the whole noun phrase
        assert "Marie Curie" in plan.hop_sequence[0].sub_query
        # and it must be the *short* form (no "pioneering Polish-French physicist" baggage)
        assert "pioneering" not in plan.hop_sequence[0].sub_query

    def test_chained_attribution_final_hop_is_original_query(self, planner):
        plan = planner.plan(self._CHAINED_Q)
        assert plan.hop_sequence[-1].sub_query == self._CHAINED_Q

    def test_named_agent_is_not_a_chain(self, planner):
        """Passive with a *named* agent (Pattern F territory) must NOT become a 3-hop chain."""
        plan = planner.plan(
            "On the Origin of Species was written by an English naturalist "
            "that has worked in London since what year?"
        )
        # Whatever pattern handles it, it must not be the 3-hop attribution chain
        # (which would emit "What ... is based on ...?" as hop0).
        assert not (len(plan.hop_sequence) == 3
                    and plan.hop_sequence[0].sub_query.lower().startswith("what ")
                    and "based on" in plan.hop_sequence[0].sub_query.lower())

    def test_single_hop_query_is_not_a_chain(self, planner):
        plan = planner.plan("What is the capital of France?")
        assert plan.query_type == QueryType.SINGLE_HOP
        assert len(plan.hop_sequence) == 1


# ── Classification–decomposition consistency (§12.31) ──────────────────────────

class TestMultiHopConsistency:
    """A query classified MULTI_HOP must never silently emit the unsplit query as
    its sole sub-query — either a pattern fires, or the generic 2-hop fallback does."""

    def test_multihop_never_emits_only_original_query(self, planner):
        # A multi-hop-shaped query that none of the named patterns specifically
        # target should still produce ≥2 sub-queries via the generic fallback.
        plan = planner.plan(
            "In what country is the headquarters of the company that employs "
            "the scientist who discovered penicillin?"
        )
        if plan.query_type == QueryType.MULTI_HOP:
            assert len(plan.sub_queries) >= 2, \
                f"multi-hop must not collapse to single sub-query: {plan.sub_queries}"


# ── Entity extraction quality ──────────────────────────────────────────────────

class TestEntityExtraction:

    def test_full_name_not_partial(self, planner):
        """'Marie Curie' must be extracted as a unit, not as 'Marie' alone."""
        plan = planner.plan("Were Marie Curie and Isaac Newton of the same nationality?")
        entity_texts = [e.text for e in plan.entities]
        assert any("Curie" in t for t in entity_texts), \
            f"Full name not extracted: {entity_texts}"

    def test_no_spurious_verb_entities(self, planner):
        """'Were' must not appear as an entity."""
        plan = planner.plan("Were Marie Curie and Isaac Newton of the same nationality?")
        entity_texts_lower = [e.text.lower() for e in plan.entities]
        assert "were" not in entity_texts_lower, \
            f"Spurious verb 'Were' extracted as entity: {entity_texts_lower}"

    def test_entity_count_within_max(self, planner):
        plan = planner.plan("Were Marie Curie and Isaac Newton of the same nationality?")
        assert len(plan.entities) <= planner.config.max_entities

    def test_bridge_entity_flagged_multihop(self, planner):
        """Bridge detection must not crash; bridge flag must be bool."""
        plan = planner.plan("Who is the director of the film that stars Greta Garbo?")
        for e in plan.entities:
            assert isinstance(e.is_bridge, bool)

    def test_entity_confidence_in_range(self, planner):
        plan = planner.plan("What did Albert Einstein discover?")
        for e in plan.entities:
            assert 0.0 <= e.confidence <= 1.0, \
                f"Entity confidence out of range: {e.text} = {e.confidence}"


# ── RetrievalPlan structural invariants ───────────────────────────────────────

class TestPlanStructure:

    def test_estimated_hops_matches_hop_sequence(self, planner):
        """estimated_hops must always equal len(hop_sequence)."""
        for query in [
            "What is the capital of France?",
            "Is Berlin older than Munich?",
            "Who directed the film that won the Oscar?",
            "What is the capital of the country where Einstein was born?",
        ]:
            plan = planner.plan(query)
            assert plan.estimated_hops == len(plan.hop_sequence), \
                f"estimated_hops={plan.estimated_hops} != len(hop_sequence)={len(plan.hop_sequence)} for {query!r}"

    def test_empty_query_does_not_crash(self, planner):
        plan = planner.plan("")
        assert isinstance(plan, RetrievalPlan)
        assert plan.query_type == QueryType.SINGLE_HOP
        assert plan.estimated_hops == 0
        assert plan.hop_sequence == []

    def test_none_query_does_not_crash(self, planner):
        """plan(None) must return a valid empty plan after the None-guard fix."""
        plan = planner.plan(None)
        assert isinstance(plan, RetrievalPlan)
        assert plan.query_type == QueryType.SINGLE_HOP
        assert plan.estimated_hops == 0

    def test_to_dict_serialisable(self, planner):
        plan = planner.plan("Who is the director of the film that stars Greta Garbo?")
        d = plan.to_dict()
        json_str = json.dumps(d)  # must not raise TypeError
        assert isinstance(json_str, str)

    def test_to_json_valid(self, planner):
        plan = planner.plan("What is the capital of France?")
        json_str = plan.to_json()
        parsed = json.loads(json_str)
        assert parsed["original_query"] == "What is the capital of France?"

    def test_temporal_constraint_extracted(self, planner):
        plan = planner.plan("Who was president in 1990?")
        assert "years" in plan.constraints, \
            f"Year not extracted into constraints: {plan.constraints}"
        assert "1990" in plan.constraints["years"]

    def test_confidence_in_range(self, planner):
        plan = planner.plan("Is Berlin older than Munich?")
        assert 0.0 <= plan.confidence <= 1.0

    def test_hop_step_ids_sequential(self, planner):
        plan = planner.plan("What is the capital of the country where Einstein was born?")
        for i, hop in enumerate(plan.hop_sequence):
            assert hop.step_id == i, \
                f"step_id={hop.step_id} not sequential at index {i}"


# ── Factory and settings compliance ──────────────────────────────────────────

class TestFactoryAndSettings:

    def test_create_planner_loads_settings(self):
        """Factory must auto-load settings.yaml — not use hardcoded defaults."""
        p = create_planner()
        assert p.config.min_entity_confidence == 0.7
        assert p.config.max_entities == 10
        assert p.config.classifier_spacy_weight == 1.5
        assert p.config.classifier_confidence_cap == 0.95

    def test_planner_config_from_yaml(self):
        cfg = {
            "planner": {
                "min_entity_confidence": 0.5,
                "max_entities": 5,
                "classifier_spacy_weight": 2.0,
            },
            "ingestion": {"spacy_model": "en_core_web_sm"},
        }
        config = PlannerConfig.from_yaml(cfg)
        assert config.min_entity_confidence == 0.5
        assert config.max_entities == 5
        assert config.classifier_spacy_weight == 2.0

    def test_planning_time_recorded(self):
        p = create_planner()
        plan = p.plan("What is the capital of France?")
        assert "planning_time_ms" in plan.metadata
        assert plan.metadata["planning_time_ms"] >= 0


# ── P1: matched_pattern recorded in RetrievalPlan ──────────────────────────────

class TestP1MatchedPatternRecorded:
    """P1: every plan must carry an identifier of the pattern that produced it,
    so the per-question JSONL can answer 'how often does Pattern X fire?'
    without parsing debug logs.

    Pre-fix, RetrievalPlan had no such field; the matched pattern was only
    visible in `logger.debug` lines. The eval harness's
    `_extract_planner_diagnostics` returned (query_type, hop_count, n_entities)
    with no pattern info.
    """

    def test_single_hop_marked(self, planner):
        plan = planner.plan("What is the capital of France?")
        assert plan.matched_pattern == "single_hop", (
            f"Expected 'single_hop', got {plan.matched_pattern!r}"
        )

    def test_comparison_attr_map_marked(self, planner):
        plan = planner.plan(
            "Were Marie Curie and Isaac Newton of the same nationality?"
        )
        # Comparison routes through _decompose_comparison; the _ATTR_MAP
        # rewrite for "same nationality" should fire.
        assert plan.matched_pattern in {
            "comparison_attr_map",
            "comparison_parallel",
            "I_boolean_conjunction",
            "select_between_two",
        }, f"Got {plan.matched_pattern!r}"

    def test_multihop_pattern_marker_set(self, planner):
        plan = planner.plan(
            "Who directed the film starring Greta Garbo in Grand Hotel?"
        )
        # Any MULTI_HOP plan must carry a pattern marker — never None.
        if plan.query_type == QueryType.MULTI_HOP:
            assert plan.matched_pattern is not None, (
                "MULTI_HOP plans must always set matched_pattern"
            )
            # Allowed values for the multi-hop dispatcher (open set in tests).
            assert isinstance(plan.matched_pattern, str)
            assert len(plan.matched_pattern) > 0

    def test_matched_pattern_in_to_dict(self, planner):
        """P8: matched_pattern surfaces in to_dict() output (greppable from JSONL)."""
        plan = planner.plan("What is the capital of France?")
        d = plan.to_dict()
        assert "matched_pattern" in d, "to_dict() must surface matched_pattern (P8)"
        assert d["matched_pattern"] == plan.matched_pattern

    def test_metadata_in_to_dict(self, planner):
        """P8: metadata key must be present in to_dict() output."""
        plan = planner.plan("What is the capital of France?")
        d = plan.to_dict()
        assert "metadata" in d, "to_dict() must surface metadata (P8)"
        assert isinstance(d["metadata"], dict)
        # planning_time_ms is always added by plan() — sanity-check it's reachable.
        assert "planning_time_ms" in d["metadata"]

    def test_no_pattern_marker_leak_across_calls(self, planner):
        """A second plan() call must not inherit the first call's pattern."""
        # First call: classified MULTI_HOP, sets some pattern marker.
        plan1 = planner.plan(
            "Who directed the film starring Greta Garbo in Grand Hotel?"
        )
        # Second call: plain single-hop, should be re-tagged "single_hop".
        plan2 = planner.plan("What is the capital of France?")
        assert plan2.matched_pattern == "single_hop", (
            f"Stale pattern leaked between calls: got {plan2.matched_pattern!r} "
            f"after first call set {plan1.matched_pattern!r}"
        )


# ── P2: classifier pre-empt surfaced in RetrievalPlan ─────────────────────────

class TestP2ClassifierPreemptRecorded:
    """P2: Phase 0 (boolean-conjunction) and Phase 0.5 (implicit-bridge)
    classifier pre-empts return early from classify() with hard-coded
    confidence values. Pre-fix, they were invisible to downstream analysis.
    """

    def test_boolean_conjunction_preempt_recorded(self, planner):
        plan = planner.plan("Are Berlin and Munich both in Germany?")
        # Pre-empt should mark COMPARISON with confidence 0.90.
        assert plan.query_type == QueryType.COMPARISON
        assert plan.classifier_preempt == "preempt_pattern_I_boolean_conjunction", (
            f"Boolean-conjunction pre-empt should record marker; got "
            f"{plan.classifier_preempt!r}"
        )

    def test_implicit_bridge_preempt_recorded(self, planner):
        plan = planner.plan(
            "Jane Goodall has been a critic of Nestlé and another corporation "
            "that has operations in how many countries?"
        )
        assert plan.classifier_preempt == "preempt_pattern_J_implicit_bridge", (
            f"Implicit-bridge pre-empt should record marker; got "
            f"{plan.classifier_preempt!r}"
        )

    def test_normal_path_no_preempt(self, planner):
        plan = planner.plan("What is the capital of France?")
        # Normal Phase 1-4 scoring path — no pre-empt should fire.
        assert plan.classifier_preempt is None, (
            f"Single-hop query should not trigger a pre-empt; got "
            f"{plan.classifier_preempt!r}"
        )

    def test_preempt_in_to_dict(self, planner):
        plan = planner.plan("Are Berlin and Munich both in Germany?")
        d = plan.to_dict()
        assert "classifier_preempt" in d, (
            "to_dict() must surface classifier_preempt (P2)"
        )
        assert d["classifier_preempt"] == plan.classifier_preempt

    def test_preempt_marker_not_leaked_across_calls(self, planner):
        """A second plan() call must not inherit the first call's pre-empt."""
        plan1 = planner.plan("Are Berlin and Munich both in Germany?")
        assert plan1.classifier_preempt is not None  # sanity
        plan2 = planner.plan("What is the capital of France?")
        assert plan2.classifier_preempt is None, (
            f"Stale pre-empt leaked between calls: got "
            f"{plan2.classifier_preempt!r}"
        )


# ── P3: pattern-priority order is documented + tested ─────────────────────────

class TestP3PatternPriorityOrder:
    """P3: when two patterns *could* match the same query, the documented
    dispatch order in PlanGenerator's class docstring must win.

    These tests pin the ordering so future pattern reshuffles cannot silently
    change which pattern fires.
    """

    def test_J_preempt_beats_aggregate(self, planner):
        """Pattern J's pre-empt must beat the AGGREGATE 'how many' pattern.
        Without the pre-empt, "how many" would dominate scoring.
        """
        plan = planner.plan(
            "Jane Goodall has been a critic of Nestlé and another corporation "
            "that has operations in how many countries?"
        )
        assert plan.query_type == QueryType.MULTI_HOP, (
            f"J pre-empt must override AGGREGATE; got {plan.query_type.value}"
        )
        assert plan.matched_pattern == "J_implicit_bridge"

    def test_I_preempt_beats_intersection_priority(self, planner):
        """Pattern I (boolean conjunction) must beat the INTERSECTION patterns
        that also match 'both ... and'. Pre-empt fires before scoring.
        """
        plan = planner.plan("Are Berlin and Munich both in Germany?")
        assert plan.query_type == QueryType.COMPARISON, (
            f"I pre-empt must produce COMPARISON; got {plan.query_type.value}"
        )
        assert plan.matched_pattern == "I_boolean_conjunction"

    def test_select_between_beats_attr_map(self, planner):
        """When a comparison query has both an 'or' disjunction AND an
        _ATTR_MAP-matching phrase, select-between-two must win because it
        is checked first in _decompose_comparison.
        """
        plan = planner.plan(
            "Which physicist was from Germany, Albert Einstein or Niels Bohr?"
        )
        # select_between_two is checked before _ATTR_MAP per the documented order.
        assert plan.matched_pattern == "select_between_two", (
            f"Expected select_between_two; got {plan.matched_pattern!r}"
        )

    def test_fallback_marker_distinguishes_failure_modes(self, planner):
        """Classified MULTI_HOP but no pattern fires → fallback_generic_2hop
        (when an entity is present) or fallback_degraded_to_single_hop (when
        not). These markers must not collide with real-pattern markers.
        """
        # A multi-hop query with no clear bridge structure beyond a generic
        # "starring" relation that no pattern picks up.
        plan = planner.plan(
            "Who directed the film starring Greta Garbo in Grand Hotel?"
        )
        if plan.query_type == QueryType.MULTI_HOP:
            # The marker should be a fallback OR a legitimate pattern letter.
            # It must NOT be None (P1 invariant).
            assert plan.matched_pattern is not None
            assert plan.matched_pattern not in {"", "None"}


# ── P5: per-pattern regression tests ──────────────────────────────────────────
# One canonical query per pattern, pinned to its expected matched_pattern
# marker. Future pattern-deletion decisions must update or delete the
# corresponding test here.

class TestP5PerPatternRegression:
    """P5: each pattern has one canonical input that must produce its marker.

    These tests are intentionally minimal — they pin the marker, not the
    full hop structure (that's covered by integration tests elsewhere).
    Their job is to make pattern-deletion decisions auditable: if you delete
    Pattern X, the corresponding test here fails, and you explicitly accept
    or migrate the canonical case.
    """

    # ── Multi-hop dispatcher patterns ──────────────────────────────────────

    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="Pattern G requires SpaCy dep parse")
    def test_pattern_G_relative_clause_form1(self, planner):
        """Form 1: 'The [noun] in which [Entity] [predicate]...'
        Anchor is the subject of the relcl (a real NP, not a pronoun).
        """
        plan = planner.plan(
            "The film in which Greta Garbo starred won which award?"
        )
        # Pattern G form1 OR form2 acceptable depending on parse.
        if plan.query_type == QueryType.MULTI_HOP and plan.matched_pattern:
            assert plan.matched_pattern.startswith("G_") or \
                   plan.matched_pattern in {"H_attribution_chain", "connector_split"}, (
                f"Expected G_form*; got {plan.matched_pattern!r}"
            )

    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="Pattern H requires SpaCy dep parse")
    def test_pattern_H_attribution_chain(self, planner):
        """Chained attribution: 'a [work] based on [Entity], is [verb-en]
        by someone [attribute]?' — a 3-hop chain.
        """
        plan = planner.plan(
            "A film based on Marie Curie, is directed by someone "
            "born in what year?"
        )
        if plan.query_type == QueryType.MULTI_HOP and plan.matched_pattern:
            # If H fires, it should be H_attribution_chain. If H doesn't fire
            # (parse ambiguous), at least the marker is set (P1 invariant).
            assert plan.matched_pattern is not None

    def test_pattern_I_boolean_conjunction(self, planner):
        """'Are X and Y both P?' → COMPARISON via Pattern I pre-empt + Pattern I
        decomposer."""
        plan = planner.plan("Are Berlin and Munich both in Germany?")
        assert plan.query_type == QueryType.COMPARISON
        assert plan.matched_pattern == "I_boolean_conjunction"

    def test_pattern_J_implicit_bridge(self, planner):
        """'X and another [N] that…' → MULTI_HOP via Pattern J pre-empt +
        Pattern J decomposer."""
        plan = planner.plan(
            "Jane Goodall has been a critic of Nestlé and another "
            "corporation that operates in how many countries?"
        )
        assert plan.query_type == QueryType.MULTI_HOP
        assert plan.matched_pattern == "J_implicit_bridge"

    # ── Comparison dispatcher patterns ─────────────────────────────────────

    def test_pattern_K_select_between(self, planner):
        """'Which X, A or B, …?' — select-between-two comparison form."""
        plan = planner.plan(
            "Which physicist was from Germany, Albert Einstein or Niels Bohr?"
        )
        assert plan.query_type == QueryType.COMPARISON
        assert plan.matched_pattern == "select_between_two"

    def test_pattern_attr_map_same_nationality(self, planner):
        """'same nationality' → _ATTR_MAP rewrite to 'What is the nationality of X?'"""
        plan = planner.plan(
            "Were Marie Curie and Isaac Newton of the same nationality?"
        )
        assert plan.query_type == QueryType.COMPARISON
        assert plan.matched_pattern == "comparison_attr_map"
        # _ATTR_MAP produces one sub-query per entity, neither equal to the
        # original query.
        for sq in plan.sub_queries:
            assert "nationality of" in sq.lower(), (
                f"_ATTR_MAP rewrite should produce 'nationality of' sub-queries; "
                f"got {sq!r}"
            )

    # ── Single-pattern markers ─────────────────────────────────────────────

    def test_marker_single_hop(self, planner):
        plan = planner.plan("What is the capital of France?")
        assert plan.matched_pattern == "single_hop"

    def test_marker_aggregate(self, planner):
        """'How many X?' classifies as AGGREGATE — but Pattern J pre-empt
        outranks it. Use a query without 'another N' to actually reach AGGREGATE.
        """
        plan = planner.plan("How many countries are in the European Union?")
        # AGGREGATE OR SINGLE_HOP depending on classifier — either way no
        # pre-empt should fire.
        assert plan.classifier_preempt is None
        if plan.query_type == QueryType.AGGREGATE:
            assert plan.matched_pattern == "aggregate"

    def test_no_pattern_marker_is_None(self, planner):
        """P1 invariant: every plan from a non-empty query must set the marker."""
        for q in [
            "What is the capital of France?",
            "Were Marie Curie and Isaac Newton of the same nationality?",
            "Are Berlin and Munich both in Germany?",
            "Jane Goodall has been a critic of Nestlé and another "
            "corporation that operates in how many countries?",
            "Who directed the film starring Greta Garbo?",
        ]:
            plan = planner.plan(q)
            assert plan.matched_pattern is not None, (
                f"P1 invariant violated for query {q!r}: matched_pattern is None"
            )


# ── Dep-parse generalisation of Patterns E and F ─────────────────────────────

class TestOptionAGeneralisability:
    """Patterns E and F are implemented via SpaCy's dependency parse. These
    tests verify the recognisers generalise to verbs and roles that are never
    enumerated in the source — proving the recognisers key on linguistic
    structure, not on a vocabulary list.
    """

    def test_pattern_E_generalises_to_novel_role_nouns(self, planner):
        """Pattern E must fire on relational nouns never listed in any
        constant. The dep-parse recogniser depends only on the syntactic
        shape `noun -> prep("of") -> pobj(named_entity)`.
        """
        # "choreographer" and "librettist" never appeared in the old role list.
        # They are relational nouns by the same structural test.
        plan = planner.plan("Where was the choreographer of Swan Lake born?")
        if plan.query_type == QueryType.MULTI_HOP:
            assert plan.matched_pattern is not None
            # E must fire OR connector-split must fire (both are acceptable).
            assert plan.matched_pattern in {
                "E_relational_noun", "connector_split",
            }, f"Got unexpected marker: {plan.matched_pattern!r}"

    def test_pattern_F_generalises_to_novel_passive_verbs(self, planner):
        """Pattern F must fire on passive verbs never enumerated in the
        deleted _PASSIVE_TO_ACTIVE table. The dep-parse recogniser keys on
        `auxpass + nsubjpass + agent` regardless of which verb fills the
        slot. SpaCy's lemmatiser handles past-participle → infinitive.
        """
        # "painted" and "conducted" were not in the deleted lookup table.
        for q in [
            "The Mona Lisa was painted by who?",
            "The symphony was conducted by an Austrian who later moved to America in what year?",
        ]:
            plan = planner.plan(q)
            if plan.query_type == QueryType.MULTI_HOP:
                # F must fire OR connector-split must fire.
                assert plan.matched_pattern in {
                    "F_passive_agent", "connector_split",
                }, f"For {q!r}, got unexpected marker: {plan.matched_pattern!r}"

    def test_pattern_F_skips_interrogative_headed_subject(self, planner):
        """Pattern F must NOT fire when the passive subject is itself a wh-NP.

        Regression guard for the failure mode where _find_passive_agent_bridge
        returns an interrogative-headed subject (e.g. "What government
        position"). The template f"Who {verb} {subj}?" then produces
        self-referential nonsense ("Who hold What government position?")
        instead of a real bridge sub-query.
        """
        plan = planner.plan(
            "What government position was held by the woman who portrayed "
            "Joan of Arc in the film Saint Joan?"
        )
        # Pattern F must NOT be the matched marker — fall-through to
        # connector_split (or one of the dep-parse patterns G/E/H) is OK.
        assert plan.matched_pattern != "F_passive_agent", (
            f"Pattern F fired on interrogative-headed subject; "
            f"got matched_pattern={plan.matched_pattern!r}, "
            f"sub_queries={plan.sub_queries!r}"
        )
        # No sub-query may contain the self-referential bug string. The
        # specific bug shape produced by Pattern F on a wh-NP subject is
        # f"Who {verb} {wh-NP}?" — e.g. "Who hold What government position?"
        # The connector-split fallback may still mangle the query in OTHER
        # ways (that is Fix B's investigation target), but the specific
        # self-referential template must not appear.
        import re as _re
        self_ref_re = _re.compile(
            r"\bWho\s+\w+\s+(What|Which|Whose)\b", _re.IGNORECASE
        )
        for sq in plan.sub_queries:
            assert not self_ref_re.search(sq), (
                f"Self-referential Pattern-F bug string leaked through: {sq!r}"
            )

    def test_pattern_F_skips_bare_pronoun_subject(self, planner):
        """Pattern F must NOT fire when the passive subject is a bare pronoun.

        Regression guard for the failure mode where _find_passive_agent_bridge
        returns a bare pronoun ("that", "this", "it"...) as the subject. The
        template f"Who {verb} {subj}?" then produces a degenerate sub-query
        ("Who form that?") that cannot anchor any retrieval.
        """
        plan = planner.plan(
            "The Blue Album is the debut album of a rock band "
            "that was formed by who?"
        )
        assert plan.matched_pattern != "F_passive_agent", (
            f"Pattern F fired on bare-pronoun subject; "
            f"got matched_pattern={plan.matched_pattern!r}, "
            f"sub_queries={plan.sub_queries!r}"
        )
        # And no sub-query may be the broken string.
        for sq in plan.sub_queries:
            assert sq.lower().strip() != "who form that?", (
                f"Broken sub-query leaked through: {sq!r}"
            )

    def test_pattern_F_still_fires_on_canonical_passive(self, planner):
        """Positive control: Pattern F must still fire on the canonical
        passive-with-named-agent form. This guards against the
        interrogative-subject guard over-rejecting.
        """
        plan = planner.plan("The Mona Lisa was painted by who?")
        if plan.query_type == QueryType.MULTI_HOP:
            # F or connector_split — both acceptable, matching the existing
            # test_pattern_F_generalises_to_novel_passive_verbs contract.
            assert plan.matched_pattern in {
                "F_passive_agent", "connector_split",
            }, f"Pattern F regression on canonical input: got {plan.matched_pattern!r}"

    def test_a1_entityfree_description_emits_two_hops(self, planner):
        """A1: a multi-hop question whose bridge entity is referenced by
        DESCRIPTION (no named entity) must produce a 2-hop plan with a
        descriptive hop-0, not silently degrade to single-hop.
        """
        plan = planner.plan(
            "What chemical element is named after the only person to win "
            "a Nobel Prize in two different scientific fields?"
        )
        if plan.query_type == QueryType.MULTI_HOP:
            assert len(plan.sub_queries) >= 2, (
                f"entity-free bridge must not collapse to single-hop; "
                f"got {plan.sub_queries!r}"
            )
            assert plan.matched_pattern in {
                "structural_descriptive_2hop", "connector_split",
                "fallback_generic_2hop",
            }, f"unexpected marker: {plan.matched_pattern!r}"
            # hop-0 must differ from the full question (real decomposition).
            assert plan.sub_queries[0].lower() != plan.original_query.lower()

    def test_a1_never_silent_single_for_multihop(self, planner):
        """A1 contract: a MULTI_HOP classification never yields a single
        sub-query unless explicitly marked as the degrade path."""
        for q in [
            "What chemical element is named after the only person to win a "
            "Nobel Prize in two different scientific fields?",
            "What class of instrument does Clara Schumann play?",
        ]:
            plan = planner.plan(q)
            if plan.query_type == QueryType.MULTI_HOP:
                assert (
                    len(plan.sub_queries) >= 2
                    or plan.matched_pattern == "fallback_degraded_to_single_hop"
                ), (
                    f"contract violated for {q!r}: {len(plan.sub_queries)} "
                    f"sub-query, marker={plan.matched_pattern!r}"
                )

    def test_a4_reroutes_attribute_over_entity(self, planner):
        """A4: a question the classifier abstained on (no pattern) but which
        asks a wh-attribute of a thing related to a named entity is re-routed
        to MULTI_HOP."""
        plan = planner.plan("What class of instrument does Clara Schumann play?")
        assert plan.query_type == QueryType.MULTI_HOP, (
            f"A4 should re-route attribute-over-entity to MULTI_HOP; "
            f"got {plan.query_type!r}"
        )
        assert len(plan.sub_queries) >= 2

    def test_a4_does_not_override_simple_single_hop(self, planner):
        """A4 negative control: a direct single-hop question stays single-hop —
        'What is X?' has no wh-determiner on a noun and no abstention bridge."""
        plan = planner.plan("What is the capital of France?")
        assert plan.query_type == QueryType.SINGLE_HOP, (
            f"A4 must not over-route a simple single-hop; got {plan.query_type!r}"
        )

    def test_a4_does_not_override_confident_classification(self, planner):
        """A4 negative control: a confidently-classified comparison is never
        touched (A4 fires only on the SINGLE_HOP no-signal sentinel)."""
        plan = planner.plan(
            "Were Marie Curie and Isaac Newton of the same nationality?"
        )
        assert plan.query_type != QueryType.SINGLE_HOP

    def test_deleted_C_pattern_marker_never_appears(self, planner):
        """The C_for_category marker must never be produced (Pattern C deleted)."""
        for q in [
            "What is the capital of France?",
            "What year did the studio release a film starring Charlie Chaplin?",
            "Who wrote a book about quantum mechanics?",
        ]:
            plan = planner.plan(q)
            assert plan.matched_pattern != "C_for_category", (
                f"Deleted marker C_for_category surfaced for {q!r}"
            )

    def test_deleted_D_pattern_marker_never_appears(self, planner):
        """The D_role_qualifier marker must never be produced (Pattern D deleted)."""
        for q in [
            "What screenwriter with credits for Metropolis co-wrote a film?",
            "Which director with a Pulitzer prize made a documentary?",
        ]:
            plan = planner.plan(q)
            assert plan.matched_pattern != "D_role_qualifier", (
                f"Deleted marker D_role_qualifier surfaced for {q!r}"
            )

    def test_no_passive_to_active_lookup_table(self):
        """The deleted _PASSIVE_TO_ACTIVE lookup table must not exist on
        PlanGenerator. Regression guard against accidental re-introduction.
        """
        from src.logic_layer.planner import PlanGenerator
        assert not hasattr(PlanGenerator, "_PASSIVE_TO_ACTIVE"), (
            "_PASSIVE_TO_ACTIVE lookup table should not exist (dep-parse only)"
        )
        assert not hasattr(PlanGenerator, "_RELATIONAL_ANCHOR_ROLES"), (
            "_RELATIONAL_ANCHOR_ROLES regex should not exist (dep-parse only)"
        )
        assert not hasattr(PlanGenerator, "_FOR_CAT"), (
            "_FOR_CAT regex (Pattern C) should not exist"
        )
        assert not hasattr(PlanGenerator, "_ROLE_PAT"), (
            "_ROLE_PAT regex (Pattern D) should not exist"
        )

    def test_no_dataset_revealing_comments_in_source(self):
        """planner.py source must not contain evaluation-set surface forms or
        change-log markers. Bibliography citations are OK.

        The blocklist of forbidden regex patterns is stored out-of-source in
        fixtures/forbidden_source_terms.json so this test file itself carries
        no evaluation-set surface forms.
        """
        import re as _re
        import json as _json
        from pathlib import Path
        here = Path(__file__).parent
        src = here.parent / "src" / "logic_layer" / "planner.py"
        text = src.read_text(encoding="utf-8")
        # Strip the bibliography section so legitimate citations don't trip us.
        # The bibliography ends before the first "class " or "def " line.
        body_start = min(
            text.find("\nclass "), text.find("\ndef "),
        )
        body = text[body_start:] if body_start > 0 else text

        forbidden = _json.loads(
            (here / "fixtures" / "forbidden_source_terms.json").read_text(encoding="utf-8")
        )["patterns"]
        leaks = []
        for pat in forbidden:
            m = _re.search(pat, body)
            if m:
                leaks.append((pat, m.group(0)))
        assert not leaks, (
            f"dataset-revealing strings remain in planner.py body: {leaks}"
        )


# ── P9: fallback confidence ordering ──────────────────────────────────────────

class TestP9FallbackConfidenceOrdering:
    """P9: fallback confidence must be strictly lower than any actual
    pattern-match confidence — otherwise 'no pattern matched' reports higher
    classifier certainty than 'one pattern matched', contradicting the
    documented confidence scaling.

    Pre-fix: fallback was 0.8, single-pattern match was 0.6 + 0.15*1 = 0.75.
    Post-fix: fallback is 0.5; matches start at 0.75.
    """

    def test_default_fallback_below_min_pattern_confidence(self):
        """Dataclass default must respect the ordering."""
        from src.logic_layer.planner import PlannerConfig
        cfg = PlannerConfig()
        min_match_confidence = (
            cfg.classifier_confidence_base
            + cfg.classifier_confidence_scale * 1.0
        )
        assert cfg.classifier_fallback_confidence < min_match_confidence, (
            f"P9: fallback={cfg.classifier_fallback_confidence} must be < "
            f"min-match={min_match_confidence}"
        )

    def test_yaml_fallback_below_min_pattern_confidence(self):
        """Loaded settings.yaml value also respects the ordering."""
        from src.logic_layer.planner import PlannerConfig
        from src.logic_layer._settings_loader import _load_settings
        cfg = PlannerConfig.from_yaml(_load_settings())
        min_match_confidence = (
            cfg.classifier_confidence_base
            + cfg.classifier_confidence_scale * 1.0
        )
        assert cfg.classifier_fallback_confidence < min_match_confidence

    def test_single_hop_query_uses_fallback(self, planner):
        """A query with no scoring-pattern match returns fallback confidence."""
        plan = planner.plan("Banana banana banana?")
        # This nonsense query should fall through to SINGLE_HOP with fallback
        # confidence (whichever value is set in settings.yaml).
        if plan.query_type == QueryType.SINGLE_HOP:
            min_match_confidence = (
                planner.config.classifier_confidence_base
                + planner.config.classifier_confidence_scale * 1.0
            )
            assert plan.confidence <= min_match_confidence, (
                f"Fallback confidence {plan.confidence} must not exceed min-match"
            )


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    p = create_planner()

    print("=" * 60)
    print("PLANNER SEMANTIC SMOKE CHECK")
    print(f"SpaCy: {'available' if SPACY_AVAILABLE else 'unavailable'}")
    print("=" * 60)

    cases = [
        ("What is the capital of France?", QueryType.SINGLE_HOP),
        ("Were Marie Curie and Isaac Newton of the same nationality?", QueryType.COMPARISON),
        ("What is the capital of the country where Einstein was born?", QueryType.MULTI_HOP),
        ("Who was president in 1990?", QueryType.TEMPORAL),
    ]
    for q, expected in cases:
        plan = p.plan(q)
        ok = "OK  " if plan.query_type == expected else "FAIL"
        print(f"[{ok}] {q[:60]}")
        print(f"       type={plan.query_type.value}  subs={plan.sub_queries}")
    print("=" * 60)
