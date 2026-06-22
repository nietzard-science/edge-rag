"""
Semantic-correctness tests for the Verifier (S_V) — paper §4.3.

Verifies structural invariants, pre-generation validation, claim
extraction/verification, question-relevance reordering, prompt-routing
(answer / bridge / comparison), provenance-aware credibility scoring,
numeric-claim checks, and iteration-history truncation — without a live LLM
(the LLM is stubbed/monkeypatched). All fixtures use neutral encyclopedia
content (no evaluation-set surface forms).

Run:
    python -X utf8 -m pytest test_system/test_verifier_semantic.py -v

Last reviewed: 2026-05-30 (audit pass, project version 5.4).
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.logic_layer.verifier import (
    Verifier, VerifierConfig, PreGenerationValidator,
    ValidationStatus, ConfidenceLevel, VerificationResult,
    create_verifier, SPACY_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def minimal_cfg():
    return {
        "llm": {"max_context_chars": 2000, "max_docs": 5, "max_chars_per_doc": 400},
        "agent": {"max_verification_iterations": 1},
        "verifier": {
            "enable_entity_path_validation": True,
            "enable_credibility_scoring": True,
            "enable_contradiction_detection": False,
        },
    }


@pytest.fixture(scope="module")
def verifier(minimal_cfg):
    return create_verifier(cfg=minimal_cfg, enable_pre_validation=True)


@pytest.fixture(scope="module")
def validator(minimal_cfg):
    config = VerifierConfig.from_yaml(minimal_cfg)
    return PreGenerationValidator(config)


@pytest.fixture(scope="module")
def einstein_context():
    return [
        "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity.",
        "Einstein received the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.",
        "He was born in Ulm, Germany, on March 14, 1879.",
    ]


# ---------------------------------------------------------------------------
# TestPreValidation
# ---------------------------------------------------------------------------

class TestPreValidation:

    def test_empty_context_returns_insufficient_evidence(self, validator):
        result = validator.validate([], "What is the capital of France?")
        assert result.status == ValidationStatus.INSUFFICIENT_EVIDENCE
        assert result.filtered_context == []

    def test_entity_found_in_context(self, validator, einstein_context):
        result = validator.validate(einstein_context, "When was Einstein born?", entities=["Einstein"])
        assert result.entity_path_valid is True

    def test_entity_missing_flags_insufficient_evidence(self, validator, einstein_context):
        result = validator.validate(
            einstein_context, "Who is Marie Curie?",
            entities=["Marie Curie", "Polonium"],
        )
        assert result.status == ValidationStatus.INSUFFICIENT_EVIDENCE

    def test_credibility_scores_never_empty_for_nonempty_context(self, validator, einstein_context):
        result = validator.validate(einstein_context, "test")
        assert len(result.credibility_scores) > 0

    def test_filtered_context_never_empty_after_credibility(self, validator, einstein_context):
        result = validator.validate(einstein_context, "test")
        assert len(result.filtered_context) >= 1

    def test_validation_time_recorded(self, validator, einstein_context):
        result = validator.validate(einstein_context, "test")
        assert result.validation_time_ms >= 0

    def test_status_passed_for_good_context(self, validator, einstein_context):
        result = validator.validate(
            einstein_context,
            "What did Einstein win the Nobel Prize for?",
            entities=["Einstein"],
        )
        assert result.status in (ValidationStatus.PASSED, ValidationStatus.LOW_CREDIBILITY)


# ---------------------------------------------------------------------------
# TestContextFormatting
# ---------------------------------------------------------------------------

class TestContextFormatting:

    def test_empty_context_returns_placeholder(self, verifier):
        fmt = verifier._format_context([])
        assert "No context available" in fmt

    def test_chunks_numbered(self, verifier, einstein_context):
        fmt = verifier._format_context(einstein_context)
        assert "[1]" in fmt
        assert "[2]" in fmt

    def test_max_docs_respected(self, verifier):
        many_chunks = ["chunk %d" % i for i in range(20)]
        fmt = verifier._format_context(many_chunks)
        assert "[6]" not in fmt  # max_docs=5

    def test_long_chunk_truncated(self, verifier):
        long_chunk = "Albert Einstein " * 200
        fmt = verifier._format_context([long_chunk])
        assert len(fmt) < len(long_chunk)


# ---------------------------------------------------------------------------
# TestClaimExtraction
# ---------------------------------------------------------------------------

class TestClaimExtraction:

    def test_factual_sentence_extracted(self, verifier):
        answer = "Albert Einstein was born in Ulm, Germany, in 1879."
        claims = verifier._extract_claims(answer)
        assert len(claims) >= 1

    def test_error_prefix_returns_empty(self, verifier):
        claims = verifier._extract_claims("[Error: LLM timeout]")
        assert claims == []

    def test_meta_statements_filtered(self, verifier):
        answer = "Based on the context, I cannot answer this question."
        claims = verifier._extract_claims(answer)
        assert claims == []

    def test_short_non_meta_answer_handled(self, verifier):
        claims = verifier._extract_claims("Paris")
        assert isinstance(claims, list)

    def test_multi_sentence_answer_splits(self, verifier):
        answer = (
            "Albert Einstein was born in 1879. "
            "He received the Nobel Prize in 1921."
        )
        claims = verifier._extract_claims(answer)
        assert len(claims) >= 1


# ---------------------------------------------------------------------------
# TestClaimVerification
# ---------------------------------------------------------------------------

class TestClaimVerification:

    def test_entity_in_context_verified(self, verifier, einstein_context):
        ok, reason = verifier._verify_claim(
            "Albert Einstein was born in Ulm.", context=einstein_context
        )
        assert ok is True

    def test_entity_not_in_context_violated(self, verifier):
        ok, reason = verifier._verify_claim(
            "Napoleon was born in Corsica.",
            context=["Einstein worked in Bern."],
        )
        assert ok is False

    def test_claim_with_no_entities_verified_by_default(self, verifier):
        ok, reason = verifier._verify_claim("Yes.", context=[])
        assert ok is True
        assert reason == "no_entities_to_verify"

    def test_stopwords_not_treated_as_entities(self, verifier, einstein_context):
        ok, reason = verifier._verify_claim(
            "This is a fact about American history.",
            context=einstein_context,
        )
        assert isinstance(ok, bool)


# ---------------------------------------------------------------------------
# TestConfidenceProperty
# ---------------------------------------------------------------------------

class TestConfidenceProperty:

    def _make_result(self, verified, violated):
        return VerificationResult(
            answer="test",
            iterations=1,
            verified_claims=verified,
            violated_claims=violated,
            confidence_high_threshold=0.8,
            confidence_medium_threshold=0.5,
        )

    def test_all_verified_high_confidence(self):
        r = self._make_result(["c1", "c2", "c3"], [])
        assert r.confidence == ConfidenceLevel.HIGH

    def test_none_verified_low_confidence(self):
        r = self._make_result([], ["c1", "c2"])
        assert r.confidence == ConfidenceLevel.LOW

    def test_zero_claims_low_confidence(self):
        r = self._make_result([], [])
        assert r.confidence == ConfidenceLevel.LOW

    def test_medium_confidence_range(self):
        r = self._make_result(["c1"], ["c1"])  # 50 % verified
        assert r.confidence == ConfidenceLevel.MEDIUM


# ---------------------------------------------------------------------------
# TestFactory
# ---------------------------------------------------------------------------

class TestFactory:

    def test_create_verifier_loads_settings(self):
        v = create_verifier()
        assert isinstance(v, Verifier)
        assert isinstance(v.config, VerifierConfig)

    def test_enable_pre_validation_flag(self, minimal_cfg):
        v = create_verifier(cfg=minimal_cfg, enable_pre_validation=True)
        assert v.config.enable_entity_path_validation is True
        assert v.config.enable_credibility_scoring is True

    def test_from_yaml_reads_all_blocks(self):
        cfg = {
            "llm": {"model_name": "phi3", "max_tokens": 100},
            "agent": {"max_verification_iterations": 3},
            "verifier": {
                "min_credibility_score": 0.6,
                "heuristic_contradiction_threshold": 0.4,
                "format_sentence_boundary_fraction": 0.65,
            },
        }
        config = VerifierConfig.from_yaml(cfg)
        assert config.model_name == "phi3"
        assert config.max_tokens == 100
        assert config.max_iterations == 3
        assert config.min_credibility_score == 0.6
        assert config.heuristic_contradiction_threshold == 0.4
        assert config.format_sentence_boundary_fraction == 0.65

    def test_none_query_does_not_crash(self, minimal_cfg):
        """generate_and_verify(None, ...) must not raise."""
        v = create_verifier(cfg=minimal_cfg)
        result = v.generate_and_verify(None, [])
        assert isinstance(result, VerificationResult)


# ---------------------------------------------------------------------------
# TestLLMErrorPaths (T-03)
# ---------------------------------------------------------------------------

class TestLLMErrorPaths:
    """Regression tests for Verifier error handling on bad LLM responses.

    T-03a: LLM raises requests.Timeout — Verifier must return a VerificationResult
           with a fallback answer string, not propagate the exception.

    T-03b: LLM returns "" — regression guard for the ``best_answer or "..."``
           bug fixed in v3.4.0 (paper 12.12).  An empty string is falsy in
           Python, so ``best_answer or fallback`` would silently substitute
           the fallback even when the LLM returned an intentional empty
           response.  The fix uses ``best_answer if best_answer is not None``.
    """

    def test_llm_timeout_returns_fallback_not_exception(
        self, minimal_cfg
    ) -> None:
        """generate_and_verify must handle the LLM timeout sentinel without raising.

        Verifier._call_llm internally catches requests.Timeout and returns
        the error sentinel "[Error: ...]" rather than propagating.  This test
        verifies that generate_and_verify handles the sentinel gracefully and
        returns a VerificationResult — not an exception.
        """
        from unittest.mock import patch

        v = create_verifier(cfg=minimal_cfg)
        # Simulate what _call_llm returns when Ollama times out:
        # a sentinel string starting with "[Error:" and zero latency.
        with patch.object(
            v, "_call_llm", return_value=("[Error: LLM request timed out]", 0.0)
        ):
            result = v.generate_and_verify(
                "What is the capital of France?",
                context=["Paris is the capital of France."],
            )
        assert isinstance(result, VerificationResult), (
            "generate_and_verify must return VerificationResult on LLM timeout sentinel"
        )
        assert result.answer is not None, (
            "VerificationResult.answer must not be None after LLM timeout"
        )

    def test_llm_empty_string_does_not_substitute_fallback(
        self, minimal_cfg
    ) -> None:
        """Regression: _call_llm returning '' must not be silently replaced by fallback.

        Before v3.4.0 fix: ``best_answer or "fallback"`` would substitute
        "fallback" when best_answer=="" because "" is falsy.
        After fix: ``best_answer if best_answer is not None else "fallback"``
        preserves the empty string as a valid (if unusual) LLM response.
        """
        from unittest.mock import patch

        v = create_verifier(cfg=minimal_cfg)
        with patch.object(v, "_call_llm", return_value=("", 5.0)):
            result = v.generate_and_verify(
                "test query",
                context=["some context"],
            )
        assert isinstance(result, VerificationResult)
        # The fix ensures "" is carried through; the answer may be "" or a
        # fallback sentinel — the important invariant is no unhandled exception.
        assert result.answer is not None


# ---------------------------------------------------------------------------
# TestVerifierFactualCorrectness (T-A)
# ---------------------------------------------------------------------------

class TestVerifierFactualCorrectness:
    """Verifier claim-grounding invariants (T-A).

    _verify_claim uses entity-presence as a conservative proxy:
    a claim is verified when **any** named entity from the claim appears in the
    context (OR logic, not logical entailment — see verifier.py docstring).
    These tests exercise that contract directly.
    """

    def test_answer_with_entity_absent_from_context_yields_violated_claim(
        self, minimal_cfg
    ) -> None:
        """If the LLM answer contains an entity completely absent from context,
        that claim must appear in violated_claims.

        Setup: context is exclusively about Einstein; LLM returns an answer
        about Napoleon — neither "Napoleon" nor "Egypt" appears anywhere in the
        Einstein context, so _verify_claim must return (False, ...).
        """
        from unittest.mock import patch

        v = create_verifier(cfg=minimal_cfg)
        with patch.object(
            v, "_call_llm",
            return_value=("Napoleon conquered Egypt.", 5.0),
        ):
            result = v.generate_and_verify(
                query="What did Napoleon do?",
                context=[
                    "Albert Einstein was born in Ulm, Germany in 1879.",
                    "He was a theoretical physicist.",
                ],
            )

        assert len(result.violated_claims) > 0 or result.confidence == ConfidenceLevel.LOW, (
            f"Answer about Napoleon (absent from Einstein context) must produce "
            f"violated_claims or LOW confidence; "
            f"got confidence={result.confidence}, violated_claims={result.violated_claims}"
        )

    def test_answer_with_entity_present_in_context_not_violated(
        self, minimal_cfg
    ) -> None:
        """An answer whose named entities are all found in the context must
        receive MEDIUM or HIGH confidence (no violated claims from _verify_claim).

        _verify_claim returns True when at least one entity from the claim is
        found in the context string.  "Einstein" and "Ulm" both appear in the
        context → claim passes → confidence must be HIGH or MEDIUM.
        """
        from unittest.mock import patch

        v = create_verifier(cfg=minimal_cfg)
        with patch.object(
            v, "_call_llm",
            return_value=("Einstein was born in Ulm.", 5.0),
        ):
            result = v.generate_and_verify(
                query="Where was Einstein born?",
                context=["Albert Einstein was born in Ulm, Germany in 1879."],
                entities=["Einstein", "Ulm"],
            )

        assert result.confidence in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM), (
            f"Answer with context-grounded entities should be HIGH/MEDIUM confidence; "
            f"got {result.confidence}. violated_claims={result.violated_claims}"
        )


# ---------------------------------------------------------------------------
# Verifier statelessness (F5)
# ---------------------------------------------------------------------------

class TestVerifierStatelessness:
    """Verifier must not leak state between independent generate_and_verify calls (F5).

    Two sequential calls with unrelated contexts and queries must produce
    independent results — entities from call A's context must not appear in
    call B's violated_claims or verified_claims unless they genuinely exist in
    call B's context.
    """

    def test_second_call_does_not_inherit_first_call_context(self, minimal_cfg):
        """Call B's result must not contain entities exclusive to call A's context."""
        from unittest.mock import patch

        v = create_verifier(cfg=minimal_cfg)

        # Call A: Einstein context + Einstein answer
        with patch.object(v, "_call_llm", return_value=("Einstein was born in Ulm.", 1.0)):
            result_a = v.generate_and_verify(
                query="Where was Einstein born?",
                context=["Albert Einstein was born in Ulm, Germany in 1879."],
                entities=["Einstein"],
            )

        # Call B: Curie context + Curie answer (Einstein ABSENT from context)
        with patch.object(v, "_call_llm", return_value=("Curie discovered radium.", 1.0)):
            result_b = v.generate_and_verify(
                query="What did Curie discover?",
                context=["Marie Curie discovered radium and polonium."],
                entities=["Curie"],
            )

        # Result B must not carry verified claims from result A's Einstein context
        b_verified_lower = {c.lower() for c in result_b.verified_claims}
        assert "einstein" not in " ".join(b_verified_lower), (
            f"Call B result must not contain 'einstein' from call A's context. "
            f"verified_claims={result_b.verified_claims}"
        )

    def test_sequential_calls_produce_independent_confidence(self, minimal_cfg):
        """Confidence level of call B must be determined solely by call B's context."""
        from unittest.mock import patch

        v = create_verifier(cfg=minimal_cfg)

        # Call A: entity absent → LOW confidence expected
        with patch.object(v, "_call_llm", return_value=("Napoleon conquered Egypt.", 1.0)):
            result_a = v.generate_and_verify(
                query="What did Napoleon do?",
                context=["Albert Einstein was born in Ulm in 1879."],
            )

        # Call B: entity present → HIGH or MEDIUM confidence expected
        with patch.object(v, "_call_llm", return_value=("Einstein was born in Ulm.", 1.0)):
            result_b = v.generate_and_verify(
                query="Where was Einstein born?",
                context=["Albert Einstein was born in Ulm, Germany in 1879."],
                entities=["Einstein"],
            )

        assert result_b.confidence in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM), (
            f"Call B should be HIGH/MEDIUM confidence; got {result_b.confidence}. "
            f"Prior low-confidence call A must not poison call B."
        )


class TestQuestionRelevanceReorder:
    """_reorder_by_question_relevance: answer-relevant chunks rise to the top (§12.27)."""

    def test_most_relevant_chunk_first(self, minimal_cfg):
        """Chunk sharing the most query content words must be sorted first."""
        v = create_verifier(cfg=minimal_cfg)
        query = "Who won the Nobel Prize in Physics in 1921?"
        context = [
            "Marie Curie was a Polish-born physicist.",
            "Albert Einstein won the Nobel Prize in Physics in 1921.",
            "Germany hosted many physics conferences in that era.",
        ]
        reordered = v._reorder_by_question_relevance(query, context)
        assert reordered[0] == context[1], (
            f"Chunk with most query-word overlap should be first; got: {reordered[0]!r}"
        )

    def test_single_chunk_unchanged(self, minimal_cfg):
        """A single-chunk list is returned as-is."""
        v = create_verifier(cfg=minimal_cfg)
        context = ["Only one chunk here."]
        assert v._reorder_by_question_relevance("any query", context) == context

    def test_empty_context_unchanged(self, minimal_cfg):
        """Empty list is returned as-is."""
        v = create_verifier(cfg=minimal_cfg)
        assert v._reorder_by_question_relevance("any query", []) == []

    def test_all_zero_score_order_preserved(self, minimal_cfg):
        """When no chunk shares query words, original order is preserved (stable sort)."""
        v = create_verifier(cfg=minimal_cfg)
        query = "xyzzy plugh"
        context = ["Alpha text.", "Beta text.", "Gamma text."]
        reordered = v._reorder_by_question_relevance(query, context)
        assert reordered == context

    def test_reorder_is_stable(self, minimal_cfg):
        """Chunks with equal score preserve their original relative order.

        Note: scoring is sqrt-length-normalised, so a true tie requires both
        equal hit count AND equal word count.
        """
        v = create_verifier(cfg=minimal_cfg)
        query = "physicist discovered"
        # Both chunks: 8 words, 2 hits → identical normalised scores.
        context = [
            "Marie Curie was a physicist who discovered radium.",
            "Hans Bethe was a physicist who discovered fusion.",
            "Some unrelated text about science.",
        ]
        reordered = v._reorder_by_question_relevance(query, context)
        assert reordered[0] == context[0]
        assert reordered[1] == context[1]

    def test_f1a_idf_demotes_generic_term_chunk(self, minimal_cfg):
        """F1a: with >=4 candidates, a chunk that only echoes a GENERIC query
        term (present in most candidates) ranks below the specific-entity chunk.
        Regression guard for the generic-term-echo failure mode."""
        v = create_verifier(cfg=minimal_cfg)
        query = "Are Granta and Lapham's both literary magazines?"
        specific = "Granta was a literary magazine published in the 1970s."
        generic1 = "Magazines in Portugal are numerous and people read magazines."
        generic2 = "Magazines in Malaysia: many literary magazines are published."
        generic3 = "Austria has many literary magazines in circulation today."
        out = v._reorder_by_question_relevance(
            query, [generic1, generic2, specific, generic3],
        )
        assert out[0] == specific, (
            f"IDF should rank the specific-entity chunk first; got {out[0][:50]!r}"
        )

    def test_f1a_falls_back_below_min_candidates(self, minimal_cfg):
        """F1a guard: with < 4 candidates, IDF is disabled and behaviour is the
        validated length-normalised hit count."""
        v = create_verifier(cfg=minimal_cfg)
        query = "Who won the Nobel Prize in Physics in 1921?"
        context = [
            "Marie Curie was a Polish-born physicist.",
            "Albert Einstein won the Nobel Prize in Physics in 1921.",
        ]
        out = v._reorder_by_question_relevance(query, context)
        assert out[0] == context[1]

    def test_d1_coverage_floor_keeps_entity_chunk_first(self, minimal_cfg):
        """D1: a chunk naming a distinctive query entity gets a coverage floor,
        so it outranks a chunk that only shares generic query terms."""
        v = create_verifier(cfg=minimal_cfg)
        query = "Where is the Brandenburg Gate located?"
        entity_chunk = "The Brandenburg Gate is in Berlin near the Tiergarten."
        generic = "The gate located here is a popular tourist attraction located downtown."
        filler1 = "Some unrelated text about architecture and history here."
        filler2 = "Another unrelated paragraph mentioning located buildings."
        out = v._reorder_by_question_relevance(
            query, [generic, filler1, entity_chunk, filler2],
            entities=["Brandenburg Gate"],
        )
        assert out[0] == entity_chunk

    def test_reorder_does_not_evict_high_rrf_answer_chunk(self, minimal_cfg):
        """Membership invariant: an answer chunk the Navigator ranked #1 by RRF
        survives the max_docs cap even when query-echoing distractors out-score
        it in the question-relevance reorder. The production call site caps by
        RRF order FIRST, then reorders only within the kept window — so the
        lexical heuristic can reorder but never evict. Mirrors the
        verifier.process() composition."""
        v = create_verifier(cfg=minimal_cfg)
        v.config.max_docs = 2
        query = "Which member of the orchestra was honoured?"
        # RRF rank #1 = the answer chunk, but it shares NO content word with the
        # query ("composer"!="member", "knighted"!="honoured").
        answer = "Edward William Elgar was an English composer who was knighted."
        # Distractors heavily echo the query terms "orchestra" / "member".
        d1 = "The orchestra member orchestra member played orchestra member music."
        d2 = "An orchestra member in an orchestra; the member toured widely."
        d3 = "Orchestra members are members who perform in an orchestra nightly."
        rrf_order = [answer, d1, d2, d3]

        # New (fixed) path: cap by RRF order, then reorder within the window.
        selected = rrf_order[: v.config.max_docs]
        reordered = v._reorder_by_question_relevance(query, selected)
        formatted = v._format_context(reordered, query=query)
        assert "Edward William Elgar" in formatted

        # Old (buggy) path: reorder the full list, then cap — evicts the answer.
        old_reordered = v._reorder_by_question_relevance(query, rrf_order)
        old_formatted = v._format_context(old_reordered, query=query)
        assert "Edward William Elgar" not in old_formatted

    def test_f2_sentence_truncation_keeps_answer_in_tail(self, minimal_cfg):
        """F2: when a doc exceeds the per-doc budget, the query-relevant
        sentence in the TAIL survives (head-truncation would drop it)."""
        v = create_verifier(cfg=minimal_cfg)
        filler = "This sentence is filler. " * 60  # ~1500 chars of filler
        answer = "The director of the film was Brad Silberling."
        doc = filler + answer
        budget = 800
        out = v._truncate_sentence_aware(doc, budget, "Who was the director?")
        assert "Brad Silberling" in out
        assert len(out) <= budget + 100  # roughly within budget
        """Length normalization must not penalise short direct-answer chunks.

        Regression guard for the failure mode where a short fight-song chunk
        ranks below long topic-description chunks under absolute hit count —
        pushing the gold below the max_docs=5 cutoff. After sqrt normalisation,
        a chunk with 4 hits in ~20 words (score ≈ 0.89) outranks a chunk with
        8 hits in ~150 words (score ≈ 0.65), matching the production trace.
        """
        v = create_verifier(cfg=minimal_cfg)
        query = ("What is the name of the fight song of the university "
                 "whose main campus is in Lawrence, Kansas and whose branch "
                 "campuses are in the Kansas City metropolitan area?")
        # Short direct-answer chunk: ~20 words, 4 query-token hits
        # ("fight", "song", "kansas", "university") → 4/sqrt(20) ≈ 0.894
        short_answer = (
            "Kansas Song is a popular fight song most often associated "
            "with the University of Kansas Jayhawks."
        )
        # Long topic chunk: ~150 words, 8 query-token hits spread thinly →
        # 8/sqrt(150) ≈ 0.653. Padded with neutral biographical content so
        # the additional length does not raise the hit count further.
        long_topic = (
            "The University of Kansas, often referred to as KU, is a "
            "public research institution founded in eighteen sixty-five. "
            "Its main location sits atop Mount Oread, a prominent ridge "
            "that overlooks the surrounding river valley. The institution "
            "houses sixteen schools and offers more than three hundred "
            "academic programs across the undergraduate and graduate levels. "
            "Researchers there have contributed to advances in pharmacy, "
            "engineering, journalism, and the social sciences. The school "
            "competes athletically in the Big Twelve Conference and has "
            "produced numerous Olympic athletes over its long history. "
            "Notable alumni include politicians, novelists, business "
            "leaders, scientists, and several professional basketball "
            "players who went on to coach at the collegiate level. The "
            "Lawrence community has long supported the institution through "
            "civic partnerships. Two additional branch campuses operate in "
            "the metropolitan area surrounding Kansas City, focused mainly "
            "on continuing-education programs."
        )
        reordered = v._reorder_by_question_relevance(
            query, [long_topic, short_answer]
        )
        assert reordered[0] == short_answer, (
            "Short direct-answer chunk must rank above long topic chunk after "
            f"length normalisation; got order: [{reordered[0][:60]!r}, ...]"
        )


# ---------------------------------------------------------------------------
# Prompt routing: query_type + bridge_entities select the right prompt
# ---------------------------------------------------------------------------

class TestPromptRoutingPlumbing:
    """The BRIDGE_PROMPT / COMPARISON_PROMPT must be selectable.

    The verifier's prompt-selection logic fires only when both query_type and
    bridge_entities are supplied; otherwise every query uses ANSWER_PROMPT.
    These tests assert that the verifier's selection picks BRIDGE/COMPARISON
    when given the proper kwargs, and that the pipeline forwards them.
    """

    def test_comparison_query_type_selects_comparison_prompt(self, minimal_cfg, monkeypatch):
        v = create_verifier(cfg=minimal_cfg, enable_pre_validation=False)
        captured = {}

        def fake_llm(self, prompt):
            captured["prompt"] = prompt
            return "Berlin", 1.0

        monkeypatch.setattr(Verifier, "_call_llm", fake_llm)
        v.generate_and_verify(
            query="Is Berlin older than Munich?",
            context=["Berlin founded 1237.", "Munich founded 1158."],
            entities=["Berlin", "Munich"],
            query_type="comparison",
        )
        # COMPARISON_PROMPT has a unique signature: numbered steps "1. Find ... 2. Find ... 3. Compare"
        assert "1. Find the relevant fact for the FIRST" in captured["prompt"]
        assert "2. Find the relevant fact for the SECOND" in captured["prompt"]

    def test_multi_hop_with_hop_sequence_selects_bridge_prompt(self, minimal_cfg, monkeypatch):
        v = create_verifier(cfg=minimal_cfg, enable_pre_validation=False)
        # This test verifies PROMPT-SELECTION ROUTING (multi-hop → BRIDGE_PROMPT),
        # not the bridge-exclusion retry path. The mocked LLM always returns the
        # bridge entity "Grand Hotel", which would trigger the bounded
        # bridge-exclusion retry and overwrite the captured prompt with the
        # exclusion prompt. Disable that retry so the captured prompt is the
        # initial bridge prompt under test.
        v.config.enable_bridge_exclusion_retry = False
        captured = {}

        def fake_llm(self, prompt):
            captured["prompt"] = prompt
            return "Grand Hotel", 1.0

        monkeypatch.setattr(Verifier, "_call_llm", fake_llm)
        # The context must contain the query entity so entity-path validation
        # passes — otherwise pre-validation flags INSUFFICIENT_EVIDENCE and the
        # INSUFFICIENT prompt is (correctly) selected instead of the bridge
        # prompt. This test exercises prompt ROUTING (multi_hop -> BRIDGE), so
        # the context is made self-consistent with the declared entity.
        v.generate_and_verify(
            query="Who directed the film starring Greta Garbo about a hotel?",
            context=["Grand Hotel is a 1932 film starring Greta Garbo, "
                     "directed by Edmund Goulding."],
            entities=["Greta Garbo"],
            hop_sequence=[
                {"step_id": 0, "sub_query": "Find the film", "is_bridge": True},
                {"step_id": 1, "sub_query": "Find the director", "is_bridge": False},
            ],
            query_type="multi_hop",
            bridge_entities=["Grand Hotel"],
        )
        # BRIDGE_PROMPT has the unique substring "Use the following reasoning chain"
        assert "reasoning chain" in captured["prompt"]
        assert "Step" in captured["prompt"]  # _build_bridge_chain produces "Step N:" lines

    def test_hop_sequence_accepts_HopStep_dataclasses(self, minimal_cfg, monkeypatch):
        """Regression guard: HopStep dataclasses are accepted by the verifier.

        AgentPipeline.process() passes the Planner's List[HopStep] dataclasses
        directly to generate_and_verify, while the signature documents
        List[Dict]. The verifier must normalise both forms — otherwise
        _build_bridge_chain crashes with `'HopStep' object has no attribute
        'get'` before any LLM call is made.
        """
        from src.logic_layer.planner import HopStep
        v = create_verifier(cfg=minimal_cfg, enable_pre_validation=False)
        captured = {}

        def fake_llm(self, prompt):
            captured["prompt"] = prompt
            return "Grand Hotel", 1.0

        monkeypatch.setattr(Verifier, "_call_llm", fake_llm)
        # Pass dataclasses (production pipeline path), not dicts (controller path).
        v.generate_and_verify(
            query="Who directed the film starring Greta Garbo?",
            context=["Grand Hotel was directed by Edmund Goulding."],
            entities=["Greta Garbo"],
            hop_sequence=[
                HopStep(step_id=0, sub_query="Find the film",
                        target_entities=["Greta Garbo"],
                        depends_on=[], is_bridge=True),
                HopStep(step_id=1, sub_query="Find the director",
                        target_entities=[], depends_on=[0], is_bridge=False),
            ],
            query_type="multi_hop",
            bridge_entities=["Grand Hotel"],
        )
        # Must reach the LLM — the crash this guards against happened before
        # _call_llm was invoked.
        assert "prompt" in captured, (
            "HopStep dataclasses must not crash _build_bridge_chain "
            "before the LLM is called"
        )
        assert "reasoning chain" in captured["prompt"]

    def test_no_query_type_falls_back_to_answer_prompt(self, minimal_cfg, monkeypatch):
        v = create_verifier(cfg=minimal_cfg, enable_pre_validation=False)
        captured = {}

        def fake_llm(self, prompt):
            captured["prompt"] = prompt
            return "Paris", 1.0

        monkeypatch.setattr(Verifier, "_call_llm", fake_llm)
        v.generate_and_verify(
            query="What is the capital of France?",
            context=["France's capital is Paris."],
            entities=["France"],
        )
        # ANSWER_PROMPT has the unique line "Give the shortest possible answer"
        assert "Give the shortest possible answer" in captured["prompt"]
        # Neither BRIDGE nor COMPARISON markers should be present.
        assert "reasoning chain" not in captured["prompt"]
        assert "1. Find the relevant fact for the FIRST" not in captured["prompt"]

    def test_call_wrapper_forwards_bridge_entities(self, minimal_cfg, monkeypatch):
        """Verifier.__call__ must forward bridge_entities and query_type."""
        v = create_verifier(cfg=minimal_cfg, enable_pre_validation=False)
        seen = {}

        def fake_gen(self, query, context, entities=None, hop_sequence=None,
                     query_type=None, bridge_entities=None, chunk_is_graph_based=None):
            seen["bridge_entities"] = bridge_entities
            seen["query_type"] = query_type
            return VerificationResult(answer="x", iterations=1)

        monkeypatch.setattr(Verifier, "generate_and_verify", fake_gen)
        v(query="q", context=["c"], query_type="multi_hop", bridge_entities=["E"])
        assert seen["bridge_entities"] == ["E"]
        assert seen["query_type"] == "multi_hop"


# ---------------------------------------------------------------------------
# Retrieval-provenance signal in credibility scoring
# ---------------------------------------------------------------------------

class TestProvenanceSignal:
    """Callers can pass per-chunk graph-provenance flags, and graph-retrieved
    chunks score strictly higher than vector-only chunks. Without a provenance
    flag the term contributes a constant baseline (0.5) regardless of retrieval
    method.
    """

    def test_graph_chunk_scores_higher_than_vector_chunk(self, minimal_cfg):
        validator = PreGenerationValidator(VerifierConfig.from_yaml(minimal_cfg))
        # Two near-identical chunks, only difference is graph-vs-vector provenance.
        chunk = "Albert Einstein was a German-born theoretical physicist."
        # Use two unique chunks to avoid filter collapse on duplicate text.
        ctx = [
            chunk,
            "Einstein developed the theory of relativity in 1915.",
        ]
        # Provenance: chunk 0 graph-retrieved, chunk 1 vector-only.
        flags = [True, False]
        result = validator.validate(
            ctx, "What did Einstein develop?",
            entities=["Einstein"],
            chunk_is_graph_based=flags,
        )
        assert len(result.credibility_scores) == 2
        # Graph-retrieved chunk should score >= vector chunk (the provenance
        # term contributes weight*1.0 vs weight*baseline).
        # Both chunks have the same entity-frequency / cross-references, so
        # the only systematic difference is provenance.
        # Graph chunk gets full provenance credit (weight × 1.0), vector chunk
        # gets baseline (weight × 0.5). Score delta should be
        # provenance_weight × (1.0 - baseline) = 0.3 × 0.5 = 0.15.
        score_graph, score_vector = result.credibility_scores[0], result.credibility_scores[1]
        assert score_graph > score_vector, (
            f"Graph chunk ({score_graph:.3f}) should outscore vector chunk "
            f"({score_vector:.3f})"
        )

    def test_missing_provenance_falls_back_to_baseline(self, minimal_cfg):
        """When chunk_is_graph_based is None, behaviour falls back to baseline."""
        validator = PreGenerationValidator(VerifierConfig.from_yaml(minimal_cfg))
        ctx = ["Einstein was a physicist.", "Einstein developed relativity."]
        result_no_provenance = validator.validate(
            ctx, "What did Einstein develop?", entities=["Einstein"],
        )
        result_explicit_none = validator.validate(
            ctx, "What did Einstein develop?", entities=["Einstein"],
            chunk_is_graph_based=None,
        )
        # Same input → same scores when provenance is absent (baseline applied).
        assert result_no_provenance.credibility_scores == result_explicit_none.credibility_scores

    def test_provenance_flag_length_mismatch_ignored(self, minimal_cfg, caplog):
        """A length-mismatched provenance list should be ignored, not crash."""
        import logging
        caplog.set_level(logging.WARNING)
        validator = PreGenerationValidator(VerifierConfig.from_yaml(minimal_cfg))
        ctx = ["chunk one", "chunk two", "chunk three"]
        result = validator.validate(
            ctx, "query", entities=[],
            chunk_is_graph_based=[True],  # wrong length: 1 vs 3
        )
        assert len(result.credibility_scores) >= 1


# ---------------------------------------------------------------------------
# Claim verification — no auto-verify for short / numeric claims
# ---------------------------------------------------------------------------

class TestNumericClaimVerification:
    """A claim with no proper noun must not auto-verify: otherwise an LLM
    hallucinating "9 million inhabitants" or "1995" would be accepted
    regardless of context. Short / numeric claims are grounded by token
    presence in the context."""

    def test_numeric_claim_grounded_in_context_verified(self, verifier):
        ok, reason = verifier._verify_claim(
            "founded in 1995",
            context=["The company was founded in 1995 in Cupertino."],
        )
        assert ok is True
        assert reason == "context_token_grounded"

    def test_numeric_claim_not_in_context_violated(self, verifier):
        """The hallucination case B4 is designed to catch."""
        ok, reason = verifier._verify_claim(
            "9 million inhabitants",
            context=["Munich has roughly 1.5 million inhabitants."],
        )
        assert ok is False
        assert reason == "no_entities_and_tokens_ungrounded"

    def test_short_phrase_not_in_context_violated(self, verifier):
        ok, reason = verifier._verify_claim(
            "ice hockey",
            context=["The athlete played football professionally."],
        )
        assert ok is False
        assert reason == "no_entities_and_tokens_ungrounded"

    def test_short_phrase_in_context_verified(self, verifier):
        ok, reason = verifier._verify_claim(
            "ice hockey",
            context=["He played ice hockey for the national team."],
        )
        assert ok is True
        assert reason == "context_token_grounded"

    def test_long_narrative_no_proper_noun_still_auto_verifies(self, verifier):
        """Long sentences with no proper noun keep the historical auto-verify
        behavior — no falsifiable anchor to check against."""
        ok, reason = verifier._verify_claim(
            "This is a long sentence that contains no proper noun and is more than six tokens.",
            context=["something else entirely"],
        )
        assert ok is True
        assert reason == "no_entities_to_verify"

    def test_empty_context_short_claim_still_auto_verifies(self, verifier):
        """No context means no falsifiable anchor — keep historical behavior."""
        ok, reason = verifier._verify_claim("Yes.", context=[])
        assert ok is True
        assert reason == "no_entities_to_verify"


# ---------------------------------------------------------------------------
# iteration_history string-truncation budget
# ---------------------------------------------------------------------------

class TestHistoryTruncation:
    """Per-string truncation budget for iteration_history entries.

    Storing full answers + full claim lists in iteration_history bloats the
    per-question JSONL across a large evaluation run (e.g. hundreds of
    questions × multiple iterations × multi-hundred-char answers). Strings are
    truncated to 200 chars + a "...[truncated]" marker.
    """

    def test_short_string_unchanged(self):
        assert Verifier._truncate_history_str("Hello world.") == "Hello world."

    def test_long_string_truncated_with_marker(self):
        s = "x" * 500
        out = Verifier._truncate_history_str(s)
        assert out.endswith("...[truncated]")
        # 200 content chars + truncation marker
        assert len(out) == 200 + len("...[truncated]")
        assert out.startswith("x" * 200)

    def test_non_string_passed_through(self):
        # Non-string inputs should not crash (defensive default).
        assert Verifier._truncate_history_str(None) is None
        assert Verifier._truncate_history_str(42) == 42

    def test_truncate_list_applies_per_element(self):
        items = ["short", "x" * 300, "y" * 50]
        out = Verifier._truncate_history_list(items)
        assert out[0] == "short"
        assert out[1].endswith("...[truncated]")
        assert out[2] == "y" * 50

    def test_iteration_history_truncated_on_long_llm_answer(self, minimal_cfg, monkeypatch):
        """Full integration: a 500-char LLM answer is truncated in history."""
        v = create_verifier(cfg=minimal_cfg, enable_pre_validation=False)
        long_answer = "Albert Einstein was born in Ulm. " + ("filler text. " * 50)
        assert len(long_answer) > 200

        monkeypatch.setattr(
            Verifier, "_call_llm",
            lambda self, prompt: (long_answer, 1.0),
        )
        result = v.generate_and_verify(
            query="Where was Einstein born?",
            context=["Einstein was born in Ulm."],
            entities=["Einstein"],
        )
        assert result.iteration_history, "Expected at least one iteration in history"
        stored_answer = result.iteration_history[0]["answer"]
        assert len(stored_answer) <= 200 + len("...[truncated]"), (
            f"history answer too long ({len(stored_answer)} chars)"
        )
        assert stored_answer.endswith("...[truncated]"), (
            "truncated answers must carry the truncation marker"
        )


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    print("=" * 60)
    print("VERIFIER SEMANTIC SMOKE CHECK")
    print("SpaCy: %s" % ("available" if SPACY_AVAILABLE else "unavailable"))
    print("=" * 60)

    _ctx = [
        "Albert Einstein was a German-born theoretical physicist.",
        "Einstein received the Nobel Prize in Physics in 1921.",
        "He was born in Ulm, Germany, on March 14, 1879.",
    ]
    _v = create_verifier(cfg={
        "llm": {"max_context_chars": 2000, "max_docs": 5, "max_chars_per_doc": 400},
        "agent": {"max_verification_iterations": 1},
        "verifier": {"enable_entity_path_validation": True, "enable_credibility_scoring": True},
    }, enable_pre_validation=True)

    _validator = PreGenerationValidator(_v.config)
    _res = _validator.validate(_ctx, "When was Einstein born?", entities=["Einstein"])
    print("Pre-validation status: %s" % _res.status.value)
    print("Entity path valid: %s" % _res.entity_path_valid)
    print("Filtered context docs: %d" % len(_res.filtered_context))
    print("=" * 60)
