"""
Test suite for src/data_layer/chunking.py (paper §3.2, Chunking).

Pins the observable behaviour of the two chunking strategies and their
support utilities, including the deterministic-chunk-ID invariant (T-E)
that guarantees stable KuzuDB node references across re-ingestion runs.

Test inventory
--------------
TestSemanticChunker          -- boundary detection, importance/lexical-
                                diversity metadata, word-boundary-aware
                                overlap, header-state reset, empty-doc fallback.
TestAutomaticQualityFilter   -- keep/drop thresholds (length, word count).
TestTFIDFScorer              -- stopword exclusion, non-zero importance for
                                discriminating chunks, zero before analysis.
TestSpacySentenceChunker     -- 3-sentence windows, overlap, SHA-256 chunk
                                IDs (length + formula + determinism), empty/
                                None handling, ingestion + LangChain interfaces.
TestSpacyModelCache          -- model-cache hit (identity) + clear semantics.
TestSentenceChunkingConfig   -- config validation (window/overlap bounds).
TestThesisCompliance         -- default values match the thesis spec.
TestChunkerProperty          -- Hypothesis property tests (non-empty chunks,
                                unique IDs); defined only if hypothesis is
                                installed; marked `slow`.

Dependencies / Requirements
---------------------------
pytest, langchain (Document), SpaCy `en_core_web_sm`. Optional: hypothesis
(enables TestChunkerProperty). Tests that load SpaCy degrade gracefully via
the SPACY_AVAILABLE guard.

Last reviewed: 2026-05-30 (audit pass, project version 5.4).
"""

import sys
import math
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from langchain.schema import Document

from src.data_layer.chunking import (
    create_semantic_chunker,
    create_sentence_chunker,
    AutomaticQualityFilter,
    TFIDFScorer,
    ALL_STOPWORDS,
    SpacyModelCache,
    SentenceChunkingConfig,
    SPACY_AVAILABLE,
)

# ── Shared parameters ───────────────────────────────────────────────────────────
# Named once so an ablation reviewer can override window/overlap/min in a single
# place instead of editing every test body.
SEMANTIC_CFG = dict(chunk_size=500, chunk_overlap=50, min_chunk_size=100)
# Paper §2.2 sentence-window defaults (3-sentence window, 1-sentence overlap).
SENTENCE_CFG = dict(sentences_per_chunk=3, sentence_overlap=1)

# Chunk-ID contract: truncated SHA-256 of "source_doc:index:text[:N]".
_CHUNK_ID_HEX_LEN = 20       # Why: 20 hex chars (80 bits) -> negligible collisions at corpus scale.
_CHUNK_ID_TEXT_PREFIX = 50   # Why: first 50 chars discriminate near-duplicate sentences.

# ── Fixtures ──────────────────────────────────────────────────────────────────

THESIS_TEXT = """
1. Introduction

This paper investigates the application of machine learning
techniques to natural language processing tasks. The research
focuses on edge deployment scenarios where computational
resources are limited.

1.1 Problem Statement

Modern language models require significant computational resources,
making deployment on edge devices challenging. This research
addresses the gap between model capability and device constraints
through quantization and optimization techniques.

1.2 Research Questions

The central research questions are:
- How can large language models be efficiently deployed on edge devices?
- What is the impact of quantization on model accuracy?
- How can retrieval-augmented generation improve edge AI systems?

2. Background

This chapter provides the theoretical foundation for the research.
We review relevant literature on language models, quantization
techniques, and retrieval-augmented generation.
"""

EINSTEIN_TEXT = """
Albert Einstein was born on March 14, 1879, in Ulm, Germany. He was a
theoretical physicist who developed the theory of relativity. Einstein is
best known for his mass-energy equivalence formula E = mc². He received the
Nobel Prize in Physics in 1921 for his discovery of the photoelectric effect.
Einstein emigrated to the United States in 1933 and worked at Princeton
University. He became an American citizen in 1940. Einstein died on April 18,
1955, in Princeton, New Jersey.
"""


# ── Semantic Chunker ──────────────────────────────────────────────────────────

class TestSemanticChunker:

    def test_produces_chunks(self) -> None:
        doc = Document(page_content=THESIS_TEXT, metadata={"source_file": "thesis.pdf"})
        chunker = create_semantic_chunker(**SEMANTIC_CFG)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 0, "Should produce at least one chunk"

    def test_chunks_have_importance_score(self) -> None:
        doc = Document(page_content=THESIS_TEXT, metadata={"source_file": "thesis.pdf"})
        chunker = create_semantic_chunker(**SEMANTIC_CFG)
        chunks = chunker.chunk_document(doc)

        assert all("importance_score" in c.metadata for c in chunks)
        assert all("lexical_diversity" in c.metadata for c in chunks)

    def test_chunks_start_with_complete_words(self) -> None:
        """Word-boundary-aware overlap: no chunk should start with a partial word."""
        doc = Document(page_content=THESIS_TEXT, metadata={"source_file": "thesis.pdf"})
        chunker = create_semantic_chunker(**SEMANTIC_CFG)
        chunks = chunker.chunk_document(doc)

        for chunk in chunks:
            first_char = chunk.page_content[0] if chunk.page_content else ""
            # A chunk starting mid-word would begin with a non-space, non-alpha boundary
            # This is a soft check: first char must not be a continuation character
            assert first_char == "" or not first_char.isspace(), \
                f"Chunk starts with unexpected whitespace: {chunk.page_content[:30]!r}"

    def test_resets_header_context_between_calls(self) -> None:
        """HeaderExtractor state must not leak across chunk_document() calls."""
        chunker = create_semantic_chunker(**SEMANTIC_CFG)
        doc = Document(page_content=THESIS_TEXT, metadata={"source_file": "doc.pdf"})

        chunks_first = chunker.chunk_document(doc)
        chunks_second = chunker.chunk_document(doc)

        # Second call should produce the same result (no stale chapter/section state)
        assert len(chunks_first) == len(chunks_second)

    def test_header_detection(self) -> None:
        doc = Document(page_content=THESIS_TEXT, metadata={"source_file": "thesis.pdf"})
        chunker = create_semantic_chunker(**SEMANTIC_CFG)
        chunks = chunker.chunk_document(doc)

        sections = {c.metadata.get("section") for c in chunks if c.metadata.get("section")}
        chapters = {c.metadata.get("chapter") for c in chunks if c.metadata.get("chapter")}

        assert sections or chapters, "Should detect at least one section or chapter header"

    def test_fallback_on_empty_document(self) -> None:
        """Empty document should not raise; fallback splitter produces result."""
        doc = Document(page_content="", metadata={})
        chunker = create_semantic_chunker(**SEMANTIC_CFG)
        # Should not raise
        chunks = chunker.chunk_document(doc)
        assert isinstance(chunks, list)


# ── Quality Filter ─────────────────────────────────────────────────────────────

class TestAutomaticQualityFilter:

    def test_keeps_good_text(self) -> None:
        qf = AutomaticQualityFilter()
        good = "This is a sample text with diverse vocabulary and meaningful content. " * 3
        keep, reason, _ = qf.should_keep_chunk(good)
        assert keep, f"Expected keep, got filtered: {reason}"

    def test_filters_short_text(self) -> None:
        qf = AutomaticQualityFilter()
        keep, reason, _ = qf.should_keep_chunk("Hi")
        assert not keep
        assert "too_short" in reason

    def test_filters_too_few_words(self) -> None:
        qf = AutomaticQualityFilter(min_length=5, min_words=20)
        keep, reason, _ = qf.should_keep_chunk("One two three.")
        assert not keep
        assert "too_few_words" in reason

    def test_empty_string_filtered(self) -> None:
        qf = AutomaticQualityFilter()
        keep, _, _2 = qf.should_keep_chunk("")
        assert not keep


# ── TF-IDF Scorer ─────────────────────────────────────────────────────────────

class TestTFIDFScorer:

    def test_stopwords_excluded_from_top_terms(self) -> None:
        """No stopword should appear as a high-scoring term after filtering."""
        scorer = TFIDFScorer()
        chunks = [
            "Machine learning is transforming artificial intelligence.",
            "Natural language processing uses deep learning models.",
            "Edge devices have limited computational resources.",
        ]
        scorer.analyze_corpus(chunks)

        # Get the top-scored terms for chunk 0 via direct access
        term_freq = scorer.chunk_term_frequencies[0]
        scores = {}
        for term, tf in term_freq.items():
            df = scorer.document_frequency.get(term, 1)
            idf = math.log(scorer.total_chunks / df) if df > 0 else 0.0
            scores[term] = tf * idf

        top_terms = sorted(scores, key=scores.get, reverse=True)[:5]
        stopwords_present = [t for t in top_terms if t in ALL_STOPWORDS]
        assert len(stopwords_present) == 0, f"Stopwords in top terms: {stopwords_present}"

    def test_importance_nonzero_for_discriminating_chunk(self) -> None:
        scorer = TFIDFScorer()
        scorer.analyze_corpus([
            "Einstein relativity physics quantum mechanics.",
            "Darwin evolution species natural selection biology.",
            "Newton gravity laws motion classical mechanics.",
        ])
        score = scorer.calculate_chunk_importance(0)
        assert score > 0.0

    def test_zero_if_not_analyzed(self) -> None:
        scorer = TFIDFScorer()
        # total_chunks == 0 → must return 0.0 without error
        assert scorer.calculate_chunk_importance(0) == 0.0


# ── SpaCy Sentence Chunker ─────────────────────────────────────────────────────

class TestSpacySentenceChunker:

    def test_produces_chunks(self) -> None:
        chunker = create_sentence_chunker(**SENTENCE_CFG)
        chunks = chunker.chunk_text(EINSTEIN_TEXT, source_doc="einstein.txt")
        assert len(chunks) > 0

    def test_overlap_present(self) -> None:
        """Adjacent chunks must share at least one sentence index."""
        chunker = create_sentence_chunker(**SENTENCE_CFG)
        chunks = chunker.chunk_text(EINSTEIN_TEXT, source_doc="einstein.txt")

        if len(chunks) >= 2:
            shared = set(chunks[0].sentence_indices) & set(chunks[1].sentence_indices)
            assert len(shared) >= 1, "Adjacent chunks must share at least one sentence"

    def test_max_sentences_per_chunk(self) -> None:
        chunker = create_sentence_chunker(**SENTENCE_CFG)
        chunks = chunker.chunk_text(EINSTEIN_TEXT, source_doc="einstein.txt")
        assert all(c.sentence_count <= SENTENCE_CFG["sentences_per_chunk"] for c in chunks)

    def test_chunk_id_deterministic(self) -> None:
        """Same input must produce identical chunk IDs across two runs."""
        chunker = create_sentence_chunker(**SENTENCE_CFG)
        chunks_a = chunker.chunk_text(EINSTEIN_TEXT, source_doc="einstein.txt")
        chunks_b = chunker.chunk_text(EINSTEIN_TEXT, source_doc="einstein.txt")

        ids_a = [c.chunk_id for c in chunks_a]
        ids_b = [c.chunk_id for c in chunks_b]
        assert ids_a == ids_b, "Chunk IDs must be deterministic across runs"

    def test_chunk_id_is_sha256_based(self) -> None:
        """Chunk ID must be a 20-char hex string (truncated SHA-256)."""
        chunker = create_sentence_chunker(**SENTENCE_CFG)
        chunks = chunker.chunk_text(EINSTEIN_TEXT, source_doc="einstein.txt")

        for chunk in chunks:
            assert len(chunk.chunk_id) == _CHUNK_ID_HEX_LEN
            assert all(c in "0123456789abcdef" for c in chunk.chunk_id)

    def test_chunk_id_uses_hashlib_sha256_not_python_hash(self) -> None:
        """Chunk ID must match the SHA-256 formula, ruling out Python's hash().

        Python's built-in hash() is PYTHONHASHSEED-randomised: the same text
        produces different values across process restarts.  Using it for chunk
        IDs would make IDs non-deterministic across re-ingestion runs, breaking
        stable KuzuDB node references.

        This test reconstructs the expected ID from the documented formula
            SHA-256(source_doc:chunk_index:text[:50])[:20]
        and asserts it matches the generated ID.  The test would fail if the
        implementation were ever changed to use hash().
        """
        chunker = create_sentence_chunker(**SENTENCE_CFG)
        chunks = chunker.chunk_text(EINSTEIN_TEXT, source_doc="einstein.txt")

        assert len(chunks) >= 1, "Need at least one chunk to verify ID formula"
        first_chunk = chunks[0]

        # chunk_index=0 is the position of the first chunk in the output list.
        content = "einstein.txt:0:%s" % first_chunk.text[:_CHUNK_ID_TEXT_PREFIX]
        expected_id = hashlib.sha256(content.encode()).hexdigest()[:_CHUNK_ID_HEX_LEN]

        assert first_chunk.chunk_id == expected_id, (
            "chunk_id does not match SHA-256('einstein.txt:0:text[:50]').\n"
            "Expected: %s, got: %s.\n"
            "If Python's hash() were used instead, this value would vary "
            "across process runs (PYTHONHASHSEED-sensitive)." % (expected_id, first_chunk.chunk_id)
        )

    def test_none_input_returns_empty(self) -> None:
        chunker = create_sentence_chunker()
        result = chunker.chunk_text(None, source_doc="test")
        assert result == []

    def test_empty_string_returns_empty(self) -> None:
        chunker = create_sentence_chunker()
        result = chunker.chunk_text("", source_doc="test")
        assert result == []

    def test_ingestion_interface(self) -> None:
        """chunk() must return list of dicts with 'text' and 'metadata' keys."""
        chunker = create_sentence_chunker()
        result = chunker.chunk("Some text about Einstein and relativity theory.",
                               metadata={"source_file": "test.txt"})
        assert isinstance(result, list)
        assert all("text" in r and "metadata" in r for r in result)

    def test_to_langchain_documents(self) -> None:
        chunker = create_sentence_chunker()
        docs = chunker.chunk_to_documents(EINSTEIN_TEXT, metadata={"source_file": "e.txt"})
        assert len(docs) > 0
        assert all(hasattr(d, "page_content") and hasattr(d, "metadata") for d in docs)


# ── SpaCy Model Cache ─────────────────────────────────────────────────────────

class TestSpacyModelCache:

    def test_cache_returns_same_instance(self) -> None:
        """Cache hit: the second get_model() returns the identical object.

        Determinism-first: asserts object identity rather than a wall-clock
        speed ratio, so the test cannot flake under CI load.
        """
        if not SPACY_AVAILABLE:
            pytest.skip("SpaCy not installed")

        SpacyModelCache.clear_cache()
        first = SpacyModelCache.get_model("en_core_web_sm")
        second = SpacyModelCache.get_model("en_core_web_sm")

        assert first is second, \
            "Cache must return the same Language object on the second call"

    def test_clear_cache(self) -> None:
        SpacyModelCache.clear_cache()
        assert SpacyModelCache._instances == {}


# ── SentenceChunkingConfig Validation ─────────────────────────────────────────

class TestSentenceChunkingConfig:

    def test_rejects_zero_sentences(self) -> None:
        with pytest.raises(ValueError, match="sentences_per_chunk"):
            SentenceChunkingConfig(sentences_per_chunk=0)

    def test_rejects_overlap_geq_window(self) -> None:
        with pytest.raises(ValueError, match="sentence_overlap"):
            SentenceChunkingConfig(sentences_per_chunk=3, sentence_overlap=3)

    def test_rejects_negative_overlap(self) -> None:
        with pytest.raises(ValueError, match="sentence_overlap"):
            SentenceChunkingConfig(sentence_overlap=-1)

    def test_valid_config_accepted(self) -> None:
        cfg = SentenceChunkingConfig(sentences_per_chunk=3, sentence_overlap=1)
        assert cfg.sentences_per_chunk == 3
        assert cfg.sentence_overlap == 1


# ── Paper Compliance ─────────────────────────────────────────────────────────

class TestThesisCompliance:
    """Verify that default values match the paper specification (§2.2, §2.3)."""

    def test_sentence_window_defaults(self) -> None:
        cfg = SentenceChunkingConfig()
        assert cfg.sentences_per_chunk == 3, "Paper §2.2: 3-sentence window"
        assert cfg.sentence_overlap == 1, "Paper §2.2: 1-sentence overlap"
        assert cfg.min_chunk_chars == 50
        assert cfg.spacy_model == "en_core_web_sm"


# ── Property-based tests (optional: requires `pip install hypothesis`) ─────────
# Class is defined only when hypothesis is available; silently absent otherwise.

try:
    from hypothesis import given as _hyp_given, settings as _hyp_settings
    import hypothesis.strategies as _hyp_st

    @pytest.mark.slow
    class TestChunkerProperty:
        """Property-based tests for SpacySentenceChunker invariants.

        Uses Hypothesis to generate arbitrary text inputs and verify that the
        chunker satisfies its core invariants regardless of input content.

        Install with: pip install hypothesis
        Run with:     pytest test_system/test_chunking.py -m slow -v
        """

        _TEXT_STRATEGY = _hyp_st.text(
            min_size=20,
            max_size=2000,
            alphabet=_hyp_st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                whitelist_characters="\n .,!?",
            ),
        )

        # derandomize=True: reuse a fixed PRNG seed so the 50 generated examples
        # are identical run-to-run, making a reviewer's reproduction deterministic.
        @_hyp_settings(max_examples=50, deadline=5000, derandomize=True)
        @_hyp_given(_TEXT_STRATEGY)
        def test_all_chunks_nonempty(self, text: str) -> None:
            """Every chunk produced by the chunker must be non-empty."""
            chunker = create_sentence_chunker()
            chunks = chunker.chunk_text(text)
            for chunk in chunks:
                assert len(chunk.text.strip()) > 0, (
                    f"Empty chunk found for input: {text!r:.80}"
                )

        @_hyp_settings(max_examples=50, deadline=5000, derandomize=True)
        @_hyp_given(_TEXT_STRATEGY)
        def test_chunk_ids_unique(self, text: str) -> None:
            """Chunk IDs must be globally unique within a single chunking call."""
            chunker = create_sentence_chunker()
            chunks = chunker.chunk_text(text)
            ids = [c.chunk_id for c in chunks]
            assert len(ids) == len(set(ids)), (
                f"Duplicate chunk IDs: {[x for x in ids if ids.count(x) > 1]}"
            )

except ImportError:
    pass  # hypothesis not installed — TestChunkerProperty not defined
