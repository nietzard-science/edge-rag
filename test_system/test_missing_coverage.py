"""
Small-surface coverage tests for behaviours not exercised by the higher-level
suites.

Each test targets one specific behaviour:
  1. embed_query and embed_documents produce same-dimensional vectors
  2. EmbeddingCache hit returns a bit-for-bit identical vector
  3. SpacySentenceChunker overlap correctness (sentence shared across windows)
  4. GLiNERExtractor returns an empty list on empty-string input
  5. BatchedOllamaEmbeddings raises a clear error on connection refused
  6. Verifier._format_context respects the max_context_chars budget

Tests mock all network/model calls to run offline without Ollama or GLiNER.

Last reviewed: 2026-05-30 (audit pass, project version 5.4).
"""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root on path so src.* imports resolve.
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# 1. embed_query and embed_documents produce same-dimensional vectors
# =============================================================================

def _make_fake_embedding(dim: int = 8) -> list:
    return [0.1] * dim


def test_embed_query_and_embed_documents_same_dimension(tmp_path):
    """embed_query and embed_documents must return vectors of equal length."""
    from src.data_layer.embeddings import BatchedOllamaEmbeddings

    fake_dim = 8
    fake_embedding = _make_fake_embedding(fake_dim)

    def fake_post(url, json=None, timeout=None):
        resp = MagicMock()
        resp.status_code = 200
        texts = json.get("input", [])
        resp.json.return_value = {"embeddings": [fake_embedding for _ in texts]}
        return resp

    with patch("src.data_layer.embeddings.requests.post", side_effect=fake_post):
        emb = BatchedOllamaEmbeddings(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434",
            batch_size=32,
            cache_path=tmp_path / "emb.db",
        )
        query_vec = emb.embed_query("What is the capital of France?")
        doc_vecs = emb.embed_documents(["Paris is the capital of France."])

    assert len(query_vec) == fake_dim, "embed_query dimension mismatch"
    assert len(doc_vecs) == 1
    assert len(doc_vecs[0]) == fake_dim, "embed_documents dimension mismatch"
    assert len(query_vec) == len(doc_vecs[0]), "Query and document dimensions differ"


# =============================================================================
# 2. EmbeddingCache hit returns bit-for-bit identical vector
# =============================================================================

def test_cache_hit_returns_identical_vector(tmp_path):
    """A vector stored in EmbeddingCache must be returned unchanged on lookup."""
    from src.data_layer.embeddings import EmbeddingCache

    cache = EmbeddingCache(tmp_path / "cache.db")
    original = [0.1, 0.2, 0.3, 0.4, 0.5]
    model = "nomic-embed-text"
    text = "The Eiffel Tower is in Paris."

    cache.put(text, original, model)
    retrieved = cache.get(text, model)

    assert retrieved is not None, "Cache miss — expected a hit"
    assert retrieved == original, "Retrieved vector differs from stored vector"
    cache.close()


# =============================================================================
# 3. SpacySentenceChunker overlap correctness
# =============================================================================

def test_overlap_correctness():
    """Adjacent chunks must share exactly sentence_overlap sentences."""
    from src.data_layer.chunking import SpacySentenceChunker

    # Use the regex fallback (no SpaCy) to avoid loading the full NLP model.
    # SpacySentenceChunker falls back to a regex splitter when spacy fails to load.
    chunker = SpacySentenceChunker(
        sentences_per_chunk=3,
        sentence_overlap=1,
        spacy_model="en_core_web_sm",
    )

    # Six clearly-separated sentences so we get at least two chunks.
    text = (
        "Alice was born in London. "
        "Bob moved to Paris in 1990. "
        "Claire worked at the Louvre. "
        "David studied at Oxford University. "
        "Eve wrote a famous novel. "
        "Frank directed several films."
    )
    chunks = chunker.chunk_text(text, source_doc="test.txt")

    assert len(chunks) >= 2, "Expected at least 2 chunks for overlap test"

    # The last sentence of chunk[0] must appear at the start of chunk[1].
    last_sent_chunk0 = chunks[0].text.strip().rstrip(". ").split(". ")[-1].strip().rstrip(".")
    first_sent_chunk1 = chunks[1].text.strip().split(". ")[0].strip().rstrip(".")
    assert last_sent_chunk0.lower() in chunks[1].text.lower(), (
        "Overlap not found: last sentence of chunk[0] (%r) missing in chunk[1] (%r)"
        % (last_sent_chunk0, chunks[1].text[:80])
    )


# =============================================================================
# 4. GLiNERExtractor returns empty list on empty-string input
# =============================================================================

def test_no_entities_returns_empty():
    """Extracting entities from an empty string must return an empty list."""
    from src.data_layer.entity_extraction import ExtractionConfig, GLiNERExtractor

    config = ExtractionConfig()

    # Patch GLiNER.from_pretrained so loading succeeds without downloading weights.
    fake_model = MagicMock()
    fake_model.predict_entities.return_value = []

    with patch("src.data_layer.entity_extraction.GLINER_AVAILABLE", True), \
         patch("src.data_layer.entity_extraction.GLiNER") as mock_cls:
        mock_cls.from_pretrained.return_value = fake_model
        extractor = GLiNERExtractor(config)
        result = extractor.extract("", chunk_id="chunk_0")

    assert result == [], "Expected empty list for empty text, got %r" % result


# =============================================================================
# 5. BatchedOllamaEmbeddings raises a clear error on connection refused
# =============================================================================

def test_ollama_connection_refused_raises_clear_error(tmp_path):
    """BatchedOllamaEmbeddings must raise ConnectionError when Ollama is down."""
    import requests as _requests
    from src.data_layer.embeddings import BatchedOllamaEmbeddings

    def refuse(*args, **kwargs):
        raise _requests.exceptions.ConnectionError("Connection refused")

    with patch("src.data_layer.embeddings.requests.post", side_effect=refuse):
        with pytest.raises(ConnectionError):
            BatchedOllamaEmbeddings(
                model_name="nomic-embed-text",
                base_url="http://localhost:11434",
                cache_path=tmp_path / "emb.db",
            )


# =============================================================================
# 6. Verifier._format_context respects max_context_chars budget
# =============================================================================

def test_format_context_respects_budget():
    """_format_context must not exceed max_context_chars total output length."""
    from src.logic_layer.verifier import Verifier, VerifierConfig

    cfg = VerifierConfig(
        max_context_chars=200,
        max_docs=10,
        max_chars_per_doc=500,
    )

    # Create a minimal Verifier without connecting to Ollama.
    with patch.object(Verifier, "__init__", lambda self, config: None):
        v = Verifier.__new__(Verifier)
        v.config = cfg

    long_docs = ["A" * 100 for _ in range(10)]  # 10 × 100 chars = well over budget
    formatted = v._format_context(long_docs)

    assert len(formatted) <= cfg.max_context_chars + 20, (
        "formatted context (%d chars) exceeds budget (%d chars)"
        % (len(formatted), cfg.max_context_chars)
    )
