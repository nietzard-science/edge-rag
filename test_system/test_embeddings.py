"""
Test suite for src/data_layer/embeddings.py (paper §3.1, Embeddings).

All Ollama API calls are mocked unless a test is marked `llm`; no network
connection is required for the default run. Pins the T-B invariant
(embed_query / embed_documents share dimensionality and yield cosine >= 0.99
for identical text) and the F2 contract (network failures wrap as RuntimeError).

Test inventory
--------------
TestEmbeddingCache                  -- put/get, batch lookup, duplicate
                                       handling, access-count consistency,
                                       model-scope isolation, clear, get_stats,
                                       deterministic 64-hex SHA-256 keys.
TestEmbeddingMetrics                -- cache_hit_rate, avg_time_per_text_ms, reset.
TestBatchedOllamaEmbeddings         -- embed_documents/embed_query cache + order
                                       semantics, total_texts counter, context
                                       manager, dim detection.
TestCreateEmbeddings                -- factory parameter mapping from a settings dict.
TestEmbedQueryDocumentsConsistency  -- T-B: shared space, List[float] dtype,
                                       cosine >= 0.99 for identical text.
TestEmbeddingSemanticQuality        -- live nomic-embed-text quality check
                                       (marked nightly + llm; documents the
                                       score-compression limitation, §4.4).
TestEmbedderErrorHandling           -- F2: ConnectionError / HTTP 500 wrap as RuntimeError.

Dependencies / Requirements
---------------------------
pytest, unittest.mock, requests. The nightly+llm test additionally needs a
live Ollama daemon serving `nomic-embed-text`; it self-skips if unreachable.

Last reviewed: 2026-05-30 (audit pass, project version 5.4).
"""

import sys
import math
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import requests

from src.data_layer.embeddings import (
    BatchedOllamaEmbeddings,
    EmbeddingCache,
    EmbeddingMetrics,
    create_embeddings,
)

# nomic-embed-text output dimensionality. Named once so an ablation that swaps
# the embedding model overrides the vector width in a single place.
EMBED_DIM = 768
# SHA-256 hex digest length; EmbeddingCache keys texts by their full digest.
_SHA256_HEX_LEN = 64


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_response(embeddings_list: List[List[float]]) -> MagicMock:
    """Return a mock requests.Response yielding the given embeddings list."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"embeddings": embeddings_list}
    return resp


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Temporary SQLite database path (isolated per test)."""
    return tmp_path / "test_embeddings.db"


@pytest.fixture
def embedder(tmp_db: Path) -> BatchedOllamaEmbeddings:
    """
    BatchedOllamaEmbeddings instance with a mocked Ollama connection.

    The _test_connection call during __init__ is satisfied by the mock
    returning a single EMBED_DIM-dim vector.
    """
    with patch("requests.post", return_value=_mock_response([[0.0] * EMBED_DIM])):
        return BatchedOllamaEmbeddings(
            model_name="test-model",
            batch_size=2,
            cache_path=tmp_db,
        )


# ── EmbeddingCache ─────────────────────────────────────────────────────────────

class TestEmbeddingCache:

    def test_put_and_get_hit(self, tmp_db: Path) -> None:
        cache = EmbeddingCache(tmp_db)
        cache.put("hello world", [0.1, 0.2, 0.3], "test-model")
        result = cache.get("hello world", "test-model")
        assert result == [0.1, 0.2, 0.3]
        cache.close()

    def test_get_miss_returns_none(self, tmp_db: Path) -> None:
        cache = EmbeddingCache(tmp_db)
        assert cache.get("unknown text", "test-model") is None
        cache.close()

    def test_model_scope_isolation(self, tmp_db: Path) -> None:
        """Entry stored for model-A must not be returned for model-B."""
        cache = EmbeddingCache(tmp_db)
        cache.put("text", [1.0], "model-A")
        assert cache.get("text", "model-B") is None
        cache.close()

    def test_hash_is_deterministic(self, tmp_db: Path) -> None:
        cache = EmbeddingCache(tmp_db)
        h1 = cache._hash_text("deterministic input")
        h2 = cache._hash_text("deterministic input")
        assert h1 == h2
        assert len(h1) == _SHA256_HEX_LEN
        assert all(c in "0123456789abcdef" for c in h1)
        cache.close()

    def test_different_texts_produce_different_hashes(self, tmp_db: Path) -> None:
        cache = EmbeddingCache(tmp_db)
        assert cache._hash_text("text A") != cache._hash_text("text B")
        cache.close()

    def test_access_count_increments_on_get(self, tmp_db: Path) -> None:
        cache = EmbeddingCache(tmp_db)
        cache.put("x", [0.5], "m")
        cache.get("x", "m")
        cache.get("x", "m")
        stats = cache.get_stats()
        # Initial access_count=1 (from put) + 2 get hits = 3
        assert stats["total_accesses"] >= 3
        cache.close()

    def test_clear_removes_all_entries(self, tmp_db: Path) -> None:
        cache = EmbeddingCache(tmp_db)
        cache.put("a", [1.0], "m")
        cache.put("b", [2.0], "m")
        cache.clear()
        assert cache.get("a", "m") is None
        assert cache.get_stats()["total_entries"] == 0
        cache.close()

    def test_get_stats_returns_expected_keys(self, tmp_db: Path) -> None:
        cache = EmbeddingCache(tmp_db)
        stats = cache.get_stats()
        for key in ("total_entries", "total_accesses", "size_bytes", "size_mb"):
            assert key in stats
        cache.close()

    def test_get_batch_returns_hits_only(self, tmp_db: Path) -> None:
        cache = EmbeddingCache(tmp_db)
        cache.put("alpha", [1.0], "m")
        cache.put("beta", [2.0], "m")
        result = cache.get_batch(["alpha", "beta", "gamma"], "m")
        assert 0 in result and result[0] == [1.0]
        assert 1 in result and result[1] == [2.0]
        assert 2 not in result, "Cache miss must not appear in result"
        cache.close()

    def test_get_batch_handles_duplicate_texts(self, tmp_db: Path) -> None:
        """Duplicate input texts must each receive an index in the result."""
        cache = EmbeddingCache(tmp_db)
        cache.put("dup", [9.9], "m")
        result = cache.get_batch(["dup", "dup"], "m")
        assert 0 in result and 1 in result
        assert result[0] == result[1] == [9.9]
        cache.close()

    def test_get_batch_updates_access_count(self, tmp_db: Path) -> None:
        """get_batch must increment access_count (consistency with get())."""
        cache = EmbeddingCache(tmp_db)
        cache.put("trackme", [1.0], "m")
        before = cache.get_stats()["total_accesses"]
        cache.get_batch(["trackme"], "m")
        after = cache.get_stats()["total_accesses"]
        assert after > before
        cache.close()

    def test_get_batch_empty_input(self, tmp_db: Path) -> None:
        cache = EmbeddingCache(tmp_db)
        assert cache.get_batch([], "m") == {}
        cache.close()

    def test_db_path_alias(self, tmp_db: Path) -> None:
        cache = EmbeddingCache(tmp_db)
        assert cache.db_path == cache.cache_path
        cache.close()


# ── EmbeddingMetrics ──────────────────────────────────────────────────────────

class TestEmbeddingMetrics:

    def test_cache_hit_rate_zero_when_empty(self) -> None:
        assert EmbeddingMetrics().cache_hit_rate == 0.0

    def test_cache_hit_rate_calculation(self) -> None:
        m = EmbeddingMetrics(cache_hits=3, cache_misses=1)
        assert m.cache_hit_rate == 75.0

    def test_cache_hit_rate_all_hits(self) -> None:
        m = EmbeddingMetrics(cache_hits=10, cache_misses=0)
        assert m.cache_hit_rate == 100.0

    def test_avg_time_zero_when_no_texts(self) -> None:
        assert EmbeddingMetrics().avg_time_per_text_ms == 0.0

    def test_avg_time_calculation(self) -> None:
        m = EmbeddingMetrics(total_texts=4, total_time_ms=200.0)
        assert m.avg_time_per_text_ms == 50.0

    def test_reset_clears_all_fields(self) -> None:
        m = EmbeddingMetrics(
            total_texts=10, cache_hits=5, cache_misses=5,
            batch_count=2, total_time_ms=100.0,
        )
        m.reset()
        assert m.total_texts == 0
        assert m.cache_hits == 0
        assert m.cache_misses == 0
        assert m.batch_count == 0
        assert m.total_time_ms == 0.0


# ── BatchedOllamaEmbeddings ───────────────────────────────────────────────────

class TestBatchedOllamaEmbeddings:

    def test_embed_documents_returns_correct_count(self, embedder: BatchedOllamaEmbeddings) -> None:
        with patch.object(
            embedder, "_embed_batch",
            return_value=[[0.1] * EMBED_DIM, [0.2] * EMBED_DIM],
        ):
            result = embedder.embed_documents(["A", "B"])
        assert len(result) == 2

    def test_embed_documents_preserves_order(self, embedder: BatchedOllamaEmbeddings) -> None:
        """Output order must match input order regardless of batch boundaries."""
        vecs = [[float(i) / 10] * EMBED_DIM for i in range(4)]
        with patch.object(
            embedder, "_embed_batch",
            side_effect=[vecs[:2], vecs[2:]],
        ):
            result = embedder.embed_documents(["a", "b", "c", "d"])
        for i, vec in enumerate(result):
            assert vec[0] == pytest.approx(i / 10)

    def test_embed_documents_cache_hit_on_second_call(self, embedder: BatchedOllamaEmbeddings) -> None:
        """Second call with same texts must not trigger any API call."""
        with patch.object(
            embedder, "_embed_batch",
            return_value=[[0.5] * EMBED_DIM, [0.6] * EMBED_DIM],
        ):
            embedder.embed_documents(["X", "Y"])

        with patch.object(
            embedder, "_embed_batch",
            side_effect=AssertionError("API must not be called on cache hit"),
        ):
            embedder.embed_documents(["X", "Y"])

        assert embedder.metrics.cache_hits == 2

    def test_embed_documents_increments_total_texts(self, embedder: BatchedOllamaEmbeddings) -> None:
        with patch.object(
            embedder, "_embed_batch",
            return_value=[[0.1] * EMBED_DIM],
        ):
            embedder.embed_documents(["one text"])
        assert embedder.metrics.total_texts == 1

    def test_embed_documents_counts_misses(self, embedder: BatchedOllamaEmbeddings) -> None:
        with patch.object(
            embedder, "_embed_batch",
            return_value=[[0.1] * EMBED_DIM, [0.2] * EMBED_DIM],
        ):
            embedder.embed_documents(["new-a", "new-b"])
        assert embedder.metrics.cache_misses == 2

    def test_embed_documents_empty_input(self, embedder: BatchedOllamaEmbeddings) -> None:
        assert embedder.embed_documents([]) == []
        assert embedder.metrics.total_texts == 0

    def test_embed_query_increments_total_texts(self, embedder: BatchedOllamaEmbeddings) -> None:
        """embed_query must increment total_texts (invariant: both paths update
        the counter, not only embed_documents)."""
        with patch.object(
            embedder, "_embed_batch",
            return_value=[[0.3] * EMBED_DIM],
        ):
            embedder.embed_query("query text")
        assert embedder.metrics.total_texts == 1

    def test_embed_query_cache_hit_does_not_call_api(self, embedder: BatchedOllamaEmbeddings) -> None:
        """After a first embed_query call, a second identical call must be
        served from cache without any API request."""
        with patch.object(
            embedder, "_embed_batch",
            return_value=[[0.7] * EMBED_DIM],
        ):
            embedder.embed_query("cached query")

        with patch.object(
            embedder, "_embed_batch",
            side_effect=AssertionError("API must not be called on cache hit"),
        ):
            embedder.embed_query("cached query")

        assert embedder.metrics.cache_hits >= 1

    def test_embed_query_total_texts_increments_on_cache_hit(self, embedder: BatchedOllamaEmbeddings) -> None:
        """total_texts must be incremented even when the result comes from cache."""
        with patch.object(
            embedder, "_embed_batch",
            return_value=[[0.7] * EMBED_DIM],
        ):
            embedder.embed_query("repeat me")
            embedder.embed_query("repeat me")
        assert embedder.metrics.total_texts == 2

    def test_clear_cache_resets_metrics(self, embedder: BatchedOllamaEmbeddings) -> None:
        embedder.metrics.total_texts = 42
        embedder.metrics.cache_hits = 10
        embedder.clear_cache()
        assert embedder.metrics.total_texts == 0
        assert embedder.metrics.cache_hits == 0

    def test_get_metrics_returns_all_keys(self, embedder: BatchedOllamaEmbeddings) -> None:
        m = embedder.get_metrics()
        for key in (
            "total_texts", "cache_hits", "cache_misses",
            "cache_hit_rate", "batch_count", "total_time_ms",
            "avg_time_per_text_ms",
        ):
            assert key in m, f"Missing key: {key}"

    def test_embedding_dim_detected_at_init(self, embedder: BatchedOllamaEmbeddings) -> None:
        """_embedding_dim must be set after the connection test."""
        assert embedder.embedding_dim == EMBED_DIM

    def test_context_manager_closes_cache(self, tmp_db: Path) -> None:
        """with-statement must close the SQLite connection cleanly."""
        with patch("requests.post", return_value=_mock_response([[0.0] * EMBED_DIM])):
            with BatchedOllamaEmbeddings(
                model_name="m", batch_size=2, cache_path=tmp_db,
            ) as emb:
                pass
        assert emb.cache.conn is None, "Cache connection must be closed after __exit__"


# ── create_embeddings factory ─────────────────────────────────────────────────

class TestCreateEmbeddings:

    def test_factory_reads_settings(self, tmp_db: Path) -> None:
        """All settings.yaml entries must be forwarded to the embedder."""
        cfg = {
            "embeddings": {
                "model_name": "custom-model",
                "base_url": "http://localhost:11434",
                "cache_path": str(tmp_db),
            },
            "performance": {"batch_size": 16, "device": "cpu"},
            "llm": {"timeout": 30},
        }
        with patch(
            "requests.post",
            return_value=_mock_response([[0.0] * EMBED_DIM]),
        ):
            emb = create_embeddings(cfg)

        assert emb.model_name == "custom-model"
        assert emb.batch_size == 16
        assert emb.timeout == 30

    def test_factory_none_uses_defaults(self, tmp_db: Path) -> None:
        """create_embeddings(None) must use class-level defaults without error."""
        with patch(
            "requests.post",
            return_value=_mock_response([[0.0] * EMBED_DIM]),
        ):
            # Redirect default cache path to temp dir to avoid touching ./cache/
            _orig = BatchedOllamaEmbeddings.__init__

            def _patched(self, *args, **kwargs):
                kwargs["cache_path"] = tmp_db
                _orig(self, *args, **kwargs)

            with patch.object(BatchedOllamaEmbeddings, "__init__", _patched):
                emb = create_embeddings(None)

        # batch_size default must match settings.yaml performance.batch_size = 64
        assert emb.batch_size == 64


# ── Query/Document consistency (T-B) ──────────────────────────────────────────

class TestEmbedQueryDocumentsConsistency:
    """embed_query and embed_documents must produce vectors in the same space.

    Consistency is a prerequisite for cosine similarity to be meaningful:
    if the two paths returned different dimensionalities or dtypes, ANN search
    results would be silently wrong.
    """

    def test_query_and_document_embeddings_have_same_dimension(self, embedder: BatchedOllamaEmbeddings) -> None:
        """embed_query and embed_documents[0] must have identical length."""
        vec = [float(i) / EMBED_DIM for i in range(EMBED_DIM)]
        with patch.object(embedder, "_embed_batch", return_value=[vec]):
            q_emb = embedder.embed_query("What is relativity?")
        with patch.object(embedder, "_embed_batch", return_value=[vec]):
            d_emb = embedder.embed_documents(["Relativity is a theory by Einstein."])[0]
        assert len(q_emb) == len(d_emb), (
            f"embed_query dim={len(q_emb)} != embed_documents dim={len(d_emb)}"
        )

    def test_query_embedding_is_list_of_floats(self, embedder: BatchedOllamaEmbeddings) -> None:
        """embed_query must return List[float]."""
        vec = [0.1] * EMBED_DIM
        with patch.object(embedder, "_embed_batch", return_value=[vec]):
            emb = embedder.embed_query("test")
        assert isinstance(emb, list)
        assert all(isinstance(v, float) for v in emb), (
            "embed_query must return a list of floats, not %s" % type(emb[0])
        )

    def test_document_embedding_is_list_of_floats(self, embedder: BatchedOllamaEmbeddings) -> None:
        """embed_documents must return List[List[float]]."""
        vec = [0.2] * EMBED_DIM
        with patch.object(embedder, "_embed_batch", return_value=[vec]):
            embs = embedder.embed_documents(["test document"])
        assert isinstance(embs, list) and len(embs) == 1
        assert all(isinstance(v, float) for v in embs[0]), (
            "embed_documents must return List[List[float]]"
        )

    def test_cosine_similarity_of_identical_text_near_one(self, embedder: BatchedOllamaEmbeddings) -> None:
        """Same text embedded via query and document paths yields cosine >= 0.99.

        Both paths call _embed_batch with the same text and (via the cache)
        return the identical vector, so cosine must be exactly 1.0 or extremely
        close.  Threshold 0.99 leaves headroom for float32 rounding artefacts.
        """
        vec = [float(i + 1) / EMBED_DIM for i in range(EMBED_DIM)]  # non-trivial vector
        mag = math.sqrt(sum(v ** 2 for v in vec))
        unit_vec = [v / mag for v in vec]

        text = "Albert Einstein was born in Ulm."
        with patch.object(embedder, "_embed_batch", return_value=[unit_vec]):
            q = embedder.embed_query(text)
        with patch.object(embedder, "_embed_batch", return_value=[unit_vec]):
            d = embedder.embed_documents([text])[0]

        dot = sum(a * b for a, b in zip(q, d))
        mag_q = math.sqrt(sum(v ** 2 for v in q)) + 1e-12
        mag_d = math.sqrt(sum(v ** 2 for v in d)) + 1e-12
        cosine = dot / (mag_q * mag_d)
        assert cosine >= 0.99, (
            f"Same text embedded via both paths: cosine={cosine:.6f} < 0.99"
        )


# ── Live semantic-quality probe (nightly + llm) ───────────────────────────────

class TestEmbeddingSemanticQuality:
    """Nightly semantic quality test using a live nomic-embed-text connection.

    Documents the known score-compression limitation (paper §4.4): ALL text
    pairs from nomic-embed-text score 0.739–0.786 on the project's corpus,
    making similarity thresholds below 0.7 effectively useless.

    Run:
        EDGE_RAG_N_SAMPLES=10 python -X utf8 -m pytest test_system/test_embeddings.py
            -m "nightly and llm" -v
    """

    @pytest.mark.nightly
    @pytest.mark.llm
    def test_similar_texts_outscore_dissimilar_texts(self, tmp_db: Path) -> None:
        """Semantically similar query must score higher against a related document
        than against a dissimilar one.

        Live-API budget: exactly 2 real inference calls (bounds the cost of this
        llm-marked test).
          Call 1: embed_documents([anchor, dissimilar])  — 2 texts, 1 batch
          Call 2: embed_query(similar)                   — 1 text (not cached)
        Expected: cosine(query_similar, doc_anchor) > cosine(query_similar, doc_dissimilar)
        """
        try:
            requests.get("http://localhost:11434/api/tags", timeout=2)
        except (requests.ConnectionError, requests.Timeout):
            pytest.skip("Ollama not reachable — skipping live embedding test")

        with patch(
            "requests.post",
            return_value=_mock_response([[0.0] * EMBED_DIM]),
        ):
            emb = BatchedOllamaEmbeddings(
                model_name="nomic-embed-text",
                batch_size=2,
                cache_path=tmp_db,
            )

        anchor = "Albert Einstein was born in Ulm, Germany."
        dissimilar = "The Eiffel Tower is a famous landmark in Paris."
        similar = "Einstein's birthplace is Ulm, a city in Germany."

        def cosine(u, v):
            dot = sum(a * b for a, b in zip(u, v))
            mag = math.sqrt(sum(x**2 for x in u)) * math.sqrt(sum(x**2 for x in v))
            return dot / (mag + 1e-12)

        # Call 1: embed anchor + dissimilar (fits in batch_size=2 → 1 API request)
        vecs = emb.embed_documents([anchor, dissimilar])
        # Call 2: embed similar query (new text, not in cache → 1 API request)
        q_similar = emb.embed_query(similar)

        sim_score = cosine(q_similar, vecs[0])   # similar ↔ anchor
        dis_score = cosine(q_similar, vecs[1])   # similar ↔ dissimilar
        assert sim_score > dis_score, (
            f"Similar-to-anchor cosine {sim_score:.4f} must exceed "
            f"similar-to-dissimilar {dis_score:.4f}. "
            f"nomic-embed-text score-compression may be extreme (see §4.4)."
        )


# ── Network-failure handling (F2 coverage) ────────────────────────────────────

class TestEmbedderErrorHandling:
    """Network-failure tests for BatchedOllamaEmbeddings._embed_batch (F2 coverage).

    Contract: _embed_batch must never propagate raw requests.ConnectionError.
    It wraps all network-layer failures as RuntimeError so callers only need
    to handle one exception type for API errors.
    """

    def test_connection_error_raises_runtime_not_raw(self, embedder: BatchedOllamaEmbeddings) -> None:
        """requests.ConnectionError from Ollama must be re-raised as RuntimeError."""
        with patch(
            "requests.post",
            side_effect=requests.ConnectionError("Connection refused"),
        ):
            with pytest.raises(RuntimeError, match="not reachable"):
                embedder.embed_query("test query")

    def test_http_500_raises_runtime_error(self, embedder: BatchedOllamaEmbeddings) -> None:
        """HTTP 500 from Ollama API must raise RuntimeError with status code."""
        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_500.text = "Internal Server Error"
        with patch("requests.post", return_value=resp_500):
            with pytest.raises(RuntimeError, match="500"):
                embedder.embed_query("test query")
