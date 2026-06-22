"""
Batched Embeddings with Persistent Content-Addressed Cache

This module implements the embedding infrastructure for the Edge-RAG pipeline.
It overcomes the performance limitations of the standard LangChain
OllamaEmbeddings implementation — one HTTP request per text, no caching —
which are critical bottlenecks during document ingestion on edge hardware where
network round-trip overhead dominates total ingestion time.

Architectural position: Data Layer (Artifact A).
Consumed by: ingestion.py (document indexing pipeline) and
hybrid_retriever.py (query-time vector search). Instantiated exclusively via
the create_embeddings(cfg) factory to ensure settings.yaml is the single source
of truth for all parameters.

Design:
  BatchedOllamaEmbeddings groups texts into configurable batches and reduces N
  sequential API calls to ceil(N / batch_size) requests. A persistent SQLite
  cache keyed by SHA-256 hashes of the input text eliminates redundant
  re-embedding across runs — essential for iterative development cycles on
  devices with limited CPU and memory.

  Batching: N=1000, batch_size=64 → 16 API calls vs 1000 sequential.
  Caching:  cache hit ~0.1 ms vs ~50 ms fresh embedding (~500× speedup).
  Note: Empirical speedup values are reported in the paper evaluation chapter.

Embedding model:
  nomic-embed-text (768 dimensions, Apache 2.0 license).
  Reference: Nussbaum, Z. et al. (2024). "Nomic Embed: Training a Reproducible
  Long Context Text Embedder." arXiv:2402.01613.

Cache design:
  Content-addressable storage: SHA-256(text.encode("utf-8")) as primary key.
  Collision resistance: 2^128 birthday bound — collision-free for all practical
  corpus sizes. Backend: SQLite in WAL mode for ACID compliance without a
  server process.

Classes:
  EmbeddingMetrics          -- dataclass for cumulative performance counters
                               (internal; not re-exported from the package)
  EmbeddingCache            -- SQLite-based persistent embedding store
                               (internal; not re-exported from the package)
  BatchedOllamaEmbeddings   -- LangChain-compatible embedding client
                               (public API; re-exported from src.data_layer)

Settings.yaml integrity check
-----------------------------
If `embeddings.embedding_dim` is set in settings.yaml and the dimension
detected from the live Ollama API differs, `create_embeddings()` raises
`RuntimeError` at startup. This prevents silent shape drift between the
embedding layer and the storage layer (which enforces `embedding_dim` on
LanceDB writes) when the embedding model is swapped.

Last reviewed: 2026-05-25 (audit pass, project version 5.4).
"""

import logging
import hashlib
import sqlite3
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

import requests
from tqdm import tqdm
from langchain_core.embeddings import Embeddings


logger = logging.getLogger(__name__)


# Public API. EmbeddingCache and EmbeddingMetrics are implementation details
# of BatchedOllamaEmbeddings and intentionally NOT re-exported -- grep confirms
# zero external import sites for either class.
__all__ = [
    "BatchedOllamaEmbeddings",
    "create_embeddings",
]


# ============================================================================
# METRICS
# ============================================================================

@dataclass
class EmbeddingMetrics:
    """
    Cumulative performance metrics for a BatchedOllamaEmbeddings session.

    Tracks texts processed by both embed_documents and embed_query.
    Useful for profiling cache effectiveness and batch throughput during
    paper evaluation.

    Attributes:
        total_texts:    Total texts processed (documents + queries).
        cache_hits:     Texts served from cache without an API call.
        cache_misses:   Texts that required a fresh API call.
        batch_count:    Number of API batch requests issued.
        total_time_ms:  Cumulative wall-clock time in milliseconds.
    """
    total_texts: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    batch_count: int = 0
    total_time_ms: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate in percent [0.0, 100.0]. Returns 0.0 if no texts processed."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100.0

    @property
    def avg_time_per_text_ms(self) -> float:
        """
        Average processing time per text in milliseconds.

        Includes both cached (fast) and non-cached (slow) texts.
        Returns 0.0 if no texts processed.
        """
        if self.total_texts == 0:
            return 0.0
        return self.total_time_ms / self.total_texts

    def reset(self) -> None:
        """Reset all counters and timers to zero."""
        self.total_texts = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.batch_count = 0
        self.total_time_ms = 0.0


# ============================================================================
# EMBEDDING CACHE
# ============================================================================

class EmbeddingCache:
    """
    SQLite-based persistent cache for text embeddings.

    DESIGN RATIONALE

    Content-Addressable Storage:
        Key = SHA-256(text.encode("utf-8")). Properties:
        - Deduplication: identical texts map to a single entry.
        - O(1) lookup via SQLite B-tree index.
        - Deterministic: same text always produces the same hash.

    Why SQLite:
        Embedded, zero-configuration, ACID-compliant, cross-platform.
        WAL journal mode gives better read concurrency without a server
        process — suitable for edge deployment with a single writer.

    Schema:
        embeddings (
            text_hash     TEXT PRIMARY KEY,  -- SHA-256 (64 hex chars)
            text_content  TEXT NOT NULL,     -- original text
            embedding     BLOB NOT NULL,     -- JSON-encoded float vector
            model_name    TEXT NOT NULL,     -- model identifier
            created_at    TIMESTAMP,
            access_count  INTEGER            -- usage frequency tracking
        )

    Cache size:
        No automatic eviction is implemented. Monitor via get_stats() and
        call clear() before experiments that require a clean timing baseline.
    """

    SCHEMA_VERSION = "2.1.0"

    def __init__(self, cache_path: Path) -> None:
        """
        Initialize embedding cache.

        Creates the database file and schema if they do not exist;
        opens an existing database otherwise. Parent directories are
        created automatically.

        Args:
            cache_path: Path to the SQLite database file.
        """
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None
        self._init_db()

    @property
    def db_path(self) -> Path:
        """Read-only alias for cache_path (the SQLite database file)."""
        return self.cache_path

    def _init_db(self) -> None:
        """
        Initialize SQLite schema: embeddings table, index, metadata table.

        WAL mode improves read throughput under the single-writer workload
        typical of edge ingestion pipelines.
        """
        self.conn = sqlite3.connect(
            str(self.cache_path),
            check_same_thread=False,
        )
        self.conn.execute("PRAGMA journal_mode=WAL")

        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash    TEXT PRIMARY KEY,
                text_content TEXT NOT NULL,
                embedding    BLOB NOT NULL,
                model_name   TEXT NOT NULL,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1
            )
        """)
        # Composite index on (model_name, text_hash) speeds up model-scoped
        # lookups in get() and get_batch().
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_hash
            ON embeddings(model_name, text_hash)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('schema_version', ?)",
            (self.SCHEMA_VERSION,),
        )
        self.conn.commit()
        logger.debug("Embedding cache initialized: %s", self.cache_path)

    def _hash_text(self, text: str) -> str:
        """
        SHA-256 hash of text for content addressing.

        Output: 64 lowercase hexadecimal characters (256 bits).
        Collision resistance: 2^128 birthday bound.
        Speed: ~500 MB/s on modern CPUs.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """
        Retrieve a single embedding from the cache.

        Updates access_count on a cache hit. Returns None on a miss or
        database error (triggers re-embedding by the caller).

        Args:
            text:       Input text whose embedding is requested.
            model_name: Embedding model identifier.

        Returns:
            Embedding vector as list of floats, or None.
        """
        text_hash = self._hash_text(text)
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT embedding FROM embeddings "
                "WHERE text_hash = ? AND model_name = ?",
                (text_hash, model_name),
            )
            row = cursor.fetchone()
            if row is not None:
                embedding = json.loads(row[0])
                cursor.execute(
                    "UPDATE embeddings "
                    "SET access_count = access_count + 1 WHERE text_hash = ?",
                    (text_hash,),
                )
                self.conn.commit()
                return embedding
            return None
        except (sqlite3.Error, AttributeError) as e:
            logger.error("Cache GET failed: %s", e)
            return None

    def put(self, text: str, embedding: List[float], model_name: str) -> None:
        """
        Store an embedding in the cache (upsert semantics).

        Args:
            text:       Original text that was embedded.
            embedding:  Embedding vector.
            model_name: Embedding model identifier.
        """
        text_hash = self._hash_text(text)
        embedding_json = json.dumps(embedding)
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO embeddings
                    (text_hash, text_content, embedding, model_name, access_count)
                VALUES (?, ?, ?, ?, 1)
                """,
                (text_hash, text, embedding_json, model_name),
            )
            self.conn.commit()
        except (sqlite3.Error, AttributeError) as e:
            logger.error("Cache PUT failed: %s", e)

    def get_batch(
        self,
        texts: List[str],
        model_name: str,
    ) -> Dict[int, List[float]]:
        """
        Retrieve embeddings for multiple texts in a single SQL query.

        More efficient than N individual get() calls: one SELECT IN (...)
        replaces N separate round-trips. Bulk-updates access_count for
        all cache hits.

        Args:
            texts:      Texts to look up (may contain duplicates).
            model_name: Embedding model identifier.

        Returns:
            Dict mapping text index (position in ``texts``) to embedding.
            Only indices with a cache hit are present.
        """
        if not texts:
            return {}

        # Map each unique hash to all indices that share it.
        # setdefault avoids silently dropping duplicate texts.
        hash_to_idxs: Dict[str, List[int]] = {}
        for i, t in enumerate(texts):
            hash_to_idxs.setdefault(self._hash_text(t), []).append(i)
        hashes = list(hash_to_idxs.keys())

        placeholders = ",".join(["?"] * len(hashes))
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                f"SELECT text_hash, embedding FROM embeddings "
                f"WHERE model_name = ? AND text_hash IN ({placeholders})",
                [model_name] + hashes,
            )
            rows = cursor.fetchall()

            results: Dict[int, List[float]] = {}
            hit_hashes: List[str] = []
            for text_hash, embedding_json in rows:
                embedding = json.loads(embedding_json)
                for idx in hash_to_idxs[text_hash]:
                    results[idx] = embedding
                hit_hashes.append(text_hash)

            # Bulk-update access counts for all cache hits.
            if hit_hashes:
                ph = ",".join(["?"] * len(hit_hashes))
                cursor.execute(
                    f"UPDATE embeddings "
                    f"SET access_count = access_count + 1 "
                    f"WHERE text_hash IN ({ph})",
                    hit_hashes,
                )
                self.conn.commit()

            return results
        except (sqlite3.Error, AttributeError) as e:
            logger.error("Cache batch GET failed: %s", e)
            return {}

    def clear(self) -> None:
        """
        Delete all cached embeddings.

        Intended for ablation studies requiring reproducible fresh timings.
        This operation is irreversible.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM embeddings")
            self.conn.commit()
            logger.info("Embedding cache cleared")
        except (sqlite3.Error, AttributeError) as e:
            logger.error("Cache CLEAR failed: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        """
        Return cache statistics.

        Returns:
            Dict with keys: total_entries, total_accesses, size_bytes, size_mb.
            Returns zero-valued dict if the connection is closed or unavailable.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT COUNT(*), COALESCE(SUM(access_count), 0) FROM embeddings"
            )
            row = cursor.fetchone()
            size_bytes = self.cache_path.stat().st_size if self.cache_path.exists() else 0
            return {
                "total_entries": row[0] or 0,
                "total_accesses": row[1] or 0,
                "size_bytes": size_bytes,
                "size_mb": size_bytes / (1024 * 1024),
            }
        except (sqlite3.Error, AttributeError) as e:
            logger.error("Cache get_stats failed (connection closed?): %s", e)
            return {"total_entries": 0, "total_accesses": 0, "size_bytes": 0, "size_mb": 0.0}

    def close(self) -> None:
        """Close the SQLite connection."""
        if self.conn:
            self.conn.close()
            self.conn = None


# ============================================================================
# BATCHED OLLAMA EMBEDDINGS
# ============================================================================

class BatchedOllamaEmbeddings(Embeddings):
    """
    High-performance Ollama embeddings with batching and persistent caching.

    Implements the LangChain Embeddings interface and can be used as a drop-in
    replacement for OllamaEmbeddings. Key improvements over the standard
    implementation:

    Batching:
        Let N = texts, B = batch_size, T = latency per API call.
        Sequential: N × T.  Batched: ceil(N/B) × T.  Speedup: min(N, B)×.
        Example (N=1000, B=64, T=50 ms): 16 × 50 ms = 0.8 s vs 50 s.

    Caching:
        Let H = cache hit rate, T_c = cache lookup time, T_e = embed time.
        Expected time per text: H × T_c + (1−H) × T_e.
        Example (H=0.8, T_c=0.1 ms, T_e=50 ms): ~10 ms vs 50 ms (5× speedup).

    Ollama /api/embed endpoint (used directly):
        POST /api/embed
        Body:     {"model": "nomic-embed-text", "input": ["t1", "t2", ...]}
        Response: {"embeddings": [[0.1, ...], [0.3, ...], ...]}

    Attributes:
        model_name:  Ollama model identifier (e.g. "nomic-embed-text").
        base_url:    Ollama API base URL.
        batch_size:  Texts per API call.
        device:      Informational label only — Ollama handles device selection.
        timeout:     HTTP request timeout in seconds.
        cache:       EmbeddingCache instance.
        metrics:     EmbeddingMetrics instance.
    """

    # Class-level defaults — used when BatchedOllamaEmbeddings is instantiated
    # directly rather than via create_embeddings().
    # Must stay aligned with settings.yaml entries:
    #   DEFAULT_MODEL      → embeddings.model_name
    #   DEFAULT_URL        → embeddings.base_url
    #   DEFAULT_BATCH_SIZE → performance.batch_size
    #   DEFAULT_TIMEOUT    → llm.timeout
    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_URL = "http://localhost:11434"
    DEFAULT_BATCH_SIZE = 64   # matches performance.batch_size in settings.yaml
    DEFAULT_TIMEOUT = 60      # matches llm.timeout in settings.yaml

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_URL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        cache_path: Path = Path("./cache/embeddings.db"),
        device: str = "cpu",   # informational only — Ollama handles device selection
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        """
        Initialize BatchedOllamaEmbeddings.

        Args:
            model_name:  Ollama embedding model name.
            base_url:    Ollama API base URL.
            batch_size:  Texts per API batch. Reduce on OOM errors.
            cache_path:  Path to the SQLite embedding cache.
            device:      Informational label ("cpu"/"gpu"). Not used at runtime.
            timeout:     API request timeout in seconds.

        Raises:
            ConnectionError: Ollama server is unreachable or timed out.
            RuntimeError:    Embedding model is not available (HTTP 404).
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.device = device
        self.timeout = timeout
        self.cache = EmbeddingCache(cache_path)
        self.metrics = EmbeddingMetrics()
        self._embedding_dim: Optional[int] = None
        self._test_connection()

    def _test_connection(self) -> None:
        """
        Verify Ollama API availability and model presence.

        Uses min(10, self.timeout) as the probe timeout to avoid blocking
        indefinitely while still respecting very short configured timeouts.

        Raises:
            ConnectionError: API unreachable or timed out.
            RuntimeError:    Model not found (HTTP 404).
        """
        connection_timeout = min(10, self.timeout)
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model_name, "input": ["connection test"]},
                timeout=connection_timeout,
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("embeddings"):
                    self._embedding_dim = len(data["embeddings"][0])
                logger.info(
                    "Ollama connection verified: model=%s, dim=%s, url=%s",
                    self.model_name, self._embedding_dim, self.base_url,
                )
            elif response.status_code == 404:
                raise RuntimeError(
                    f"Model '{self.model_name}' not found. "
                    f"Pull it with: ollama pull {self.model_name}"
                )
            else:
                raise ConnectionError(
                    f"Ollama API error: HTTP {response.status_code}"
                )
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Ensure Ollama is running: ollama serve"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise ConnectionError(
                "Ollama connection timeout. Server may be overloaded."
            ) from exc

    @property
    def embedding_dim(self) -> Optional[int]:
        """Embedding dimensionality, detected at initialization."""
        return self._embedding_dim

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Send a single batch to the Ollama /api/embed endpoint.

        Args:
            texts: Up to batch_size texts to embed.

        Returns:
            List of embedding vectors, same length as input.

        Raises:
            RuntimeError: API error, count mismatch, or request timeout.
        """
        if not texts:
            return []
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model_name, "input": texts},
                timeout=self.timeout,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama API error: HTTP {response.status_code} - "
                    f"{response.text[:200]}"
                )
            data = response.json()
            embeddings = data.get("embeddings", [])
            if len(embeddings) != len(texts):
                raise RuntimeError(
                    f"Embedding count mismatch: expected {len(texts)}, "
                    f"got {len(embeddings)}"
                )
            if self._embedding_dim is None and embeddings:
                self._embedding_dim = len(embeddings[0])
            return embeddings
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama API timeout after {self.timeout}s. "
                f"Consider reducing batch_size or increasing timeout."
            )
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Ollama not reachable at {self.base_url} (connection refused). "
                f"Ensure Ollama is running: {e}"
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Embedding API error: {e}") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents with batching and caching.

        Algorithm:
          Phase 1 — Batch cache lookup: single SQL query for all N texts.
          Phase 2 — API batching: ceil((N−H) / batch_size) requests for cache
                    misses; results stored in cache immediately.
          Phase 3 — Reassembly: results sorted to restore original input order.

        Complexity:
          Cache lookup: O(N) hashes + O(1) SQL query.
          API calls:    ceil((N−H) / B) HTTP requests.
          Assembly:     O(N log N).

        Args:
            texts: Text strings to embed.

        Returns:
            Embedding vectors in the same order as the input list.
        """
        if not texts:
            return []

        start_time = time.time()
        results: List[Tuple[int, List[float]]] = []
        texts_to_embed: List[str] = []
        indices_to_embed: List[int] = []
        cache_hits_this_run = 0
        cache_misses_this_run = 0

        # Phase 1: Batch cache lookup — one SQL query replaces N round-trips.
        cached = self.cache.get_batch(texts, self.model_name)
        for i, text in enumerate(texts):
            if i in cached:
                results.append((i, cached[i]))
                cache_hits_this_run += 1
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
                cache_misses_this_run += 1

        self.metrics.total_texts += len(texts)
        self.metrics.cache_hits += cache_hits_this_run
        self.metrics.cache_misses += cache_misses_this_run

        # Phase 2: Batch API calls for cache misses.
        batches_this_run = 0
        if texts_to_embed:
            num_batches = (len(texts_to_embed) + self.batch_size - 1) // self.batch_size
            logger.info(
                "Generating embeddings for %d texts in %d batches...",
                len(texts_to_embed), num_batches,
            )
            for batch_idx in tqdm(
                range(num_batches),
                desc="Embedding batches",
                unit="batch",
                disable=num_batches < 3,
            ):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(texts_to_embed))
                batch_texts = texts_to_embed[start_idx:end_idx]
                batch_embeddings = self._embed_batch(batch_texts)
                for j, (text, embedding) in enumerate(
                    zip(batch_texts, batch_embeddings)
                ):
                    self.cache.put(text, embedding, self.model_name)
                    results.append((indices_to_embed[start_idx + j], embedding))
                batches_this_run += 1
            self.metrics.batch_count += batches_this_run

        # Phase 3: Restore original input order.
        results.sort(key=lambda x: x[0])
        embeddings = [emb for _, emb in results]

        elapsed_ms = (time.time() - start_time) * 1000
        self.metrics.total_time_ms += elapsed_ms
        cache_hit_rate = cache_hits_this_run / len(texts) * 100
        logger.info(
            "Embedded %d texts: cache_hit_rate=%.1f%%, batches=%d, "
            "time=%.1fms, avg=%.2fms/text",
            len(texts), cache_hit_rate, batches_this_run,
            elapsed_ms, elapsed_ms / len(texts),
        )
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string with caching.

        Args:
            text: Query text.

        Returns:
            Embedding vector as list of floats.
        """
        self.metrics.total_texts += 1  # count queries alongside documents
        cached = self.cache.get(text, self.model_name)
        if cached is not None:
            self.metrics.cache_hits += 1
            return cached
        self.metrics.cache_misses += 1
        embedding = self._embed_batch([text])[0]
        # Count the API round-trip in batch_count so cumulative metrics stay
        # accurate when queries dominate the workload (embed_documents accounts
        # for its own batches via the inner loop).
        self.metrics.batch_count += 1
        self.cache.put(text, embedding, self.model_name)
        return embedding

    def clear_cache(self) -> None:
        """
        Clear the embedding cache and reset session metrics.

        Call before ablation studies to ensure reproducible timing baselines.
        """
        self.cache.clear()
        self.metrics.reset()
        logger.info("Embedding cache and metrics cleared")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return current session metrics as a dictionary.

        Returns:
            Dict with keys: total_texts, cache_hits, cache_misses,
            cache_hit_rate, batch_count, total_time_ms, avg_time_per_text_ms.
        """
        return {
            "total_texts": self.metrics.total_texts,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "cache_hit_rate": self.metrics.cache_hit_rate,
            "batch_count": self.metrics.batch_count,
            "total_time_ms": self.metrics.total_time_ms,
            "avg_time_per_text_ms": self.metrics.avg_time_per_text_ms,
        }

    def print_metrics(self) -> None:
        """
        Log formatted performance metrics at INFO level.

        Useful for profiling sessions and documenting paper evaluation results.
        """
        cache_stats = self.cache.get_stats()
        msg = "\n".join([
            "",
            "=" * 70,
            "EMBEDDING PERFORMANCE METRICS",
            "=" * 70,
            f"Model:        {self.model_name}",
            f"Dimension:    {self._embedding_dim}",
            f"Batch Size:   {self.batch_size}",
            "",
            "Session Metrics:",
            f"  Total Texts:     {self.metrics.total_texts}",
            f"  Cache Hits:      {self.metrics.cache_hits}",
            f"  Cache Misses:    {self.metrics.cache_misses}",
            f"  Cache Hit Rate:  {self.metrics.cache_hit_rate:.1f}%",
            f"  Batch Count:     {self.metrics.batch_count}",
            f"  Total Time:      {self.metrics.total_time_ms:.1f}ms",
            f"  Avg Time/Text:   {self.metrics.avg_time_per_text_ms:.2f}ms",
            "",
            "Cache Statistics:",
            f"  Cached Entries:  {cache_stats['total_entries']}",
            f"  Total Accesses:  {cache_stats['total_accesses']}",
            f"  Cache Size:      {cache_stats['size_mb']:.2f} MB",
            "=" * 70,
        ])
        logger.info(msg)

    # ---- Context Manager Protocol ----------------------------------------

    def __enter__(self) -> "BatchedOllamaEmbeddings":
        """Support ``with BatchedOllamaEmbeddings(...) as emb:`` usage."""
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        """Close the cache connection on context manager exit."""
        self.cache.close()

    def __del__(self) -> None:
        """Safety net: close cache connection if context manager was not used."""
        if hasattr(self, "cache") and self.cache is not None:
            self.cache.close()


# ============================================================================
# FACTORY
# ============================================================================

def create_embeddings(
    cfg: Optional[Dict[str, Any]] = None,
) -> BatchedOllamaEmbeddings:
    """
    Factory for BatchedOllamaEmbeddings that reads all parameters from the
    settings.yaml configuration dictionary.

    Parameter mapping from settings.yaml:
        embeddings.model_name     -> model_name
        embeddings.base_url       -> base_url
        embeddings.cache_path     -> cache_path
        embeddings.embedding_dim  -> startup cross-check (see below)
        performance.batch_size    -> batch_size
        performance.device        -> device
        llm.timeout               -> timeout

    Settings.yaml integrity check
    -----------------------------
    If `embeddings.embedding_dim` is set in settings.yaml, the value is
    compared against the dimension the Ollama API actually returned during
    `_test_connection()`. A mismatch raises `RuntimeError` at construction
    time. This catches model-swap inconsistencies (where the embedding
    model now serves a different dim than `storage.py` is configured to
    accept on LanceDB writes) at the point where the error message is
    actionable, rather than as a cryptic LanceDB shape error later.

    Args:
        cfg: Full settings dict as loaded from config/settings.yaml.
             Pass None or {} to fall back to class-level defaults.

    Returns:
        Configured BatchedOllamaEmbeddings instance.

    Raises:
        RuntimeError: configured `embeddings.embedding_dim` does not match
                      the dim returned by the live Ollama API.
    """
    cfg = cfg or {}
    emb_cfg = cfg.get("embeddings", {})
    perf_cfg = cfg.get("performance", {})
    llm_cfg = cfg.get("llm", {})

    _D = BatchedOllamaEmbeddings
    emb = BatchedOllamaEmbeddings(
        model_name=emb_cfg.get("model_name", _D.DEFAULT_MODEL),
        base_url=emb_cfg.get("base_url", _D.DEFAULT_URL),
        batch_size=perf_cfg.get("batch_size", _D.DEFAULT_BATCH_SIZE),
        cache_path=Path(emb_cfg.get("cache_path", "./cache/embeddings.db")),
        device=perf_cfg.get("device", "cpu"),
        timeout=llm_cfg.get("timeout", _D.DEFAULT_TIMEOUT),
    )

    configured_dim = emb_cfg.get("embedding_dim")
    if configured_dim is not None and emb.embedding_dim is not None:
        if int(configured_dim) != emb.embedding_dim:
            raise RuntimeError(
                f"Embedding dimension mismatch: settings.yaml declares "
                f"embeddings.embedding_dim={configured_dim} but the Ollama "
                f"model '{emb.model_name}' returned vectors of dimension "
                f"{emb.embedding_dim}. Either pull the correct model "
                f"(ollama pull {emb.model_name}) or update "
                f"embeddings.embedding_dim in config/settings.yaml to match."
            )
    return emb


# ============================================================================
# SMOKE DEMO
# ============================================================================

if __name__ == "__main__":
    import tempfile
    from unittest.mock import MagicMock, patch

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    def _make_response(vecs):
        r = MagicMock()
        r.status_code = 200
        r.json.return_value = {"embeddings": vecs}
        return r

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db_path = Path(tmp) / "demo.db"
        texts = [
            "Edge AI on resource-constrained devices.",
            "Hybrid retrieval-augmented generation.",
            "Quantization reduces model memory footprint.",
        ]

        with patch(
            "requests.post",
            side_effect=[
                _make_response([[0.0] * 768]),           # _test_connection
                _make_response([[float(i) / 10] * 768 for i in range(3)]),  # batch
            ],
        ):
            # Use context manager so the SQLite connection is closed before
            # TemporaryDirectory cleanup — avoids WinError 32 on Windows.
            with BatchedOllamaEmbeddings(
                model_name="nomic-embed-text",
                batch_size=8,
                cache_path=db_path,
            ) as emb:
                vecs = emb.embed_documents(texts)
                logger.info("-- embed_documents --  %d vectors, dim=%d", len(vecs), len(vecs[0]))

                # Second call -- all from cache, no API side_effect needed.
                with patch("requests.post", side_effect=RuntimeError("should not call API")):
                    vecs2 = emb.embed_documents(texts)
                    logger.info("-- cache-only call --  %d vectors (all from cache)", len(vecs2))

                # print_metrics() must be called while the context manager is still open
                # so the SQLite connection is available for get_stats().
                emb.print_metrics()
