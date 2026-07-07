"""
Hybrid Storage Module: Vector Store (LanceDB) + Knowledge Graph (KuzuDB)

Author: Jan Nietzard

===============================================================================
OVERVIEW
===============================================================================

This module implements the dual-storage persistence layer of the Edge-RAG
system.  Two embedded databases are combined under a single `HybridStore`
facade:

  - **LanceDB** (vector store): approximate nearest-neighbour search over
    dense embeddings produced by nomic-embed-text via Ollama.
    Reference: Malkov & Yashunin (2018). "Efficient and robust approximate
    nearest neighbor search using hierarchical navigable small world graphs."
    IEEE Transactions on Pattern Analysis and Machine Intelligence.

  - **KuzuDB** (knowledge graph): native Cypher-based multi-hop traversal
    over a DocumentChunk–Entity graph constructed during ingestion.
    Reference: Feng et al. (2023). "Kùzu Graph Database Management System."
    CIDR 2023.

The combination supports the hybrid retrieval strategy described in paper
section 2.4: dense vector recall is augmented with graph-based entity
expansion, enabling bridge-entity reasoning across documents without loading
full document text into memory (< 16 GB RAM constraint).

NetworkX is retained as a fallback for environments where KuzuDB is
unavailable, but multi-hop Cypher retrieval is only available under KuzuDB.

===============================================================================
GRAPH SCHEMA
===============================================================================

Node Tables:
    DocumentChunk(chunk_id PK, text, page_number, chunk_index, source_file)
    SourceDocument(doc_id PK, filename, total_pages)
    Entity(entity_id PK, name, type, confidence)

Relationship Tables:
    FROM_SOURCE:  DocumentChunk → SourceDocument
    NEXT_CHUNK:   DocumentChunk → DocumentChunk  (sequential ordering)
    MENTIONS:     DocumentChunk → Entity
    RELATED_TO:   Entity → Entity                (with relation_type, confidence)

===============================================================================
USAGE
===============================================================================

    from storage import HybridStore, StorageConfig
    from langchain_core.embeddings import Embeddings

    config = StorageConfig(
        vector_db_path=Path("./data/vector"),
        graph_db_path=Path("./data/graph"),
    )
    store = HybridStore(config, embeddings)
    store.add_documents(documents)
    results = store.vector_search(query_embedding, top_k=5)

Review History:
    Last Reviewed: 2026-06-12
    Review Result: 1 CRITICAL, 3 IMPORTANT, 6 RECOMMENDED (all addressed)
    Reviewer: Code Review Prompt v2.1
    Next Review: After the next graph-subsystem change
    Changes applied: error-sentinel fix in _triple_frequency_confidence
        (now -1.0, WARNING, sorts last); _integrate_entities routed through the
        bulk writers; bulk-writer commit-count fix via shared _bulk_execute;
        HUB_FANOUT_CAP + GRAPH_SEARCH_BUDGET_S promoted to settings-overridable
        class attributes; per-source-doc MERGE hoisted out of the chunk loop;
        math import hoisted; conn typed; threshold-default mismatch documented.
        graph_traversal / run_diagnostics / add_relation / add_entity_from_metadata
        verified retained (tested public/diagnostic API, no production caller).
    ---
    Previous Review: 2026-05-25 (audit pass, project version 5.4)
    Previous Result: no itemized action list recorded
"""

import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

if TYPE_CHECKING:
    from .entity_extraction import EntityExtractionPipeline

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import lancedb

# KuzuDB for graph storage
try:
    import kuzu
    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False

logger = logging.getLogger(__name__)

# Expected storage-layer exception types.
#
# Used in all except-clauses throughout this module instead of bare
# `except Exception` so that programming bugs (AttributeError, NameError,
# IndexError) are NOT silently swallowed alongside legitimate DB/IO errors.
#
# SystemExit, KeyboardInterrupt, and MemoryError are already excluded by
# `Exception`, but this alias further excludes AssertionError and the broad
# catch-all for library-internal exceptions that should propagate.
_STORAGE_ERRORS: tuple = (
    OSError,        # file-system / IPC failures
    IOError,        # alias for OSError, explicit for clarity
    RuntimeError,   # KuzuDB / LanceDB internal errors
    ValueError,     # type mismatch, out-of-range inputs
    TypeError,      # wrong argument type from DB layer
    KeyError,       # missing column / field in result row
)
if KUZU_AVAILABLE and hasattr(kuzu, "Error"):
    _STORAGE_ERRORS = (*_STORAGE_ERRORS, kuzu.Error)

if not KUZU_AVAILABLE:
    logger.warning("KuzuDB not available. Install with: pip install kuzu")


# Triple-frequency-confidence sentinels (review 2026-06-12, finding #1).
#   _TRIPLE_CONF_NEUTRAL — empty-input prior (not an error): a partially
#       specified bridge is neither boosted nor buried. Mirrors REBEL's old
#       0.5 mid-point but is reached ONLY on missing-name input, never on error.
#   _TRIPLE_CONF_ERROR — query-failure sentinel OUTSIDE the valid (0, 1] range,
#       so an error can never be mistaken for an honest confidence. Callers sort
#       by -triple_confidence, so this sorts an un-scorable bridge to the bottom.
_TRIPLE_CONF_NEUTRAL: float = 0.5
_TRIPLE_CONF_ERROR: float = -1.0


# Public API. Module-internal helpers (_entity_name_variants,
# _get_hub_entity_names, _is_graph_worthy_entity, _STORAGE_ERRORS,
# _HUB_NAMES_CACHE) remain accessible by direct-name import for tests.
__all__ = [
    "HybridStore",
    "StorageConfig",
    "VectorStoreAdapter",
    "KuzuGraphStore",
    "create_storage_config",
]


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class StorageConfig:
    """
    Configuration for the Hybrid Storage System.

    Frozen dataclass: fields are immutable after construction, preventing
    accidental mutation of shared config objects (e.g. in HybridStore.__init__).

    All configurable thresholds should be read from config/settings.yaml
    via a factory (e.g. create_hybrid_retriever in hybrid_retriever.py).
    Defaults here serve as documented emergency fallbacks only.

    Attributes:
        vector_db_path: Directory path for LanceDB.
        graph_db_path: Container directory for KuzuDB
                       (actual DB stored as graph_KuzuDB/ inside it).
        embedding_dim: Embedding vector dimensionality (None = auto-detect).
        similarity_threshold: Minimum cosine similarity for vector results.
        normalize_embeddings: L2-normalise vectors before storage and search.
        distance_metric: LanceDB distance metric ("cosine" or "l2").
        graph_backend: Storage backend — only "kuzu" is supported.
        overfetch_factor: ANN over-fetch multiplier. LanceDB retrieves
            top_k * overfetch_factor candidates; Python then re-ranks and
            filters by similarity_threshold. Factor 3 balances recall vs
            latency for typical top_k=5–10 queries (empirically validated).
        graph_text_max_chars: Maximum characters stored in graph node text
            field. Full text lives in the vector store; the graph field is
            used only for lightweight context display.
        enable_entity_extraction: Enable GLiNER + REBEL entity extraction
            during ingestion (opt-in; can be injected via entity_pipeline).
        entity_cache_path: Path to entity SQLite cache (None = auto-generate).
        read_only: Open the KuzuDB graph store read-only. KuzuDB's write mode
            takes an exclusive file lock, so only ONE process can hold a
            store; read-only mode allows several concurrent readers (e.g.
            two demo terminals on the same dataset). Retrieval-only entry
            points (demo_app, diagnostics) should set True; ingestion must
            leave it False.
    """
    vector_db_path: Path
    graph_db_path: Path
    embedding_dim: Optional[int] = None
    similarity_threshold: float = 0.3
    normalize_embeddings: bool = True
    distance_metric: str = "cosine"
    graph_backend: str = "kuzu"
    overfetch_factor: int = 3
    graph_text_max_chars: int = 500
    enable_entity_extraction: bool = False
    entity_cache_path: Optional[Path] = None
    read_only: bool = False

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.embedding_dim is not None and self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive: %d" % self.embedding_dim)

        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError(
                "similarity_threshold must be in [0,1]: %f" % self.similarity_threshold
            )

        if self.distance_metric not in ("cosine", "l2"):
            raise ValueError(
                "distance_metric must be 'cosine' or 'l2': %s" % self.distance_metric
            )

        if self.graph_backend != "kuzu":
            raise ValueError(
                "graph_backend must be 'kuzu' (NetworkX fallback removed): %s" % self.graph_backend
            )

        if self.overfetch_factor < 1:
            raise ValueError(
                "overfetch_factor must be >= 1: %d" % self.overfetch_factor
            )


# ============================================================================
# VECTOR STORE ADAPTER (LanceDB)
# ============================================================================

class VectorStoreAdapter:
    """
    LanceDB Vector Store Adapter with distance-to-similarity conversion.

    LanceDB returns raw distances (lower = more similar for cosine/L2).
    This adapter converts them to similarity scores in [0, 1] so that
    downstream components can apply a uniform threshold.

    The ANN search over-fetches by `overfetch_factor` to allow threshold
    filtering after similarity conversion without losing top candidates.
    """

    SCHEMA_VERSION = "3.1.0"
    TABLE_NAME = "documents"

    def __init__(
        self,
        db_path: Path,
        embedding_dim: Optional[int] = None,
        normalize_embeddings: bool = True,
        distance_metric: str = "cosine",
        overfetch_factor: int = 3,
    ) -> None:
        if distance_metric not in ("cosine", "l2"):
            raise ValueError("Unsupported distance metric: %s" % distance_metric)

        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(str(db_path))
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.normalize_embeddings = normalize_embeddings
        self.distance_metric = distance_metric
        self.overfetch_factor = overfetch_factor
        self.table = None

        logger.info(
            "VectorStoreAdapter initialised: path=%s, metric=%s",
            db_path,
            distance_metric,
        )

        self._load_metadata()

        try:
            table_names = self.db.table_names()
            if self.TABLE_NAME in table_names:
                self.table = self.db.open_table(self.TABLE_NAME)
                logger.info(
                    "Opened existing table '%s' with %d rows",
                    self.TABLE_NAME,
                    len(self.table),
                )
        except (RuntimeError, OSError) as exc:
            logger.warning("Could not open existing table: %s", exc)

    def _get_metadata_path(self) -> Path:
        return self.db_path / "vector_store_metadata.json"

    def _load_metadata(self) -> None:
        metadata_path = self._get_metadata_path()
        if not metadata_path.exists():
            return
        try:
            with open(metadata_path, "r", encoding="utf-8") as fh:
                metadata = json.load(fh)
            stored_dim = metadata.get("embedding_dim")
            if stored_dim and self.embedding_dim is None:
                self.embedding_dim = stored_dim
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Could not load metadata: %s", exc)

    def _save_metadata(self) -> None:
        if self.embedding_dim is None:
            return
        metadata = {
            "schema_version": self.SCHEMA_VERSION,
            "embedding_dim": self.embedding_dim,
            "distance_metric": self.distance_metric,
            "normalize_embeddings": self.normalize_embeddings,
            "num_documents": len(self.table) if self.table else 0,
            "timestamp": time.time(),
        }
        metadata_path = self._get_metadata_path()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        if not self.normalize_embeddings:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Guard against zero-norm vectors (e.g. all-zero padding)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def _validate_embedding_dimension(self, embeddings: List[List[float]]) -> None:
        if not embeddings:
            return
        actual_dim = len(embeddings[0])
        if self.embedding_dim is None:
            self.embedding_dim = actual_dim
            self._save_metadata()
            return
        if actual_dim != self.embedding_dim:
            raise ValueError(
                "EMBEDDING DIMENSION MISMATCH: expected %d, got %d"
                % (self.embedding_dim, actual_dim)
            )

    def _distance_to_similarity(self, distance: float) -> float:
        """
        Convert a raw LanceDB distance to a similarity score in [0, 1].

        Cosine: similarity = 1 - distance  (distance ∈ [0, 2] in unnormalised
                space; ∈ [0, 1] after L2 normalisation).
        L2:     similarity = 1 / (1 + distance).
        """
        if self.distance_metric == "cosine":
            return max(0.0, min(1.0, 1.0 - distance))
        elif self.distance_metric == "l2":
            return max(0.0, min(1.0, 1.0 / (1.0 + distance)))
        else:
            raise ValueError("Unknown distance metric: %s" % self.distance_metric)

    def add_documents_with_embeddings(
        self,
        documents: List[Document],
        embeddings: Embeddings,
    ) -> None:
        """Embed documents and insert them into the LanceDB table."""
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        logger.info("Generating embeddings for %d documents...", len(texts))

        t0 = time.time()
        embeddings_list = embeddings.embed_documents(texts)
        logger.info("Embeddings generated in %.2fs", time.time() - t0)

        self._validate_embedding_dimension(embeddings_list)

        embeddings_array = np.array(embeddings_list, dtype=np.float32)
        if self.normalize_embeddings:
            embeddings_array = self._normalize_vectors(embeddings_array)
        embeddings_list = embeddings_array.tolist()

        data = []
        for doc, emb in zip(documents, embeddings_list):
            data.append({
                "document_id": str(doc.metadata.get("chunk_id", "unknown")),
                "text": doc.page_content,
                "vector": emb,
                "metadata": json.dumps(doc.metadata),
                "source_file": doc.metadata.get("source_file", "unknown"),
            })

        try:
            if self.table is None:
                self.table = self.db.create_table(
                    self.TABLE_NAME, data=data, mode="overwrite"
                )
            else:
                self.table.add(data)
            self._save_metadata()
        except _STORAGE_ERRORS as exc:
            logger.error("Failed to insert documents: %s", exc)
            raise

    def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Perform ANN vector search with similarity threshold filtering.

        Over-fetches by `overfetch_factor` before threshold filtering to
        avoid losing top-k candidates when many results fall below threshold.
        """
        if self.table is None:
            return []

        try:
            if self.normalize_embeddings:
                query_array = np.array([query_embedding], dtype=np.float32)
                query_array = self._normalize_vectors(query_array)
                query_embedding = query_array[0].tolist()

            raw_results = (
                self.table
                .search(query_embedding)
                .metric(self.distance_metric)
                .limit(top_k * self.overfetch_factor)
                .to_list()
            )

            filtered_results = []
            for result in raw_results:
                distance = result.get("_distance", 0.0)
                similarity = self._distance_to_similarity(distance)

                if similarity >= threshold:
                    try:
                        metadata = json.loads(result.get("metadata", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}

                    filtered_results.append({
                        "text": result.get("text", ""),
                        "similarity": similarity,
                        "document_id": result.get("document_id", "unknown"),
                        "metadata": metadata,
                    })

            filtered_results.sort(key=lambda x: x["similarity"], reverse=True)
            return filtered_results[:top_k]

        except (OSError, ValueError, RuntimeError) as exc:
            logger.error("Vector search failed: %s", exc)
            return []


def _entity_name_variants(name: str) -> List[str]:
    """
    Build a prioritised list of name variants for alias-tolerant entity lookup.

    Priority order (most → least specific):
      1. Full name                     "Marie Curie"
      2. Last token (surname)          "Curie"             — skipped if ≤4 chars
      3. Individual tokens ≥7 chars    (middle names, long first names)

    Tokens ≤6 chars are excluded from single-token fallbacks because short
    given names match unrelated hub entities of the same first name and
    flood results before the correct surname hit reaches the top-K cap.
    The only exception is step 1 (full name always tried first).

    Examples:
      "Marie Curie"        → ["Marie Curie", "Curie"]
      "Albert Einstein"    → ["Albert Einstein", "Einstein"]
      "Leonardo da Vinci"  → ["Leonardo da Vinci", "Vinci"]
      "Akira Kurosawa"     → ["Akira Kurosawa", "Kurosawa", "Akira"]
    """
    variants: List[str] = [name]
    tokens = name.split()
    if len(tokens) < 2:
        return variants

    # Surname (last token) — always add if long enough to be discriminative.
    last = tokens[-1]
    if len(last) >= 4 and last not in variants:
        variants.append(last)

    # Remaining tokens that are long enough to be discriminative on their own.
    for tok in tokens[:-1]:
        if len(tok) >= 7 and tok not in variants:
            variants.append(tok)

    return variants


# ---------------------------------------------------------------------------
# Entity-key filtering: which entity strings are worth a graph lookup?
# ---------------------------------------------------------------------------
# `e.name CONTAINS $x` in KuzuDB is an un-indexed substring scan over every
# Entity node, run once per query-entity × per hop × per name-variant. A
# low-value string ("Italian", "Physics", "2014", a question-word like "what
# year") matches many high-degree hub nodes and the hop-2/3 expansion from a
# hub blows up to tens of seconds. We therefore (a) skip the lookup entirely
# for such strings, and (b) cap the hop-2/3 hub expansion below.
#
# This stoplist is intentionally small and conservative — generic nationality
# adjectives, broad academic-field nouns, and a few question fragments. Real
# named entities (people, organisations, works, specific places) are never on
# it. Co-occurrence/cleanup already drop most of these from the graph, but the
# query-side filter is cheaper and catches NORP/field-noun query terms that
# the Planner emits.
_GRAPH_LOOKUP_STOPWORDS: frozenset = frozenset({
    # nationality / NORP-style adjectives commonly extracted from queries
    "american", "british", "english", "irish", "italian", "french", "german",
    "spanish", "russian", "chinese", "japanese", "canadian", "australian",
    "scottish", "welsh", "dutch", "swedish", "norwegian", "danish", "polish",
    "european", "asian", "african", "indian", "mexican", "brazilian",
    # broad field / category nouns that fragment off real titles
    "physics", "chemistry", "biology", "mathematics", "history", "science",
    "music", "art", "film", "movie", "book", "novel", "band", "album", "song",
    "writer", "actor", "director", "singer", "physicist", "scientist", "author",
    # question-word fragments that sometimes survive as "entities"
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "what year", "which year", "what city", "what country", "the same",
})


_HUB_NAMES_CACHE: Dict[Tuple[str, int], List[str]] = {}


def _get_hub_entity_names(conn: Any, mention_cap: int, db_key: str = "") -> List[str]:
    """
    Hub suppression: return entity names whose mention-degree exceeds
    `mention_cap`. Cached per (database, cap).

    Rationale: KuzuDB does not support subquery COUNT { ... } in
    WHERE clauses (Neo4j-5 syntax). We compute the hub set once and pass
    it as a query parameter so the hop-2/hop-3 traversal can NOT IN-filter
    against it. The set is tiny (typically <100 entities at cap=280) so
    parameter overhead is negligible.

    Cache is keyed by (db_key, mention_cap) — ``db_key`` is the store's
    database path — so a process that opens several datasets (e.g. an
    ablation suite iterating hotpotqa → musique) never reuses one
    dataset's hub list on another. If you tune HUB_MENTION_CAP at
    runtime, restart the process to clear.
    """
    cache_key = (db_key, mention_cap)
    cached = _HUB_NAMES_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        res = conn.execute(
            """
            MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity)
            WITH e.name AS name, COUNT(c) AS deg
            WHERE deg > $cap
            RETURN name
            """,
            {"cap": mention_cap},
        )
        names: List[str] = []
        while res.has_next():
            row = res.get_next()
            if row and row[0]:
                names.append(row[0])
        _HUB_NAMES_CACHE[cache_key] = names
        logger.info("Hub-entity cache built for %s: %d names with "
                    "mention-degree > %d", db_key or "<default>", len(names),
                    mention_cap)
        return names
    except (RuntimeError, ValueError, AttributeError, ConnectionError) as exc:
        logger.warning("Hub-entity cache build failed (%s) — proceeding "
                       "without hub filter (graph quality may degrade)", exc)
        # Cache empty list so we don't retry the failing query every call.
        _HUB_NAMES_CACHE[cache_key] = []
        return []


def _is_graph_worthy_entity(name: str) -> bool:
    """True if `name` should be used as a graph-lookup key. Rejects empty/short
    strings, bare numbers/years, and known generic single-word stop-terms."""
    s = (name or "").strip()
    if len(s) < 3:
        return False
    low = s.lower()
    if low in _GRAPH_LOOKUP_STOPWORDS:
        return False
    # bare year / number, with optional thousands separators or s-suffix ("1990s")
    bare = low.replace(",", "").rstrip("s")
    if bare.isdigit():
        return False
    return True


# ============================================================================
# KUZU GRAPH STORE
# ============================================================================

class KuzuGraphStore:
    """
    KuzuDB-based Knowledge Graph Store.

    Provides Cypher-based multi-hop traversal over a DocumentChunk–Entity
    graph.  Advantages over the NetworkX fallback:

    - Native Cypher support for complex path queries.
    - Columnar, vectorised query execution (10–100x faster for large graphs).
    - ACID transactions and crash recovery.
    - Out-of-core processing via memory-mapped files (handles graphs > RAM).

    Reference: Feng et al. (2023). "Kùzu Graph Database Management System."
               CIDR 2023.

    Schema:
        Nodes:  DocumentChunk, SourceDocument, Entity
        Edges:  FROM_SOURCE, NEXT_CHUNK, MENTIONS, RELATED_TO
    """

    SCHEMA_VERSION = "3.1.0"
    # Subdirectory inside graph_db_path where KuzuDB stores its files.
    # Keeping it in a subdirectory allows sibling files (e.g. entity_cache.db)
    # to share the same parent without confusing KuzuDB.
    KUZU_DIR_NAME = "graph_KuzuDB"

    # I-3: retrieval-time hub-mention cap. Class attribute (not local constant)
    # so the hybrid_retriever factory can override it from settings.yaml
    # (`graph.hub_mention_cap`). Default 280 = 3% of 9412 HotpotQA chunks;
    # matches the cleanup-time hub_threshold_ratio=0.03. Entities with more
    # mentions than this are excluded as bridge targets.
    HUB_MENTION_CAP: int = 280

    # Retrieval-time hub fan-out cap (review 2026-06-12, finding #4). Class
    # attribute (not a local constant) so the hybrid_retriever factory can
    # override it from settings.yaml (`graph.hub_fanout_cap`). Limits how many
    # `e1.name CONTAINS $x` matches feed the hop-2/3 bridge expansion: a generic
    # query string can match dozens of hub entities, and expanding RELATED_TO
    # from each (every hub has ~30 co-occurrence neighbours) is what produces
    # multi-second graph queries. Bridging through a hub adds no signal, so only
    # the first few matches are ever expanded.
    HUB_FANOUT_CAP: int = 5

    def __init__(self, db_path: Path, read_only: bool = False) -> None:
        """
        Initialise KuzuDB graph store.

        Args:
            db_path: Container directory.  KuzuDB files are stored at
                     db_path / KUZU_DIR_NAME.
            read_only: Open the database read-only. Write mode takes an
                     exclusive OS file lock (one process per store);
                     read-only permits multiple concurrent reader
                     processes. Schema creation is skipped in this mode
                     (DDL would fail), so the store must already exist.
        """
        self.db_path = Path(db_path)
        self.read_only = read_only

        if not KUZU_AVAILABLE:
            raise ImportError("KuzuDB not installed. Install with: pip install kuzu")

        try:
            self.db_path.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            pass  # WinError 183: directory already exists — normal on --resume
        kuzu_file = self.db_path / self.KUZU_DIR_NAME

        self.db = kuzu.Database(str(kuzu_file), read_only=read_only)
        self.conn = kuzu.Connection(self.db)

        if not read_only:
            self._init_schema()
        logger.info("KuzuGraphStore initialised: %s%s", kuzu_file,
                    " (read-only)" if read_only else "")

    def close(self) -> None:
        """Release the KuzuDB connection and database handle.

        Explicitly deleting the connection before the database is required
        because KuzuDB's C++ bindings use reference counting internally; if
        the Connection is still alive when the Database is deleted, the OS
        file lock is not released promptly, causing the next test that opens
        a new KuzuDB in the same process to see a stale lock.
        """
        try:
            if getattr(self, "conn", None) is not None:
                del self.conn
                self.conn = None  # type: ignore[assignment]
        except (RuntimeError, AttributeError, ReferenceError):
            pass
        try:
            if getattr(self, "db", None) is not None:
                del self.db
                self.db = None  # type: ignore[assignment]
        except (RuntimeError, AttributeError, ReferenceError):
            pass

    def _init_schema(self) -> None:
        """
        Create node and relationship tables if they do not yet exist.

        KuzuDB requires explicit schema definition before INSERT/MERGE.
        IF NOT EXISTS prevents errors on repeated initialisation; any
        exception here is therefore unexpected and logged as a warning.
        """
        try:
            self.conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS DocumentChunk(
                    chunk_id STRING,
                    text STRING,
                    page_number INT64,
                    chunk_index INT64,
                    source_file STRING,
                    PRIMARY KEY (chunk_id)
                )
            """)
            self.conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS SourceDocument(
                    doc_id STRING,
                    filename STRING,
                    total_pages INT64,
                    PRIMARY KEY (doc_id)
                )
            """)
            self.conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Entity(
                    entity_id STRING,
                    name STRING,
                    type STRING,
                    confidence DOUBLE,
                    PRIMARY KEY (entity_id)
                )
            """)
            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS FROM_SOURCE(
                    FROM DocumentChunk TO SourceDocument
                )
            """)
            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS NEXT_CHUNK(
                    FROM DocumentChunk TO DocumentChunk
                )
            """)
            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS MENTIONS(
                    FROM DocumentChunk TO Entity
                )
            """)
            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS RELATED_TO(
                    FROM Entity TO Entity,
                    relation_type STRING,
                    confidence DOUBLE,
                    source_chunks STRING
                )
            """)
            logger.debug("Graph schema initialised")

        except _STORAGE_ERRORS as exc:
            # IF NOT EXISTS makes duplicate-table errors impossible;
            # any exception here signals an unexpected problem.
            logger.warning("Unexpected error during schema init: %s", exc)

    # ========================================================================
    # NODE WRITERS
    # ========================================================================

    def add_document_chunk(
        self,
        chunk_id: str,
        text: str,
        page_number: int,
        chunk_index: int,
        source_file: str,
        max_text_chars: int = 500,
    ) -> None:
        """
        MERGE a DocumentChunk node.

        Full text is stored in the vector store; `text` here is a truncated
        preview used for lightweight graph-side context display only.
        """
        truncated = text[:max_text_chars] if len(text) > max_text_chars else text
        try:
            self.conn.execute(
                """
                MERGE (c:DocumentChunk {chunk_id: $chunk_id})
                SET c.text = $text,
                    c.page_number = $page_number,
                    c.chunk_index = $chunk_index,
                    c.source_file = $source_file
                """,
                {
                    "chunk_id": chunk_id,
                    "text": truncated,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "source_file": source_file,
                },
            )
        except _STORAGE_ERRORS as exc:
            logger.error("Failed to add chunk %s: %s", chunk_id, exc)
            raise

    def add_source_document(
        self,
        doc_id: str,
        filename: str,
        total_pages: int = 0,
    ) -> None:
        """MERGE a SourceDocument node."""
        try:
            self.conn.execute(
                """
                MERGE (d:SourceDocument {doc_id: $doc_id})
                SET d.filename = $filename,
                    d.total_pages = $total_pages
                """,
                {"doc_id": doc_id, "filename": filename, "total_pages": total_pages},
            )
        except _STORAGE_ERRORS as exc:
            logger.error("Failed to add source doc %s: %s", doc_id, exc)
            raise

    def add_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str = "unknown",
        confidence: float = 0.0,
    ) -> None:
        """MERGE an Entity node."""
        try:
            self.conn.execute(
                """
                MERGE (e:Entity {entity_id: $entity_id})
                SET e.name = $name,
                    e.type = $entity_type,
                    e.confidence = $confidence
                """,
                {
                    "entity_id": entity_id,
                    "name": name,
                    "entity_type": entity_type,
                    "confidence": confidence,
                },
            )
        except _STORAGE_ERRORS as exc:
            logger.error("Failed to add entity %s: %s", entity_id, exc)
            raise

    def add_entity_from_metadata(
        self,
        entity_id: str,
        entity_type: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Compatibility wrapper: routes entity_type string to the typed
        add_document_chunk or add_source_document method.
        """
        if entity_type == "document_chunk":
            self.add_document_chunk(
                chunk_id=entity_id,
                text=metadata.get("text", ""),
                page_number=metadata.get("page_number", 0),
                chunk_index=metadata.get("chunk_index", 0),
                source_file=metadata.get("source_file", ""),
            )
        elif entity_type == "source_document":
            self.add_source_document(
                doc_id=entity_id,
                filename=metadata.get("filename", entity_id),
                total_pages=metadata.get("total_pages", 0),
            )
        else:
            self.add_entity(
                entity_id=entity_id,
                name=metadata.get("name", entity_id),
                entity_type=entity_type,
            )

    # ========================================================================
    # EDGE WRITERS
    # ========================================================================

    def add_from_source_relation(self, chunk_id: str, doc_id: str) -> None:
        """MERGE a FROM_SOURCE edge: DocumentChunk → SourceDocument."""
        try:
            self.conn.execute(
                """
                MATCH (c:DocumentChunk {chunk_id: $chunk_id})
                MATCH (d:SourceDocument {doc_id: $doc_id})
                MERGE (c)-[:FROM_SOURCE]->(d)
                """,
                {"chunk_id": chunk_id, "doc_id": doc_id},
            )
        except _STORAGE_ERRORS as exc:
            logger.warning("FROM_SOURCE relation failed (%s → %s): %s", chunk_id, doc_id, exc)

    def add_next_chunk_relation(self, chunk_id: str, next_chunk_id: str) -> None:
        """MERGE a NEXT_CHUNK edge for sequential ordering."""
        try:
            self.conn.execute(
                """
                MATCH (c1:DocumentChunk {chunk_id: $chunk_id})
                MATCH (c2:DocumentChunk {chunk_id: $next_chunk_id})
                MERGE (c1)-[:NEXT_CHUNK]->(c2)
                """,
                {"chunk_id": chunk_id, "next_chunk_id": next_chunk_id},
            )
        except _STORAGE_ERRORS as exc:
            logger.warning(
                "NEXT_CHUNK relation failed (%s → %s): %s", chunk_id, next_chunk_id, exc
            )

    def add_mentions_relation(self, chunk_id: str, entity_id: str) -> None:
        """MERGE a MENTIONS edge: DocumentChunk → Entity."""
        try:
            self.conn.execute(
                """
                MATCH (c:DocumentChunk {chunk_id: $chunk_id})
                MATCH (e:Entity {entity_id: $entity_id})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                {"chunk_id": chunk_id, "entity_id": entity_id},
            )
        except _STORAGE_ERRORS as exc:
            logger.warning(
                "MENTIONS relation failed (%s → %s): %s", chunk_id, entity_id, exc
            )

    # -- Batch transaction helpers ---------------------------------------------
    # KuzuDB auto-commits every conn.execute() individually, which means one
    # fsync per statement.  Wrapping N inserts in a single BEGIN/COMMIT reduces
    # the fsync count from N to 1 and speeds up bulk imports by 10–30×.
    # Usage:
    #     graph_store.batch_begin()
    #     for item in items:
    #         graph_store.add_entity(...)   # no individual fsync
    #     graph_store.batch_commit()       # single fsync for the whole batch

    def batch_begin(self) -> None:
        """Open an explicit transaction (suppresses auto-commit per statement)."""
        if self.conn is not None:
            self.conn.execute("BEGIN TRANSACTION")

    def batch_commit(self) -> None:
        """Commit the current transaction (single fsync for all pending writes)."""
        if self.conn is not None:
            self.conn.execute("COMMIT")

    def batch_rollback(self) -> None:
        """Roll back the current transaction (discard all pending writes)."""
        if self.conn is not None:
            try:
                self.conn.execute("ROLLBACK")
            except (RuntimeError, AttributeError):
                pass

    def _bulk_execute(
        self,
        query: str,
        items: List[Any],
        param_builder,
        batch_size: int,
        label: str,
    ) -> int:
        """Execute ``query`` over ``items`` in batched explicit transactions.

        Shared engine for all bulk writers (review 2026-06-12, findings #3/#6):
        one BEGIN/COMMIT per ``batch_size`` items, ``param_builder(item)`` mapping
        each item to the Cypher parameter dict. ``label`` names the writer in
        warnings.

        Counting contract (fixes finding #3 — the previous per-row ``written +=
        1`` over-counted on a rolled-back batch): a batch's items are added to
        the running total ONLY after its ``COMMIT`` succeeds. A batch that
        raises is rolled back and contributes ZERO, so the return value is the
        number of items actually *committed*, exactly as the docstrings promise.

        Returns:
            Number of items successfully committed.
        """
        written = 0
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            try:
                self.conn.execute("BEGIN TRANSACTION")
                for item in batch:
                    self.conn.execute(query, param_builder(item))
                self.conn.execute("COMMIT")
                # Count only after a successful COMMIT — a rolled-back batch
                # contributes nothing.
                written += len(batch)
            except _STORAGE_ERRORS as exc:
                logger.warning("Bulk %s batch failed at offset %d: %s", label, i, exc)
                try:
                    self.conn.execute("ROLLBACK")
                except (RuntimeError, AttributeError):
                    pass
        return written

    def add_related_to_relation(
        self,
        entity1_id: str,
        entity2_id: str,
        relation_type: str = "related",
        confidence: float = 0.0,
        source_chunks: Optional[List[str]] = None,
    ) -> None:
        """
        MERGE a RELATED_TO edge: Entity -> Entity.

        The schema (see `_init_schema`) defines three properties on this edge:
        `relation_type`, `confidence`, and `source_chunks`. All three are
        populated on insertion so that downstream consumers (graph retrieval,
        analysis tools) can rank/filter relations by REBEL confidence and
        trace them back to the source chunk(s).

        Args:
            entity1_id:    Source entity_id (subject).
            entity2_id:    Target entity_id (object).
            relation_type: Wikidata-style or REBEL relation label, or
                           "cooccurs" for co-occurrence edges.
            confidence:    Score in [0, 1]. REBEL emits 0.5 as a sentinel
                           (no per-triplet score); co-occurrence edges
                           pass 1.0 by convention.
            source_chunks: List of chunk_ids that produced this relation.
                           Stored as a comma-separated string (KuzuDB does
                           not have a native list column on rel tables).
                           None / empty -> empty string.
        """
        chunks_str = ",".join(source_chunks) if source_chunks else ""
        try:
            self.conn.execute(
                """
                MATCH (e1:Entity {entity_id: $entity1_id})
                MATCH (e2:Entity {entity_id: $entity2_id})
                MERGE (e1)-[r:RELATED_TO {relation_type: $relation_type}]->(e2)
                SET r.confidence = $confidence,
                    r.source_chunks = $source_chunks
                """,
                {
                    "entity1_id": entity1_id,
                    "entity2_id": entity2_id,
                    "relation_type": relation_type,
                    "confidence": float(confidence),
                    "source_chunks": chunks_str,
                },
            )
        except _STORAGE_ERRORS as exc:
            logger.warning(
                "RELATED_TO relation failed (%s -> %s): %s", entity1_id, entity2_id, exc
            )

    def add_related_to_relations_bulk(
        self,
        pairs: List[Tuple[str, str, str, float, str]],
        batch_size: int = 500,
        use_create: bool = False,
    ) -> int:
        """
        Bulk-insert RELATED_TO edges using explicit transactions.

        Grouped commits (500 edges per transaction) are ~20-50× faster than
        auto-commit on Windows because KuzuDB only fsyncs once per commit
        instead of once per MERGE statement.

        Args:
            pairs:       List of (entity1_id, entity2_id, relation_type,
                         confidence, source_chunks_str).
            batch_size:  Edges per transaction commit.
            use_create:  Use CREATE instead of MERGE.  Safe when the caller
                         has already deduplicated pairs in Python (e.g.
                         co-occurrence edges).  CREATE skips KuzuDB's
                         adjacency-list existence scan, which makes it
                         ~100× faster on large tables.

        Returns:
            Number of edges successfully written.
        """
        # CREATE skips the per-edge adjacency-list scan; MERGE checks for an
        # existing edge with matching relation_type before inserting.
        if use_create:
            _query = """
                MATCH (e1:Entity {entity_id: $entity1_id})
                MATCH (e2:Entity {entity_id: $entity2_id})
                CREATE (e1)-[:RELATED_TO {
                    relation_type: $relation_type,
                    confidence:    $confidence,
                    source_chunks: $source_chunks
                }]->(e2)
            """
        else:
            _query = """
                MATCH (e1:Entity {entity_id: $entity1_id})
                MATCH (e2:Entity {entity_id: $entity2_id})
                MERGE (e1)-[r:RELATED_TO {relation_type: $relation_type}]->(e2)
                SET r.confidence    = $confidence,
                    r.source_chunks = $source_chunks
            """

        return self._bulk_execute(
            _query,
            pairs,
            lambda p: {
                "entity1_id": p[0],
                "entity2_id": p[1],
                "relation_type": p[2],
                "confidence": float(p[3]),
                "source_chunks": p[4],
            },
            batch_size,
            "RELATED_TO",
        )

    def add_entities_bulk(
        self,
        entities: List[Tuple[str, str, str, float]],
        batch_size: int = 500,
    ) -> int:
        """
        Bulk-insert Entity nodes using explicit transactions.

        Args:
            entities:   List of (entity_id, name, entity_type, confidence).
                        The caller is expected to deduplicate by entity_id
                        in Python; MERGE here is therefore safe and cheap
                        (the per-node primary-key index makes the existence
                        check O(log N), not the O(degree) adjacency-list
                        scan that MERGE on an EDGE incurs).
            batch_size: Entities per transaction commit. 500 keeps the
                        Windows fsync count manageable without bloating
                        memory.

        Returns:
            Number of entity nodes the call attempted to insert (the actual
            count of new nodes is `unique_entities` tracked by the caller).
        """
        if not entities:
            return 0
        _query = """
            MERGE (e:Entity {entity_id: $entity_id})
            SET e.name = $name,
                e.type = $entity_type,
                e.confidence = $confidence
        """
        return self._bulk_execute(
            _query,
            entities,
            lambda e: {
                "entity_id":   e[0],
                "name":        e[1],
                "entity_type": e[2],
                "confidence":  float(e[3]),
            },
            batch_size,
            "Entity",
        )

    def add_mentions_relations_bulk(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 500,
    ) -> int:
        """
        Bulk-insert MENTIONS edges (DocumentChunk -> Entity) using
        transactions and CREATE — the same fast path the co-occurrence
        phase uses for RELATED_TO edges.

        Why this exists:
            ``add_mentions_relation`` issues
                MATCH (c) MATCH (e) MERGE (c)-[:MENTIONS]->(e)
            per call. MERGE on an EDGE does an adjacency-list existence
            scan whose cost scales with the entity's degree. For a popular
            entity that grows to thousands of mentions, every subsequent
            MENTIONS insert against that entity gets quadratically slower.
            On HotpotQA this manifested as a ~87-hour Phase 3b vs. an
            expected ~5-10 minutes.

        Args:
            pairs:      List of (chunk_id, entity_id). The CALLER MUST
                        DEDUPLICATE the list — CREATE writes a separate
                        edge for every call. Duplicates produce multiple
                        identical MENTIONS edges, which inflate the graph
                        and break MENTIONS-degree statistics.
            batch_size: Edges per transaction commit.

        Returns:
            Number of edges written.
        """
        if not pairs:
            return 0
        _query = """
            MATCH (c:DocumentChunk {chunk_id: $chunk_id})
            MATCH (e:Entity        {entity_id: $entity_id})
            CREATE (c)-[:MENTIONS]->(e)
        """
        return self._bulk_execute(
            _query,
            pairs,
            lambda p: {"chunk_id": p[0], "entity_id": p[1]},
            batch_size,
            "MENTIONS",
        )

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Compatibility method: routes relation_type to the appropriate
        typed add_*_relation method.
        """
        if relation_type == "from_source":
            self.add_from_source_relation(source_id, target_id)
        elif relation_type == "next_chunk":
            self.add_next_chunk_relation(source_id, target_id)
        elif relation_type == "mentions":
            self.add_mentions_relation(source_id, target_id)
        else:
            self.add_related_to_relation(source_id, target_id, relation_type)

    # ========================================================================
    # GRAPH TRAVERSAL (Cypher)
    # ========================================================================

    def graph_traversal(
        self,
        start_entity: str,
        relation_types: Optional[List[str]] = None,
        max_hops: int = 2,
    ) -> Dict[str, int]:
        """
        Multi-hop graph traversal via Cypher.

        Returns a mapping of node_id → hop_distance for all nodes reachable
        from start_entity within max_hops steps.  start_entity is included
        only if it exists as a DocumentChunk or Entity node.

        Args:
            start_entity: Starting node ID (chunk_id or entity_id).
            relation_types: Reserved for future filtering; not yet applied
                            in the Cypher query.
            max_hops: Maximum traversal depth.

        Returns:
            Dict mapping node_id -> hop_distance.
        """
        visited: Dict[str, int] = {}

        # Attempt 1: start_entity as DocumentChunk.
        # If the query executes but returns 0 rows (e.g. start_entity is an
        # entity ID, not a chunk ID), visited remains empty and Attempt 2
        # runs unconditionally below.
        try:
            result = self.conn.execute(
                f"""
                MATCH (start:DocumentChunk {{chunk_id: $start_id}})
                MATCH path = (start)-[*1..{max_hops}]->(connected:DocumentChunk)
                RETURN DISTINCT
                    connected.chunk_id AS node_id,
                    length(path) AS hops
                """,
                {"start_id": start_entity},
            )
            # Include start node only if it actually exists in the graph
            start_exists = False
            while result.has_next():
                row = result.get_next()
                node_id, hops = row[0], row[1]
                if node_id:
                    start_exists = True
                    if node_id not in visited or visited[node_id] > hops:
                        visited[node_id] = hops
            if start_exists:
                visited[start_entity] = 0

        except _STORAGE_ERRORS as exc:
            logger.debug("Traversal as DocumentChunk failed: %s", exc)

        # Attempt 2: start_entity as Entity.
        # Runs whenever Attempt 1 found nothing (empty result OR exception),
        # so entity-based traversal is not gated on a DocumentChunk error.
        if not visited:
            try:
                result = self.conn.execute(
                    f"""
                    MATCH (start:Entity {{entity_id: $start_id}})
                    MATCH path = (start)-[*1..{max_hops}]-(connected)
                    RETURN DISTINCT
                        CASE
                            WHEN connected:DocumentChunk THEN connected.chunk_id
                            WHEN connected:Entity THEN connected.entity_id
                        END AS node_id,
                        length(path) AS hops
                    """,
                    {"start_id": start_entity},
                )
                start_exists = False
                while result.has_next():
                    row = result.get_next()
                    node_id, hops = row[0], row[1]
                    if node_id:
                        start_exists = True
                        if node_id not in visited or visited[node_id] > hops:
                            visited[node_id] = hops
                if start_exists:
                    visited[start_entity] = 0

            except _STORAGE_ERRORS as exc2:
                logger.debug("Traversal as Entity failed: %s", exc2)

        return visited

    def find_related_chunks(
        self,
        chunk_id: str,
        max_hops: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Find chunks reachable from chunk_id through any graph path.

        Useful for manual context expansion in diagnostic tools.

        Returns:
            List of dicts with chunk_id, text, source_file, hops.
        """
        related: List[Dict[str, Any]] = []
        try:
            result = self.conn.execute(
                f"""
                MATCH (start:DocumentChunk {{chunk_id: $chunk_id}})
                MATCH path = (start)-[*1..{max_hops}]-(related:DocumentChunk)
                WHERE related.chunk_id <> $chunk_id
                RETURN DISTINCT
                    related.chunk_id AS chunk_id,
                    related.text AS text,
                    related.source_file AS source_file,
                    min(length(path)) AS hops
                ORDER BY hops ASC
                """,
                {"chunk_id": chunk_id},
            )
            while result.has_next():
                row = result.get_next()
                related.append({
                    "chunk_id": row[0],
                    "text": row[1],
                    "source_file": row[2],
                    "hops": row[3],
                })
        except _STORAGE_ERRORS as exc:
            logger.error("find_related_chunks failed: %s", exc)
        return related

    def get_context_chunks(self, chunk_id: str, window: int = 2) -> List[str]:
        """
        Return chunk IDs within ±window positions via NEXT_CHUNK edges.

        Args:
            chunk_id: Centre chunk ID.
            window: Number of neighbours before and after to include.

        Returns:
            List of chunk IDs in document order (prev … centre … next).
        """
        context_chunks = [chunk_id]
        try:
            # Forward: follow NEXT_CHUNK edges
            result = self.conn.execute(
                f"""
                MATCH path = (start:DocumentChunk {{chunk_id: $chunk_id}})
                    -[:NEXT_CHUNK*1..{window}]->(next:DocumentChunk)
                RETURN next.chunk_id AS chunk_id, length(path) AS distance
                ORDER BY distance ASC
                """,
                {"chunk_id": chunk_id},
            )
            while result.has_next():
                row = result.get_next()
                if row[0] and row[0] not in context_chunks:
                    context_chunks.append(row[0])

            # Backward: reverse NEXT_CHUNK edges
            result = self.conn.execute(
                f"""
                MATCH path = (prev:DocumentChunk)
                    -[:NEXT_CHUNK*1..{window}]->(end:DocumentChunk {{chunk_id: $chunk_id}})
                RETURN prev.chunk_id AS chunk_id, length(path) AS distance
                ORDER BY distance DESC
                """,
                {"chunk_id": chunk_id},
            )
            backward: List[str] = []
            while result.has_next():
                row = result.get_next()
                if row[0] and row[0] not in context_chunks:
                    backward.append(row[0])
            context_chunks = backward + context_chunks

        except _STORAGE_ERRORS as exc:
            logger.debug("get_context_chunks note: %s", exc)

        return context_chunks

    # Relation-type weights (§11.6) — used by _triple_frequency_confidence to
    # rank bridge candidates. Semantic REBEL relations (named predicate)
    # score full weight; cooccurs heavily down-weighted but not dropped.
    # This preserves bridge connectivity (84.7% of entity-pairs have ONLY
    # cooccurs edges) while ensuring semantic bridges outrank statistical-
    # noise bridges when both exist.
    #
    # Refs:
    #   - Cooccurrence weighting in graph IR follows the PMI tradition (Church
    #     & Hanks, 1990, "Word Association Norms, Mutual Information and
    #     Lexicography", Computational Linguistics 16(1)) — co-mention edges
    #     carry weak but non-zero signal; the right choice is to down-weight,
    #     not delete.
    #   - The 0.25 multiplier down-weights non-typed (co-mention) edges
    #     relative to typed semantic relations, preserving their weak signal
    #     rather than deleting it.
    #   - Confidence via corpus support count follows DeepDive (Niu et al.,
    #     2012, "Elementary: Large-scale Knowledge-base Construction via
    #     Machine Learning and Statistical Inference", AI Magazine 33(3))
    #     and KnowledgeVault (Dong et al., 2014, KDD).
    # Origin-based weighting. SVO triples come from a SpaCy dependency parse
    # (single-token verb lemmas like "direct", "win", "bear", "found") and are
    # noisier than REBEL's seq2seq predictions on Wikidata-style relations
    # ("date_of_birth", "member_of_sports_team", "place_of_birth"). We discriminate
    # by string shape: REBEL relations contain "_" or " " (multi-token Wikidata
    # IDs); SVO relations are single lowercase tokens. Cooccurs is the explicit
    # sentinel value used at ingestion.
    #
    # Refs:
    #   - Multi-extractor pooling: Knowledge Vault (Dong et al., 2014, KDD)
    #     weights per-extractor accuracy in the fusion step.
    _RELATION_TYPE_WEIGHTS = {
        "cooccurs": 0.25,    # statistical co-mention — weak signal, last resort
    }
    _RELATION_TYPE_DEFAULT_WEIGHT = 1.0   # REBEL Wikidata-style relations

    # Heuristic to identify SVO vs REBEL relation strings at retrieval time
    # (without a separate edge property). REBEL's Wikidata-style relations
    # are multi-token: "date_of_birth", "member of sports team", "place_of_death".
    # SVO predicates are single verb lemmas: "direct", "win", "bear", "lead".
    # The cooccurs sentinel is matched literally by the dict above.
    _SVO_WEIGHT = 0.6        # dependency-parse heuristic — between cooccurs and REBEL

    @classmethod
    def _classified_weight(cls, relation_type: Optional[str]) -> float:
        """Return the retrieval-time weight for a relation_type string.

        Cooccurs is matched literally. REBEL Wikidata-style relations
        (containing '_' or whitespace) get the default 1.0 weight. SVO
        single-token verb lemmas get _SVO_WEIGHT.
        """
        rel = (relation_type or "").strip().lower()
        if not rel:
            return cls._RELATION_TYPE_DEFAULT_WEIGHT
        explicit = cls._RELATION_TYPE_WEIGHTS.get(rel)
        if explicit is not None:
            return explicit
        # Multi-token => REBEL Wikidata-style
        if "_" in rel or " " in rel:
            return cls._RELATION_TYPE_DEFAULT_WEIGHT
        # Single-token => SVO verb lemma
        return cls._SVO_WEIGHT

    def _triple_frequency_confidence(
        self,
        e1_name: str,
        relation_type: Optional[str],
        e2_name: str,
    ) -> float:
        """
        Triple-frequency confidence: compute a confidence score from triple
        co-occurrence frequency, replacing REBEL's constant 0.5 sentinel.

        Refs:
            - DeepDive corpus-support inference: Niu et al. (2012). AI
              Magazine 33(3). "Elementary: Large-scale KB Construction via
              Machine Learning and Statistical Inference."
            - Knowledge Vault confidence pooling: Dong et al. (2014). KDD.
            - Log-scaled support normalisation: log frequency is the standard
              transform for highly-skewed count distributions in IR (Manning,
              Raghavan & Schütze, 2008, "Introduction to Information
              Retrieval", §6.2.1 on sublinear tf scaling).

        Relation-type weighting (§11.6): multiplied by a relation-type weight
        so cooccurs bridges (weight=0.25) are ranked far below semantic bridges
        (weight=1.0) at the same support level.

        Counts how many distinct chunks mention BOTH e1 and e2 (proxy for
        "how often is this relation supported by the corpus"). Returns a
        normalised score in (0, 1]:
            base_conf = min(1.0, log(1 + n_supporting_chunks) / log(10))
            conf = base_conf * relation_type_weight
        - cooccurs at 1 chunk    → 0.30 × 0.25 = 0.075
        - cooccurs at 9+ chunks  → 1.00 × 0.25 = 0.250
        - semantic at 1 chunk    → 0.30 × 1.00 = 0.300
        - semantic at 9+ chunks  → 1.00 × 1.00 = 1.000
        ⇒ any semantic bridge beats any cooccurs bridge on confidence.

        Error contract (review 2026-06-12, finding #1)
        ----------------------------------------------
        On a *query failure* this returns ``_TRIPLE_CONF_ERROR`` (-1.0), a
        sentinel OUTSIDE the valid (0, 1] range, and logs at WARNING — so an
        error can never masquerade as an honest confidence (the whole point of
        replacing REBEL's constant-0.5). Callers sort by ``-triple_confidence``;
        a negative sentinel sorts an un-scorable bridge to the BOTTOM, which is
        the correct conservative behaviour (do not promote a bridge we could
        not score). An *empty-input* (missing entity name) is not an error — it
        returns the neutral 0.5 prior so a partially-specified bridge is neither
        boosted nor buried.
        """
        if not e1_name or not e2_name:
            return _TRIPLE_CONF_NEUTRAL
        try:
            res = self.conn.execute(
                """
                MATCH (e1:Entity {name: $e1_name})
                MATCH (e2:Entity {name: $e2_name})
                MATCH (c:DocumentChunk)-[:MENTIONS]->(e1)
                MATCH (c)-[:MENTIONS]->(e2)
                RETURN COUNT(DISTINCT c)
                """,
                {"e1_name": e1_name, "e2_name": e2_name},
            )
            if res.has_next():
                n = int(res.get_next()[0] or 0)
                # Origin-based weight: cooccurs 0.25, SVO 0.6, REBEL 1.0.
                weight = self._classified_weight(relation_type)
                if n <= 0:
                    return 0.3 * weight  # one-sided support — still some signal
                base = min(1.0, math.log(1 + n) / math.log(10))
                return base * weight
        except (RuntimeError, ValueError, AttributeError, ConnectionError) as exc:
            logger.warning(
                "_triple_frequency_confidence failed for (%r, %r): %s — "
                "returning error sentinel %.1f (bridge sorts last)",
                e1_name, e2_name, exc, _TRIPLE_CONF_ERROR,
            )
        return _TRIPLE_CONF_ERROR

    def find_chunks_by_entity_multihop(
        self,
        entity_name: str,
        max_results: int = 5,
        enable_hop3: bool = False,
        max_hops: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Multi-hop entity search: returns chunks related to entity_name through
        1–3 hops of RELATED_TO edges.

        Hop=0 (direct):   DocumentChunk -[MENTIONS]-> Entity (name = / CONTAINS query)
        Hop=2 (1 bridge): e1 -[RELATED_TO]- e2 <-[MENTIONS]- Chunk
        Hop=3 (2 bridges): e1 -[REL]-> e2 -[REL]-> e3 <-[MENTIONS]- Chunk
                           — only run when enable_hop3=True (the two-edge
                           expansion is the dominant cost and rarely needed;
                           off by default).

        This design enables bridge-entity reasoning as described in paper
        section 2.4 (original architectural contribution): a query about
        one entity can reach chunks about a second entity via an
        intermediate semantic relation in the graph (e.g. PersonA →
        directed → MovieM ← directed ← PersonB), without loading all
        document text into memory (< 16 GB RAM constraint).

        Args:
            entity_name: Entity name substring to match.
            max_results: Maximum total results across all hops.
            enable_hop3: Run the 2-bridge hop as well (slow; default False).
            max_hops: Cap on graph-traversal depth. 0 disables graph search,
                1 returns Hop-0 (direct mentions) only, ≥2 includes Hop-2 bridges.
                Hop-3 is additionally gated by enable_hop3. Used by the
                graph-only ablation baseline to disable bridge expansion.

        Returns:
            List of dicts: chunk_id, text, source_file, matched_entity,
            hops, bridge_entity, relation_type.
        """
        if not entity_name.strip():
            return []

        chunks: List[Dict[str, Any]] = []
        seen: set[str] = set()

        name_variants = _entity_name_variants(entity_name)

        # Fan-out cap (see KuzuGraphStore.HUB_FANOUT_CAP — overridable from
        # settings.yaml `graph.hub_fanout_cap`). Limits how many `e1` matches
        # feed the hop-2/3 bridge expansion.
        _HUB_FANOUT_CAP = self.HUB_FANOUT_CAP
        # Hub suppression: reject bridges through high-degree hub
        # entities. Read from the class attribute so the hybrid_retriever
        # factory can override it from settings.yaml (`graph.hub_mention_cap`).
        # See KuzuGraphStore.HUB_MENTION_CAP for the rationale and refs.
        _HUB_MENTION_CAP = self.HUB_MENTION_CAP

        try:
            # Hop 0: direct MENTIONS — try name variants until we get a hit.
            # Cascade (most precise → most permissive):
            #   1. e.name = $query                 (exact match)
            #   2. e.name CONTAINS $query          (stored entity contains query)
            #   3. $query CONTAINS e.name          (I-3, query contains stored entity)
            #      — implemented as `e.name IN $substrings` where substrings
            #      are the multi-token prefixes/suffixes of the query.
            # I-3 (publication-recommended): bidirectional alias matching.
            # Without direction (3), a query that uses a generic role noun
            # ("the King", "the director") cannot match a stored entity
            # whose name embeds the role token but with an additional
            # proper-noun specifier; only the reverse direction worked.
            effective_name = entity_name  # will be updated on first hit

            # Pre-compute multi-token sub-phrases of the candidate for the
            # reverse-CONTAINS direction. Single tokens are too noisy (would
            # match "the" entity). Sub-phrases of ≥2 tokens and ≥4 chars only.
            def _multi_token_subphrases(text: str) -> list:
                tokens = text.split()
                subs: list = []
                for i in range(len(tokens)):
                    for j in range(i + 2, len(tokens) + 1):
                        s = " ".join(tokens[i:j])
                        if len(s) >= 4:
                            subs.append(s)
                # Sort by length descending so the most specific match wins.
                return sorted(set(subs), key=len, reverse=True)

            for candidate in name_variants:
                hop0_rows = []
                reverse_subs = _multi_token_subphrases(candidate)
                cascade = [
                    ("e.name = $entity_name",
                     {"entity_name": candidate, "limit": max_results}),
                    ("e.name CONTAINS $entity_name",
                     {"entity_name": candidate, "limit": max_results}),
                ]
                # I-3 reverse direction: only attempt when query has 2+ tokens
                # AND the previous two clauses didn't match. The IN-list keeps
                # the query indexable and avoids a full table scan.
                if reverse_subs:
                    cascade.append((
                        "e.name IN $substrings",
                        {"substrings": reverse_subs[:20], "limit": max_results},
                    ))
                for clause, params in cascade:
                    res = self.conn.execute(
                        f"""
                        MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity)
                        WHERE {clause}
                        RETURN c.chunk_id, c.text, c.source_file, e.name
                        LIMIT $limit
                        """,
                        params,
                    )
                    while res.has_next():
                        hop0_rows.append(res.get_next())
                    if hop0_rows:
                        break  # earliest cascade stage hit — stop here
                if hop0_rows:
                    effective_name = candidate
                    if candidate != entity_name:
                        logger.info(
                            "Entity alias resolved: %r → %r", entity_name, candidate
                        )
                    for row in hop0_rows:
                        cid = row[0]
                        if cid and cid not in seen:
                            seen.add(cid)
                            chunks.append({
                                "chunk_id": cid,
                                "text": row[1],
                                "source_file": row[2],
                                "matched_entity": row[3],
                                "hops": 0,
                                "bridge_entity": None,
                                "relation_type": None,
                                # Hop-0 is a direct mention — full confidence,
                                # no triple-frequency lookup needed (the chunk
                                # IS the supporting evidence).
                                "triple_confidence": 1.0,
                            })
                    break  # stop trying variants once we have results

            # Use effective_name (the variant that matched) for hop 2/3 queries
            entity_name = effective_name  # noqa: PLW2901 — intentional rebinding for hop queries

            # Hop 2: one RELATED_TO bridge (bidirectional), capped fan-out.
            #
            # Weighted cooccurs, not binary drop (§11.6). An earlier
            # binary cooccurs-filter eliminated 84.7% of bridge connectivity
            # (255,838 of 302,184 entity-pairs had ONLY cooccurs edges) —
            # empirically broke graph retrieval for bridge questions whose two
            # entities are connected ONLY by a co-occurrence edge (no named
            # semantic relation between them). Current behaviour:
            #   - Cooccurs edges are KEPT but ranked lower (relation-type
            #     weighting happens at confidence time, not retrieval time).
            #   - Semantic relations (REBEL/SVO with named predicate) still
            #     dominate via higher triple_confidence × relation_type_weight.
            # Hub exclusion still applies: hubs are excluded as bridge targets.
            remaining = max_results - len(chunks)
            if remaining > 0 and max_hops >= 2:
                hub_names = _get_hub_entity_names(
                    self.conn, _HUB_MENTION_CAP, db_key=str(self.db_path))
                res = self.conn.execute(
                    """
                    MATCH (e1:Entity)
                    WHERE e1.name CONTAINS $entity_name
                    WITH e1 LIMIT $hub_cap
                    MATCH (e1)-[r:RELATED_TO]-(e2:Entity)
                    WHERE NOT (e2.name IN $hub_names)
                    WITH e2, r.relation_type AS rel
                    MATCH (c:DocumentChunk)-[:MENTIONS]->(e2)
                    RETURN c.chunk_id, c.text, c.source_file,
                           e2.name AS bridge, rel
                    LIMIT $limit
                    """,
                    {"entity_name": entity_name, "limit": remaining,
                     "hub_cap": _HUB_FANOUT_CAP,
                     "hub_names": hub_names},
                )
                while res.has_next():
                    row = res.get_next()
                    cid = row[0]
                    if cid and cid not in seen:
                        seen.add(cid)
                        # Triple-frequency confidence replaces REBEL's
                        # constant-0.5 sentinel. Computed lazily per
                        # returned chunk (not per edge) so cost is bounded.
                        bridge_name = row[3]
                        triple_conf = self._triple_frequency_confidence(
                            entity_name, row[4], bridge_name
                        )
                        chunks.append({
                            "chunk_id": cid,
                            "text": row[1],
                            "source_file": row[2],
                            "matched_entity": entity_name,
                            "hops": 2,
                            "bridge_entity": bridge_name,
                            "relation_type": row[4],
                            "triple_confidence": triple_conf,
                        })

            # Hop 3: two RELATED_TO bridges, capped fan-out (opt-in — slow).
            # Cooccurs edges are weighted-low, not dropped (§11.6).
            # Hop-3 STILL excludes cooccurs because the noise compounds:
            # two cooccurs hops = anything-to-anything in the corpus. Only
            # semantic→semantic chains are admitted at depth 3.
            # Hub suppression: reject bridges through hub entities (e2 AND e3).
            remaining = max_results - len(chunks)
            if enable_hop3 and remaining > 0 and max_hops >= 3:
                hub_names = _get_hub_entity_names(
                    self.conn, _HUB_MENTION_CAP, db_key=str(self.db_path))
                res = self.conn.execute(
                    """
                    MATCH (e1:Entity)
                    WHERE e1.name CONTAINS $entity_name
                    WITH e1 LIMIT $hub_cap
                    MATCH (e1)-[r1:RELATED_TO]-(e2:Entity)-[r2:RELATED_TO]-(e3:Entity)
                    WHERE e3.name <> e1.name
                      AND (r1.relation_type IS NULL OR r1.relation_type <> 'cooccurs')
                      AND (r2.relation_type IS NULL OR r2.relation_type <> 'cooccurs')
                      AND NOT (e2.name IN $hub_names)
                      AND NOT (e3.name IN $hub_names)
                    WITH e2, e3, r2.relation_type AS rel
                    MATCH (c:DocumentChunk)-[:MENTIONS]->(e3)
                    RETURN c.chunk_id, c.text, c.source_file,
                           e3.name AS bridge, e2.name AS mid_entity, rel
                    LIMIT $limit
                    """,
                    {"entity_name": entity_name, "limit": remaining,
                     "hub_cap": _HUB_FANOUT_CAP,
                     "hub_names": hub_names},
                )
                while res.has_next():
                    row = res.get_next()
                    cid = row[0]
                    if cid and cid not in seen:
                        seen.add(cid)
                        # Same triple-frequency confidence as Hop-2 (computed on
                        # the second edge, mid-entity -> bridge) so Hop-3 rows
                        # carry the key too instead of silently defaulting to
                        # 1.0 downstream.
                        triple_conf = self._triple_frequency_confidence(
                            row[4], row[5], row[3]
                        )
                        chunks.append({
                            "chunk_id": cid,
                            "text": row[1],
                            "source_file": row[2],
                            "matched_entity": entity_name,
                            "hops": 3,
                            "bridge_entity": row[3],
                            # row[4] is the intermediate entity (mid_entity);
                            # row[5] is the relation type of the second edge.
                            "relation_type": row[5],
                            "triple_confidence": triple_conf,
                        })

        except _STORAGE_ERRORS as exc:
            logger.error("find_chunks_by_entity_multihop failed: %s", exc)

        return chunks

    def get_document_structure(self, source_file: str) -> List[Dict[str, Any]]:
        """
        Return ordered chunks for a document (useful for diagnostic tools).

        Args:
            source_file: Source document filename.

        Returns:
            Ordered list of chunk dicts (chunk_id, text, page_number, chunk_index).
        """
        chunks: List[Dict[str, Any]] = []
        try:
            result = self.conn.execute(
                """
                MATCH (c:DocumentChunk {source_file: $source_file})
                RETURN c.chunk_id, c.text, c.page_number, c.chunk_index
                ORDER BY c.chunk_index ASC
                """,
                {"source_file": source_file},
            )
            while result.has_next():
                row = result.get_next()
                chunks.append({
                    "chunk_id": row[0],
                    "text": row[1],
                    "page_number": row[2],
                    "chunk_index": row[3],
                })
        except _STORAGE_ERRORS as exc:
            logger.error("get_document_structure failed: %s", exc)
        return chunks

    # ========================================================================
    # STATISTICS AND UTILITIES
    # ========================================================================

    def get_statistics(self) -> Dict[str, int]:
        """Return node and edge counts for all table types."""
        stats: Dict[str, int] = {
            "document_chunks": 0,
            "source_documents": 0,
            "entities": 0,
            "from_source_edges": 0,
            "next_chunk_edges": 0,
            "mentions_edges": 0,
            "related_to_edges": 0,
        }

        try:
            for label, key in [
                ("DocumentChunk", "document_chunks"),
                ("SourceDocument", "source_documents"),
                ("Entity", "entities"),
            ]:
                result = self.conn.execute(
                    "MATCH (n:%s) RETURN count(n)" % label
                )
                if result.has_next():
                    stats[key] = result.get_next()[0]

            for rel_type, key in [
                ("FROM_SOURCE", "from_source_edges"),
                ("NEXT_CHUNK", "next_chunk_edges"),
                ("MENTIONS", "mentions_edges"),
                ("RELATED_TO", "related_to_edges"),
            ]:
                try:
                    result = self.conn.execute(
                        "MATCH ()-[r:%s]->() RETURN count(r)" % rel_type
                    )
                    if result.has_next():
                        stats[key] = result.get_next()[0]
                except _STORAGE_ERRORS as exc:
                    logger.debug("Edge count for %s failed: %s", rel_type, exc)

        except _STORAGE_ERRORS as exc:
            logger.error("get_statistics failed: %s", exc)

        return stats

    def clear(self) -> None:
        """Delete all nodes and edges from the graph."""
        for rel_type in ["FROM_SOURCE", "NEXT_CHUNK", "MENTIONS", "RELATED_TO"]:
            try:
                self.conn.execute(
                    "MATCH ()-[r:%s]->() DELETE r" % rel_type
                )
            except _STORAGE_ERRORS as exc:
                logger.warning("Failed to clear edges %s: %s", rel_type, exc)

        for label in ["DocumentChunk", "SourceDocument", "Entity"]:
            try:
                self.conn.execute("MATCH (n:%s) DELETE n" % label)
            except _STORAGE_ERRORS as exc:
                logger.warning("Failed to clear nodes %s: %s", label, exc)

        logger.info("Graph cleared")

    def save(self) -> None:
        """No-op: KuzuDB auto-persists after every transaction."""
        logger.debug("KuzuDB auto-persists; no explicit save required")


# ============================================================================
# HYBRID STORE (Facade)
# ============================================================================

class HybridStore:
    """
    Unified facade over VectorStoreAdapter (LanceDB) and KuzuDB.

    Design pattern: Facade — hides storage heterogeneity from callers.
    Callers interact exclusively with `add_documents`, `vector_search`, and
    `graph_search`.

    KuzuDB is the only supported graph backend. If KuzuDB is not installed,
    HybridStore raises RuntimeError immediately rather than falling back to an
    alternative.  This is intentional: the paper evaluates a specific system
    configuration; silent fallbacks would invalidate the evaluation results.
    """

    # Per-call wall-clock budget for graph_search (review 2026-06-12, finding
    # #4/#5). Class attribute so the hybrid_retriever factory can override it
    # from settings.yaml (`graph.search_budget_seconds`). A pathological hub
    # query that exceeds this returns PARTIAL results with a logged WARNING —
    # so graph_search output is budget-bounded and therefore hardware-dependent
    # at the tail. For a strictly reproducible run, raise this high enough that
    # it never fires (the multihop helper's internal hub-fan-out cap already
    # bounds the common case).
    GRAPH_SEARCH_BUDGET_S: float = 8.0

    def __init__(
        self,
        config: StorageConfig,
        embeddings: Embeddings,
        entity_pipeline: "Optional[EntityExtractionPipeline]" = None,
    ) -> None:
        """
        Initialise the hybrid store.

        Args:
            config: StorageConfig instance.
            embeddings: Embeddings model used for both ingestion and
                        dimension auto-detection.
            entity_pipeline: Optional pre-constructed EntityExtractionPipeline.
                             If None and config.enable_entity_extraction is True,
                             the pipeline is constructed internally.
        """
        self.config = config
        self.embeddings = embeddings

        # Auto-detect embedding dimension from the live model if not set.
        # Use a local variable to avoid mutating the caller's StorageConfig
        # object — a side effect that could confuse shared config instances.
        embedding_dim = config.embedding_dim
        if embedding_dim is None:
            test_emb = embeddings.embed_query("dimension test")
            embedding_dim = len(test_emb)
            logger.info("Detected embedding dim: %d", embedding_dim)
        self.embedding_dim: int = embedding_dim

        # Vector store
        self.vector_store = VectorStoreAdapter(
            db_path=config.vector_db_path,
            embedding_dim=embedding_dim,
            normalize_embeddings=config.normalize_embeddings,
            distance_metric=config.distance_metric,
            overfetch_factor=config.overfetch_factor,
        )

        # Graph store — KuzuDB only. No fallback by design.
        if not KUZU_AVAILABLE:
            raise RuntimeError(
                "KuzuDB not available. Install with: pip install kuzu. "
                "No fallback is provided — the paper evaluation requires KuzuDB."
            )
        self.graph_store: KuzuGraphStore = KuzuGraphStore(
            config.graph_db_path, read_only=config.read_only,
        )
        logger.info("Using KuzuDB for graph storage%s",
                    " (read-only)" if config.read_only else "")

        # Entity extraction pipeline (optional)
        self.entity_pipeline: Optional["EntityExtractionPipeline"] = entity_pipeline
        if self.entity_pipeline is None and config.enable_entity_extraction:
            self.entity_pipeline = self._init_entity_pipeline(config)

        logger.info("HybridStore initialised: dim=%d", embedding_dim)

    def close(self) -> None:
        """Release all underlying store connections.

        Delegates to KuzuGraphStore.close() so that KuzuDB's C++ file lock
        is released promptly rather than waiting for garbage collection.
        Call this at the end of test teardown or when the store is no longer
        needed to prevent lock-related flicker in the test suite.
        """
        if hasattr(self, "graph_store") and self.graph_store is not None:
            self.graph_store.close()

    # ---- Context Manager Protocol ----------------------------------------
    # Forgetting to call close() leaks the KuzuDB C++ file lock until garbage
    # collection. Use `with HybridStore(...) as store:` to release it
    # deterministically when leaving the scope.

    def __enter__(self) -> "HybridStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _init_entity_pipeline(self, config: StorageConfig) -> "Optional[EntityExtractionPipeline]":
        """
        Construct EntityExtractionPipeline from settings when not injected.

        Separated from __init__ to keep the constructor readable and to make
        the pipeline injectable for testing (Dependency Inversion Principle).
        """
        try:
            from .entity_extraction import EntityExtractionPipeline, ExtractionConfig

            cache_path = config.entity_cache_path
            if cache_path is None:
                cache_path = Path(config.graph_db_path).parent / "entity_cache.db"

            extraction_config = ExtractionConfig(
                cache_enabled=True,
                cache_path=str(cache_path),
                # All NER/RE parameters sourced from settings.yaml via ExtractionConfig
                # defaults; override by passing a pre-configured pipeline instead.
            )
            pipeline = EntityExtractionPipeline(extraction_config)
            logger.info("Entity extraction pipeline initialised (cache: %s)", cache_path)
            return pipeline

        except ImportError as exc:
            logger.warning(
                "Entity extraction not available: %s. "
                "Install with: pip install gliner transformers",
                exc,
            )
        except _STORAGE_ERRORS as exc:
            logger.warning("Failed to initialise entity pipeline: %s", exc)

        return None

    def add_documents(self, documents: List[Document]) -> None:
        """
        Ingest documents into both the vector store and the knowledge graph.

        Ingests into both the vector store (LanceDB) and the knowledge graph
        (KuzuDB): creates DocumentChunk, SourceDocument nodes, FROM_SOURCE
        and NEXT_CHUNK edges, and optionally MENTIONS / RELATED_TO edges via
        the entity extraction pipeline.
        """
        if not documents:
            return

        # Embed and insert into vector store
        self.vector_store.add_documents_with_embeddings(documents, self.embeddings)

        max_chars = self.config.graph_text_max_chars
        prev_chunk_id: Optional[str] = None
        # MERGE each SourceDocument node once per unique source_file rather than
        # once per chunk (review 2026-06-12, finding #9). The old per-chunk
        # MERGE was idempotent but issued N redundant executes for an N-chunk
        # document.
        seen_sources: set = set()

        prev_source: Optional[str] = None
        for doc in documents:
            chunk_id = str(doc.metadata.get("chunk_id", "unknown"))
            source_file = doc.metadata.get("source_file", "unknown")

            self.graph_store.add_document_chunk(
                chunk_id=chunk_id,
                text=doc.page_content,
                page_number=doc.metadata.get("page_number", 0),
                chunk_index=doc.metadata.get("chunk_index", 0),
                source_file=source_file,
                max_text_chars=max_chars,
            )
            if source_file not in seen_sources:
                seen_sources.add(source_file)
                self.graph_store.add_source_document(
                    doc_id=source_file,
                    filename=source_file,
                    total_pages=doc.metadata.get("total_pages", 0),
                )
            self.graph_store.add_from_source_relation(chunk_id, source_file)
            # NEXT_CHUNK encodes sequential order WITHIN one document only —
            # reset the chain at every source_file boundary so a batch that
            # mixes documents never links the last chunk of doc A to the
            # first chunk of doc B (a spurious edge that would corrupt
            # get_context_chunks window expansion).
            if prev_chunk_id and source_file == prev_source:
                self.graph_store.add_next_chunk_relation(prev_chunk_id, chunk_id)

            prev_chunk_id = chunk_id
            prev_source = source_file

        # Entity extraction and graph integration
        if self.entity_pipeline:
            try:
                entity_stats = self._integrate_entities(documents)
                logger.info(
                    "Entity integration: %d entities, %d mentions, %d relations",
                    entity_stats.get("unique_entities", 0),
                    entity_stats.get("total_mentions", 0),
                    entity_stats.get("total_relations", 0),
                )
            except _STORAGE_ERRORS as exc:
                logger.warning("Entity integration failed: %s", exc)

        logger.info("Added %d documents to hybrid store", len(documents))

    def _integrate_entities(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Extract entities and relations and integrate them into the graph.

        Pipeline (paper section 2.5):
        1. Batch entity extraction with GLiNER (configured batch size).
        2. Add unique Entity nodes (deduplicated by entity_id).
        3. Create MENTIONS edges: DocumentChunk → Entity.
        4. Create RELATED_TO edges from REBEL relation extraction.

        Write path (review 2026-06-12, finding #2): entities/mentions/relations
        are accumulated and written via the bulk writers
        (``add_entities_bulk`` / ``add_mentions_relations_bulk`` /
        ``add_related_to_relations_bulk``) — one transaction per table instead
        of one auto-commit per edge. MENTIONS uses CREATE under the hood, so
        pairs are de-duplicated WITHIN this call; like the bulk writers'
        documented contract, re-ingesting the same documents across calls
        requires a ``reset_*`` first (the per-edge MERGE idempotency of the old
        path no longer applies). Production graph ingestion
        (``local_importingestion.py``) already uses these bulk writers; this is
        the convenience ``add_documents()`` path brought to parity.

        Args:
            documents: Documents already present in vector and graph stores.

        Returns:
            Statistics dict with extraction metrics.
        """
        if not self.entity_pipeline:
            return {}

        stats: Dict[str, Any] = {
            "total_chunks": len(documents),
            "chunks_with_entities": 0,
            "total_entities": 0,
            "unique_entities": 0,
            "total_mentions": 0,
            "total_relations": 0,
        }

        if not documents:
            return stats

        texts = [doc.page_content for doc in documents]
        chunk_ids = [
            str(doc.metadata.get("chunk_id", "chunk_%d" % i))
            for i, doc in enumerate(documents)
        ]

        logger.debug("Extracting entities from %d chunks...", len(documents))
        extraction_results = self.entity_pipeline.process_chunks_batch(texts, chunk_ids)

        # Collect-then-bulk-write (review 2026-06-12, finding #2). The previous
        # implementation issued one auto-committed conn.execute() per entity and
        # per MENTIONS edge — the exact per-edge MERGE path the bulk writers
        # (used by local_importingestion.py) exist to avoid. We accumulate
        # deduplicated batches and write each table once. Production graph
        # ingestion already routes through the bulk path; this brings the
        # convenience add_documents() path to parity.
        entity_name_to_id: Dict[str, str] = {}
        entities_by_id: Dict[str, Tuple[str, str, str, float]] = {}
        mentions_seen: set = set()
        mentions_pairs: List[Tuple[str, str]] = []

        for result in extraction_results:
            if not result.entities:
                continue
            stats["chunks_with_entities"] += 1
            for entity in result.entities:
                stats["total_entities"] += 1
                entity_name_to_id[entity.name.lower()] = entity.entity_id
                # Dedup Entity nodes by primary key (MERGE upsert).
                entities_by_id[entity.entity_id] = (
                    entity.entity_id, entity.name,
                    entity.entity_type, entity.confidence,
                )
                # Dedup MENTIONS pairs — the bulk writer uses CREATE, so an
                # un-deduplicated list would produce duplicate edges and corrupt
                # MENTIONS-degree statistics (hub detection, triple confidence).
                pair = (result.chunk_id, entity.entity_id)
                if pair not in mentions_seen:
                    mentions_seen.add(pair)
                    mentions_pairs.append(pair)

        # Build RELATED_TO tuples for the bulk writer.
        # NOTE (known limitation): entity_name_to_id is populated only from
        # the current batch. If a relation's subject or object was extracted in
        # a previous ingestion batch, its entity_id will not be found here and
        # the relation is silently skipped. Full cross-document relation
        # coverage requires single-batch ingestion (or a post-hoc graph repair
        # pass). Documented in paper section 2.5.
        rel_tuples: List[Tuple[str, str, str, float, str]] = []
        for result in extraction_results:
            for relation in result.relations:
                subject_id = entity_name_to_id.get(relation.subject_entity.lower())
                object_id = entity_name_to_id.get(relation.object_entity.lower())
                if subject_id and object_id:
                    rel_tuples.append(
                        (subject_id, object_id, relation.relation_type, 0.0, "")
                    )

        # Single bulk write per table. The return values are the COMMITTED
        # counts (finding #3), so a partial failure is visible in the stats
        # rather than silently absorbed at debug level (finding #8).
        n_entities = self.graph_store.add_entities_bulk(list(entities_by_id.values()))
        n_mentions = self.graph_store.add_mentions_relations_bulk(mentions_pairs)
        n_relations = self.graph_store.add_related_to_relations_bulk(rel_tuples)
        stats["unique_entities"] = n_entities
        stats["total_mentions"] = n_mentions
        stats["total_relations"] = n_relations
        if n_entities < len(entities_by_id):
            logger.warning("_integrate_entities: %d/%d entity nodes committed",
                           n_entities, len(entities_by_id))
        if n_mentions < len(mentions_pairs):
            logger.warning("_integrate_entities: %d/%d MENTIONS edges committed",
                           n_mentions, len(mentions_pairs))

        return stats

    def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Delegate to VectorStoreAdapter.vector_search.

        Args:
            query_embedding: Dense query vector (same dim as stored embeddings).
            top_k: Maximum results to return.
            threshold: Minimum similarity score. NOTE (review 2026-06-12,
                finding #10): this facade default is 0.3 (the product default),
                whereas the underlying ``VectorStoreAdapter.vector_search``
                default is 0.0 (no filtering). A caller that bypasses this
                facade and hits the adapter directly therefore filters nothing
                unless it passes a threshold explicitly. Production retrieval
                always goes through the facade / the configured
                similarity_threshold, so the effective floor is the configured
                one — the adapter's 0.0 is an "unfiltered primitive" default.

        Returns:
            List of dicts with text, similarity, document_id, metadata.
        """
        return self.vector_store.vector_search(
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=threshold,
        )

    def graph_search(
        self,
        entities: List[str],
        max_hops: int = 2,
        top_k: int = 5,
        enable_hop3: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Entity-driven graph retrieval.

        Uses KuzuDB find_chunks_by_entity_multihop.
        Results are sorted by (hops, -triple_confidence).

        Args:
            entities: Entity name strings extracted from the query.
            max_hops: Cap on graph-traversal depth. 1 = Hop-0 only (direct
                mentions, no bridges); 2 = adds Hop-2 (one-bridge);
                3 = adds Hop-3 (two-bridge, also requires enable_hop3=True).
                Default 2. Used by ablation studies to isolate the
                bridge-expansion contribution.
            top_k: Maximum results to return.
            enable_hop3: I-2 (publication-recommended). When True, run the
                2-bridge Hop-3 traversal in addition to Hop-0/Hop-2. Off
                by default because Hop-3 adds 200-1000ms latency. Opt in
                via `graph.enable_hop3: true` in settings.yaml.

        Returns:
            List of dicts: chunk_id, text, source_file, matched_entity,
            hops, bridge_entity, relation_type, triple_confidence.

        Reproducibility note (review 2026-06-12, finding #5): results are
        bounded by ``HybridStore.GRAPH_SEARCH_BUDGET_S`` (settings.yaml
        `graph.search_budget_seconds`). If the per-call budget is exhausted the
        method returns the chunks gathered so far and logs a WARNING — so on
        slow hardware the tail of the result list can differ run-to-run. Raise
        the budget to make the path deterministic.
        """
        results: List[Dict[str, Any]] = []
        seen_chunks: set = set()

        # Entity-key filtering: drop low-value graph-lookup keys (NORP adjectives,
        # broad field nouns, bare years, question-word fragments). These match
        # high-degree hub nodes and make the un-indexed CONTAINS scan + hop-2/3
        # expansion catastrophically slow without adding signal. Specific named
        # entities are kept; if *everything* is filtered out we fall back to the
        # original list so the call still returns something.
        worthy = [e for e in entities if _is_graph_worthy_entity(e)]
        if not worthy and entities:
            logger.debug("graph_search: all %d entities below worthiness bar; "
                         "using full list", len(entities))
            worthy = list(entities)
        elif len(worthy) < len(entities):
            dropped = [e for e in entities if e not in worthy]
            logger.debug("graph_search: skipping low-value entities %s", dropped)

        # Per-call wall-clock budget so a pathological hub query can't stall the
        # whole pipeline (the multihop helper also caps hub expansion internally).
        # Class attribute so it is overridable from settings.yaml
        # (`graph.search_budget_seconds`) — review 2026-06-12, finding #4.
        budget_s = self.GRAPH_SEARCH_BUDGET_S
        t_start = time.time()

        for entity_name in worthy:
            if time.time() - t_start > budget_s:
                logger.warning("graph_search: time budget (%.0fs) exhausted after "
                               "%d/%d entities — returning partial results",
                               budget_s, len(results), len(worthy))
                break
            entity_chunks = self.graph_store.find_chunks_by_entity_multihop(
                entity_name=entity_name,
                max_results=top_k,
                enable_hop3=enable_hop3,
                max_hops=max_hops,
            )

            for chunk in entity_chunks:
                chunk_id = chunk.get("chunk_id")
                if chunk_id and chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    results.append({
                        "chunk_id": chunk_id,
                        "text": chunk.get("text", ""),
                        "source_file": chunk.get("source_file", ""),
                        "matched_entity": chunk.get("matched_entity", entity_name),
                        "hops": chunk.get("hops", 0),
                        "bridge_entity": chunk.get("bridge_entity"),
                        "relation_type": chunk.get("relation_type"),
                        # Triple-frequency confidence. Hop-0 (direct
                        # mention) defaults to 1.0 since no bridge is needed.
                        "triple_confidence": chunk.get("triple_confidence", 1.0),
                    })

        # Sort by (hops, -triple_confidence) so direct hits come first,
        # then bridges ranked by corpus support strength rather than REBEL's
        # constant 0.5 sentinel. The triple_confidence is floored at 0.0 in the
        # sort key so the error sentinel (_TRIPLE_CONF_ERROR = -1.0, set when the
        # support query fails — review 2026-06-12) sorts an un-scorable bridge to
        # the BOTTOM rather than (via the negation) to the top.
        results.sort(
            key=lambda x: (x.get("hops", 999),
                           -max(0.0, x.get("triple_confidence", 0.0)))
        )
        return results[:top_k]

    def save(self) -> None:
        """Persist the graph store (no-op for KuzuDB — data is written on each operation)."""
        self.graph_store.save()

    def reset_vector_store(self) -> None:
        """Wipe and re-create the vector store (used for ablation studies)."""
        if self.config.vector_db_path.exists():
            shutil.rmtree(self.config.vector_db_path)
        self.vector_store = VectorStoreAdapter(
            self.config.vector_db_path,
            self.embedding_dim,
            self.config.normalize_embeddings,
            self.config.distance_metric,
            self.config.overfetch_factor,
        )
        logger.info("Vector store reset")

    def reset_graph_store(self) -> None:
        """Clear all graph data (used for ablation studies)."""
        self.graph_store.clear()
        logger.info("Graph store reset")

    def reset_all(self) -> None:
        """Reset both vector and graph stores."""
        self.reset_vector_store()
        self.reset_graph_store()


# ============================================================================
# DIAGNOSTICS
# ============================================================================

def run_diagnostics(config: StorageConfig, embeddings: Embeddings) -> Dict[str, Any]:
    """
    Run diagnostic checks on storage configuration.

    Args:
        config: StorageConfig to validate.
        embeddings: Embeddings model to probe for dimension detection.

    Returns:
        Dict with embedding_dim, graph_backend, availability flags, issues list.
    """
    results: Dict[str, Any] = {
        "embedding_dim": None,
        "kuzu_available": KUZU_AVAILABLE,
        "issues": [],
    }

    try:
        test_emb = embeddings.embed_query("diagnostic test")
        results["embedding_dim"] = len(test_emb)
    except (RuntimeError, ValueError, AttributeError, ConnectionError) as exc:
        results["issues"].append("Embedding failed: %s" % exc)

    if not KUZU_AVAILABLE:
        results["issues"].append("KuzuDB not installed — install with: pip install kuzu")

    return results


# ============================================================================
# FACTORY
# ============================================================================

def create_storage_config(
    cfg: Optional[Dict[str, Any]] = None,
    dataset: str = "default",
) -> "StorageConfig":
    """
    Build a StorageConfig from a settings.yaml configuration dictionary.

    This is the canonical way to construct StorageConfig in production code.
    It keeps storage.py decoupled from YAML-loading and ensures all paths and
    thresholds are sourced from config/settings.yaml rather than scattered
    across caller code.

    Parameter mapping from settings.yaml:
        vector_store.db_path          → vector_db_path  (with dataset sub-dir)
        graph.graph_path              → graph_db_path   (with dataset sub-dir)
        embeddings.embedding_dim      → embedding_dim
        vector_store.similarity_threshold  → similarity_threshold
        vector_store.normalize_embeddings  → normalize_embeddings
        vector_store.distance_metric       → distance_metric
        graph.backend                      → graph_backend
        ingestion.extract_entities         → enable_entity_extraction

    Args:
        cfg:     Full settings dict as loaded from config/settings.yaml.
                 Pass None or {} to fall back to class-level defaults.
                 Paths will then resolve to ./data/default/vector_db and
                 ./data/default/knowledge_graph.
        dataset: Dataset sub-directory name (e.g. "hotpotqa"). Appended to
                 the base paths from settings.yaml so each dataset gets an
                 isolated store, preventing cross-dataset data leakage.

    Returns:
        StorageConfig (frozen dataclass).
    """
    cfg = cfg or {}
    vs_cfg = cfg.get("vector_store", {})
    gr_cfg = cfg.get("graph", {})
    emb_cfg = cfg.get("embeddings", {})
    ing_cfg = cfg.get("ingestion", {})

    base_vector = Path(vs_cfg.get("db_path", "./data/vector_db"))
    base_graph = Path(gr_cfg.get("graph_path", "./data/knowledge_graph"))

    # Append dataset sub-directory when a named dataset is specified
    if dataset and dataset != "default":
        vector_db_path = base_vector.parent / dataset / base_vector.name
        graph_db_path = base_graph.parent / dataset / base_graph.name
    else:
        vector_db_path = base_vector
        graph_db_path = base_graph

    return StorageConfig(
        vector_db_path=vector_db_path,
        graph_db_path=graph_db_path,
        embedding_dim=emb_cfg.get("embedding_dim", None),
        similarity_threshold=vs_cfg.get("similarity_threshold", 0.3),
        normalize_embeddings=vs_cfg.get("normalize_embeddings", True),
        distance_metric=vs_cfg.get("distance_metric", "cosine"),
        # Fallbacks mirror the StorageConfig dataclass defaults so settings.yaml
        # remains authoritative for the ANN over-fetch and graph-node text caps.
        overfetch_factor=vs_cfg.get("overfetch_factor", 3),
        graph_text_max_chars=vs_cfg.get("graph_text_max_chars", 500),
        graph_backend=gr_cfg.get("backend", "kuzu"),
        enable_entity_extraction=ing_cfg.get("extract_entities", False),
    )


# ============================================================================
# SELF-VERIFICATION
# ============================================================================

def _main() -> None:
    """Smoke demo and test runner for direct module invocation."""
    import sys
    import subprocess
    import tempfile

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # --- smoke demo -----------------------------------------------------------
    class _MockEmbeddings:
        """Deterministic mock embeddings for smoke testing."""
        def __init__(self, dim: int = 64) -> None:
            self.dim = dim

        def embed_documents(self, texts):
            results = []
            for text in texts:
                # Content-based seed: deterministic across processes, no global RNG mutation.
                seed = sum(ord(c) for c in text[:64]) % 10000
                rng = np.random.default_rng(seed)
                vec = rng.standard_normal(self.dim).astype(np.float32)
                vec /= np.linalg.norm(vec) + 1e-8
                results.append(vec.tolist())
            return results

        def embed_query(self, text):
            return self.embed_documents([text])[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        config = StorageConfig(
            vector_db_path=Path(tmpdir) / "vector",
            graph_db_path=Path(tmpdir) / "graph",
            embedding_dim=64,
        )
        store = HybridStore(config, _MockEmbeddings(dim=64))

        docs = [
            Document(
                page_content="Albert Einstein developed the theory of relativity.",
                metadata={"chunk_id": "c1", "source_file": "physics.pdf",
                          "chunk_index": 0, "page_number": 1},
            ),
            Document(
                page_content="He received the Nobel Prize in Physics in 1921.",
                metadata={"chunk_id": "c2", "source_file": "physics.pdf",
                          "chunk_index": 1, "page_number": 1},
            ),
        ]
        store.add_documents(docs)

        query_emb = _MockEmbeddings(dim=64).embed_query("relativity")
        # threshold=0.0: smoke test verifies plumbing, not semantic quality.
        results = store.vector_search(query_emb, top_k=2, threshold=0.0)
        assert results, "Expected at least one vector result"
        logger.info("Smoke demo: vector_search returned %d results", len(results))

        if KUZU_AVAILABLE:
            graph_results = store.graph_search(["Einstein"], top_k=5)
            logger.info(
                "Smoke demo: graph_search returned %d results", len(graph_results)
            )

    logger.info("Smoke demo passed.")

    # --- pytest ---------------------------------------------------------------
    # Canonical test suite lives under <project-root>/test_system/.
    test_file = Path(__file__).resolve().parents[2] / "test_system" / "test_data_layer.py"
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "pytest", str(test_file),
         "-v", "-k", "storage or Storage or VectorStore or HybridStore"],
        check=False,
    )
    sys.exit(proc.returncode)


if __name__ == "__main__":
    _main()
