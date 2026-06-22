"""
Entity Extraction Pipeline: GLiNER + REBEL

This module implements Named Entity Recognition (NER) and Relation Extraction
(RE) for the Edge-RAG pipeline. It populates the KuzuDB knowledge graph during
document ingestion, enabling graph-augmented hybrid retrieval in the Navigator
agent (S_N). Architectural position: Data Layer (Artifact A).

Data flow:
  raw chunk text
    → GLiNERExtractor (zero-shot NER)
    → REBELExtractor (relation extraction, conditional on entity count)
    → EntityExtractionPipeline (cache + orchestration)
    → KuzuDB knowledge graph

NER fallback chain (activated when GLiNER is unavailable or fails):
  1. GLiNER zero-shot NER — primary backend (bidirectional transformer)
  2. SpaCy en_core_web_sm — supervised fallback, loaded once at init
  3. Regex proper-noun heuristic — last resort, CONCEPT type only

Lightweight alternative:
  SpacyEntityPipeline implements the same interface as EntityExtractionPipeline
  using only SpaCy (no GLiNER/REBEL). Recommended for memory-constrained
  deployments where the combined ~3.1 GB footprint of GLiNER + REBEL is
  infeasible on the target hardware (< 16 GB RAM).

Entity ID design:
  Each entity is identified by SHA-256(f"{name.lower()}:{type}")[:24].
  96-bit output space → birthday-bound collision probability < 1 in 10^14
  for any realistic corpus. See _generate_entity_id() for rationale.

Cache design:
  EntityCache is a hybrid in-memory LRU + SQLite persistent store, scoped
  by (text_hash, model_name) to invalidate stale entries when the NER model
  or configuration changes between ingestion runs.

Scientific references:
  NER:   Zaratiana, U. et al. (2023). "GLiNER: Generalist Model for Named
         Entity Recognition using Bidirectional Transformer." arXiv:2311.08526.
  RE:    Cabot, P. L. H., & Navigli, R. (2021). "REBEL: Relation Extraction
         by End-to-end Language generation." EMNLP 2021 Findings.
  SpaCy: Honnibal, M. et al. (2020). "spaCy: Industrial-strength Natural
         Language Processing in Python." https://spacy.io

Confidence sentinels (module-level constants, see below):
    _CONF_GLINER_SPACY_FALLBACK   SpaCy used as GLiNER fallback
    _CONF_GLINER_REGEX_FALLBACK   Regex proper-noun heuristic (last resort)
    _CONF_SPACY_LIGHT             Lightweight SpacyEntityPipeline
    _CONF_CACHE_DEFAULT           Missing-field default in legacy cache entries

Last reviewed: 2026-05-25 (audit pass, project version 5.4).
"""

import logging
import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
import time

logger = logging.getLogger(__name__)

# Public API. The other classes defined in this module (ExtractedEntity,
# ExtractedRelation, ChunkExtractionResult, GLiNERExtractor, REBELExtractor,
# SpacyEntityPipeline, EntityCache) are implementation details. They remain
# importable by direct name from this module (tests rely on this), but are
# intentionally NOT re-exported from the data_layer package.
__all__ = [
    "EntityExtractionPipeline",
    "ExtractionConfig",
    "create_extraction_pipeline",
    "normalize_entity_name",
]

# ---------------------------------------------------------------------------
# Confidence sentinels for non-probabilistic NER backends.
# ---------------------------------------------------------------------------
# Lifted out of in-method literals so the calibration is greppable and
# documented in one place. The values reflect relative trust placed in each
# backend, not measured probabilities:
#   GLINER_SPACY_FALLBACK > SPACY_LIGHT > CACHE_DEFAULT >= GLINER_REGEX_FALLBACK
# GLiNER itself emits per-span scores and is NOT covered by these constants.
_CONF_GLINER_SPACY_FALLBACK: float = 0.7   # SpaCy en_core_web_sm under GLiNER
_CONF_GLINER_REGEX_FALLBACK: float = 0.5   # regex proper-noun heuristic
_CONF_SPACY_LIGHT: float = 0.8             # SpacyEntityPipeline (no GLiNER stack)
_CONF_CACHE_DEFAULT: float = 0.5           # missing-field fallback for old cache rows


# ---------------------------------------------------------------------------
# Optional dependency flags
# ---------------------------------------------------------------------------

from .entity_types import GLINER_LABEL_MAP, SPACY_LABEL_MAP, SPACY_LABEL_MAP_FLAT

try:
    from gliner import GLiNER
    GLINER_AVAILABLE = True
except ImportError:
    GLINER_AVAILABLE = False
    logger.warning("GLiNER not available. Install with: pip install gliner")

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Install with: pip install transformers")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("SpaCy not available. Install with: pip install spacy")


# -- Module-level entity ID helper (shared by all extractor classes) ------------

def _generate_entity_id(name: str, entity_type: str) -> str:
    """
    Deterministic entity identifier via SHA-256 content addressing.

    Key properties:
      - Stable:              same (name, type) always yields the same ID.
      - Deduplicating:       identical strings share one knowledge-graph node.
      - Collision-resistant: 24 hex chars = 96-bit output space.
        Birthday bound at 50% collision probability: √(2^96) ≈ 9 × 10^14
        distinct (name, type) pairs — safe for any realistic corpus.

    Design choice: 24 hex chars balances storage compactness with collision
    safety. MD5 (previously used, 48-bit truncated) had a birthday bound of
    ~16 million entries — insufficient for large Wikipedia-scale corpora.

    Args:
        name:        Normalized entity name (lowercase, stripped).
        entity_type: Canonical type string (e.g. "PERSON", "GPE").

    Returns:
        24-character lowercase hexadecimal string.
    """
    combined = f"{name.lower().strip()}:{entity_type}"
    return hashlib.sha256(combined.encode()).hexdigest()[:24]


# -- Data classes ---------------------------------------------------------------

@dataclass
class ExtractedEntity:
    """Container for a single named entity extracted from a text chunk."""
    entity_id: str
    name: str
    entity_type: str           # Canonical type: PERSON, GPE, LOCATION, etc.
    confidence: float
    mention_span: Tuple[int, int]   # Character offsets (start, end)
    source_chunk_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type,   # canonical key — no "type" alias
            "confidence": self.confidence,
            "mention_span": list(self.mention_span),
            "source_chunk_id": self.source_chunk_id,
        }


@dataclass
class ExtractedRelation:
    """Container for a subject–relation–object triple."""
    subject_entity: str
    relation_type: str
    object_entity: str
    confidence: float
    source_chunk_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_entity": self.subject_entity,
            "relation_type": self.relation_type,
            "object_entity": self.object_entity,
            "confidence": self.confidence,
            "source_chunk_ids": self.source_chunk_ids,
        }


@dataclass
class ChunkExtractionResult:
    """Aggregated extraction result for one text chunk."""
    chunk_id: str
    text: str
    entities: List[ExtractedEntity]
    relations: List[ExtractedRelation]
    extraction_time_ms: float

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def relation_count(self) -> int:
        return len(self.relations)


@dataclass
class ExtractionConfig:
    """
    Configuration for the entity extraction pipeline.

    All parameters that appear here also exist in config/settings.yaml
    → entity_extraction, which is the single source of truth during a run.
    The values below serve as documented emergency fallbacks when no config
    file is provided (e.g. in unit tests).
    """
    # -- GLiNER NER ------------------------------------------------------------
    # Reference: Zaratiana et al. (2023). arXiv:2311.08526.
    gliner_model: str = "urchade/gliner_small-v2.1"
    # --------------------------------------------------------------------------
    # FIXED: OntoNotes-5 core entity-type set (Weischedel et al. 2013,
    # LDC2013T19). Mirrors config/settings.yaml exactly.
    # 9 GLiNER prompts -> 8 canonical types via GLINER_LABEL_MAP:
    #   person, organization, location, city, country, date, event,
    #   work of art, product.
    #
    # IMPORTANT: DO NOT change this list. It is the scientifically
    # defensible label set across HotpotQA + 2WikiMultiHopQA + StrategyQA.
    # Adding domain-specific types (state, landmark, album, film, movie,
    # award, ...) breaks cross-dataset transfer and the reproducibility
    # of the paper.
    # The multi-prompt expansion (city + country alongside location) is a
    # recall optimisation, not a domain specialisation - all prompts map
    # to the same OntoNotes-5 canonical types in the graph.
    # --------------------------------------------------------------------------
    entity_types: List[str] = field(default_factory=lambda: [
        "person",
        "organization",
        "location",
        "city",
        "country",
        "date",
        "event",
        "work of art",
        "product",
    ])
    ner_confidence_threshold: float = 0.15   # recall-optimised; junk filtered downstream
    ner_batch_size: int = 16

    # -- REBEL Relation Extraction ---------------------------------------------
    # Reference: Cabot & Navigli (2021). EMNLP 2021 Findings.
    rebel_model: str = "Babelscape/rebel-large"
    re_confidence_threshold: float = 0.5   # uniform sentinel — REBEL emits no per-triplet score
    re_batch_size: int = 8
    min_entities_for_re: int = 2           # skip RE when fewer entities are found
    rebel_max_input_length: int = 256      # tokenizer truncation length
    rebel_max_output_length: int = 256     # generation length cap
    rebel_num_beams: int = 5               # beam search width
    device: str = "cpu"                    # "cuda" if GPU is available

    # -- SpaCy fallback --------------------------------------------------------
    spacy_fallback_model: str = "en_core_web_sm"

    # -- Caching ---------------------------------------------------------------
    cache_enabled: bool = True
    cache_path: str = "./data/entity_cache.db"
    lru_cache_size: int = 10000

    # -- Fallback-path confidence sentinels ------------------------------------
    # settings.yaml entity_extraction.fallback_confidence. Used only when the
    # GLiNER+REBEL primary path is unavailable; the reproducible GLiNER-on run
    # never reads them. Defaults mirror the module-level sentinels.
    conf_gliner_spacy_fallback: float = _CONF_GLINER_SPACY_FALLBACK
    conf_gliner_regex_fallback: float = _CONF_GLINER_REGEX_FALLBACK
    conf_spacy_light: float = _CONF_SPACY_LIGHT
    conf_cache_default: float = _CONF_CACHE_DEFAULT


# -- Entity Cache ---------------------------------------------------------------

class EntityCache:
    """
    Hybrid in-memory LRU + SQLite persistent cache for extraction results.

    DESIGN

    Content-addressable storage:
        Cache key = (SHA-256(text.encode())[:16], model_name).
        Scoping by model_name ensures cache invalidation when the NER model
        or entity type list changes between ingestion runs.

    In-memory LRU:
        Implemented via collections.OrderedDict (move_to_end on access).
        In-memory key = f"{text_hash}:{model_name}".
        Evicts the least-recently-used entry when max_size is reached.

    SQLite backend:
        WAL mode for improved read concurrency.
        check_same_thread=False for thread-safe access.

    Schema:
        entity_cache (
            text_hash     TEXT NOT NULL,   -- SHA-256[:16] of input text
            model_name    TEXT NOT NULL,   -- NER model identifier
            entities_json TEXT NOT NULL,   -- JSON-encoded extraction dict
            hit_count     INTEGER DEFAULT 1,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (text_hash, model_name)
        )
    """

    def __init__(self, cache_path: str, max_size: int = 10000) -> None:
        self.cache_path = Path(cache_path)
        self.max_size = max_size
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_cache: OrderedDict = OrderedDict()
        self.conn: Optional[sqlite3.Connection] = None
        self._init_db()
        logger.info("EntityCache initialized: %s", self.cache_path)

    def _init_db(self) -> None:
        """Create schema if absent; migrate legacy schema; enable WAL mode."""
        self.conn = sqlite3.connect(
            str(self.cache_path),
            check_same_thread=False,
        )
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS entity_cache (
                text_hash     TEXT NOT NULL,
                model_name    TEXT NOT NULL,
                entities_json TEXT NOT NULL,
                hit_count     INTEGER DEFAULT 1,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (text_hash, model_name)
            )
        """)
        # Migration: add model_name column to pre-v3.5.0 cache databases that
        # were created without it.  ALTER TABLE is a no-op on fresh DBs because
        # the column is already present; the try/except handles the
        # "duplicate column" error from SQLite without needing a schema-version
        # table.
        try:
            self.conn.execute(
                "ALTER TABLE entity_cache ADD COLUMN model_name TEXT NOT NULL DEFAULT ''"
            )
            logger.info("EntityCache: migrated legacy schema — added model_name column")
        except sqlite3.OperationalError:
            pass  # column already exists
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_hitcount
            ON entity_cache(model_name, hit_count DESC)
        """)
        self.conn.commit()

    _CACHE_HASH_LEN: int = 16   # 64-bit key; birthday bound at ~4 billion entries

    def _text_hash(self, text: str) -> str:
        """SHA-256 of text, truncated to _CACHE_HASH_LEN hex chars (64-bit cache key)."""
        return hashlib.sha256(text.encode()).hexdigest()[:self._CACHE_HASH_LEN]

    def get(self, text: str, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached extraction result for a single text.

        Args:
            text:       Input text.
            model_name: NER model identifier (cache scope key).

        Returns:
            Dict with "entities" and "relations" lists, or None on a miss.
        """
        key = self._text_hash(text)
        mem_key = f"{key}:{model_name}"

        # In-memory LRU check
        if mem_key in self._memory_cache:
            self._memory_cache.move_to_end(mem_key)
            return self._memory_cache[mem_key]

        # SQLite check
        try:
            cursor = self.conn.execute(
                "SELECT entities_json FROM entity_cache "
                "WHERE text_hash = ? AND model_name = ?",
                (key, model_name),
            )
            row = cursor.fetchone()
            if row is not None:
                data = json.loads(row[0])
                self.conn.execute(
                    "UPDATE entity_cache "
                    "SET hit_count = hit_count + 1, "
                    "last_accessed = CURRENT_TIMESTAMP "
                    "WHERE text_hash = ? AND model_name = ?",
                    (key, model_name),
                )
                self.conn.commit()
                self._add_to_memory(mem_key, data)
                return data
            return None
        except (sqlite3.Error, AttributeError) as e:
            logger.error("EntityCache GET failed: %s", e)
            return None

    def put(self, text: str, data: Dict[str, Any], model_name: str) -> None:
        """
        Store an extraction result (upsert).

        Args:
            text:       Original input text.
            data:       Dict with "entities" and "relations" lists.
            model_name: NER model identifier.
        """
        key = self._text_hash(text)
        mem_key = f"{key}:{model_name}"
        self._add_to_memory(mem_key, data)
        try:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO entity_cache
                    (text_hash, model_name, entities_json, hit_count)
                VALUES (?, ?, ?, 1)
                """,
                (key, model_name, json.dumps(data)),
            )
            self.conn.commit()
        except (sqlite3.Error, AttributeError) as e:
            logger.error("EntityCache PUT failed: %s", e)

    def get_batch(
        self,
        texts: List[str],
        model_name: str,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Retrieve cached results for multiple texts in a single SQL query.

        More efficient than N individual get() calls: one SELECT IN (...)
        plus memory-cache checks replaces N sequential round-trips.

        Args:
            texts:      Input texts (may contain duplicates).
            model_name: NER model identifier.

        Returns:
            Dict mapping text index (position in ``texts``) to cached result.
            Only cache-hit indices are present.
        """
        if not texts:
            return {}

        hash_to_idxs: Dict[str, List[int]] = {}
        for i, t in enumerate(texts):
            hash_to_idxs.setdefault(self._text_hash(t), []).append(i)

        # Check memory cache first — avoids SQL for hot entries.
        results: Dict[int, Dict[str, Any]] = {}
        db_hashes: List[str] = []
        for h, idxs in hash_to_idxs.items():
            mem_key = f"{h}:{model_name}"
            if mem_key in self._memory_cache:
                self._memory_cache.move_to_end(mem_key)
                for idx in idxs:
                    results[idx] = self._memory_cache[mem_key]
            else:
                db_hashes.append(h)

        # Fetch remaining misses from SQLite in one query.
        if db_hashes:
            placeholders = ",".join(["?"] * len(db_hashes))
            try:
                cursor = self.conn.execute(
                    f"SELECT text_hash, entities_json FROM entity_cache "
                    f"WHERE model_name = ? AND text_hash IN ({placeholders})",
                    [model_name] + db_hashes,
                )
                rows = cursor.fetchall()
                hit_hashes: List[str] = []
                for text_hash, entities_json in rows:
                    data = json.loads(entities_json)
                    for idx in hash_to_idxs[text_hash]:
                        results[idx] = data
                    self._add_to_memory(f"{text_hash}:{model_name}", data)
                    hit_hashes.append(text_hash)

                if hit_hashes:
                    ph = ",".join(["?"] * len(hit_hashes))
                    self.conn.execute(
                        f"UPDATE entity_cache "
                        f"SET hit_count = hit_count + 1, "
                        f"last_accessed = CURRENT_TIMESTAMP "
                        f"WHERE model_name = ? AND text_hash IN ({ph})",
                        [model_name] + hit_hashes,
                    )
                    self.conn.commit()
            except (sqlite3.Error, AttributeError) as e:
                logger.error("EntityCache batch GET failed: %s", e)

        return results

    def _add_to_memory(self, mem_key: str, value: Dict[str, Any]) -> None:
        """Add to in-memory LRU; evict least-recently-used entry if at capacity."""
        if len(self._memory_cache) >= self.max_size:
            self._memory_cache.popitem(last=False)
        self._memory_cache[mem_key] = value

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        try:
            cursor = self.conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM entity_cache"
            )
            row = cursor.fetchone()
            size_bytes = self.cache_path.stat().st_size if self.cache_path.exists() else 0
            return {
                "total_entries": row[0] or 0,
                "total_hits": row[1] or 0,
                "memory_entries": len(self._memory_cache),
                "size_mb": size_bytes / (1024 * 1024),
            }
        except (sqlite3.Error, AttributeError) as e:
            logger.error("EntityCache get_stats failed (connection closed?): %s", e)
            return {"total_entries": 0, "total_hits": 0, "memory_entries": 0, "size_mb": 0.0}

    def close(self) -> None:
        """Close the SQLite connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "EntityCache":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# -- Shared normalization (used by GLiNERExtractor AND hybrid_retriever) --------

# Types for which leading articles are grammatical, not part of the official name.
_STRIP_ARTICLE_TYPES: frozenset = frozenset({"GPE", "LOCATION", "EVENT"})

# Abbreviation suffixes whose trailing period must NOT be stripped.
_ABBREV_SUFFIXES: Tuple[str, ...] = (
    " Inc.", " Ltd.", " Bros.", " Corp.", " Co.", " Jr.", " Sr.", " Dr.",
)


def normalize_entity_name(name: str, entity_type: Optional[str] = None) -> str:
    """
    Normalise an entity name for consistent knowledge-graph lookups.

    Used at both ingestion time (GLiNERExtractor) and query time
    (hybrid_retriever.ImprovedQueryEntityExtractor) so the same
    normalisation is applied to both stored and queried names.

    Transformations (applied in order):
      1. Strip leading/trailing whitespace.
      2. Remove trailing , ; :
      3. Remove trailing period unless the name ends with a known
         abbreviation suffix (Inc., Ltd., etc.).
      4. Strip leading articles ("The ", "A ", "An ") for
         GPE/LOCATION/EVENT types only.
    """
    name = name.strip().rstrip(",;:")
    if name.endswith(".") and not any(name.endswith(s) for s in _ABBREV_SUFFIXES):
        name = name[:-1]
    if entity_type in _STRIP_ARTICLE_TYPES:
        for article in ("The ", "A ", "An "):
            if name.startswith(article) and len(name) > len(article) + 1:
                name = name[len(article):]
                break
    return name


# -- GLiNER NER Extractor -------------------------------------------------------

class GLiNERExtractor:
    """
    Named Entity Recognition via GLiNER (zero-shot).

    GLiNER predicts entity spans by treating entity-type descriptions as
    natural-language labels fed to a bidirectional encoder (DeBERTa-v3-small),
    eliminating the need for domain-specific fine-tuning data.

    Reference: Zaratiana, U. et al. (2023). "GLiNER: Generalist Model for
    Named Entity Recognition using Bidirectional Transformer." arXiv:2311.08526.

    Fallback chain (activated when GLiNER model is unavailable or raises):
      1. SpaCy en_core_web_sm — loaded once at __init__ to avoid repeated
         12 MB model-load overhead in the per-chunk hot path.
      2. Regex proper-noun heuristic — last resort.
    """

    # Natural-language GLiNER output label → canonical internal type.
    _LABEL_MAP: Dict[str, str] = GLINER_LABEL_MAP

    # Types and abbreviation suffixes are module-level constants shared with
    # hybrid_retriever._normalize_query_entity (see normalize_entity_name()).

    @classmethod
    def _normalize_label(cls, label: str) -> str:
        """Map a GLiNER output label to the canonical internal type."""
        return cls._LABEL_MAP.get(label.lower(), label.upper())

    @classmethod
    def _normalize_entity_name(cls, name: str, entity_type: Optional[str] = None) -> str:
        """Delegate to module-level normalize_entity_name() for DRY normalisation."""
        return normalize_entity_name(name, entity_type)

    def __init__(self, config: ExtractionConfig) -> None:
        self.config = config
        self.model: Optional[Any] = None
        self._nlp: Optional[Any] = None   # SpaCy model — loaded once if needed
        self._fallback_warned: bool = False
        self._load_model()
        # Pre-load the SpaCy fallback so _spacy_extract does not call
        # spacy.load() on every chunk invocation (12 MB model, ~100 ms load).
        if self.model is None and SPACY_AVAILABLE:
            self._nlp = self._load_spacy()

    def _load_model(self) -> None:
        """Load GLiNER model weights. Sets self.model = None on failure."""
        if not GLINER_AVAILABLE:
            logger.warning("GLiNER package not installed; fallback chain active")
            return
        try:
            logger.info("Loading GLiNER model: %s", self.config.gliner_model)
            self.model = GLiNER.from_pretrained(self.config.gliner_model)
            logger.info("GLiNER model loaded successfully")
        except (OSError, ValueError, RuntimeError) as e:
            logger.error(
                "Failed to load GLiNER model '%s': %s", self.config.gliner_model, e
            )
            self.model = None
        except Exception as _net_err:
            # transformers ≥ 4.45 calls huggingface_hub.model_info() even for
            # cached models; raises httpx.ConnectError in offline environments.
            # Retry with HF_HUB_OFFLINE=1 to load from local cache.
            logger.warning(
                "GLiNER online load raised %s; retrying offline.",
                type(_net_err).__name__,
            )
            _prev_hf = os.environ.get("HF_HUB_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                self.model = GLiNER.from_pretrained(self.config.gliner_model)
                logger.info("GLiNER model loaded from local cache")
            except Exception as e2:
                logger.error(
                    "GLiNER offline load failed for '%s': %s",
                    self.config.gliner_model, e2,
                )
                self.model = None
            finally:
                if _prev_hf is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = _prev_hf

    def _load_spacy(self) -> Optional[Any]:
        """Load SpaCy model once; return None on failure."""
        model_name = self.config.spacy_fallback_model
        try:
            nlp = spacy.load(model_name)
            logger.info("SpaCy fallback model loaded (%s)", model_name)
            return nlp
        except (OSError, ImportError) as e:
            logger.warning("SpaCy model '%s' unavailable: %s", model_name, e)
            return None

    def extract(self, text: str, chunk_id: str) -> List[ExtractedEntity]:
        """
        Extract named entities from a single text chunk.

        Args:
            text:     Input text.
            chunk_id: Unique identifier of the source chunk.

        Returns:
            List of ExtractedEntity objects.
        """
        if self.model is None:
            if not self._fallback_warned:
                logger.warning(
                    "GLiNER model unavailable — using %s fallback for NER",
                    "SpaCy" if self._nlp is not None else "regex",
                )
                self._fallback_warned = True
            return self._fallback_extract(text, chunk_id)

        try:
            entities = self.model.predict_entities(
                text,
                self.config.entity_types,
                threshold=self.config.ner_confidence_threshold,
            )
            return self._build_entities(entities, chunk_id)
        except (RuntimeError, ValueError) as e:
            logger.error("GLiNER extraction failed: %s", e)
            return self._fallback_extract(text, chunk_id)

    def extract_batch(
        self,
        texts: List[str],
        chunk_ids: List[str],
    ) -> List[List[ExtractedEntity]]:
        """
        Batch NER for multiple texts.

        Processes texts in batches of config.ner_batch_size to amortise
        GLiNER inference overhead. Falls back per-batch on exception.

        Args:
            texts:     Input texts.
            chunk_ids: Corresponding chunk identifiers.

        Returns:
            List of entity lists in the same order as the input.
        """
        if self.model is None:
            logger.warning(
                "GLiNER model unavailable — fallback NER for all %d chunks",
                len(texts),
            )
            return [self._fallback_extract(t, c) for t, c in zip(texts, chunk_ids)]

        all_results: List[List[ExtractedEntity]] = []
        for i in range(0, len(texts), self.config.ner_batch_size):
            batch_texts = texts[i : i + self.config.ner_batch_size]
            batch_ids = chunk_ids[i : i + self.config.ner_batch_size]
            try:
                batch_entities = self.model.inference(
                    batch_texts,
                    self.config.entity_types,
                    threshold=self.config.ner_confidence_threshold,
                )
                for text_entities, chunk_id in zip(batch_entities, batch_ids):
                    all_results.append(self._build_entities(text_entities, chunk_id))
            except (RuntimeError, ValueError) as e:
                logger.error(
                    "GLiNER batch extraction failed (batch starting at %d): %s", i, e
                )
                for t, c in zip(batch_texts, batch_ids):
                    all_results.append(self._fallback_extract(t, c))

        return all_results

    def _build_entities(
        self, raw_entities: List[Dict[str, Any]], chunk_id: str
    ) -> List[ExtractedEntity]:
        """Convert GLiNER prediction dicts to ExtractedEntity objects."""
        results = []
        for ent in raw_entities:
            canonical_type = self._normalize_label(ent["label"])
            norm_name = self._normalize_entity_name(ent["text"], canonical_type)
            results.append(ExtractedEntity(
                entity_id=_generate_entity_id(norm_name, canonical_type),
                name=norm_name,
                entity_type=canonical_type,
                confidence=ent["score"],
                mention_span=(ent["start"], ent["end"]),
                source_chunk_id=chunk_id,
            ))
        return results

    def _fallback_extract(self, text: str, chunk_id: str) -> List[ExtractedEntity]:
        """Route to SpaCy (if loaded) or regex fallback."""
        if self._nlp is not None:
            return self._spacy_extract(text, chunk_id)
        return self._regex_extract(text, chunk_id)

    # SpaCy label → canonical internal type.
    # GPE maps to "GPE" (not "LOCATION") so entity IDs are consistent with the
    # primary GLiNER path, which normalises "gpe" → "GPE" via _LABEL_MAP.
    _SPACY_FALLBACK_TYPE_MAP: Dict[str, str] = SPACY_LABEL_MAP

    def _spacy_extract(self, text: str, chunk_id: str) -> List[ExtractedEntity]:
        """
        SpaCy-based NER fallback.

        Confidence is set to 0.7 as a sentinel value: SpaCy en_core_web_sm
        is a supervised pipeline-model but does not emit per-span probabilities.
        """
        try:
            doc = self._nlp(text)
            results = []
            for ent in doc.ents:
                ent_type = self._SPACY_FALLBACK_TYPE_MAP.get(ent.label_, "CONCEPT")
                results.append(ExtractedEntity(
                    entity_id=_generate_entity_id(ent.text, ent_type),
                    name=ent.text,
                    entity_type=ent_type,
                    confidence=self.config.conf_gliner_spacy_fallback,
                    mention_span=(ent.start_char, ent.end_char),
                    source_chunk_id=chunk_id,
                ))
            return results
        except (RuntimeError, AttributeError) as e:
            logger.error("SpaCy extraction failed: %s", e)
            return self._regex_extract(text, chunk_id)

    def _regex_extract(self, text: str, chunk_id: str) -> List[ExtractedEntity]:
        """
        Regex proper-noun heuristic — last-resort fallback.

        Matches multi-word sequences of Title-Cased tokens. Confidence is
        set to 0.5 as a sentinel — this is a heuristic, not a model score.
        All entities are assigned the CONCEPT type.
        """
        results = []
        for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text):
            results.append(ExtractedEntity(
                entity_id=_generate_entity_id(match.group(1), "CONCEPT"),
                name=match.group(1),
                entity_type="CONCEPT",
                confidence=self.config.conf_gliner_regex_fallback,
                mention_span=(match.start(), match.end()),
                source_chunk_id=chunk_id,
            ))
        return results


# -- REBEL Relation Extractor ---------------------------------------------------

class REBELExtractor:
    """
    Relation Extraction via REBEL (Relation Extraction By End-to-end Language).

    REBEL casts RE as a seq2seq task: the input is raw text; the output is a
    linearized sequence of (subject, relation, object) triplets encoded with
    special tokens (<triplet>, <subj>, <obj>).

    Reference: Cabot, P. L. H., & Navigli, R. (2021). "REBEL: Relation
    Extraction by End-to-end Language generation." EMNLP 2021 Findings.

    Implementation note:
        model.generate() is used directly rather than the HuggingFace
        pipeline() abstraction to avoid task-label inference errors from the
        REBEL model card.

    Memory note:
        rebel-large requires ~2.7 GB RAM. On edge hardware with < 16 GB,
        evaluate whether RE is required for the use case. SpacyEntityPipeline
        (relations=[]) eliminates this footprint entirely.

    Non-determinism:
        Beam search on CPU is deterministic given identical inputs and weights.
        On GPU, set torch.backends.cudnn.deterministic = True if strict
        reproducibility across runs is required.
    """

    def __init__(self, config: ExtractionConfig) -> None:
        self.config = config
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        # re_batch_size is preserved in ExtractionConfig for settings.yaml
        # compatibility but is currently a no-op: REBEL's variable-length
        # seq2seq output does not support true input batching without
        # additional padding logic. Warn loudly so a user who flips the
        # setting expecting an effect does not get silent acceptance.
        if config.re_batch_size != 1:
            logger.warning(
                "REBELExtractor: re_batch_size=%d requested but is a no-op in "
                "this release; relation extraction is processed sequentially. "
                "See REBELExtractor.extract_sequential() for the rationale.",
                config.re_batch_size,
            )
        self._load_model()

    def _load_model(self) -> None:
        """Load REBEL tokenizer and model. Sets self.model = None on failure."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers package not installed; REBEL disabled")
            return
        try:
            logger.info("Loading REBEL model: %s", self.config.rebel_model)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.rebel_model)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.rebel_model)
            self.model.to(self.config.device)
            logger.info("REBEL model loaded successfully")
        except (OSError, ValueError, RuntimeError) as e:
            logger.error(
                "Failed to load REBEL model '%s': %s", self.config.rebel_model, e
            )
            self.model = None

    def extract(
        self,
        text: str,
        entities: List[ExtractedEntity],
        chunk_id: str,
    ) -> List[ExtractedRelation]:
        """
        Extract relations from text given the entities identified by NER.

        Selective application: returns [] immediately if fewer than
        config.min_entities_for_re entities were found, avoiding ~60% of
        REBEL calls on HotpotQA where most chunks mention only one entity.

        Args:
            text:     Input chunk text.
            entities: NER-extracted entities for this chunk.
            chunk_id: Source chunk identifier.

        Returns:
            List of ExtractedRelation objects, empty if REBEL is unavailable
            or too few entities are present.
        """
        if len(entities) < self.config.min_entities_for_re:
            return []

        if self.model is None:
            logger.warning(
                "REBEL model unavailable — relation extraction skipped (chunk %s)",
                chunk_id,
            )
            return []

        try:
            inputs = self.tokenizer(
                text,
                max_length=self.config.rebel_max_input_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.config.device)

            generated_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.config.rebel_max_output_length,
                num_beams=self.config.rebel_num_beams,
                num_return_sequences=1,
            )

            raw_text = self.tokenizer.decode(
                generated_ids[0], skip_special_tokens=False
            )
            raw_text = (
                raw_text.replace("<s>", "")
                .replace("</s>", "")
                .replace("<pad>", "")
            )

            triplets = self._parse_triplets(raw_text)
            entity_names = {e.name.lower() for e in entities}

            results = []
            for subj, rel, obj in triplets:
                # Relevance filter: keep triplets referencing at least one
                # known entity (substring check to handle partial name matches).
                if any(e in subj.lower() or e in obj.lower() for e in entity_names):
                    results.append(ExtractedRelation(
                        subject_entity=subj.strip(),
                        relation_type=rel.strip(),
                        object_entity=obj.strip(),
                        # REBEL does not emit per-triplet scores; re_confidence_threshold
                        # is used as a uniform sentinel value for all extracted triplets.
                        confidence=self.config.re_confidence_threshold,
                        source_chunk_ids=[chunk_id],
                    ))
            return results

        except (RuntimeError, ValueError) as e:
            logger.error("REBEL extraction failed for chunk %s: %s", chunk_id, e)
            return []

    def _parse_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Parse REBEL output format into (subject, relation, object) tuples.

        REBEL linearization format:
          <triplet> subject <subj> object <obj> relation [<triplet> ...]
        """
        triplets = []
        try:
            for part in text.split("<triplet>"):
                if "<subj>" not in part or "<obj>" not in part:
                    continue
                subj_parts = part.split("<subj>")
                subject = subj_parts[0].strip()
                obj_rel_parts = subj_parts[1].split("<obj>")
                object_ = obj_rel_parts[0].strip()
                relation = obj_rel_parts[1].strip()
                if subject and relation and object_:
                    triplets.append((subject, relation, object_))
        except (IndexError, AttributeError) as e:
            logger.debug("Triplet parsing skipped (malformed REBEL output): %s", e)
        return triplets

    def extract_sequential(
        self,
        texts: List[str],
        entities_per_text: List[List[ExtractedEntity]],
        chunk_ids: List[str],
    ) -> List[List[ExtractedRelation]]:
        """
        Apply relation extraction to a list of texts, one at a time.

        Note: REBEL's variable-length seq2seq output does not support true
        input batching without additional padding logic. This method processes
        texts sequentially. The re_batch_size parameter in ExtractionConfig
        is reserved for a future batched implementation.

        Args:
            texts:             Input texts.
            entities_per_text: Parallel entity lists.
            chunk_ids:         Parallel chunk identifiers.

        Returns:
            List of relation lists in the same order as the input.
        """
        return [
            self.extract(t, e, c)
            for t, e, c in zip(texts, entities_per_text, chunk_ids)
        ]


# -- Unified Extraction Pipeline ------------------------------------------------

class EntityExtractionPipeline:
    """
    Orchestrates GLiNER NER + REBEL RE with cache-first batch processing.

    Processing stages per batch (process_chunks_batch):
      Phase 1 — Batch cache lookup: single SQL query for all N texts.
      Phase 2 — Batch NER for cache misses (GLiNERExtractor.extract_batch).
      Phase 3 — Sequential RE for chunks with ≥ min_entities_for_re.
      Phase 4 — Cache write for newly extracted results.
      Phase 5 — Result reassembly in original input order.

    Typical latency (CPU, gliner_small-v2.1, rebel-large):
      Cache hit:           ~0.5 ms/chunk (memory) | ~1 ms/chunk (SQLite)
      Cache miss, NER:     ~6–10 ms/chunk
      Cache miss, NER+RE:  ~80–120 ms/chunk
    """

    def __init__(self, config: Optional[ExtractionConfig] = None) -> None:
        self.config = config or ExtractionConfig()
        self.ner_extractor = GLiNERExtractor(self.config)
        self.re_extractor = REBELExtractor(self.config)
        self.cache: Optional[EntityCache] = (
            EntityCache(self.config.cache_path, self.config.lru_cache_size)
            if self.config.cache_enabled
            else None
        )
        self.stats: Dict[str, Any] = {
            "total_chunks": 0,
            "cache_hits": 0,
            "ner_calls": 0,
            "re_calls": 0,
            "total_entities": 0,
            "total_relations": 0,
            # Note: last_batch_avg_ms reflects the most recent batch only,
            # not a session-wide average.
            "last_batch_avg_ms": 0.0,
        }
        logger.info("EntityExtractionPipeline initialized")

    # -- Cache reconstruction -------------------------------------------------

    def _reconstruct_from_cache(
        self,
        cached: Dict[str, Any],
        chunk_id: str,
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """
        Deserialize a cached extraction result into typed objects.

        Handles both the current serialization format (entity_type key) and
        the legacy format (type key) for backwards compatibility with caches
        written by earlier versions of this module.
        """
        entities = [
            ExtractedEntity(
                entity_id=e.get("entity_id", ""),
                name=e.get("name", ""),
                entity_type=e.get("entity_type", e.get("type", "CONCEPT")),
                confidence=e.get("confidence", self.config.conf_cache_default),
                mention_span=tuple(e.get("mention_span", [0, 0])),
                source_chunk_id=chunk_id,
            )
            for e in cached.get("entities", [])
        ]
        relations = [
            ExtractedRelation(
                subject_entity=r.get("subject_entity", r.get("subject", "")),
                relation_type=r.get("relation_type", r.get("relation", "")),
                object_entity=r.get("object_entity", r.get("object", "")),
                confidence=r.get("confidence", self.config.conf_cache_default),
                source_chunk_ids=[chunk_id],
            )
            for r in cached.get("relations", [])
        ]
        return entities, relations

    # -- Public API -----------------------------------------------------------

    def process_chunk(self, text: str, chunk_id: str) -> ChunkExtractionResult:
        """
        Extract entities and relations from a single chunk.

        Checks the entity cache first; runs GLiNER + REBEL on a cache miss.
        """
        start = time.time()
        self.stats["total_chunks"] += 1

        if self.cache:
            cached = self.cache.get(text, self.config.gliner_model)
            if cached is not None:
                self.stats["cache_hits"] += 1
                entities, relations = self._reconstruct_from_cache(cached, chunk_id)
                return ChunkExtractionResult(
                    chunk_id=chunk_id, text=text,
                    entities=entities, relations=relations,
                    extraction_time_ms=(time.time() - start) * 1000,
                )

        self.stats["ner_calls"] += 1
        entities = self.ner_extractor.extract(text, chunk_id)

        relations: List[ExtractedRelation] = []
        if len(entities) >= self.config.min_entities_for_re:
            self.stats["re_calls"] += 1
            relations = self.re_extractor.extract(text, entities, chunk_id)

        if self.cache:
            self.cache.put(
                text,
                {"entities": [e.to_dict() for e in entities],
                 "relations": [r.to_dict() for r in relations]},
                self.config.gliner_model,
            )

        self.stats["total_entities"] += len(entities)
        self.stats["total_relations"] += len(relations)
        return ChunkExtractionResult(
            chunk_id=chunk_id, text=text,
            entities=entities, relations=relations,
            extraction_time_ms=(time.time() - start) * 1000,
        )

    def process_chunks_batch(
        self,
        texts: List[str],
        chunk_ids: List[str],
    ) -> List[ChunkExtractionResult]:
        """
        Batch entity extraction with cache-first strategy.

        Phase 1 — Batch cache lookup: single SQL query for all N texts,
                   replacing N individual round-trips.
        Phase 2 — Batch NER for cache misses (GLiNER batches ner_batch_size).
        Phase 3 — Sequential RE for chunks with ≥ min_entities_for_re.
        Phase 4 — Cache write for newly extracted results.
        Phase 5 — Reassembly in original input order.

        Args:
            texts:     Chunk texts in processing order.
            chunk_ids: Corresponding chunk identifiers.

        Returns:
            ChunkExtractionResult list in the same order as the input.
        """
        start = time.time()
        results: List[Tuple[int, ChunkExtractionResult]] = []
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []
        uncached_ids: List[str] = []

        self.stats["total_chunks"] += len(texts)

        # Phase 1: Batch cache lookup
        cached_map = (
            self.cache.get_batch(texts, self.config.gliner_model)
            if self.cache else {}
        )
        for i, (text, chunk_id) in enumerate(zip(texts, chunk_ids)):
            if i in cached_map:
                self.stats["cache_hits"] += 1
                entities, relations = self._reconstruct_from_cache(
                    cached_map[i], chunk_id
                )
                results.append((i, ChunkExtractionResult(
                    chunk_id=chunk_id, text=text,
                    entities=entities, relations=relations,
                    extraction_time_ms=0.0,
                )))
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
                uncached_ids.append(chunk_id)

        # Phase 2 + 3: NER + RE for cache misses
        if uncached_texts:
            self.stats["ner_calls"] += len(uncached_texts)
            all_entities = self.ner_extractor.extract_batch(uncached_texts, uncached_ids)
            all_relations = self.re_extractor.extract_sequential(
                uncached_texts, all_entities, uncached_ids
            )
            self.stats["re_calls"] += sum(
                1 for ents in all_entities
                if len(ents) >= self.config.min_entities_for_re
            )

            for idx, text, chunk_id, entities, relations in zip(
                uncached_indices, uncached_texts, uncached_ids,
                all_entities, all_relations,
            ):
                for rel in relations:
                    rel.source_chunk_ids = [chunk_id]

                # Phase 4: Cache write
                if self.cache:
                    self.cache.put(
                        text,
                        {"entities": [e.to_dict() for e in entities],
                         "relations": [r.to_dict() for r in relations]},
                        self.config.gliner_model,
                    )

                self.stats["total_entities"] += len(entities)
                self.stats["total_relations"] += len(relations)
                results.append((idx, ChunkExtractionResult(
                    chunk_id=chunk_id, text=text,
                    entities=entities, relations=relations,
                    extraction_time_ms=0.0,
                )))

        # Phase 5: Restore original input order
        results.sort(key=lambda x: x[0])
        final_results = [r[1] for r in results]

        total_elapsed = (time.time() - start) * 1000
        uncached_count = len(uncached_texts)
        avg_per_chunk = total_elapsed / uncached_count if uncached_count else 0.0
        self.stats["last_batch_avg_ms"] = avg_per_chunk
        # Only stamp uncached results — cached hits truthfully cost 0 ms.
        uncached_idx_set = set(uncached_indices)
        for i, (orig_idx, result) in enumerate(results):
            if orig_idx in uncached_idx_set:
                result.extraction_time_ms = avg_per_chunk

        logger.info(
            "Batch extraction: %d chunks, %d entities, %d relations, "
            "%.0fms total (%.1fms/chunk)",
            len(texts),
            sum(len(r.entities) for r in final_results),
            sum(len(r.relations) for r in final_results),
            total_elapsed, avg_per_chunk,
        )
        return final_results

    def get_stats(self) -> Dict[str, Any]:
        """
        Return cumulative pipeline statistics.

        Note: last_batch_avg_ms reflects the most recent process_chunks_batch
        call, not a session-wide average.
        """
        total = self.stats["total_chunks"]
        ner = self.stats["ner_calls"]
        return {
            **self.stats,
            "cache_hit_rate": self.stats["cache_hits"] / total if total > 0 else 0.0,
            "re_skip_rate": (
                1 - (self.stats["re_calls"] / ner) if ner > 0 else 0.0
            ),
        }

    def close(self) -> None:
        """Release resources (SQLite connection)."""
        if self.cache:
            self.cache.close()

    def __enter__(self) -> "EntityExtractionPipeline":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# -- Lightweight SpaCy-only Pipeline -------------------------------------------

class SpacyEntityPipeline:
    """
    Lightweight NER pipeline using SpaCy only (no GLiNER/REBEL).

    Intended for memory-constrained deployments where the combined ~3.1 GB
    footprint of GLiNER + REBEL is infeasible on the target hardware.
    Uses SpaCy en_core_web_sm (12 MB, typically already installed).

    Interface: process_chunks_batch is compatible with EntityExtractionPipeline
    so the two classes are interchangeable from the pipeline caller's perspective.
    All returned ChunkExtractionResult objects have relations=[].

    Performance: 3–5 ms/chunk via nlp.pipe() (vs. 80–120 ms with GLiNER + REBEL).

    Reference: Honnibal, M. et al. (2020). "spaCy: Industrial-strength Natural
    Language Processing in Python." https://spacy.io

    SpaCy label → internal type mapping:
      PERSON      → PERSON          ORG         → ORGANIZATION
      GPE         → LOCATION        LOC         → LOCATION
      DATE        → DATE            EVENT       → EVENT
      WORK_OF_ART → WORK_OF_ART     FAC         → LOCATION
      (CARDINAL, ORDINAL, MONEY, PERCENT, QUANTITY are omitted)
    """

    _LABEL_MAP: Dict[str, str] = SPACY_LABEL_MAP_FLAT

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        batch_size: int = 64,
        confidence: float = _CONF_SPACY_LIGHT,
    ) -> None:
        """
        Args:
            spacy_model: SpaCy model name (must be installed).
            batch_size:  Chunks per nlp.pipe() call.
            confidence:  Uniform confidence assigned to every entity (spaCy
                emits no per-span probability). Defaults to the settings.yaml
                entity_extraction.fallback_confidence.spacy_light sentinel.
        """
        import spacy as _spacy
        self.nlp = _spacy.load(spacy_model)
        self.batch_size = batch_size
        self.confidence = confidence
        logger.info("SpacyEntityPipeline initialized: model=%s", spacy_model)

    def process_chunks_batch(
        self,
        texts: List[str],
        chunk_ids: List[str],
    ) -> List[ChunkExtractionResult]:
        """
        Extract named entities from a batch of chunks via nlp.pipe().

        Args:
            texts:     Chunk texts.
            chunk_ids: Corresponding chunk identifiers.

        Returns:
            List[ChunkExtractionResult] with relations=[] for all results.
        """
        results: List[ChunkExtractionResult] = []
        for doc, chunk_id, text in zip(
            self.nlp.pipe(texts, batch_size=self.batch_size),
            chunk_ids,
            texts,
        ):
            t0 = time.time()
            entities = self._extract_from_doc(doc, chunk_id)
            results.append(ChunkExtractionResult(
                chunk_id=chunk_id,
                text=text,
                entities=entities,
                relations=[],   # SpaCy provides no relation extraction
                extraction_time_ms=(time.time() - t0) * 1000,
            ))
        return results

    def _extract_from_doc(
        self,
        doc: Any,   # spacy.tokens.Doc — not annotated to avoid hard import
        chunk_id: str,
    ) -> List[ExtractedEntity]:
        """
        Convert a SpaCy Doc to ExtractedEntity objects.

        Deduplicates by entity_id within the chunk: the same entity string
        appearing multiple times produces only one ExtractedEntity per chunk.
        """
        entities: List[ExtractedEntity] = []
        seen_ids: set[str] = set()

        for ent in doc.ents:
            ent_type = self._LABEL_MAP.get(ent.label_)
            if ent_type is None:
                continue   # Skip CARDINAL, ORDINAL, MONEY, etc.

            entity_id = _generate_entity_id(ent.text, ent_type)
            if entity_id in seen_ids:
                continue   # Deduplicate within chunk
            seen_ids.add(entity_id)

            entities.append(ExtractedEntity(
                entity_id=entity_id,
                name=ent.text,
                entity_type=ent_type,
                confidence=self.confidence,
                mention_span=(ent.start_char, ent.end_char),
                source_chunk_id=chunk_id,
            ))
        return entities


# -- Factory --------------------------------------------------------------------

def create_extraction_pipeline(
    cfg: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> EntityExtractionPipeline:
    """
    Factory for EntityExtractionPipeline.

    Reads all parameters from the settings.yaml configuration dictionary.
    The **kwargs allow individual parameter overrides, which is useful for
    test setups (e.g. cache_enabled=False).

    Parameter mapping from settings.yaml → entity_extraction:
      gliner.model_name              → gliner_model
      gliner.entity_types            → entity_types
      gliner.confidence_threshold    → ner_confidence_threshold
      gliner.batch_size              → ner_batch_size
      rebel.model_name               → rebel_model
      rebel.confidence_threshold     → re_confidence_threshold
      rebel.batch_size               → re_batch_size
      rebel.min_entities_for_re      → min_entities_for_re
      rebel.max_input_length         → rebel_max_input_length
      rebel.max_output_length        → rebel_max_output_length
      rebel.num_beams                → rebel_num_beams
      caching.enabled                → cache_enabled
      caching.cache_path             → cache_path
      caching.lru_cache_size         → lru_cache_size
      performance.device             → device

    Args:
        cfg:     Full settings dict as loaded from config/settings.yaml.
                 Pass None or {} to fall back to ExtractionConfig defaults.
        **kwargs: Override individual ExtractionConfig fields by name.

    Returns:
        Configured EntityExtractionPipeline instance.
    """
    cfg = cfg or {}
    ee = cfg.get("entity_extraction", {})
    gliner_cfg = ee.get("gliner", {})
    rebel_cfg = ee.get("rebel", {})
    cache_cfg = ee.get("caching", {})
    fb_cfg = ee.get("fallback_confidence", {})
    perf_cfg = cfg.get("performance", {})

    _d = ExtractionConfig()   # default values as authoritative reference
    config = ExtractionConfig(
        gliner_model=gliner_cfg.get("model_name", _d.gliner_model),
        entity_types=gliner_cfg.get("entity_types", _d.entity_types),
        ner_confidence_threshold=gliner_cfg.get(
            "confidence_threshold", _d.ner_confidence_threshold
        ),
        ner_batch_size=gliner_cfg.get("batch_size", _d.ner_batch_size),
        rebel_model=rebel_cfg.get("model_name", _d.rebel_model),
        re_confidence_threshold=rebel_cfg.get(
            "confidence_threshold", _d.re_confidence_threshold
        ),
        re_batch_size=rebel_cfg.get("batch_size", _d.re_batch_size),
        min_entities_for_re=rebel_cfg.get(
            "min_entities_for_re", _d.min_entities_for_re
        ),
        rebel_max_input_length=rebel_cfg.get(
            "max_input_length", _d.rebel_max_input_length
        ),
        rebel_max_output_length=rebel_cfg.get(
            "max_output_length", _d.rebel_max_output_length
        ),
        rebel_num_beams=rebel_cfg.get("num_beams", _d.rebel_num_beams),
        cache_enabled=cache_cfg.get("enabled", _d.cache_enabled),
        cache_path=cache_cfg.get("cache_path", _d.cache_path),
        lru_cache_size=cache_cfg.get("lru_cache_size", _d.lru_cache_size),
        conf_gliner_spacy_fallback=fb_cfg.get("gliner_spacy", _d.conf_gliner_spacy_fallback),
        conf_gliner_regex_fallback=fb_cfg.get("gliner_regex", _d.conf_gliner_regex_fallback),
        conf_spacy_light=fb_cfg.get("spacy_light", _d.conf_spacy_light),
        conf_cache_default=fb_cfg.get("cache_default", _d.conf_cache_default),
        device=perf_cfg.get("device", _d.device),
    )

    # Apply kwarg overrides (useful for test setups).
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning("create_extraction_pipeline: unknown kwarg '%s' ignored", key)

    return EntityExtractionPipeline(config)


# -- Smoke Demo + Test Runner ----------------------------------------------------

def _main() -> None:
    """Smoke demo + test runner invoked when the module is run directly."""
    import sys
    import subprocess
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # -- 1. Smoke Demo --------------------------------------------------------
    pipeline = create_extraction_pipeline(cache_enabled=False)
    demo_texts = [
        "Albert Einstein was born in Ulm, Germany in 1879.",
        "Microsoft was founded by Bill Gates and Paul Allen in Albuquerque.",
        "The Eiffel Tower is located in Paris, France.",
    ]
    logger.info("-- Entity Extraction Smoke Demo --")
    for idx, demo_text in enumerate(demo_texts):
        demo_result = pipeline.process_chunk(demo_text, f"chunk_{idx}")
        logger.info(
            "chunk_%d: %d entities, %d relations, %.1fms",
            idx, demo_result.entity_count, demo_result.relation_count,
            demo_result.extraction_time_ms,
        )
        for ent in demo_result.entities:
            logger.info("  %-30s [%-12s] conf=%.2f", ent.name, ent.entity_type, ent.confidence)
    pipeline.close()

    # -- 2. Pytest Test Suite -------------------------------------------------
    # Runs the entity-related tests from test_data_layer.py so that
    # `python entity_extraction.py` serves as a self-contained verification.
    test_file = Path(__file__).parent / "test_data_layer.py"
    logger.info("-- Running pytest: %s --", test_file)
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "pytest", str(test_file),
         "-v", "-k", "entity"],
        check=False,
    )
    sys.exit(proc.returncode)


if __name__ == "__main__":
    _main()
