"""
Local import ingestion: Phase 3 of the decoupled ingestion pipeline.

Imports the chunks (Phase 1) and the Colab extraction artefact (Phase 2)
into the local stores: LanceDB (vector) + KuzuDB (knowledge graph).

Phases (numbering matches the checkpoint keys)
----------------------------------------------
  3a   Vector store ingestion (LanceDB, batched embeddings)
  3b   Knowledge graph base structure (DocumentChunk / SourceDocument /
       FROM_SOURCE / NEXT_CHUNK / Entity / MENTIONS / REBEL RELATED_TO /
       SVO RELATED_TO)
  3c   Co-occurrence edges (RELATED_TO {relation_type='cooccurs'})
  3c.5 Subsumptive co-occurrence cleanup (drop cooccurs covered by a
       semantic edge on the same pair)
  3d   Graph cleanup: orphans, hubs, duplicate-canonical merge
  3d.5 Embedding-based entity linking (alias resolution beyond
       canonical_form, e.g. short-form / long-form aliases)
  3e   Baseline metrics + invariant assertion (read-only)
  3f   Post-link isolated-entity drop (entities with zero RELATED_TO
       after linking)

The checkpoint at data/<dataset>/graph/.import_checkpoint.json records
per-phase completion + summary stats; --resume skips phases already
flagged done so a crash mid-run does not require restarting the (long)
import.

Inputs
------
- chunks_export.json                              (Phase 1)
- data/<dataset>/graph/extraction_results.json    (Phase 2 / Colab)
- config/settings.yaml                            (embedding model, storage
                                                  thresholds, batch sizes)

Outputs
-------
- data/<dataset>/vector/                          (LanceDB)
- data/<dataset>/graph/                           (KuzuDB + checkpoint)

Exports
-------
- load_config / load_chunks / load_extractions / chunks_to_documents
- ingest_vector_store(...)                        -- Phase 3a
- ingest_knowledge_graph(...)                     -- Phase 3b
- _canonical_entity_id(name, type)                -- SHA-256-bound id
- _is_plausible_concept(text)                     -- structural CONCEPT filter
- run_full_import(...)                            -- top-level orchestrator
- main()                                          -- CLI entry point

Dependencies / Requirements
---------------------------
- src.data_layer (HybridStore, KuzuGraphStore, StorageConfig,
                  BatchedOllamaEmbeddings, VectorStoreAdapter)
- src.data_layer.graph_quality (canonical_form, cleanup_graph,
                                 build_cooccurrence_edges,
                                 link_entities_by_embedding,
                                 drop_isolated_entities, ...)
- src.data_layer.svo_extraction (extract_svo_relations)
- ollama server reachable at config.embeddings.base_url (Phase 3a,
  when --embeddings-backend=ollama)
- sentence-transformers (optional, Phase 3a alternative backend)
- yaml, tqdm

Graph-quality post-processing (always applied unless --no-cleanup)
------------------------------------------------------------------
  1. Canonical entity IDs           -- deduplicate at MERGE time using
                                       canonical surface form (parentheticals,
                                       honorifics, suffixes, possessives).
  2. Co-occurrence edges            -- every pair of entities co-mentioned
                                       in the same chunk gets a RELATED_TO
                                       edge (relation_type='cooccurs').
  3. Cleanup pass                   -- drop orphan entities, drop hubs that
                                       exceed the mention-count threshold,
                                       merge duplicates sharing canonical form.
  4. Baseline + invariant assertion -- print metrics and warn on threshold
                                       violations; never aborts the import.

Usage (single line; -X utf8 required on Windows / PowerShell):
    python -X utf8 local_importingestion.py --chunks data/<ds>/chunks_export.json --extractions data/<ds>/graph/extraction_results.json --dataset <ds>
    python -X utf8 local_importingestion.py --chunks <p1> --extractions <p2> --dataset <ds> --graph-only --resume
    python -X utf8 local_importingestion.py --chunks <p1> --extractions <p2> --dataset <ds> --config config/settings.yaml --clear

See `python local_importingestion.py --help` for the full flag set.

Last reviewed: 2026-05-30 (audit pass, project version 5.4)
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

# Why: anchor data / cache / settings paths on the project root so the CLI
# works regardless of cwd, and put _PROJECT_ROOT on sys.path so the
# `from src.*` imports below resolve from any directory.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_DATA_ROOT = _PROJECT_ROOT / "data"
_CACHE_ROOT = _PROJECT_ROOT / "cache"
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"


# ---------------------------------------------------------------------------
# Module constants (algorithm-side hyperparameters)
# ---------------------------------------------------------------------------

# Why: nomic-embed-text emits 768-dim vectors. Hard-coded by the embedding
# model choice; surfaced as a constant so a model swap touches one line.
_NOMIC_EMBED_DIM = 768

# Why: HF / Ollama defaults used as emergency fallbacks if settings.yaml
# is missing. Production deployments should set these in config.
_HF_DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1"
_OLLAMA_DEFAULT_MODEL = "nomic-embed-text"
_OLLAMA_DEFAULT_URL = "http://localhost:11434"
_HF_DEFAULT_BATCH_SIZE = 64
_OLLAMA_DEFAULT_BATCH_SIZE = 32

# Why: vector-store ingestion batch size. 100 keeps one batch's embeddings
# (100 * 768 * 4B ~= 0.3 MiB) comfortably under RAM pressure on edge.
_VECTOR_INGEST_BATCH_SIZE = 100

# Why: KuzuDB bulk-write batch size. 500 trades one fsync per 500 statements
# against statement-batch memory; established empirically against the
# Windows + KuzuDB Phase 3b regression (45 min -> 5-8 min).
_KUZU_BULK_BATCH_SIZE = 500

# Why: NER confidence floor applied at entity-node insertion. Recall-optimised;
# below this threshold the GLiNER output is dominated by junk that the
# Phase-3d cleanup pass cannot economically absorb downstream.
_ENTITY_CONFIDENCE_THRESHOLD_DEFAULT = 0.5

# Why: confidence assigned to SVO dependency-parse triples (config fallback for
# settings.yaml entity_extraction.svo.confidence). SVO triples carry no
# per-triple model score, so a uniform sentinel is used, mirroring REBEL.
_SVO_CONFIDENCE_DEFAULT = 0.7

# Why: vector-store similarity threshold (config fallback). 0.3 is the
# nomic-embed-text-v1 production setting that admits paraphrase recall
# without flooding the LLM context with semantic near-duplicates.
_SIMILARITY_THRESHOLD_DEFAULT = 0.3

# Why: REBEL relation confidence fallback when the field is missing. 0.5
# mirrors the legacy decoder sentinel so older extraction artefacts still
# import; the constant-confidence detector below surfaces the case.
_REBEL_CONFIDENCE_FALLBACK = 0.5

# Why: minimum NER confidence for an entity to participate in
# co-occurrence edges (Phase 3c CLI default).
_COOCC_MIN_CONFIDENCE_DEFAULT = 0.5

# Why: hub-suppression cutoff (Phase 3d). Entities mentioned in more than
# ratio * total_chunks chunks are dropped as overly generic nodes that
# link unrelated chunks and reduce retrieval precision.
_HUB_THRESHOLD_RATIO_DEFAULT = 0.03

# Why: cosine similarity threshold for embedding-based entity linking
# (Phase 3d.5). 0.92 is in the strict end of the [0.85, 0.97] safe range
# -- aggressive merging without false-positive name collisions.
_LINKING_THRESHOLD_DEFAULT = 0.92

# Why: max entity-bucket size for embedding linking. 8000 entities ~=
# 256 MB float32 similarity matrix; buckets larger than this are skipped
# (PERSON / ORG on large datasets) so the linker does not OOM the runner.
_LINKING_MAX_TYPE_SIZE_DEFAULT = 8000

# Why: synthetic, lower-cased "concept" type for REBEL relation objects
# that are attribute values rather than named entities. Tagged distinctly
# from the GLiNER entity types so cleanup and retrieval can treat them
# differently if needed.
_CONCEPT_ENTITY_TYPE = "CONCEPT"

# Why: rule width for printed report sections.
_BAR_WIDTH = 70

# Why: minimum entity-name length used by `_is_plausible_concept` to reject
# single-character / digit-only relation objects that would otherwise spawn
# spurious CONCEPT hubs.
_MIN_CONCEPT_LEN = 3

# Why: checkpoint keys.
_CP_DONE = "done"
_CP_TS = "ts"

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Quiet sub-modules
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# ============================================================================
# IMPORTS WITH FALLBACKS
# ============================================================================

try:
    from langchain.schema import Document
except ImportError:
    @dataclass
    class Document:
        page_content: str
        metadata: dict

try:
    from sentence_transformers import SentenceTransformer as _ST
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

try:
    from src.data_layer import (
        BatchedOllamaEmbeddings,
        HybridStore,
        KuzuGraphStore,
        StorageConfig,
        VectorStoreAdapter,
    )
    from src.data_layer.graph_quality import (
        assert_graph_invariants,
        build_cooccurrence_edges,
        canonical_form,
        cleanup_graph,
        compute_graph_baseline,
        drop_isolated_entities,
        drop_subsumed_cooccurrence_edges,
        format_baseline_report,
        link_entities_by_embedding,
    )
    from src.data_layer.svo_extraction import (
        extract_svo_relations,
        is_available as svo_available,
    )
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    logger.error(
        "Storage modules not found! "
        "Make sure you are in the project root and "
        "src/data_layer/storage.py exists."
    )


# ============================================================================
# PHASE CHECKPOINTING
# ============================================================================
# After each phase completes successfully, we write a small JSON file to disk.
# On re-run with --resume, phases that already have "done": true are skipped.
# This means a crash (power cut, Ctrl-C, etc.) after Phase 3b doesn't require
# re-running the full 45-minute entity import — you just resume from 3c.
#
# Checkpoint file location:  data/<dataset>/graph/.import_checkpoint.json
# name_to_id cache:          data/<dataset>/graph/.name_to_id.json
#
# The checkpoint is deleted by --clear.

def _checkpoint_path(graph_path: Path) -> Path:
    return graph_path / ".import_checkpoint.json"

def _name_to_id_path(graph_path: Path) -> Path:
    return graph_path / ".name_to_id.json"

def _load_checkpoint(graph_path: Path) -> Dict[str, Any]:
    """Load existing checkpoint or return empty dict."""
    p = _checkpoint_path(graph_path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_checkpoint(graph_path: Path, phase: str, data: Dict[str, Any]) -> None:
    """Mark a phase as done and persist phase-level stats to the checkpoint."""
    cp = _load_checkpoint(graph_path)
    cp[phase] = {_CP_DONE: True, _CP_TS: time.time(), **data}
    _checkpoint_path(graph_path).write_text(
        json.dumps(cp, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("  Checkpoint saved: phase=%s", phase)


def _save_partial_checkpoint(
    graph_path: Path, phase: str, data: Dict[str, Any],
) -> None:
    """Persist intra-phase progress WITHOUT marking the phase as done.

    Used by per-bucket checkpointing inside Phase 3d.5 — the phase only
    flips to `_CP_DONE=True` when the linker returns successfully (see
    `_save_checkpoint`). If the process dies in the middle, the partial
    entry survives so the next `--resume` run can skip already-finished
    buckets.
    """
    cp = _load_checkpoint(graph_path)
    cp[phase] = {_CP_DONE: False, _CP_TS: time.time(), **data}
    _checkpoint_path(graph_path).write_text(
        json.dumps(cp, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _phase_done(checkpoint: Dict[str, Any], phase: str) -> bool:
    return checkpoint.get(phase, {}).get(_CP_DONE, False)

def _phase_eta(label: str, elapsed_s: float) -> str:
    """Format a phase timing line for the console."""
    m, s = divmod(int(elapsed_s), 60)
    return f"  [{label}]  elapsed: {m}m {s:02d}s"


# ============================================================================
# CANONICAL ENTITY IDENTIFIERS
# ============================================================================

def _canonical_entity_id(name: str, entity_type: str) -> str:
    """
    Deterministic entity identifier based on the canonical surface form.

    Differs from `entity_extraction._generate_entity_id` only in the
    normalisation applied to the name: this version uses
    `graph_quality.canonical_form`, which strips parentheticals,
    honorifics, name suffixes, possessives, and applies NFKC + casefold.

    Two surface forms that collapse under `canonical_form` (e.g.
    "<PERSON>", "<PERSON> (occupation)", "<PERSON>  ") produce the same
    entity_id, so the KuzuDB MERGE on entity_id deduplicates them at
    insert time.
    """
    canon = canonical_form(name)
    combined = f"{canon}:{entity_type or 'UNKNOWN'}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:24]


# Why:    structural filter for "is this REBEL relation object plausibly a
#         standalone CONCEPT node?".
# What:   reject surface forms shorter than _MIN_CONCEPT_LEN OR with no
#         alphabetic character (digits-only, bare punctuation, empty string).
# Misses: surface forms that pass the structural test but still describe a
#         single sense rather than a named-entity-like concept ("dog", "old")
#         -- those produce singleton CONCEPT nodes that the cleanup pass
#         absorbs as orphans.
def _is_plausible_concept(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < _MIN_CONCEPT_LEN:
        return False
    if not any(ch.isalpha() for ch in t):
        return False
    return True

class _HFEmbeddings:
    """Thin wrapper so SentenceTransformer has the same interface as BatchedOllamaEmbeddings."""

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1", batch_size: int = 64):
        if not _HF_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        logger.info("  Loading HuggingFace model: %s", model_name)
        self._model = _ST(model_name, trust_remote_code=True)
        self._batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        vec = self._model.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec[0].tolist()


try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, **kwargs):
            self._iterable = iterable
        def __iter__(self):
            return iter(self._iterable) if self._iterable else iter([])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            pass
        def set_postfix(self, **kwargs):
            pass


# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config(config_path: Optional[Path] = None) -> Dict:
    """Load config from YAML or fall back to in-code defaults."""
    if config_path and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    if _DEFAULT_CONFIG_PATH.exists():
        with open(_DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    logger.warning("No config found -- using emergency defaults")
    return {
        "embeddings": {
            "model_name": _OLLAMA_DEFAULT_MODEL,
            "base_url": _OLLAMA_DEFAULT_URL,
            "embedding_dim": _NOMIC_EMBED_DIM,
        },
        "vector_store": {
            "similarity_threshold": _SIMILARITY_THRESHOLD_DEFAULT,
            "distance_metric": "cosine",
            "normalize_embeddings": True,
        },
        "performance": {
            "batch_size": _OLLAMA_DEFAULT_BATCH_SIZE,
            "device": "cpu",
        },
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_chunks(chunks_path: Path) -> List[Dict]:
    """Read chunks_export.json (Phase 1 output)."""
    logger.info("Loading chunks: %s", chunks_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info("  %d chunks loaded", len(chunks))
    return chunks


def load_extractions(extractions_path: Path) -> Dict:
    """Read extraction_results.json (Phase 2 / Colab output)."""
    logger.info("Loading extraction results: %s", extractions_path)
    with open(extractions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})

    logger.info("  %s chunks processed", meta.get("total_chunks", "?"))
    logger.info("  %s entities extracted", meta.get("total_entities", "?"))
    logger.info("  %s unique entities", meta.get("unique_entities", "?"))
    logger.info("  %s relations extracted", meta.get("total_relations", "?"))
    logger.info("  Device: %s", meta.get("device", "?"))
    logger.info(
        "  NER: %ss, RE: %ss",
        meta.get("ner_time_seconds", "?"), meta.get("re_time_seconds", "?"),
    )

    return data


def chunks_to_documents(chunks: List[Dict]) -> List[Document]:
    """Convert the chunks_export.json schema into LangChain `Document`s."""
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"],
        )
        documents.append(doc)
    return documents


# ============================================================================
# PHASE 3a: VECTOR STORE INGESTION (Embeddings via Ollama)
# ============================================================================

def ingest_vector_store(
    documents: List[Document],
    vector_path: Path,
    config: Dict,
    dataset_name: str,
    embeddings=None,
) -> None:
    """
    Ingest chunks into the LanceDB vector store.

    Uses BatchedOllamaEmbeddings by default; pass a pre-built embeddings
    object (e.g. _HFEmbeddings) to switch backends.
    """
    if not STORAGE_AVAILABLE:
        logger.error("Storage modules not available!")
        return

    bar = "-" * _BAR_WIDTH
    logger.info("\n%s", bar)
    logger.info("PHASE 3a: VECTOR STORE INGESTION (%d chunks)", len(documents))
    logger.info("%s", bar)

    embedding_config = config.get("embeddings", {})
    if embeddings is None:
        perf_config = config.get("performance", {})
        cache_path = _CACHE_ROOT / f"{dataset_name}_embeddings.db"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings = BatchedOllamaEmbeddings(
            model_name=embedding_config.get("model_name", _OLLAMA_DEFAULT_MODEL),
            base_url=embedding_config.get("base_url", _OLLAMA_DEFAULT_URL),
            batch_size=perf_config.get("batch_size", _OLLAMA_DEFAULT_BATCH_SIZE),
            cache_path=cache_path,
            device=perf_config.get("device", "cpu"),
        )

    # Why: build StorageConfig WITHOUT entity extraction -- the Phase-2
    # Colab notebook produced the entities; we are loading them, not
    # re-running NER. graph_db_path is required by the dataclass but not
    # used by this phase.
    vector_config = config.get("vector_store", {})
    storage_config = StorageConfig(
        vector_db_path=vector_path,
        graph_db_path=vector_path.parent / "graph",
        embedding_dim=embedding_config.get("embedding_dim", _NOMIC_EMBED_DIM),
        similarity_threshold=vector_config.get(
            "similarity_threshold", _SIMILARITY_THRESHOLD_DEFAULT,
        ),
        normalize_embeddings=vector_config.get("normalize_embeddings", True),
        distance_metric=vector_config.get("distance_metric", "cosine"),
        enable_entity_extraction=False,
    )

    vector_store = VectorStoreAdapter(
        db_path=str(vector_path),
        embedding_dim=storage_config.embedding_dim,
        distance_metric=storage_config.distance_metric,
    )

    # Batched ingest
    start_time = time.time()
    for i in tqdm(
        range(0, len(documents), _VECTOR_INGEST_BATCH_SIZE),
        desc="Vector Store",
        unit="batch",
    ):
        batch = documents[i : i + _VECTOR_INGEST_BATCH_SIZE]
        vector_store.add_documents_with_embeddings(batch, embeddings)

    elapsed = time.time() - start_time
    logger.info("  OK vector store: %d chunks in %.1fs", len(documents), elapsed)
    logger.info("    path: %s", vector_path)


# ============================================================================
# PHASE 3b: KNOWLEDGE GRAPH INGESTION (KuzuDB)
# ============================================================================

def ingest_knowledge_graph(
    documents: List[Document],
    extraction_results: List[Dict],
    graph_path: Path,
    dataset_name: str,
    entity_confidence_threshold: float = _ENTITY_CONFIDENCE_THRESHOLD_DEFAULT,
    capture_attribute_objects: bool = False,
    svo_confidence: float = _SVO_CONFIDENCE_DEFAULT,
) -> Tuple[Optional["KuzuGraphStore"], Dict[str, Any]]:
    """Import entities and relations into the KuzuDB knowledge graph.

    Steps:
        1. DocumentChunk nodes
        2. SourceDocument nodes
        3. FROM_SOURCE + NEXT_CHUNK edges
        4. Entity nodes from the extraction artefact
        5. MENTIONS edges (chunk -> entity)
        6. RELATED_TO edges (entity -> entity) -- second pass, after the
           full `name_to_id` map is built, so a relation in chunk N can be
           resolved against an entity first seen in chunk N+k.

    Args:
        capture_attribute_objects: When True, a REBEL relation whose object
            is a free-text attribute value rather than a named entity (e.g.
            "<RELATION_TYPE> -> <free_text_value>") still produces a graph
            edge -- the object is materialised as a synthetic CONCEPT-typed
            Entity node. Default False: such relations are dropped, keeping
            the graph within the OntoNotes-5 entity-type taxonomy (PERSON,
            GPE, ORGANIZATION, LOCATION, DATE, EVENT, WORK_OF_ART, PRODUCT)
            used by GLiNER. Enable with --attribute-objects only after
            considering the impact on co-occurrence edge quality and hub
            detection.
    """
    if not STORAGE_AVAILABLE:
        logger.error("Storage modules not available!")
        return None, {}

    bar = "-" * _BAR_WIDTH
    logger.info("\n%s", bar)
    logger.info("PHASE 3b: KNOWLEDGE GRAPH INGESTION")
    logger.info("%s", bar)

    # Initialise the KuzuDB graph store.
    graph_store = KuzuGraphStore(str(graph_path))

    stats = {
        "document_chunks": 0,
        "source_documents": 0,
        "from_source": 0,
        "next_chunk": 0,
        "entities": 0,
        "unique_entities": 0,
        "mentions": 0,
        "relations": 0,
        "rebel_duplicates_skipped": 0,
        "rebel_unresolved_dropped": 0,
        "attribute_relations": 0,
        "concept_nodes": 0,
        "rebel_confidence_is_constant": False,
    }

    # Index: chunk_id → extraction result
    extraction_by_chunk = {}
    for result in extraction_results:
        extraction_by_chunk[str(result["chunk_id"])] = result

    # ── Step 1-3: Document Structure ─────────────────────────────────────

    logger.info("  Steps 1-3: building document structure...")
    seen_sources = set()
    prev_chunk_id = None

    def _to_int(value, default: int = 0) -> int:
        """Coerce a metadata value (int OR str, since Phase 1 stringifies)."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    for doc in tqdm(documents, desc="Graph Nodes", unit="doc"):
        chunk_id = str(doc.metadata.get("chunk_id", "unknown"))
        source_file = doc.metadata.get("source_file", "unknown")

        # Phase 1 writes the chunk's order-within-document under "position";
        # the storage schema calls the same field "chunk_index". Map either
        # name to the schema column. All Phase-1 metadata values are strings
        # (see chunks_export serialiser), so coerce.
        chunk_index = _to_int(
            doc.metadata.get("chunk_index", doc.metadata.get("position", 0))
        )
        page_number = _to_int(doc.metadata.get("page_number", 0))

        # DocumentChunk Node — pass the FULL text; the storage layer's
        # max_text_chars=500 default applies the truncation.
        try:
            graph_store.add_document_chunk(
                chunk_id=chunk_id,
                text=doc.page_content,
                page_number=page_number,
                chunk_index=chunk_index,
                source_file=source_file,
            )
            stats["document_chunks"] += 1
        except Exception as e:  # noqa: BLE001 -- best-effort: skip on duplicate / KuzuDB error
            logger.debug("    Chunk %s: %s", chunk_id, e)

        # SourceDocument node (one per source).
        if source_file not in seen_sources:
            try:
                graph_store.add_source_document(
                    doc_id=source_file,
                    filename=source_file,
                    total_pages=int(doc.metadata.get("total_pages", 0)),
                )
                seen_sources.add(source_file)
                stats["source_documents"] += 1
            except Exception as e:  # noqa: BLE001 -- best-effort: existing node is harmless
                logger.debug("    Source %s: %s", source_file, e)

        # FROM_SOURCE edge.
        try:
            graph_store.add_from_source_relation(chunk_id, source_file)
            stats["from_source"] += 1
        except Exception as e:  # noqa: BLE001 -- best-effort: duplicate edge is harmless
            logger.debug("    FROM_SOURCE: %s", e)

        # NEXT_CHUNK edge (only within the same source document).
        if prev_chunk_id is not None:
            prev_doc = documents[stats["document_chunks"] - 2] if stats["document_chunks"] > 1 else None
            if prev_doc and prev_doc.metadata.get("source_file") == source_file:
                try:
                    graph_store.add_next_chunk_relation(prev_chunk_id, chunk_id)
                    stats["next_chunk"] += 1
                except Exception:  # noqa: BLE001 -- best-effort: NEXT_CHUNK is decorative
                    pass

        prev_chunk_id = chunk_id

    logger.info(
        "    OK %d chunks, %d sources",
        stats["document_chunks"], stats["source_documents"],
    )

    # ── Step 4-6: Entities & Relations ───────────────────────────────────

    logger.info("  Steps 4-6: importing entities and relations...")

    seen_entities: set = set()
    # Map: canonical_form(name) -> canonical entity_id (used for relation
    # resolution AND co-occurrence edge construction). One key per canonical
    # surface form; collisions are intentional and produce the deduplication.
    name_to_id: Dict[str, str] = {}

    # Build a lookup: chunk_id -> raw text (used for SVO extraction below).
    chunk_id_to_text: Dict[str, str] = {
        str(doc.metadata.get("chunk_id", f"chunk_{i}")): doc.page_content
        for i, doc in enumerate(documents)
    }

    # -- Pass A: Entity nodes + MENTIONS edges (BULK PATH) -------------------
    # Build the COMPLETE name_to_id map before touching relations. Relations
    # are resolved in Pass B, so a relation in chunk N can reference an
    # entity first extracted in chunk N+k (forward reference) -- previously
    # those were silently dropped because the map was only partially
    # populated.
    #
    # The original implementation issued one Cypher statement per entity
    # and one per MENTIONS edge inside a `batch_begin/batch_commit`
    # envelope. The transactions reduced fsyncs but did NOT eliminate
    # Phase 3b's hot path: MERGE on a MENTIONS edge performs an adjacency-
    # list existence scan on the target Entity. For popular entities
    # (frequent named-entity / common-noun heads) the adjacency list grows
    # to thousands of edges, and every subsequent MENTIONS insert against
    # them scales with the existing degree, producing a quadratic
    # ingestion time.
    #
    # The fix: collect entities and mention pairs in Python first,
    # deduplicate by primary key (entity_id for nodes, (chunk_id,
    # entity_id) for edges), then issue:
    #   - `add_entities_bulk(...)`             -- MERGE on Entity
    #     (primary-key index, O(log N))
    #   - `add_mentions_relations_bulk(...)`   -- CREATE on MENTIONS
    #     (no scan)
    # CREATE is safe ONLY because the pair list is deduplicated in Python;
    # CREATE without dedup would write multiple identical edges.
    pending_entities: List[Tuple[str, str, str, float]] = []
    pending_mentions: List[Tuple[str, str]] = []
    seen_mention_pairs: Set[Tuple[str, str]] = set()

    for result in tqdm(extraction_results, desc="Collecting entities & mentions",
                        unit="chunk"):
        chunk_id = str(result["chunk_id"])

        for ent in result.get("entities", []):
            entity_name = (ent.get("name") or "").strip()
            entity_type = ent.get("entity_type") or ent.get("type", "UNKNOWN")
            confidence  = ent.get("confidence", 0.5)

            # Skip low-confidence entities (recall-vs-noise trade-off).
            if confidence < entity_confidence_threshold or not entity_name:
                continue

            # Canonical entity_id: identical canonical surface forms produce
            # the same id, so the bulk MERGE deduplicates at insert time.
            entity_id = _canonical_entity_id(entity_name, entity_type)
            canon_key = canonical_form(entity_name)
            name_to_id[canon_key] = entity_id

            if entity_id not in seen_entities:
                seen_entities.add(entity_id)
                pending_entities.append(
                    (entity_id, entity_name, entity_type, float(confidence))
                )
                stats["unique_entities"] += 1
            stats["entities"] += 1

            # Deduplicate MENTIONS at (chunk_id, entity_id) granularity:
            # GLiNER may emit the same entity twice within a chunk; we want
            # ONE MENTIONS edge per pair, not N. (CREATE is non-idempotent.)
            pair = (chunk_id, entity_id)
            if pair not in seen_mention_pairs:
                seen_mention_pairs.add(pair)
                pending_mentions.append(pair)
                stats["mentions"] += 1

    # Bulk-insert nodes first (MENTIONS edges MATCH against existing nodes).
    logger.info("  Bulk-inserting %d Entity nodes...", len(pending_entities))
    graph_store.add_entities_bulk(pending_entities, batch_size=_KUZU_BULK_BATCH_SIZE)

    logger.info("  Bulk-inserting %d MENTIONS edges...", len(pending_mentions))
    graph_store.add_mentions_relations_bulk(
        pending_mentions, batch_size=_KUZU_BULK_BATCH_SIZE,
    )

    # ── Pass B: RELATED_TO edges (REBEL relations) ───────────────────────────
    # Resolution is strict on canonical_form (no lossy substring fallback).
    # Two-pass means `name_to_id` is now complete, so both endpoints can be a
    # named entity from ANY chunk. A relation whose object resolves to no named
    # entity is treated as a free-text attribute value and (when
    # capture_attribute_objects=True) materialised as a CONCEPT node, so the
    # genre/sport/occupation knowledge is not lost. Within-run triple
    # deduplication keeps the relation count honest (REBEL beam search emits
    # the same triple multiple times — KuzuDB MERGE collapses them, but the
    # counter must not double-count).
    seen_triples: Set[Tuple[str, str, str]] = set()
    _rel_confs: Set[float] = set()
    _concept_ids: Dict[str, str] = {}  # canonical_form(value) -> concept entity_id
    _concept_mentions: Set[Tuple[str, str]] = set()  # (chunk_id, concept_id) already linked

    def _ensure_concept_node(value: str, src_chunk: str) -> Optional[str]:
        """MERGE a CONCEPT entity for a free-text relation object; link it from src_chunk."""
        if not _is_plausible_concept(value):
            return None
        key = canonical_form(value)
        cid = _concept_ids.get(key)
        if cid is None:
            cid = _canonical_entity_id(value, _CONCEPT_ENTITY_TYPE)
            _concept_ids[key] = cid
            try:
                graph_store.add_entity(
                    entity_id=cid,
                    name=value,
                    entity_type=_CONCEPT_ENTITY_TYPE,
                    confidence=_REBEL_CONFIDENCE_FALLBACK,
                )
                stats["concept_nodes"] += 1
            except Exception as e:  # noqa: BLE001 -- best-effort: skip on KuzuDB error
                logger.debug("    CONCEPT node %r: %s", value, e)
                return None
        # Make the concept reachable: MENTIONS edge from the originating
        # chunk (once per chunk/concept pair -- KuzuDB MERGE is idempotent
        # but the counter must not double-count repeated references in the
        # same chunk).
        mkey = (src_chunk, cid)
        if mkey not in _concept_mentions:
            _concept_mentions.add(mkey)
            try:
                graph_store.add_mentions_relation(chunk_id=src_chunk, entity_id=cid)
                stats["mentions"] += 1
            except Exception:  # noqa: BLE001 -- best-effort: MENTIONS may already exist
                pass
        # Keep it resolvable by later relations referencing the same value.
        name_to_id.setdefault(key, cid)
        return cid

    # ── Pass B: RELATED_TO edges via the bulk-CREATE path ───────────────────
    # Same fix as Pass A — collect, deduplicate, then issue the writes via
    # add_related_to_relations_bulk(use_create=True), which the co-occurrence
    # phase has used successfully for >170k edges in <20 min.
    pending_relations: List[Tuple[str, str, str, float, str]] = []
    pending_attribute_count = 0

    for result in tqdm(extraction_results, desc="Collecting RELATED_TO (REBEL)",
                        unit="chunk"):
        chunk_id = str(result["chunk_id"])
        for rel in result.get("relations", []):
            subject  = (rel.get("subject_entity") or rel.get("subject") or "").strip()
            obj      = (rel.get("object_entity")  or rel.get("object")  or "").strip()
            rel_type = rel.get("relation_type")   or rel.get("relation") or "related_to"
            rel_conf = float(rel.get("confidence", _REBEL_CONFIDENCE_FALLBACK))
            _rel_confs.add(rel_conf)
            rel_sources = (rel.get("source_chunk_ids")
                           or rel.get("source_chunks")
                           or [chunk_id])
            if isinstance(rel_sources, str):
                rel_sources = [rel_sources]
            if not subject or not obj:
                continue

            subject_id = name_to_id.get(canonical_form(subject))
            if not subject_id:
                # A relation whose SUBJECT is not a named entity is almost
                # always a parse artefact ("biographical -> ..."); drop it.
                stats["rebel_unresolved_dropped"] += 1
                continue

            object_id = name_to_id.get(canonical_form(obj))
            is_attribute = False
            if not object_id:
                if capture_attribute_objects:
                    object_id = _ensure_concept_node(obj, chunk_id)
                    is_attribute = object_id is not None
                if not object_id:
                    stats["rebel_unresolved_dropped"] += 1
                    continue

            if subject_id == object_id:
                continue

            triple = (subject_id, rel_type, object_id)
            if triple in seen_triples:
                stats["rebel_duplicates_skipped"] += 1
                continue
            seen_triples.add(triple)

            pending_relations.append((
                subject_id, object_id, rel_type, rel_conf,
                ",".join(str(c) for c in rel_sources),
            ))
            stats["relations"] += 1
            if is_attribute:
                pending_attribute_count += 1

    stats["attribute_relations"] = pending_attribute_count

    logger.info("  Bulk-inserting %d RELATED_TO (REBEL) edges...",
                len(pending_relations))
    graph_store.add_related_to_relations_bulk(
        pending_relations, batch_size=_KUZU_BULK_BATCH_SIZE, use_create=True,
    )

    # When the Phase-2 extraction artefact predates the calibrated log-prob
    # confidence (see colab_extraction.py / Critical Fix 4), REBEL emits a
    # single sentinel value for every triple. Detect that case so downstream
    # consumers do not mistake the value for a real ranking signal.
    stats["rebel_confidence_is_constant"] = len(_rel_confs) <= 1
    if stats["rebel_confidence_is_constant"] and _rel_confs:
        logger.warning(
            "    REBEL relation confidence is CONSTANT (%.2f for all %d edges) "
            "-- do not filter graph relations on this value; it carries no signal.",
            next(iter(_rel_confs)), stats["relations"],
        )

    logger.info(
        "    OK %d REBEL relations  (+%d attribute/CONCEPT, %d dup skipped, "
        "%d unresolved dropped, %d concept nodes)",
        stats["relations"], stats["attribute_relations"],
        stats["rebel_duplicates_skipped"], stats["rebel_unresolved_dropped"],
        stats["concept_nodes"],
    )
    logger.info(
        "    OK %d entities, %d mentions",
        stats["unique_entities"], stats["mentions"],
    )

    # ── SVO extraction: narrative relations from the SpaCy dependency parse ──
    # Adds RELATED_TO edges with relation_type=verb_lemma and the configured
    # SVO confidence (settings.yaml entity_extraction.svo.confidence).
    # Both subject and object must resolve to a known entity in name_to_id;
    # unmatched triples are dropped silently.
    stats["svo_relations"] = 0
    if svo_available():
        svo_added = 0
        for result in tqdm(extraction_results, desc="SVO relations", unit="chunk"):
            chunk_id = str(result.get("chunk_id", ""))
            text = chunk_id_to_text.get(chunk_id, "")
            if not text:
                continue
            triples = extract_svo_relations(
                text=text,
                name_to_id=name_to_id,
                canonical_form_fn=canonical_form,
                confidence=svo_confidence,
            )
            for subj_id, verb_lemma, obj_id, conf in triples:
                try:
                    graph_store.add_related_to_relation(
                        entity1_id=subj_id,
                        entity2_id=obj_id,
                        relation_type=verb_lemma,
                        confidence=conf,
                        source_chunks=[chunk_id],
                    )
                    svo_added += 1
                except Exception as exc:  # noqa: BLE001 -- best-effort: skip bad triple
                    logger.debug("SVO RELATED_TO failed: %s", exc)
        stats["svo_relations"] = svo_added
        logger.info("    OK %d SVO narrative relations", svo_added)
    else:
        logger.info("    SVO extraction unavailable (spaCy missing); skipped")

    # Expose the canonical-id map to the caller so co-occurrence edges and
    # cleanup can use the same lookup convention.
    stats["_name_to_id"] = name_to_id
    return graph_store, stats


# ============================================================================
# FULL IMPORT PIPELINE
# ============================================================================

def run_full_import(
    chunks_path: Path,
    extractions_path: Path,
    dataset_name: str,
    config: Dict,
    graph_only: bool = False,
    clear: bool = False,
    skip_cleanup: bool = False,
    skip_cooccurrence: bool = False,
    skip_subsumptive_cleanup: bool = False,
    skip_isolated_drop: bool = False,
    cleanup_dry_run: bool = False,
    cooccurrence_min_confidence: float = _COOCC_MIN_CONFIDENCE_DEFAULT,
    hub_threshold_ratio: float = _HUB_THRESHOLD_RATIO_DEFAULT,
    enable_entity_linking: bool = False,
    linking_threshold: float = _LINKING_THRESHOLD_DEFAULT,
    linking_max_type_size: int = _LINKING_MAX_TYPE_SIZE_DEFAULT,
    resume: bool = False,
    embeddings_backend: str = "ollama",
    capture_attribute_objects: bool = False,
) -> None:
    """Run the full Phase-3 import (vector store + knowledge graph + cleanup).

    Main entry point for Phase 3 of the three-phase ingestion pipeline.
    Phases 1 and 2 ran earlier:

        Phase 1 (benchmark_datasets.py)
            Splits raw source text into overlapping chunks and writes
            chunks_export.json. CPU, fast (~1 min).

        Phase 2 (Colab notebook)
            Runs GLiNER (NER) and REBEL (relation extraction) on every
            chunk using a GPU. Writes extraction_results.json (~2 h on
            Colab T4).

        Phase 3 (this file) -- local machine
            Imports the GPU outputs into the local databases:

            3a   Vector store    -> LanceDB (ANN index for dense
                                    retrieval). Skipped with --graph-only.
            3b   Knowledge graph -> KuzuDB. Sub-steps:
                                    1. DocumentChunk nodes
                                    2. SourceDocument nodes
                                    3. FROM_SOURCE + NEXT_CHUNK edges
                                    4. Entity nodes from extraction artefact
                                    5. MENTIONS edges
                                    6. RELATED_TO edges from REBEL relations
                                    7. SVO narrative edges
            3c   Co-occurrence   -- every pair of entities co-mentioned in
                                    the same chunk gets a RELATED_TO
                                    {cooccurs} edge.
            3c.5 Subsumptive co-occurrence cleanup.
            3d   Graph cleanup   -- orphans, hubs, duplicate-canonical merge.
            3d.5 Entity linking  -- embed entity names via the configured
                                    embedder and merge near-duplicates within
                                    each type bucket (cosine >=
                                    _LINKING_THRESHOLD_DEFAULT). Handles
                                    short-form / long-form aliases.
            3f   Post-link isolated-entity drop.
            3e   Baseline metrics -- print graph health, warn on
                                    threshold violations.

    Checkpointing (--resume)
    ------------------------
    After each phase completes, a checkpoint is saved to
    `data/<dataset>/graph/.import_checkpoint.json`. If a phase crashes
    (Ctrl-C, power cut, KuzuDB lock error) restart with --resume and the
    completed phases are skipped. The name_to_id mapping needed by Phase 3c
    is persisted to `data/<dataset>/graph/.name_to_id.json`.

    --clear deletes the checkpoint, forcing a full fresh import. --resume
    and --clear are mutually exclusive.


    Args:
        chunks_path:                  Path to chunks_export.json (Phase 1).
        extractions_path:             Path to extraction_results.json (Phase 2).
        dataset_name:                 Dataset name (e.g. "hotpotqa").
        config:                       Settings dict (from settings.yaml or defaults).
        graph_only:                   Skip the vector-store ingestion (Phase 3a).
        clear:                        Delete existing stores before import.
        skip_cleanup:                 Disable Phase 3d cleanup pass.
        skip_cooccurrence:            Disable Phase 3c co-occurrence edges.
        skip_subsumptive_cleanup:     Disable Phase 3c.5 (delete cooccurs
                                      edges where a semantic edge already
                                      covers the same pair).
        skip_isolated_drop:           Disable Phase 3f (drop entities with
                                      zero RELATED_TO edges after linking).
        cleanup_dry_run:              Cleanup pass counts only, no DB writes.
        cooccurrence_min_confidence:  Min NER confidence for co-occurrence.
        hub_threshold_ratio:          Hub cutoff: ratio × total_chunks.
        enable_entity_linking:        Run Phase 3d.5 embedding-based linking.
        linking_threshold:            Cosine threshold for entity merging.
        resume:                       Skip phases recorded as done in checkpoint.
    """
    if not STORAGE_AVAILABLE:
        logger.error("Storage modules not available - aborting")
        sys.exit(1)

    if resume and clear:
        logger.error("--resume and --clear are mutually exclusive. Use one or the other.")
        sys.exit(1)

    total_start = time.time()

    # Paths (defined early so --clear and checkpoint logic can use them)
    base_path = _DATA_ROOT / dataset_name
    vector_path = base_path / "vector"
    graph_path = base_path / "graph"

    # -- Load checkpoint (--resume) -----------------------------------------
    checkpoint = _load_checkpoint(graph_path) if resume else {}
    if resume and checkpoint:
        done = [p for p, v in checkpoint.items() if v.get(_CP_DONE)]
        logger.info("  Resuming: phases already done: %s", done)

    bar = "=" * _BAR_WIDTH
    print()
    print(bar)
    print("DECOUPLED INGESTION  -  PHASE 3: LOCAL IMPORT")
    print(bar)
    print(f"  Dataset:      {dataset_name}")
    print(f"  Chunks:       {chunks_path}")
    print(f"  Extractions:  {extractions_path}")
    print(f"  Graph only:   {graph_only}")
    print(f"  Resume:       {'on' if resume else 'off'}")
    print(f"  Cleanup:      {'off' if skip_cleanup else ('dry-run' if cleanup_dry_run else 'on')}")
    print(f"  Co-occur:     {'off' if skip_cooccurrence else 'on'} "
          f"(conf >= {cooccurrence_min_confidence})")
    print(f"  Attr objects: {'CONCEPT nodes (--attribute-objects)' if capture_attribute_objects else 'dropped (default -- OntoNotes-5 taxonomy)'}")
    print(f"  Embeddings:   {embeddings_backend}")
    print(bar)

    # Build the shared embeddings object once; both Phase 3a and 3d.5 use it.
    if embeddings_backend == "huggingface":
        embedding_config = config.get("embeddings", {})
        perf_config = config.get("performance", {})
        _hf_model = embedding_config.get("hf_model_name", _HF_DEFAULT_MODEL)
        _shared_embeddings = _HFEmbeddings(
            model_name=_hf_model,
            batch_size=perf_config.get("batch_size", _HF_DEFAULT_BATCH_SIZE),
        )
        logger.info("  HuggingFace embeddings ready (%s)", _hf_model)
    else:
        _shared_embeddings = None  # Phase 3a/3d.5 build BatchedOllamaEmbeddings lazily

    # Clear if requested.
    # IMPORTANT: extraction_results.json and chunks_export.json are NEVER
    # deleted - they are source artifacts (Phase 1 / Colab output) and must
    # be preserved. Only the derived stores (vector, KuzuDB graph files) are
    # removed.
    if clear:
        import shutil, os, stat

        # Windows / KuzuDB compatibility: KuzuDB holds an OS-level lock on
        # its .lock and shadow files until the holding process exits.  A
        # bare shutil.rmtree raises PermissionError [WinError 5] on any
        # locked file.  This handler chmod+retries each failed entry and
        # skips the ones that remain locked, so a partial clean still
        # succeeds and the next run can recreate what was missed.
        def _rm_retry(func, path):
            try:
                os.chmod(path, stat.S_IWRITE)
            except OSError:
                pass
            for _ in range(3):
                try:
                    func(path)
                    return
                except PermissionError:
                    time.sleep(0.4)
                except FileNotFoundError:
                    return
            logger.warning(
                "  Skipped (still locked, close Python/IDE holders and retry): %s",
                path,
            )

        # Python 3.12 deprecates onerror in favour of onexc; support both.
        if sys.version_info >= (3, 12):
            def _rm_handler(func, path, exc):  # onexc signature
                _rm_retry(func, path)
            _rmtree_kwargs = {"onexc": _rm_handler}
        else:
            def _rm_handler(func, path, exc_info):  # onerror signature
                _rm_retry(func, path)
            _rmtree_kwargs = {"onerror": _rm_handler}

        if graph_only:
            targets = [graph_path]
        else:
            targets = [vector_path, graph_path]
        for target in targets:
            if not target.exists():
                continue
            # ── SAFETY: filesystem sidecar for every .json file ───────────────
            # JSON files are SOURCE ARTIFACTS (extraction_results.json,
            # chunks_export.json) that take ~30 min of GPU time to regenerate
            # and MUST survive --clear.
            #
            # CRITICAL: rescue must be filesystem-based, not in-memory.  If
            # rmtree raises mid-way (Windows DB lock, Ctrl-C, OOM, segfault
            # in a C extension, etc.) the in-memory dict is lost with the
            # process while the on-disk JSON is already deleted.  Sidecar
            # the files OUT of the target first, then delete, then move
            # them back — at every point the data lives on disk somewhere.
            sidecar = target.parent / f".{target.name}_rescue"
            if sidecar.exists():
                # Leftover from a prior crashed run — restore those first
                # before we touch anything else, so we never overwrite a
                # rescued file with a fresh one.
                logger.warning(
                    "  Found prior rescue dir %s; restoring before --clear.",
                    sidecar,
                )
                target.mkdir(parents=True, exist_ok=True)
                for f in sidecar.iterdir():
                    dst = target / f.name
                    if not dst.exists():
                        f.replace(dst)
                        logger.info("  Recovered from prior rescue: %s", dst)
                # Remove leftover empty sidecar; ignore if anything remains.
                try:
                    sidecar.rmdir()
                except OSError:
                    pass

            sidecar.mkdir(parents=True, exist_ok=True)
            json_files = list(target.rglob("*.json"))
            for json_file in json_files:
                dst = sidecar / json_file.name
                # os.replace is atomic on a single filesystem
                os.replace(json_file, dst)
                logger.info("  Sidecarred before --clear: %s", json_file.name)

            # ── DELETE database directory (try/finally so restore ALWAYS
            #    runs, even if rmtree raises) ─────────────────────────────────
            rmtree_error: Optional[BaseException] = None
            try:
                shutil.rmtree(target, **_rmtree_kwargs)
                logger.info("  Cleared: %s", target)
            except BaseException as exc:   # noqa: BLE001 — must restore on ANY failure
                rmtree_error = exc
                logger.error(
                    "  rmtree failed (%s); will restore JSON sidecar before re-raising.",
                    exc,
                )
            finally:
                # ── RESTORE rescued JSON files ────────────────────────────
                target.mkdir(parents=True, exist_ok=True)
                for f in sidecar.iterdir():
                    dst = target / f.name
                    os.replace(f, dst)
                    logger.info("  Restored: %s", dst)
                try:
                    sidecar.rmdir()
                except OSError:
                    logger.warning(
                        "  Sidecar dir %s not empty after restore — inspect manually.",
                        sidecar,
                    )

            if rmtree_error is not None:
                raise rmtree_error

    # --clear also invalidates the checkpoint so the next run starts fresh.
    if clear:
        for _cp in [_checkpoint_path(graph_path), _name_to_id_path(graph_path)]:
            try:
                _cp.unlink(missing_ok=True)
            except OSError:
                pass

    base_path.mkdir(parents=True, exist_ok=True)

    # Load source data (needed by all phases).
    chunks = load_chunks(chunks_path)
    extraction_data = load_extractions(extractions_path)
    extraction_results = extraction_data.get("results", [])

    if len(chunks) != len(extraction_results):
        logger.warning(
            "  WARNING: chunk count mismatch (chunks=%d, extractions=%d)",
            len(chunks), len(extraction_results),
        )
        logger.warning("  Continuing anyway (using chunk_id intersection).")

    documents = chunks_to_documents(chunks)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3a — VECTOR STORE  (skipped with --graph-only)
    # ══════════════════════════════════════════════════════════════════════
    # Writes the chunk embeddings into the LanceDB table for dense retrieval.
    # No ANN index is built — query-time search is an exact cosine scan over
    # the full table (fast enough at ~9 000 chunks; see TECHNICAL_ARCHITECTURE
    # §3.4). Typical time: ~2-5 min for 9 000 chunks on CPU.
    # ══════════════════════════════════════════════════════════════════════
    _t0 = time.time()
    if not graph_only:
        if _phase_done(checkpoint, "3a"):
            logger.info("  Phase 3a (vector store): SKIPPED -- already in checkpoint")
        else:
            try:
                ingest_vector_store(documents, vector_path, config, dataset_name,
                                    embeddings=_shared_embeddings)
                _save_checkpoint(graph_path, "3a", {"chunks": len(documents)})
                logger.info(_phase_eta("3a vector store", time.time() - _t0))
            except Exception as e:
                logger.error("Vector store ingestion failed: %s", e)
                logger.error("Use --graph-only when the vector store is built separately.")
                raise
    else:
        logger.info("  Phase 3a (vector store): SKIPPED -- --graph-only")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3b — KNOWLEDGE GRAPH  (KuzuDB)
    # ══════════════════════════════════════════════════════════════════════
    # Imports ALL entities, relations, and chunk-graph structure into KuzuDB.
    # Sub-steps:
    #   1-3  DocumentChunk + SourceDocument nodes + FROM_SOURCE / NEXT_CHUNK
    #   4    Entity nodes (deduplicated via canonical_form + SHA-256 id)
    #   5    MENTIONS edges: chunk → entity
    #   6    RELATED_TO edges from REBEL relation extraction
    #   7    SVO narrative edges from SpaCy dependency parse
    #
    # Typical time (9 000 chunks, Windows, SSD):
    #   Without batch transactions:  ~45 min  (one fsync per statement)
    #   With batch transactions:     ~5-8 min (one fsync per 200 chunks)
    # ══════════════════════════════════════════════════════════════════════
    _t0 = time.time()
    graph_store = None
    stats: Dict[str, Any] = {}
    name_to_id: Dict[str, str] = {}

    if _phase_done(checkpoint, "3b"):
        logger.info("  Phase 3b (knowledge graph): SKIPPED -- already in checkpoint")
        # Re-open the existing graph store so downstream phases can use it.
        # KuzuGraphStore takes the *container* directory and appends
        # KUZU_DIR_NAME itself -- pass graph_path, not graph_path/"graph_KuzuDB",
        # otherwise the path is doubled (.../graph_KuzuDB/graph_KuzuDB).
        graph_store = KuzuGraphStore(str(graph_path))
        stats = checkpoint["3b"].get("stats", {})
        # Reload name_to_id from the sidecar file saved by the previous run.
        n2i_path = _name_to_id_path(graph_path)
        if n2i_path.exists():
            name_to_id = json.loads(n2i_path.read_text(encoding="utf-8"))
            logger.info("  Loaded name_to_id: %d entries", len(name_to_id))
        else:
            logger.warning(
                "  name_to_id cache not found -- Phase 3c (co-occurrence) "
                "will produce 0 edges. Re-run without --resume to rebuild."
            )
    else:
        try:
            entity_conf = (
                config.get("entity_extraction", {})
                      .get("gliner", {})
                      .get("confidence_threshold", _ENTITY_CONFIDENCE_THRESHOLD_DEFAULT)
            )
            svo_conf = (
                config.get("entity_extraction", {})
                      .get("svo", {})
                      .get("confidence", _SVO_CONFIDENCE_DEFAULT)
            )
            graph_store, stats = ingest_knowledge_graph(
                documents, extraction_results, graph_path, dataset_name,
                entity_confidence_threshold=entity_conf,
                capture_attribute_objects=capture_attribute_objects,
                svo_confidence=svo_conf,
            )
            name_to_id = stats.pop("_name_to_id", {})
            # Persist name_to_id so --resume can reload it for Phase 3c.
            _name_to_id_path(graph_path).write_text(
                json.dumps(name_to_id, ensure_ascii=False), encoding="utf-8"
            )
            _save_checkpoint(graph_path, "3b", {"stats": stats})
            logger.info(_phase_eta("3b knowledge graph", time.time() - _t0))
        except Exception as e:
            logger.error("Knowledge graph ingestion failed: %s", e)
            raise

    # =====================================================================
    # PHASE 3c -- CO-OCCURRENCE EDGES
    # =====================================================================
    # Every pair of entities that appear in the SAME chunk gets a
    # RELATED_TO(cooccurs) edge. This is the primary mechanism for
    # increasing graph density.
    #
    # Example: chunk with 3 named entities [A, B, C]
    #   -> 3 new pairwise edges: A<->B, A<->C, B<->C
    #
    # Typical time: ~3-5 min (bulk transactional writes, batched).
    # =====================================================================
    _t0 = time.time()
    cooccurrence_edges = 0
    bar_dash = "-" * _BAR_WIDTH
    if graph_store is not None and not skip_cooccurrence:
        if _phase_done(checkpoint, "3c"):
            logger.info("  Phase 3c (co-occurrence): SKIPPED -- already in checkpoint")
            cooccurrence_edges = checkpoint["3c"].get("edges", 0)
        else:
            logger.info("\n%s", bar_dash)
            logger.info("PHASE 3c: CO-OCCURRENCE EDGES")
            logger.info("%s", bar_dash)
            try:
                cooccurrence_edges = build_cooccurrence_edges(
                    graph_store=graph_store,
                    extraction_results=extraction_results,
                    name_to_id=name_to_id,
                    min_confidence=cooccurrence_min_confidence,
                    relation_type="cooccurs",
                )
                logger.info("  OK %d co-occurrence edges added", cooccurrence_edges)
                _save_checkpoint(graph_path, "3c", {"edges": cooccurrence_edges})
                logger.info(_phase_eta("3c co-occurrence", time.time() - _t0))
            except Exception as e:  # noqa: BLE001 -- best-effort: log and continue
                logger.error("Co-occurrence edge construction failed: %s", e)

    # =====================================================================
    # PHASE 3c.5 -- SUBSUMPTIVE CO-OCCURRENCE CLEANUP (semantic wins)
    # =====================================================================
    # For every entity-pair that has BOTH a REBEL/SVO semantic edge AND a
    # cooccurs edge, delete the cooccurs edge. The semantic relation already
    # entails co-occurrence, so the cooccurs row is redundant -- keeping it
    # inflates edge counts and pollutes visualisation + ablation metrics.
    # Pairs whose ONLY signal is co-occurrence are kept (the cooccurs row
    # is still the only edge available to retrieval for them).
    # =====================================================================
    _t0 = time.time()
    subsumed_dropped = 0
    if (
        graph_store is not None
        and not skip_cooccurrence
        and not skip_subsumptive_cleanup
    ):
        if _phase_done(checkpoint, "3c5"):
            logger.info("  Phase 3c.5 (subsumptive cleanup): SKIPPED -- already in checkpoint")
            subsumed_dropped = checkpoint["3c5"].get("dropped", 0)
        else:
            logger.info("\n%s", bar_dash)
            logger.info(
                "PHASE 3c.5: SUBSUMPTIVE CO-OCCURRENCE CLEANUP%s",
                " (DRY RUN)" if cleanup_dry_run else "",
            )
            logger.info("%s", bar_dash)
            try:
                subsumed_dropped = drop_subsumed_cooccurrence_edges(
                    graph_store=graph_store,
                    cooccurs_relation_type="cooccurs",
                    dry_run=cleanup_dry_run,
                )
                logger.info(
                    "  Subsumed cooccurs deleted: %d (semantic relation already exists)",
                    subsumed_dropped,
                )
                _save_checkpoint(graph_path, "3c5", {"dropped": subsumed_dropped})
                logger.info(_phase_eta("3c.5 subsumptive cleanup", time.time() - _t0))
            except Exception as e:  # noqa: BLE001 -- best-effort: log and continue
                logger.error("Subsumptive cleanup failed: %s", e)

    # =====================================================================
    # PHASE 3d -- GRAPH CLEANUP
    # =====================================================================
    # Four-pass cleanup to improve graph quality:
    #   Pass 1  Stop-list:   Drop entities matching DEFAULT_STOPLIST
    #           (closed-class pronouns, nationality adjectives, ambiguous
    #            abbreviations -- see graph_quality.DEFAULT_STOPLIST).
    #   Pass 2  Orphans:     Drop entities with 0 MENTIONS edges.
    #   Pass 3  Hubs:        Drop entities mentioned in more than
    #           _HUB_THRESHOLD_RATIO_DEFAULT of all chunks. Overly generic
    #           nodes that link unrelated chunks and reduce retrieval
    #           precision (high-frequency named entities + frequent
    #           toponyms).
    #   Pass 4  Duplicates:  Merge entities sharing canonical_form within
    #           a type bucket.
    # Typical time: < 1 min.
    # =====================================================================
    _t0 = time.time()
    cleanup_ops = {"orphans_dropped": 0, "hubs_dropped": 0,
                   "duplicates_merged": 0, "stoplist_dropped": 0}
    if graph_store is not None and not skip_cleanup:
        if _phase_done(checkpoint, "3d"):
            logger.info("  Phase 3d (cleanup): SKIPPED -- already in checkpoint")
            cleanup_ops = checkpoint["3d"].get("ops", cleanup_ops)
        else:
            logger.info("\n%s", bar_dash)
            logger.info(
                "PHASE 3d: GRAPH CLEANUP%s",
                " (DRY RUN)" if cleanup_dry_run else "",
            )
            logger.info("%s", bar_dash)
            try:
                cleanup_ops = cleanup_graph(
                    graph_store=graph_store,
                    drop_orphans=True,
                    hub_threshold_ratio=hub_threshold_ratio,
                    merge_duplicates=True,
                    dry_run=cleanup_dry_run,
                )
                logger.info(
                    "  Stop-list dropped:  %d (pronouns, nationality adjectives)",
                    cleanup_ops["stoplist_dropped"],
                )
                logger.info("  Orphans dropped:    %d", cleanup_ops["orphans_dropped"])
                logger.info(
                    "  Hubs dropped:       %d (threshold ratio = %.1f%% of chunks)",
                    cleanup_ops["hubs_dropped"], hub_threshold_ratio * 100,
                )
                logger.info("  Duplicates merged:  %d", cleanup_ops["duplicates_merged"])
                _save_checkpoint(graph_path, "3d", {"ops": cleanup_ops})
                logger.info(_phase_eta("3d cleanup", time.time() - _t0))
            except Exception as e:  # noqa: BLE001 -- best-effort: log and continue
                logger.error("Cleanup pass failed: %s", e)

    # =====================================================================
    # PHASE 3d.5 -- EMBEDDING-BASED ENTITY LINKING
    # =====================================================================
    # After canonical_form deduplication, some aliases still differ:
    # short-form / long-form abbreviation pairs, partial vs. full name,
    # and other surface-form variations the deterministic normaliser cannot
    # collapse. This phase embeds every entity name using the configured
    # embedding model and merges pairs with cosine similarity above
    # `linking_threshold` within the same type bucket.
    # Typical time: ~10 min for ~5 000 entities on CPU.
    # =====================================================================
    _t0 = time.time()
    linked_count = 0
    if graph_store is not None and enable_entity_linking:
        if _phase_done(checkpoint, "3d5"):
            prev_3d5 = checkpoint["3d5"]
            prev_note = prev_3d5.get("note", "")
            prev_max_type_size = prev_3d5.get("max_type_size", 0)
            logger.info("  Phase 3d.5 (entity linking): SKIPPED -- already in checkpoint")
            linked_count = prev_3d5.get("linked", 0)
            if "partial" in prev_note.lower() or (
                prev_max_type_size and linking_max_type_size > prev_max_type_size
            ):
                logger.warning(
                    "  Phase 3d.5 prior run was PARTIAL (note='%s', max_type_size=%d) "
                    "but current --linking-max-type-size=%d is larger. "
                    "Delete the '3d5' entry from .import_checkpoint.json and re-run "
                    "with --resume to process the previously-skipped type buckets "
                    "(typically PERSON, ORG).",
                    prev_note, prev_max_type_size, linking_max_type_size,
                )
        else:
            logger.info("\n%s", bar_dash)
            logger.info("PHASE 3d.5: EMBEDDING-BASED ENTITY LINKING")
            logger.info("%s", bar_dash)
            try:
                if _shared_embeddings is not None:
                    embedder = _shared_embeddings
                else:
                    embedding_config = config.get("embeddings", {})
                    perf_config = config.get("performance", {})
                    cache_path = _CACHE_ROOT / f"{dataset_name}_embeddings.db"
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    embedder = BatchedOllamaEmbeddings(
                        model_name=embedding_config.get("model_name", _OLLAMA_DEFAULT_MODEL),
                        base_url=embedding_config.get("base_url", _OLLAMA_DEFAULT_URL),
                        batch_size=perf_config.get("batch_size", _HF_DEFAULT_BATCH_SIZE),
                        cache_path=cache_path,
                        device=perf_config.get("device", "cpu"),
                    )

                # -- Per-bucket resume ------------------------------------
                # If a previous run died mid-phase (Ollama OOM, kill, etc.)
                # it left a non-done `3d5` entry with `done_buckets` listed.
                # Read those and skip them in this call; accumulate already
                # merged counts so the final summary is accurate.
                prev_partial = checkpoint.get("3d5", {})
                done_buckets: List[str] = list(
                    prev_partial.get("done_buckets", [])
                )
                prev_linked: int = int(prev_partial.get("linked", 0))
                if done_buckets:
                    logger.info(
                        "  Resuming entity linking -- %d bucket(s) already "
                        "done in a prior run: %s",
                        len(done_buckets), ", ".join(sorted(done_buckets)),
                    )

                # Mutable cell so the closure can update the running total
                # (Python 2-style nonlocal-via-list; works on all versions).
                linked_count_holder = [0]

                def _on_bucket_done(etype: str, merged_in_type: int) -> None:
                    """Persist this bucket's completion immediately.

                    Writes a *partial* checkpoint (no `_CP_DONE`) after every
                    bucket so a crash mid-phase does not lose progress on the
                    buckets that already finished.
                    """
                    linked_count_holder[0] += merged_in_type
                    if etype not in done_buckets:
                        done_buckets.append(etype)
                    _save_partial_checkpoint(
                        graph_path,
                        "3d5",
                        {
                            "linked": prev_linked + linked_count_holder[0],
                            "done_buckets": list(done_buckets),
                            "max_type_size": linking_max_type_size,
                            "threshold": linking_threshold,
                            "note": "partial -- bucket-level checkpoint",
                        },
                    )

                linked_count = link_entities_by_embedding(
                    graph_store=graph_store,
                    embedder=embedder,
                    similarity_threshold=linking_threshold,
                    max_type_size=linking_max_type_size,
                    dry_run=cleanup_dry_run,
                    done_buckets=done_buckets,
                    on_bucket_done=_on_bucket_done,
                )
                # Linker's return value covers ONLY buckets processed in this
                # call; add any merges recorded by a prior partial run.
                total_linked = prev_linked + linked_count
                logger.info(
                    "  Embedding-linked entities merged: %d "
                    "(this run: %d, prior partial: %d, threshold=%.2f)",
                    total_linked, linked_count, prev_linked, linking_threshold,
                )
                _save_checkpoint(
                    graph_path,
                    "3d5",
                    {
                        "linked": total_linked,
                        "done_buckets": list(done_buckets),
                        "max_type_size": linking_max_type_size,
                        "threshold": linking_threshold,
                    },
                )
                linked_count = total_linked
                logger.info(_phase_eta("3d.5 entity linking", time.time() - _t0))
            except Exception as exc:  # noqa: BLE001 -- best-effort: log and continue
                logger.error("Entity linking failed: %s", exc)

    # =====================================================================
    # PHASE 3f -- POST-LINK ISOLATED-ENTITY DROP
    # =====================================================================
    # After Phase 3d.5 (alias resolution) some entities end up with MENTIONS
    # edges but ZERO RELATED_TO edges (their cluster-mates absorbed the
    # connectivity during the merge, or REBEL/SVO never produced a triple
    # for them). These dead-leaf nodes contribute nothing to graph
    # traversal -- every multi-hop search rooted on them returns the empty
    # set -- so we drop them here, before the baseline-metrics phase, so
    # that phase reports the correctly-pruned state.
    # =====================================================================
    _t0 = time.time()
    isolated_dropped = 0
    if (
        graph_store is not None
        and not skip_isolated_drop
    ):
        if _phase_done(checkpoint, "3f"):
            logger.info("  Phase 3f (drop-isolated): SKIPPED -- already in checkpoint")
            isolated_dropped = checkpoint["3f"].get("dropped", 0)
        else:
            logger.info("\n%s", bar_dash)
            logger.info(
                "PHASE 3f: POST-LINK ISOLATED-ENTITY DROP%s",
                " (DRY RUN)" if cleanup_dry_run else "",
            )
            logger.info("%s", bar_dash)
            try:
                isolated_dropped = drop_isolated_entities(
                    graph_store=graph_store,
                    dry_run=cleanup_dry_run,
                )
                logger.info(
                    "  Isolated entities dropped: %d (zero RELATED_TO edges)",
                    isolated_dropped,
                )
                _save_checkpoint(graph_path, "3f", {"dropped": isolated_dropped})
                logger.info(_phase_eta("3f drop-isolated", time.time() - _t0))
            except Exception as exc:  # noqa: BLE001 -- best-effort: log and continue
                logger.error("Drop-isolated failed: %s", exc)

    # =====================================================================
    # PHASE 3e -- BASELINE METRICS
    # =====================================================================
    # Computes graph health statistics and checks invariants:
    #   - total nodes / edges / densities
    #   - isolated entity rate (should be small after co-occurrence)
    #   - duplicate cluster rate (should be small)
    #   - relations per chunk
    # Warnings are printed but never abort the import.
    # Typical time: < 30 seconds.
    # =====================================================================
    baseline: Dict[str, Any] = {}
    violations: List[str] = []
    if graph_store is not None:
        try:
            baseline = compute_graph_baseline(graph_store)
            print()
            print(format_baseline_report(baseline))
            violations = assert_graph_invariants(baseline, strict=False)
            if violations:
                print()
                print("  INVARIANT VIOLATIONS (warning, not fatal):")
                for v in violations:
                    print(f"    - {v}")
        except Exception as e:  # noqa: BLE001 -- best-effort: log and continue
            logger.error("Baseline computation failed: %s", e)

    # Summary
    total_elapsed = time.time() - total_start

    print()
    print(bar)
    print("IMPORT COMPLETE")
    print(bar)
    print(f"  Total time:       {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Vector store:     {vector_path}")
    print(f"  Knowledge graph:  {graph_path}")
    print()
    print("  Ingestion stats:")
    for key, val in stats.items():
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        print(f"    {key:<26}: {val:>10,}")
    print(f"    {'cooccurrence_edges':<26}: {cooccurrence_edges:>10,}")
    print(f"    {'cooccurs_subsumed':<26}: {subsumed_dropped:>10,}")
    print(f"    {'svo_relations':<26}: {stats.get('svo_relations', 0):>10,}")
    print(f"    {'stoplist_dropped':<26}: {cleanup_ops['stoplist_dropped']:>10,}")
    print(f"    {'orphans_dropped':<26}: {cleanup_ops['orphans_dropped']:>10,}")
    print(f"    {'hubs_dropped':<26}: {cleanup_ops['hubs_dropped']:>10,}")
    print(f"    {'duplicates_merged':<26}: {cleanup_ops['duplicates_merged']:>10,}")
    print(f"    {'embedding_linked':<26}: {linked_count:>10,}")
    print(f"    {'post_link_isolated':<26}: {isolated_dropped:>10,}")
    if stats.get("rebel_confidence_is_constant"):
        print()
        print("    NOTE: REBEL relation confidence is constant for all edges --")
        print("          it is a decoder sentinel, not a ranking signal. Graph")
        print("          retrieval must not filter RELATED_TO edges on confidence.")
    print()
    print(bar)
    print("  Next steps:")
    print(f"    python -X utf8 -m src.thesis_evaluations.benchmark_datasets evaluate --dataset {dataset_name} --samples 100")
    print(f"    python -X utf8 -m src.thesis_evaluations.benchmark_datasets ablation --dataset {dataset_name} --samples 100")
    print(f"    python -X utf8 diagnose_graph_baseline.py --dataset {dataset_name}")
    print(bar)

    # Persist extraction + import metadata.
    meta_path = base_path / "graph" / "extraction_metadata.json"
    meta = extraction_data.get("metadata", {})
    meta["import_time_seconds"] = round(total_elapsed, 1)
    meta["graph_stats"] = stats
    meta["cleanup_ops"] = cleanup_ops
    meta["cooccurrence_edges"] = cooccurrence_edges
    meta["cooccurs_subsumed_dropped"] = subsumed_dropped
    meta["post_link_isolated_dropped"] = isolated_dropped
    if baseline:
        # Strip top_clusters because they hold raw entity_id/name tuples that
        # bloat the metadata file. Keep the summary numbers.
        meta["graph_baseline"] = {
            "totals": baseline["totals"],
            "densities": baseline["densities"],
            "isolated": baseline["isolated"],
            "duplicates": {
                k: v for k, v in baseline["duplicates"].items()
                if k != "top_clusters"
            },
            "violations": violations,
        }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info("  Metadata written: %s", meta_path)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Decoupled ingestion - Phase 3: local import",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard import (vector store + knowledge graph + cleanup)
  python local_importingestion.py \\
      --chunks data/hotpotqa/chunks_export.json \\
      --extractions data/hotpotqa/graph/extraction_results.json \\
      --dataset hotpotqa

  # Knowledge graph only (vector store already exists)
  python local_importingestion.py \\
      --chunks data/hotpotqa/chunks_export.json \\
      --extractions data/hotpotqa/graph/extraction_results.json \\
      --dataset hotpotqa \\
      --graph-only

  # With explicit YAML config
  python local_importingestion.py \\
      --chunks data/hotpotqa/chunks_export.json \\
      --extractions data/hotpotqa/graph/extraction_results.json \\
      --dataset hotpotqa \\
      --config config/settings.yaml

  # Preview cleanup without mutating the graph
  python local_importingestion.py \\
      --chunks data/hotpotqa/chunks_export.json \\
      --extractions data/hotpotqa/graph/extraction_results.json \\
      --dataset hotpotqa \\
      --cleanup-dry-run
        """,
    )

    parser.add_argument(
        "--chunks", "-c",
        type=Path,
        required=True,
        help="Path to chunks_export.json (Phase 1 output)",
    )
    parser.add_argument(
        "--extractions", "-e",
        type=Path,
        required=True,
        help="Path to extraction_results.json (Phase 2 / Colab output)",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="Dataset name (e.g. hotpotqa, 2wikimultihop)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to settings.yaml (optional)",
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="Import only the knowledge graph; skip the vector store",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete existing stores before import",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Disable the post-ingestion cleanup pass (orphans, hubs, duplicate merge)",
    )
    parser.add_argument(
        "--no-cooccurrence",
        action="store_true",
        help="Disable co-occurrence edge construction (RELATED_TO {cooccurs})",
    )
    parser.add_argument(
        "--no-subsumptive-cleanup",
        action="store_true",
        help="Disable Phase 3c.5 -- keep cooccurs edges even when a semantic "
             "edge already covers the same entity-pair. Default is ON: any "
             "cooccurs edge that has a paired semantic edge in either "
             "direction is deleted as redundant.",
    )
    parser.add_argument(
        "--no-isolated-drop",
        action="store_true",
        help="Disable Phase 3f -- keep entities with zero RELATED_TO edges "
             "after entity linking. Default is ON: dead-leaf entities "
             "(MENTIONS present but RELATED_TO empty in both directions) "
             "are removed so the Phase-3e baseline reports the correctly-"
             "pruned state.",
    )
    parser.add_argument(
        "--cleanup-dry-run",
        action="store_true",
        help="Run the cleanup pass in dry-run mode (count operations, no mutation)",
    )
    parser.add_argument(
        "--cooccurrence-min-confidence",
        type=float,
        default=_COOCC_MIN_CONFIDENCE_DEFAULT,
        help=f"Minimum NER confidence required for an entity to participate "
             f"in co-occurrence edges (default: {_COOCC_MIN_CONFIDENCE_DEFAULT}).",
    )
    parser.add_argument(
        "--hub-threshold-ratio",
        type=float,
        default=_HUB_THRESHOLD_RATIO_DEFAULT,
        help=f"Drop entities mentioned in more than ratio*total_chunks chunks "
             f"(default: {_HUB_THRESHOLD_RATIO_DEFAULT} = "
             f"{_HUB_THRESHOLD_RATIO_DEFAULT * 100:.1f}%% of the corpus). "
             "Tuned to keep frequent-pronoun and common-toponym noise "
             "reachable for downstream cleanup rather than dropped at the "
             "hub-suppression step.",
    )
    parser.add_argument(
        "--no-entity-linking",
        action="store_true",
        help="Disable embedding-based entity linking (alias resolution "
             "beyond canonical_form, e.g. abbreviation <-> full name).",
    )
    parser.add_argument(
        "--linking-threshold",
        type=float,
        default=_LINKING_THRESHOLD_DEFAULT,
        help=f"Cosine similarity threshold for embedding-based entity linking "
             f"(default: {_LINKING_THRESHOLD_DEFAULT}). Higher = stricter "
             "merging. Range [0.85, 0.97].",
    )
    parser.add_argument(
        "--linking-max-type-size",
        type=int,
        default=_LINKING_MAX_TYPE_SIZE_DEFAULT,
        help=f"Maximum entity-bucket size for embedding-based linking "
             f"(default: {_LINKING_MAX_TYPE_SIZE_DEFAULT}). Buckets larger "
             "than this are skipped to avoid OOM (8000 entities ~= 256 MB "
             "float32 matrix). Raise to 50000+ only if you have sufficient "
             "RAM/VRAM.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip phases already recorded as complete in the checkpoint file "
             "(data/<dataset>/graph/.import_checkpoint.json). "
             "Mutually exclusive with --clear.",
    )
    parser.add_argument(
        "--embeddings-backend",
        choices=["ollama", "huggingface"],
        default="ollama",
        help="Embedding backend: 'ollama' (default, requires Ollama running locally) "
             "or 'huggingface' (uses sentence-transformers, no Ollama needed — "
             "ideal for Colab/GPU environments).",
    )
    parser.add_argument(
        "--attribute-objects",
        action="store_true",
        help="Opt-in: capture REBEL relations whose object is a free-text "
             "attribute value (e.g. category / sport / occupation values "
             "that are common nouns rather than named entities) as "
             "synthetic CONCEPT entity nodes. Default OFF -- keeps the "
             "graph within the OntoNotes-5 GLiNER taxonomy. Enable only "
             "for exploratory runs or ablation studies.",
    )

    args = parser.parse_args()

    # Validation
    if args.resume and args.clear:
        logger.error("--resume and --clear are mutually exclusive.")
        sys.exit(1)
    if not args.chunks.exists():
        logger.error("Chunks file not found: %s", args.chunks)
        sys.exit(1)
    if not args.extractions.exists():
        logger.error("Extractions file not found: %s", args.extractions)
        sys.exit(1)

    config = load_config(args.config)

    run_full_import(
        chunks_path=args.chunks,
        extractions_path=args.extractions,
        dataset_name=args.dataset,
        config=config,
        graph_only=args.graph_only,
        clear=args.clear,
        skip_cleanup=args.no_cleanup,
        skip_cooccurrence=args.no_cooccurrence,
        skip_subsumptive_cleanup=args.no_subsumptive_cleanup,
        skip_isolated_drop=args.no_isolated_drop,
        cleanup_dry_run=args.cleanup_dry_run,
        cooccurrence_min_confidence=args.cooccurrence_min_confidence,
        hub_threshold_ratio=args.hub_threshold_ratio,
        enable_entity_linking=not args.no_entity_linking,
        linking_threshold=args.linking_threshold,
        linking_max_type_size=args.linking_max_type_size,
        resume=args.resume,
        embeddings_backend=args.embeddings_backend,
        capture_attribute_objects=args.attribute_objects,
    )


if __name__ == "__main__":
    main()