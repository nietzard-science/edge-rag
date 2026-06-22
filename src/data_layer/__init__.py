"""
Data Layer -- Artifact A of the Edge-RAG System.

This package encapsulates all data operations for the three-agent
(Planner -> Navigator -> Verifier) Edge-RAG pipeline. It is a thin
re-export shim: every symbol here is implemented in a submodule, and
the symbol-to-submodule mapping is the public stable contract that
downstream layers (logic_layer, pipeline, thesis_evaluations) import
against.

Public submodules (in dependency order)
---------------------------------------
storage.py
    HybridStore facade over LanceDB (vector) and KuzuDB (graph),
    plus the StorageConfig dataclass and the per-store adapters.
embeddings.py
    BatchedOllamaEmbeddings with SQLite-backed cache; nomic-embed-text
    via the local Ollama daemon.
entity_extraction.py
    EntityExtractionPipeline (GLiNER NER + REBEL relation extraction)
    and the canonical normalize_entity_name() used by both ingestion
    and query-time entity matching.
chunking.py
    SpacySentenceChunker is the production chunker; SemanticChunker /
    FixedSizeChunker / RecursiveChunker are kept as internal variants
    consumed only by ingestion.py for ablation studies.
graph_quality.py
    Post-ingestion graph audit, co-occurrence edge construction,
    cleanup pipeline, and embedding-based entity linking.
hybrid_retriever.py
    HybridRetriever with RRF fusion of dense + sparse (BM25) + graph;
    consumed by both AgentPipeline (production) and benchmark_datasets
    (evaluation).
ingestion.py
    DocumentIngestionPipeline orchestrates chunking -> embeddings ->
    entity extraction -> storage with per-phase checkpointing.
coreference.py, svo_extraction.py
    Optional pipeline stages. Each module handles its own dependency
    availability internally and exposes is_available() so the rest of
    the pipeline degrades gracefully if the optional packages are
    absent.

Data flow
---------
Ingestion: raw text -> ingestion.py -> chunking.py -> embeddings.py
                    -> entity_extraction.py -> storage.py
Query:     query -> hybrid_retriever.py -> (embeddings.py + storage.py)
                 -> RetrievalResult[]

Required local services (no cloud dependency)
---------------------------------------------
- Ollama at http://localhost:11434 serving `nomic-embed-text` and
  `qwen2:1.5b` (LLM consumed by the logic layer).
- SpaCy `en_core_web_sm` (`python -m spacy download en_core_web_sm`).
- GLiNER `urchade/gliner_small-v2.1` (auto-downloaded by HuggingFace
  on first use; cached under HF_HOME).

Reproducibility note
--------------------
This file is a stable public interface. The 35 settings keys that
parameterise these submodules are validated at startup by
src/logic_layer/_settings_loader.py against config/settings.yaml; importing
this package does not silently fall back on defaults.

Common import patterns
----------------------
Retrieval (logic_layer.navigator, benchmark_datasets):
    from src.data_layer import HybridRetriever, RetrievalConfig
    from src.data_layer import HybridStore, StorageConfig

Ingestion (local_importingestion.py, pipeline layer):
    from src.data_layer import DocumentIngestionPipeline, DocumentIngestionConfig
    from src.data_layer import create_ingestion_config
    from src.data_layer import BatchedOllamaEmbeddings
    from src.data_layer import EntityExtractionPipeline

Chunking (direct use, tests):
    from src.data_layer import SpacySentenceChunker, create_sentence_chunker

Last reviewed: 2026-05-25 (audit pass, project version 5.4).
"""

__version__ = "5.4.0"
__author__ = "Jan Nietzard"

# ---- Storage ---------------------------------------------------------------
from .storage import (
    HybridStore,
    StorageConfig,
    VectorStoreAdapter,
    KuzuGraphStore,
    create_storage_config,
)

# ---- Embeddings ------------------------------------------------------------
from .embeddings import BatchedOllamaEmbeddings, create_embeddings

# ---- Retrieval -------------------------------------------------------------
from .hybrid_retriever import (
    HybridRetriever,
    RetrievalConfig,
    RetrievalMode,
    RetrievalResult,
    RetrievalMetrics,
    ImprovedQueryEntityExtractor,
)

# ---- Entity Extraction -----------------------------------------------------
from .entity_extraction import (
    EntityExtractionPipeline,
    ExtractionConfig,
    create_extraction_pipeline,
    normalize_entity_name,
)

# ---- Chunking --------------------------------------------------------------
# Public API: SpacySentenceChunker and create_sentence_chunker.
# SemanticChunker / FixedSizeChunker / RecursiveChunker / SentenceChunker are
# internal implementation details consumed only by ingestion.py.
from .chunking import (
    SpacySentenceChunker,
    SentenceChunkingConfig,
    SentenceChunk,
    create_sentence_chunker,
)

# ---- Ingestion Pipeline ----------------------------------------------------
from .ingestion import (
    DocumentIngestionPipeline,
    DocumentIngestionConfig,
    ChunkingStrategy,
    create_ingestion_config,
    create_data_layer_pipeline,
)

# ---- Graph Quality (post-ingestion analysis + cleanup) ---------------------
from .graph_quality import (
    canonical_form,
    compute_graph_baseline,
    format_baseline_report,
    assert_graph_invariants,
    GraphQualityViolation,
    build_cooccurrence_edges,
    cleanup_graph,
    link_entities_by_embedding,
    DEFAULT_STOPLIST,
)

# ---- Optional pipeline stages (coreference, SVO) ---------------------------
# Both modules handle their own optional-dependency fallbacks internally and
# expose is_available() so callers can branch without try/except at import.
from .coreference import (
    resolve_coreferences,
    is_available as coreference_available,
)
from .svo_extraction import (
    extract_svo_relations,
    is_available as svo_available,
)

__all__ = [
    # Storage
    "HybridStore",
    "StorageConfig",
    "VectorStoreAdapter",
    "KuzuGraphStore",
    "create_storage_config",
    # Embeddings
    "BatchedOllamaEmbeddings",
    "create_embeddings",
    # Retrieval
    "HybridRetriever",
    "RetrievalConfig",
    "RetrievalMode",
    "RetrievalResult",
    "RetrievalMetrics",
    "ImprovedQueryEntityExtractor",
    # Entity Extraction
    "EntityExtractionPipeline",
    "ExtractionConfig",
    "create_extraction_pipeline",
    "normalize_entity_name",
    # Chunking (public API only -- internal variants are not re-exported)
    "SpacySentenceChunker",
    "SentenceChunkingConfig",
    "SentenceChunk",
    "create_sentence_chunker",
    # Ingestion pipeline
    "DocumentIngestionPipeline",
    "DocumentIngestionConfig",
    "ChunkingStrategy",
    "create_ingestion_config",
    "create_data_layer_pipeline",
    # Graph quality
    "canonical_form",
    "compute_graph_baseline",
    "format_baseline_report",
    "assert_graph_invariants",
    "GraphQualityViolation",
    "build_cooccurrence_edges",
    "cleanup_graph",
    "link_entities_by_embedding",
    "DEFAULT_STOPLIST",
    # Coreference + SVO (optional pipeline stages)
    "resolve_coreferences",
    "coreference_available",
    "extract_svo_relations",
    "svo_available",
]
