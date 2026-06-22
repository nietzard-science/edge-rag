"""
Pipeline package — unified orchestration layer.

Holds the two top-level orchestrators of the paper system:

  * IngestionPipeline  Document corpus → chunks → entities → storage
                       (LanceDB vector store + KuzuDB knowledge graph).
                       Built once per dataset; streamed one document at a time.
  * AgentPipeline      Query → S_P (Planner) → S_N (Navigator) → S_V
                       (Verifier) → Answer. The runtime entry point used by
                       every evaluation script.

Both orchestrators are constructed from a single ``config/settings.yaml``
via factory functions (``create_ingestion_pipeline``, ``create_full_pipeline``)
so the settings file is the only adjustable surface.

Exports
-------
    AgentPipeline, AgentPipelineConfig, PipelineResult, BatchProcessor,
        create_full_pipeline                       (from agent_pipeline)
    IngestionPipeline, IngestionConfig, IngestionMetrics, DocumentLoader,
        create_ingestion_pipeline                  (from ingestion_pipeline)

Dependencies
------------
    src.data_layer (storage, retrieval, embeddings, entity extraction)
    src.logic_layer (Planner, Navigator, Verifier, settings loader)
    stdlib otherwise.

Last reviewed: 2026-05-27 (audit pass, project version 5.4).
"""

from .agent_pipeline import (
    AgentPipeline,
    AgentPipelineConfig,
    PipelineResult,
    BatchProcessor,
    create_full_pipeline,
)
from .ingestion_pipeline import (
    IngestionPipeline,
    IngestionConfig,
    IngestionMetrics,
    DocumentLoader,
    create_ingestion_pipeline,
)

__version__ = "5.4.0"

__all__ = [
    # Agent Pipeline
    "AgentPipeline",
    "AgentPipelineConfig",
    "PipelineResult",
    "BatchProcessor",
    "create_full_pipeline",

    # Ingestion Pipeline
    "IngestionPipeline",
    "IngestionConfig",
    "IngestionMetrics",
    "DocumentLoader",
    "create_ingestion_pipeline",
]
