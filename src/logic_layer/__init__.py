"""
Logic Layer — Agent-Based Query Processing

Implements Artifact B of the paper:
Edge-RAG: Empirical Characterization of When Knowledge-Graph Lanes
Add Value in CPU-Only Hybrid Retrieval

Three-agent architecture:
    S_P (Planner)   — Query classification, entity extraction, plan generation
    S_N (Navigator) — Hybrid retrieval, RRF fusion, pre-generative filtering
    S_V (Verifier)  — Pre-validation, answer generation, self-correction loop

Usage:
    from src.pipeline import AgentPipeline, create_full_pipeline

    pipeline = create_full_pipeline()
    result = pipeline.process("What is the capital of France?")

Note:
    AgenticController is a static-helper container for bridge-entity
    extraction (see src/logic_layer/controller.py); it is not the
    orchestrator. The production entry point is AgentPipeline.process().

Last reviewed: 2026-05-25 (audit pass, project version 5.4).
"""

# =============================================================================
# PLANNER (S_P) — Query Analysis & Planning
# =============================================================================
from .planner import (
    Planner,
    create_planner,
    QueryType,
    RetrievalStrategy,
    EntityInfo,
    HopStep,
    RetrievalPlan,
    # QueryClassifier, EntityExtractor, PlanGenerator are internal
    # sub-components of Planner — not part of the public API.
)

# =============================================================================
# NAVIGATOR (S_N) — Retrieval & Pre-Generative Filtering
# =============================================================================
from .navigator import (
    Navigator,
    NavigatorResult,
)

# ControllerConfig is defined in _config.py; import from there to keep the
# public API stable regardless of which production file is refactored.
from ._config import ControllerConfig

# =============================================================================
# VERIFIER (S_V) — Validation & Generation
# =============================================================================
from .verifier import (
    Verifier,
    create_verifier,
    VerifierConfig,
    ValidationStatus,
    ConfidenceLevel,
    SourceCredibility,
    PreValidationResult,
    VerificationResult,
    PreGenerationValidator,
)

# =============================================================================
# CONTROLLER — static helpers only
# =============================================================================
# AgenticController is exported as a static-helper container (bridge-entity
# extraction + hop-query rewriting) consumed by
# AgentPipeline._iterative_navigate; it is not an orchestrator. For the
# production entry point use src.pipeline.AgentPipeline / create_full_pipeline.
from .controller import AgenticController

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Planner
    "Planner",
    "create_planner",
    "QueryType",
    "RetrievalStrategy",
    "EntityInfo",
    "HopStep",
    "RetrievalPlan",
    # Navigator
    "Navigator",
    "NavigatorResult",
    "ControllerConfig",
    # Verifier
    "Verifier",
    "create_verifier",
    "VerifierConfig",
    "ValidationStatus",
    "ConfidenceLevel",
    "SourceCredibility",
    "PreValidationResult",
    "VerificationResult",
    "PreGenerationValidator",
    # Controller (static-helper container)
    "AgenticController",
]

__version__ = "5.4.0"
__author__ = "Jan Nietzard"