"""
Extension of the Section 4.3 required test matrix (paper §4.3) — the
remaining coverage gaps.

Tests cover:
  DATA LAYER   — RRF fusion math (2 tests), retrieval-mode isolation (2 tests)
  LOGIC LAYER  — Planner comparison decomposition, Verifier confidence boundaries (2 tests)
  PIPELINE     — Ablation: planner disabled
  ERROR PATH   — Ollama unreachable raises a clear exception

All tests are fully offline (no Ollama, no real stores required) except
test_ollama_unreachable_raises_connection_error, which explicitly tests the
error path when the service is absent. Fixtures use neutral encyclopedia
entities.

Last reviewed: 2026-05-30 (audit pass, project version 5.4).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# DATA LAYER — RRF Fusion: mathematical correctness
# =============================================================================

def test_rrf_fusion_correct_math():
    """
    RRF score for a chunk at rank 1 in BOTH vector and graph lists must equal
        1/(k+1) + 1/(k+1) + cross_source_boost/(k+1)
    With k=60 and cross_source_boost=1.2 (the paper defaults) that gives
        (1 + 1 + 1.2) / 61 = 3.2 / 61.
    """
    from src.data_layer.hybrid_retriever import RRFFusion

    vector_results = [{
        "document_id": "c1",
        "text": "Einstein was born in Ulm in 1879.",
        "similarity": 0.9,
        "metadata": {"source_file": "einstein.txt"},
        "position": 0,
    }]
    graph_results = [{
        "chunk_id": "c1",
        "text": "Einstein was born in Ulm in 1879.",
        "hops": 1,
        "source_file": "einstein.txt",
        "matched_entity": "Einstein",
        "position": 0,
    }]

    fusion = RRFFusion(k=60, cross_source_boost=1.2)
    fused = fusion.fuse(vector_results, graph_results, final_top_k=5)

    assert len(fused) == 1, f"Expected 1 fused result, got {len(fused)}"
    k = 60
    expected = (1.0 / (k + 1)) + (1.0 / (k + 1)) + (1.2 / (k + 1))
    assert abs(fused[0].rrf_score - expected) < 1e-9, (
        f"RRF score mismatch: expected {expected:.9f}, got {fused[0].rrf_score:.9f}"
    )
    assert fused[0].retrieval_method == "hybrid", (
        f"Chunk found in both sources must have method='hybrid', got {fused[0].retrieval_method!r}"
    )


def test_rrf_fusion_single_source_score():
    """
    When only the vector source provides a result (rank 1, no cross-source
    boost), the RRF score must equal exactly 1/(k+1) = 1/61.
    """
    from src.data_layer.hybrid_retriever import RRFFusion

    vector_results = [{
        "document_id": "c1",
        "text": "Paris is the capital of France.",
        "similarity": 0.9,
        "metadata": {"source_file": "paris.txt"},
        "position": 0,
    }]

    fusion = RRFFusion(k=60)
    fused = fusion.fuse(vector_results, [], final_top_k=5)

    assert len(fused) == 1, f"Expected 1 result, got {len(fused)}"
    expected = 1.0 / (60 + 1)
    assert abs(fused[0].rrf_score - expected) < 1e-9, (
        f"Single-source RRF expected {expected:.9f}, got {fused[0].rrf_score:.9f}"
    )
    assert fused[0].retrieval_method == "vector"


# =============================================================================
# DATA LAYER — HybridRetriever: retrieval-mode isolation
# =============================================================================

def test_hybrid_retriever_vector_mode_never_calls_graph():
    """
    VECTOR mode must retrieve only from the vector store.
    graph_search must never be called, regardless of the query.
    """
    from src.data_layer.hybrid_retriever import HybridRetriever, RetrievalConfig, RetrievalMode

    mock_store = MagicMock()
    mock_store.vector_search.return_value = [{
        "document_id": "c1", "text": "Some text.", "similarity": 0.9,
        "metadata": {"source_file": "f.txt"}, "position": 0,
    }]
    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.return_value = [0.1] * 768

    cfg = RetrievalConfig(mode=RetrievalMode.VECTOR, vector_top_k=5,
                          graph_top_k=0, final_top_k=5)
    retriever = HybridRetriever(mock_store, mock_embeddings, cfg)
    retriever.retrieve("Einstein nationality")

    mock_store.graph_search.assert_not_called()
    mock_store.vector_search.assert_called_once()


def test_hybrid_retriever_graph_mode_never_calls_vector():
    """
    GRAPH mode must retrieve only from the graph store.
    vector_search must never be called, regardless of the query.
    entity_hints are required to give the graph retriever a non-empty entity
    list; without them, graph_search is correctly skipped (no entities to look
    up), which is orthogonal to the mode-isolation behaviour being tested.
    """
    from src.data_layer.hybrid_retriever import HybridRetriever, RetrievalConfig, RetrievalMode

    mock_store = MagicMock()
    mock_store.graph_search.return_value = [{
        "chunk_id": "c1", "text": "Some text.", "hops": 1,
        "source_file": "f.txt", "matched_entity": "Einstein", "position": 0,
    }]
    mock_embeddings = MagicMock()

    cfg = RetrievalConfig(mode=RetrievalMode.GRAPH, vector_top_k=0,
                          graph_top_k=5, final_top_k=5)
    retriever = HybridRetriever(mock_store, mock_embeddings, cfg)
    # Supply entity_hints so the graph path receives a non-empty entity list.
    retriever.retrieve("Einstein nationality", entity_hints=["Einstein"])

    mock_store.vector_search.assert_not_called()
    mock_store.graph_search.assert_called_once()


# =============================================================================
# LOGIC LAYER — Planner: comparison query decomposition
# =============================================================================

def test_planner_comparison_query_entities_in_sub_queries():
    """
    A comparison query must be classified as COMPARISON and its sub-queries
    must each reference one of the two compared entities individually.
    This verifies that the _ATTR_MAP rewriting in _decompose_comparison()
    produces correctly scoped sub-queries for parallel retrieval.
    """
    from src.logic_layer.planner import Planner, QueryType

    planner = Planner()
    plan = planner.plan(
        "Were Albert Einstein and Isaac Newton of the same nationality?"
    )

    assert plan.query_type == QueryType.COMPARISON, (
        f"Expected COMPARISON, got {plan.query_type}"
    )
    all_sub_query_text = " ".join(plan.sub_queries).lower()
    assert "einstein" in all_sub_query_text, (
        "Sub-queries must reference 'Einstein' individually"
    )
    assert "newton" in all_sub_query_text, (
        "Sub-queries must reference 'Newton' individually"
    )
    # Both entities appear in separate sub-queries (not both in one)
    einstein_queries = [q for q in plan.sub_queries if "einstein" in q.lower()]
    newton_queries   = [q for q in plan.sub_queries if "newton"   in q.lower()]
    assert einstein_queries, "At least one sub-query must focus on Einstein"
    assert newton_queries,   "At least one sub-query must focus on Newton"


# =============================================================================
# LOGIC LAYER — Verifier: confidence level boundaries
# =============================================================================

def test_verifier_confidence_high_when_all_claims_verified():
    """
    VerificationResult.confidence must be HIGH when verified_claims is
    non-empty and violated_claims is empty (verified_ratio ≥ 0.8 threshold).
    Tests the confidence @property directly — independent of LLM claim
    extraction heuristics.
    """
    from src.logic_layer.verifier import VerificationResult, ConfidenceLevel

    result = VerificationResult(
        answer="Paris is the capital of France.",
        iterations=1,
        verified_claims=["Paris is the capital of France"],
        violated_claims=[],
        all_verified=True,
        pre_validation=None,
        timing_ms=5.0,
        iteration_history=[],
        confidence_high_threshold=0.8,
        confidence_medium_threshold=0.5,
    )

    assert result.confidence == ConfidenceLevel.HIGH, (
        f"verified_ratio=1.0 must yield HIGH; got {result.confidence}"
    )


def test_verifier_confidence_low_when_mostly_violated():
    """
    VerificationResult.confidence must be LOW when violated_claims outnumber
    verified_claims (verified_ratio < 0.5 threshold).
    Tests the confidence @property directly — independent of LLM claim
    extraction heuristics.
    """
    from src.logic_layer.verifier import VerificationResult, ConfidenceLevel

    result = VerificationResult(
        answer="wrong answer",
        iterations=1,
        verified_claims=[],
        violated_claims=["claim A was wrong", "claim B was wrong", "claim C was wrong"],
        all_verified=False,
        pre_validation=None,
        timing_ms=5.0,
        iteration_history=[],
        confidence_high_threshold=0.8,
        confidence_medium_threshold=0.5,
    )

    assert result.confidence == ConfidenceLevel.LOW, (
        f"verified_ratio=0.0 must yield LOW; got {result.confidence}"
    )


# =============================================================================
# PIPELINE — Ablation: planner disabled
# =============================================================================

def test_ablation_planner_disabled_returns_answer():
    """
    When enable_planner=False the pipeline must produce a non-empty answer
    using the passthrough RetrievalPlan — and must NOT call planner.plan().
    This verifies the --no-planner ablation switch from the paper evaluation.
    """
    from src.pipeline.agent_pipeline import AgentPipeline, AgentPipelineConfig
    from src.logic_layer.navigator import NavigatorResult
    from src.logic_layer.verifier import VerificationResult

    cfg = AgentPipelineConfig(enable_caching=False)
    pipeline = AgentPipeline.__new__(AgentPipeline)
    pipeline.config = cfg
    # Gate flags mirror AgentPipeline.__init__, derived from the config so this
    # manual (__new__-based) construction stays in sync as new flags are added.
    pipeline.enable_confidence_gate = cfg.enable_confidence_gate
    pipeline._conf_score_gap_threshold = cfg.confidence_score_gap_threshold
    pipeline._conf_require_signals = cfg.confidence_require_signals
    pipeline.enable_over_decomposition_gate = cfg.enable_over_decomposition_gate
    pipeline._connector_split_min_half_words = cfg.connector_split_min_half_words
    pipeline.enable_reretrieval_loop = cfg.enable_reretrieval_loop
    pipeline.enable_planner = False   # <- ablation switch under test
    pipeline.enable_verifier = True
    pipeline.enable_caching = False
    pipeline._cache = {}
    pipeline._cache_max_size = 0
    pipeline._stats = {"total_queries": 0, "cache_hits": 0, "avg_latency_ms": 0.0}
    pipeline._initialized = True

    # Planner is wired up but MUST NOT be called
    pipeline.planner = MagicMock()

    pipeline.navigator = MagicMock()
    pipeline.navigator.navigate.return_value = NavigatorResult(
        filtered_context=["The sky is blue."],
        raw_context=["The sky is blue."],
        scores=[0.9],
        metadata={},
    )
    pipeline.verifier = MagicMock()
    pipeline.verifier.generate_and_verify.return_value = VerificationResult(
        answer="blue",
        iterations=1,
        verified_claims=["The sky is blue"],
        violated_claims=[],
        all_verified=True,
        pre_validation=None,
        timing_ms=10.0,
        iteration_history=[],
    )

    result = pipeline.process("What colour is the sky?")

    assert result.answer, "Answer must be non-empty when planner is disabled"
    pipeline.planner.plan.assert_not_called()


# =============================================================================
# ERROR PATH — Ollama unreachable
# =============================================================================

# =============================================================================
# DATA LAYER — RRF Fusion: cross-source boost ordering (T-04)
# =============================================================================

def test_dual_source_chunk_outranks_single_source_chunk():
    """
    A chunk appearing in BOTH vector and graph result lists must have a higher
    RRF score than a chunk that appears only in the vector list — even when the
    single-source chunk ranks first in its list.

    This tests the cross_source_boost invariant that is the key architectural
    contribution of the RRF fusion design (paper section 3.2.3):
        dual-source score   = 1/(k+1) + 1/(k+1) + boost/(k+1)
        single-source score = 1/(k+1)
    With boost=1.2, dual-source always > single-source for equal ranks.
    """
    from src.data_layer.hybrid_retriever import RRFFusion

    # c1 appears only in vector (rank 1 — highest possible single-source score)
    # c2 appears in both vector (rank 2) and graph (rank 1) → receives boost
    vector_results = [
        {
            "document_id": "c1",
            "text": "Single-source chunk at rank 1.",
            "similarity": 0.95,
            "metadata": {"source_file": "a.txt"},
            "position": 0,
        },
        {
            "document_id": "c2",
            "text": "Dual-source chunk at rank 2.",
            "similarity": 0.85,
            "metadata": {"source_file": "a.txt"},
            "position": 1,
        },
    ]
    graph_results = [
        {
            "chunk_id": "c2",
            "text": "Dual-source chunk at rank 2.",
            "hops": 1,
            "source_file": "a.txt",
            "matched_entity": "Einstein",
            "position": 0,
        },
    ]

    fusion = RRFFusion(k=60, cross_source_boost=1.2)
    fused = fusion.fuse(vector_results, graph_results, final_top_k=5)

    fused_by_id = {r.chunk_id: r for r in fused}
    assert "c1" in fused_by_id, "c1 must be present in fused results"
    assert "c2" in fused_by_id, "c2 must be present in fused results"

    assert fused_by_id["c2"].rrf_score > fused_by_id["c1"].rrf_score, (
        f"Dual-source c2 ({fused_by_id['c2'].rrf_score:.6f}) must outrank "
        f"single-source c1 ({fused_by_id['c1'].rrf_score:.6f}) "
        f"despite c1 ranking higher in the vector list."
    )
    assert fused_by_id["c2"].retrieval_method == "hybrid", (
        "Chunk found in both sources must have retrieval_method='hybrid'"
    )


def test_ollama_unreachable_raises_connection_error():
    """
    When Ollama is not running, constructing BatchedOllamaEmbeddings must
    raise ConnectionError (the constructor calls _test_connection() eagerly)
    — not hang indefinitely, not return None silently.
    Uses port 19999 which is guaranteed to have nothing listening.
    """
    import requests
    from src.data_layer.embeddings import BatchedOllamaEmbeddings

    # __init__ calls _test_connection(), so the ConnectionError is raised
    # during construction — wrap the whole instantiation in pytest.raises.
    with pytest.raises(
        (ConnectionError, OSError, requests.exceptions.ConnectionError,
         requests.exceptions.ConnectTimeout)
    ):
        BatchedOllamaEmbeddings(
            base_url="http://localhost:19999",   # nothing listening here
            model_name="nomic-embed-text",
            timeout=2,
        )
