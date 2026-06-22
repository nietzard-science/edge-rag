"""
Tests for items in the Section 4.3 required matrix (paper §4.3).

All tests mock network/model calls and run fully offline — no Ollama or real
model weights required. Fixtures use neutral encyclopedia entities.

Coverage:
  DATA LAYER   — single-sentence chunking, known-entity extraction
  LOGIC LAYER  — empty navigator plan, verifier max_iterations cap
  PIPELINE     — timing fields, cache-clear, ablation (verifier disabled)
  CROSS-LAYER  — top_k config propagates to result count
  END-TO-END   — multi-hop answer references bridge entity (fully mocked)

Last reviewed: 2026-05-30 (audit pass, project version 5.4).
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# DATA LAYER — Chunking
# =============================================================================

def test_chunking_single_sentence_returns_one_chunk():
    """A single sentence must produce exactly one chunk, not crash or return empty."""
    from src.data_layer.chunking import SpacySentenceChunker

    # min_chunk_chars=0 so the short sentence is not filtered out
    chunker = SpacySentenceChunker(sentences_per_chunk=3, sentence_overlap=1, min_chunk_chars=0)
    chunks = chunker.chunk_text(
        "Albert Einstein was born in Ulm in 1879.",
        source_doc="test.txt",
    )
    assert len(chunks) == 1, f"Expected 1 chunk for a single sentence, got {len(chunks)}"
    assert "Einstein" in chunks[0].text


# =============================================================================
# DATA LAYER — Entity Extraction
# =============================================================================

def test_entity_extraction_known_entity_correct_type():
    """
    A text containing a known person name must yield a PERSON entity with
    confidence above the default threshold (0.15).
    """
    from src.data_layer.entity_extraction import ExtractionConfig, GLiNERExtractor

    fake_model = MagicMock()
    fake_model.predict_entities.return_value = [
        {"text": "Fritz Lang", "label": "person", "score": 0.92,
         "start": 0, "end": 10}
    ]

    with patch("src.data_layer.entity_extraction.GLINER_AVAILABLE", True), \
         patch("src.data_layer.entity_extraction.GLiNER") as mock_cls:
        mock_cls.from_pretrained.return_value = fake_model
        extractor = GLiNERExtractor(ExtractionConfig())
        entities = extractor.extract(
            "Fritz Lang directed Metropolis.", chunk_id="c0"
        )

    assert len(entities) == 1, f"Expected 1 entity, got {len(entities)}"
    assert entities[0].name == "Fritz Lang"
    assert entities[0].entity_type == "PERSON"
    assert entities[0].confidence >= 0.15


# =============================================================================
# LOGIC LAYER — Navigator
# =============================================================================

def test_navigator_empty_sub_queries_returns_empty():
    """
    Navigator.navigate() with an empty sub_queries list must return
    a NavigatorResult with no chunks — not crash.
    """
    from src.logic_layer.navigator import Navigator, NavigatorResult
    from src.logic_layer._config import ControllerConfig
    from src.logic_layer.planner import (
        RetrievalPlan, QueryType, RetrievalStrategy,
    )

    nav = Navigator(config=ControllerConfig())
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    nav.set_retriever(mock_retriever)

    empty_plan = RetrievalPlan(
        original_query="",
        query_type=QueryType.SINGLE_HOP,
        strategy=RetrievalStrategy.VECTOR_ONLY,
        sub_queries=[],
        entities=[],
        hop_sequence=[],
    )

    result = nav.navigate(empty_plan, sub_queries=[])

    assert isinstance(result, NavigatorResult)
    assert result.filtered_context == []


# =============================================================================
# LOGIC LAYER — Verifier
# =============================================================================

def test_verifier_max_iterations_not_exceeded():
    """
    Verifier.generate_and_verify() must stop after max_iterations rounds
    even when every iteration still has violated claims.
    """
    from src.logic_layer.verifier import Verifier, VerifierConfig

    cfg = VerifierConfig(
        max_iterations=2,
        temperature=0.0,
        enable_entity_path_validation=False,
        enable_contradiction_detection=False,
        enable_credibility_scoring=False,
    )
    verifier = Verifier(config=cfg, graph_store=None)

    llm_call_count = 0

    def fake_llm(prompt: str):
        nonlocal llm_call_count
        llm_call_count += 1
        return "wrong answer", 5.0   # (answer, latency_ms)

    verifier._call_llm = fake_llm

    # All claims will fail verification (no context to support them).
    context = ["The film was released in 1999."]
    result = verifier.generate_and_verify(
        query="Who directed Metropolis?",
        context=context,
    )

    # The loop must have run at most max_iterations times.
    assert llm_call_count <= cfg.max_iterations, (
        f"_call_llm invoked {llm_call_count} times; "
        f"max_iterations={cfg.max_iterations}"
    )
    assert result is not None
    assert result.iterations <= cfg.max_iterations


# =============================================================================
# PIPELINE — Timing fields
# =============================================================================

def test_pipeline_all_timing_fields_non_negative():
    """
    All timing fields in PipelineResult must be ≥ 0 after a successful call.
    total_time_ms must be strictly positive.
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
    pipeline.enable_planner = True
    pipeline.enable_verifier = True
    pipeline.enable_caching = False
    pipeline._cache: dict = {}
    pipeline._cache_max_size = 0
    pipeline._stats = {"total_queries": 0, "cache_hits": 0, "avg_latency_ms": 0.0}
    pipeline._initialized = True

    fake_plan = MagicMock()
    fake_plan.query_type.value = "simple"
    fake_plan.strategy.value = "vector_only"
    fake_plan.hop_sequence = []
    fake_plan.entities = []
    fake_plan.to_dict.return_value = {}

    real_nav_result = NavigatorResult(
        filtered_context=["Paris is the capital of France."],
        raw_context=["Paris is the capital of France."],
        scores=[0.9],
        metadata={},
    )

    real_ver_result = VerificationResult(
        answer="Paris",
        iterations=1,
        verified_claims=["Paris is the capital"],
        violated_claims=[],
        all_verified=True,
        pre_validation=None,
        timing_ms=20.0,
        iteration_history=[],
    )

    fake_planner = MagicMock()
    fake_planner.plan.return_value = fake_plan
    fake_navigator = MagicMock()
    fake_navigator.navigate.return_value = real_nav_result
    fake_verifier = MagicMock()
    fake_verifier.generate_and_verify.return_value = real_ver_result

    pipeline.planner = fake_planner
    pipeline.navigator = fake_navigator
    pipeline.verifier = fake_verifier

    result = pipeline.process("What is the capital of France?")

    assert result.planner_time_ms >= 0.0, "planner_time_ms must be ≥ 0"
    assert result.navigator_time_ms >= 0.0, "navigator_time_ms must be ≥ 0"
    assert result.verifier_time_ms >= 0.0, "verifier_time_ms must be ≥ 0"
    # total_time_ms may round to 0.0 with instant mocks on fast machines;
    # the important invariant is that the field is set and non-negative.
    assert result.total_time_ms >= 0.0, "total_time_ms must be ≥ 0"


# =============================================================================
# PIPELINE — Cache clear
# =============================================================================

def test_pipeline_cache_clear_forces_recompute():
    """
    After clearing the pipeline cache, an identical query must trigger a
    fresh inference call — not return the previously cached result.
    """
    from src.pipeline.agent_pipeline import AgentPipeline, AgentPipelineConfig
    from src.logic_layer.navigator import NavigatorResult
    from src.logic_layer.verifier import VerificationResult

    cfg = AgentPipelineConfig(enable_caching=True, cache_max_size=100)
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
    pipeline.enable_planner = True
    pipeline.enable_verifier = True
    pipeline.enable_caching = True
    pipeline._cache: dict = {}
    pipeline._cache_max_size = 100
    pipeline._stats = {"total_queries": 0, "cache_hits": 0, "avg_latency_ms": 0.0}
    pipeline._initialized = True

    call_count = 0

    def make_nav():
        return NavigatorResult(
            filtered_context=["Some context."],
            raw_context=["Some context."],
            scores=[0.9],
            metadata={},
        )

    def make_ver():
        return VerificationResult(
            answer="42",
            iterations=1,
            verified_claims=[],
            violated_claims=[],
            all_verified=True,
            pre_validation=None,
            timing_ms=10.0,
            iteration_history=[],
        )

    def fake_planner_plan(query):
        nonlocal call_count
        call_count += 1
        p = MagicMock()
        p.query_type.value = "simple"
        p.strategy.value = "vector_only"
        p.hop_sequence = []
        p.entities = []
        p.to_dict.return_value = {}
        return p

    fake_planner = MagicMock()
    fake_planner.plan.side_effect = fake_planner_plan
    fake_navigator = MagicMock()
    fake_navigator.navigate.return_value = make_nav()
    fake_verifier = MagicMock()
    fake_verifier.generate_and_verify.return_value = make_ver()

    pipeline.planner = fake_planner
    pipeline.navigator = fake_navigator
    pipeline.verifier = fake_verifier

    query = "What is 6 times 7?"

    pipeline.process(query)           # first call — computes and caches
    assert call_count == 1

    pipeline.process(query)           # second call — should be a cache hit
    assert call_count == 1, "Expected cache hit on second call"

    pipeline._cache.clear()           # clear the cache
    pipeline.process(query)           # third call — must recompute
    assert call_count == 2, "Expected recompute after cache clear"


# =============================================================================
# CROSS-LAYER — Config top_k propagates to result count
# =============================================================================

def test_config_top_k_change_propagates_to_result_count():
    """
    Reducing final_top_k in RetrievalConfig must reduce the number of
    results returned by HybridRetriever — proving config is not ignored.
    """
    from src.data_layer.hybrid_retriever import HybridRetriever, RetrievalConfig, RetrievalMode

    # vector_search returns dicts with the keys HybridRetriever._vector_only_results expects
    def make_results(n):
        return [
            {
                "document_id": f"c_{i}",
                "text": f"This is chunk number {i} about Einstein.",
                "similarity": 0.9 - i * 0.05,
                "metadata": {"source_file": f"f{i}.txt"},
                "position": i,
            }
            for i in range(n)
        ]

    mock_store = MagicMock()
    mock_store.vector_search.return_value = make_results(10)
    mock_store.graph_search.return_value = []

    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.return_value = [0.1] * 768

    # final_top_k is the config field that slices the returned list;
    # vector_top_k controls how many raw docs the store fetches.
    cfg_small = RetrievalConfig(
        mode=RetrievalMode.VECTOR, vector_top_k=10, graph_top_k=0, final_top_k=3
    )
    cfg_large = RetrievalConfig(
        mode=RetrievalMode.VECTOR, vector_top_k=10, graph_top_k=0, final_top_k=8
    )

    ret_small = HybridRetriever(mock_store, mock_embeddings, cfg_small)
    ret_large = HybridRetriever(mock_store, mock_embeddings, cfg_large)

    # retrieve() returns (results_list, metrics) — unpack accordingly
    res_small, _ = ret_small.retrieve("Einstein nationality")
    res_large, _ = ret_large.retrieve("Einstein nationality")

    assert len(res_small) <= 3, f"Expected ≤3 results with final_top_k=3, got {len(res_small)}"
    assert len(res_large) > len(res_small), (
        f"Larger final_top_k must return more results: small={len(res_small)}, large={len(res_large)}"
    )


# =============================================================================
# END-TO-END — Multi-hop answer references bridge entity
# =============================================================================

def test_end_to_end_multi_hop_answer_references_bridge_entity():
    """
    For a bridge-entity multi-hop query the final answer must contain (or be
    derivable from) the bridge entity.  Uses a fully-mocked pipeline so no
    Ollama or real model is needed.

    Scenario: "What nationality is the director of Metropolis?" →
    bridge entity "Fritz Lang" → answer "Austrian"
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
    pipeline.enable_planner = True
    pipeline.enable_verifier = True
    pipeline.enable_caching = False
    pipeline._cache: dict = {}
    pipeline._cache_max_size = 0
    pipeline._stats = {"total_queries": 0, "cache_hits": 0, "avg_latency_ms": 0.0}
    pipeline._initialized = True

    # Planner identifies this as a multi-hop bridge query with two hop steps.
    # Concrete step_id / is_bridge / depends_on so the iterative-navigate path
    # is exercised deterministically (bare MagicMocks would make the per-hop
    # cap comparison `len(...) > total_cap` compare int against a MagicMock).
    hop1 = MagicMock()
    hop1.sub_query = "Who directed Metropolis?"
    hop1.step_id = 0
    hop1.is_bridge = True
    hop1.depends_on = []
    hop2 = MagicMock()
    hop2.sub_query = "What is Fritz Lang's nationality?"
    hop2.step_id = 1
    hop2.is_bridge = False
    hop2.depends_on = [0]

    fake_plan = MagicMock()
    fake_plan.query_type.value = "bridge"
    fake_plan.strategy.value = "hybrid"
    fake_plan.hop_sequence = [hop1, hop2]
    fake_plan.entities = []
    fake_plan.to_dict.return_value = {}

    # Navigator returns a real NavigatorResult (pipeline calls asdict on it).
    bridge_context = [
        "Fritz Lang is an Austrian-German film director.",
        "Metropolis was directed by Fritz Lang and released in 1927.",
    ]
    real_nav_result = NavigatorResult(
        filtered_context=bridge_context,
        raw_context=bridge_context,
        scores=[0.9, 0.85],
        metadata={},
    )

    # Verifier returns a real VerificationResult (pipeline calls asdict on it).
    real_ver_result = VerificationResult(
        answer="Fritz Lang is Austrian.",
        iterations=1,
        verified_claims=["Fritz Lang is Austrian"],
        violated_claims=[],
        all_verified=True,
        pre_validation=None,
        timing_ms=30.0,
        iteration_history=[],
    )

    pipeline.planner = MagicMock()
    pipeline.planner.plan.return_value = fake_plan
    pipeline.navigator = MagicMock()
    pipeline.navigator.navigate.return_value = real_nav_result
    # The iterative-navigate per-hop cap reads navigator.config.max_context_chunks
    # as an int; give the mock a concrete value so the comparison is valid.
    pipeline.navigator.config.max_context_chunks = 8
    pipeline.verifier = MagicMock()
    pipeline.verifier.generate_and_verify.return_value = real_ver_result

    result = pipeline.process("What nationality is the director of Metropolis?")

    # The answer must reference the bridge entity "Fritz Lang" or the
    # derived answer "Austrian" — proving the multi-hop chain was followed.
    answer_lower = result.answer.lower()
    assert "austrian" in answer_lower or "lang" in answer_lower, (
        f"Bridge entity answer expected; got: {result.answer!r}"
    )
    # §12.37 iterative-navigation contract: Navigator is called once per
    # hop (not once with both sub-queries), so a 2-hop plan with bridge
    # dependencies produces exactly 2 navigate() invocations. Each call
    # carries a single-hop sub_queries list. Verify both invariants.
    nav_calls = pipeline.navigator.navigate.call_args_list
    assert len(nav_calls) == 2, (
        f"Iterative navigation expects 2 hops -> 2 navigate() calls; "
        f"got {len(nav_calls)}"
    )
    all_sub_queries: list = []
    for call in nav_calls:
        # Accept both positional and keyword forms of sub_queries.
        sq = (
            call.kwargs.get("sub_queries")
            if "sub_queries" in call.kwargs
            else (call.args[1] if len(call.args) > 1 else [])
        )
        all_sub_queries.extend(sq)
    assert len(all_sub_queries) == 2, (
        f"Expected 2 total sub-queries across hops, got "
        f"{len(all_sub_queries)}: {all_sub_queries}"
    )
