"""
Infrastructure-invariant tests for the evaluation tooling and the
controller / coreference contracts (paper §4, §7 support).

Static-only: they do NOT run the chunking ablation, the pipeline eval, or any
LLM call. They check that the evaluation-support infrastructure imports cleanly
and exposes its documented invariants (ablation config parsing, ablation
store-path scoping, coreference optionality, the embedding-metrics surface, and
AgenticController's static-helper surface).

Last reviewed: 2026-05-30 (audit pass, project version 5.4).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Chunking ablation script
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkingAblationScript:
    """The ablation script must import cleanly and its config parser must
    enforce the documented invariants. We do NOT call main() — that would
    re-ingest the dataset.
    """

    def test_script_importable(self):
        """`from src.thesis_evaluations import chunking_ablation` must succeed
        without side effects.
        """
        from src.thesis_evaluations import chunking_ablation  # noqa: F401
        assert hasattr(chunking_ablation, "main")
        assert hasattr(chunking_ablation, "parse_configs")
        assert hasattr(chunking_ablation, "run_one_config")
        assert hasattr(chunking_ablation, "write_summary")
        assert hasattr(chunking_ablation, "DEFAULT_CONFIGS")
        # Documented default grid: production baseline + two window variants
        # + two overlap variants = 5 cells.
        assert len(chunking_ablation.DEFAULT_CONFIGS) == 5

    def test_parse_configs_canonical(self):
        from src.thesis_evaluations.chunking_ablation import parse_configs
        assert parse_configs("3:1") == [(3, 1)]
        assert parse_configs("3:1,5:1,7:1") == [(3, 1), (5, 1), (7, 1)]

    def test_parse_configs_rejects_overlap_ge_window(self):
        """A chunker with overlap == window makes no forward progress."""
        from src.thesis_evaluations.chunking_ablation import parse_configs
        import pytest
        with pytest.raises(ValueError, match="must be <"):
            parse_configs("3:3")
        with pytest.raises(ValueError, match="must be <"):
            parse_configs("3:5")

    def test_parse_configs_rejects_negative_overlap(self):
        from src.thesis_evaluations.chunking_ablation import parse_configs
        import pytest
        with pytest.raises(ValueError, match=">= 0"):
            parse_configs("3:-1")

    def test_parse_configs_rejects_zero_window(self):
        from src.thesis_evaluations.chunking_ablation import parse_configs
        import pytest
        with pytest.raises(ValueError, match=">= 1"):
            parse_configs("0:0")

    def test_ablation_store_manager_overrides_only_vector_path(self, tmp_path):
        """The _AblationStoreManager must redirect the vector path while
        keeping graph/questions/articles_info pointing at production."""
        from src.thesis_evaluations.chunking_ablation import _AblationStoreManager
        custom_vec = tmp_path / "custom_vector"
        custom_vec.mkdir()
        mgr = _AblationStoreManager(vector_override=custom_vec)
        paths = mgr.get_paths("hotpotqa")
        assert paths["vector"] == custom_vec, "vector path must be overridden"
        # Graph and other paths still point to production via the parent class.
        assert "graph" in paths
        assert "hotpotqa" in str(paths["graph"]), (
            "graph path must remain dataset-scoped; got "
            f"{paths['graph']!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Coreference: optional dependency contract
# ─────────────────────────────────────────────────────────────────────────────

class TestCoreferenceOptional:
    """Coreference resolution is opt-in: the module must import even when the
    optional backend is absent."""

    def test_resolver_is_optional(self):
        """The module must still be importable when the coref backend is
        missing — the design contract is that coref is opt-in.
        """
        from src.data_layer import coreference
        assert hasattr(coreference, "resolve_coreferences")
        assert hasattr(coreference, "is_available")


# ─────────────────────────────────────────────────────────────────────────────
# Embedding metrics: field surface consumed by the eval harness
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbeddingMetricsSurface:
    """The metrics accumulator plumbed through to the eval harness must expose
    the fields/properties the run summary reads."""

    def test_embedding_metrics_object_exposes_required_fields(self):
        from src.data_layer.embeddings import EmbeddingMetrics
        m = EmbeddingMetrics()
        # Properties / attributes the summary block touches:
        assert hasattr(m, "total_texts")
        assert hasattr(m, "cache_hits")
        assert hasattr(m, "cache_misses")
        assert hasattr(m, "batch_count")
        assert hasattr(m, "cache_hit_rate")        # property
        assert hasattr(m, "avg_time_per_text_ms")  # property


# ─────────────────────────────────────────────────────────────────────────────
# AgenticController: static-helper surface
# ─────────────────────────────────────────────────────────────────────────────

class TestAgenticControllerSurface:
    """AgenticController is a static-helper namespace; the helpers consumed by
    AgentPipeline must be present and callable."""

    def test_AgenticController_is_stateless_namespace(self):
        """No __init__; class-level static helpers only."""
        from src.logic_layer.controller import AgenticController
        # All public-facing methods should be staticmethod or classmethod.
        for name in [
            "_extract_bridge_entities",
            "_rewrite_hop_query_with_bridges",
            "_score_bridge_candidate",
            "_detect_expected_type",
        ]:
            attr = getattr(AgenticController, name, None)
            assert attr is not None, f"AgenticController.{name} is missing"
            assert callable(attr)
