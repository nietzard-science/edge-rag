"""
Robustness tests for ControllerConfig.from_yaml() (paper §4, config loader).

Validates that the config loader degrades gracefully when settings.yaml keys
are missing, partially present, or contain edge-case values — negative and
boundary coverage beyond the happy path.

Run:
    pytest test_system/test_config_robustness.py -v

Last reviewed: 2026-05-30 (audit pass, project version 5.4).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logic_layer._config import ControllerConfig


# =============================================================================
# TestControllerConfigFromYaml
# =============================================================================

class TestControllerConfigFromYaml:
    """Boundary and negative tests for ControllerConfig.from_yaml()."""

    def test_empty_dict_returns_all_defaults(self) -> None:
        """from_yaml({}) must return a config with every field at its default value."""
        cfg = ControllerConfig.from_yaml({})
        assert cfg.model_name == "qwen2:1.5b"
        assert cfg.temperature == pytest.approx(0.0)
        assert cfg.rrf_k == 60
        assert cfg.max_verification_iterations == 2
        assert cfg.relevance_threshold_factor == pytest.approx(0.85)
        assert cfg.redundancy_threshold == pytest.approx(0.8)
        assert cfg.max_context_chunks == 10
        assert cfg.top_k_per_subquery == 10

    def test_missing_navigator_block_uses_nav_defaults(self) -> None:
        """When 'navigator' block is absent, all navigator fields fall back to defaults."""
        cfg = ControllerConfig.from_yaml({"llm": {"model_name": "phi3"}})
        assert cfg.rrf_k == 60
        assert cfg.max_context_chunks == 10
        assert cfg.top_k_per_subquery == 10
        assert cfg.relevance_threshold_factor == pytest.approx(0.85)

    def test_missing_llm_block_uses_llm_defaults(self) -> None:
        """When 'llm' block is absent, all LLM fields fall back to defaults."""
        cfg = ControllerConfig.from_yaml({"navigator": {"rrf_k": 30}})
        assert cfg.model_name == "qwen2:1.5b"
        assert cfg.temperature == pytest.approx(0.0)
        assert cfg.max_chars_per_doc == 500

    def test_missing_agent_block_uses_agent_defaults(self) -> None:
        """When 'agent' block is absent, max_verification_iterations falls back to 2."""
        cfg = ControllerConfig.from_yaml({"llm": {}, "navigator": {}})
        assert cfg.max_verification_iterations == 2

    def test_partial_navigator_honours_supplied_values(self) -> None:
        """Partially-specified navigator: supplied values honoured, unspecified at defaults."""
        cfg = ControllerConfig.from_yaml({
            "navigator": {
                "rrf_k": 30,
                "max_context_chunks": 5,
            }
        })
        assert cfg.rrf_k == 30
        assert cfg.max_context_chunks == 5
        # Unspecified fields must still have their defaults.
        assert cfg.relevance_threshold_factor == pytest.approx(0.85)
        assert cfg.redundancy_threshold == pytest.approx(0.8)

    def test_all_blocks_provided_uses_all_values(self) -> None:
        """When all three settings blocks are present, every supplied value is applied."""
        cfg = ControllerConfig.from_yaml({
            "llm": {
                "model_name": "phi3",
                "temperature": 0.0,
                "max_chars_per_doc": 300,
            },
            "agent": {"max_verification_iterations": 3},
            "navigator": {
                "rrf_k": 45,
                "relevance_threshold_factor": 0.9,
                "redundancy_threshold": 0.7,
                "max_context_chunks": 8,
            },
        })
        assert cfg.model_name == "phi3"
        assert cfg.max_verification_iterations == 3
        assert cfg.rrf_k == 45
        assert cfg.relevance_threshold_factor == pytest.approx(0.9)
        assert cfg.redundancy_threshold == pytest.approx(0.7)
        assert cfg.max_context_chunks == 8
        assert cfg.max_chars_per_doc == 300

    def test_unknown_keys_in_blocks_are_silently_ignored(self) -> None:
        """Extra keys in a settings block must not raise an error."""
        cfg = ControllerConfig.from_yaml({
            "navigator": {
                "rrf_k": 60,
                "nonexistent_future_key": "some_value",
            }
        })
        assert cfg.rrf_k == 60


# =============================================================================
# TestControllerConfigDefaults
# =============================================================================

class TestControllerConfigDefaults:
    """Unit tests for ControllerConfig direct-construction defaults."""

    def test_default_model_is_thesis_model(self) -> None:
        """ControllerConfig() default model must be the paper evaluation model."""
        cfg = ControllerConfig()
        assert cfg.model_name == "qwen2:1.5b"

    def test_nonzero_temperature_emits_user_warning(self) -> None:
        """ControllerConfig(temperature=0.5) must emit UserWarning about non-determinism."""
        with pytest.warns(UserWarning, match="temperature"):
            ControllerConfig(temperature=0.5)

    def test_zero_temperature_does_not_warn(self) -> None:
        """ControllerConfig(temperature=0.0) must not emit any warning."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cfg = ControllerConfig(temperature=0.0)
        assert cfg.temperature == pytest.approx(0.0)


# =============================================================================
# File-level config loading
# =============================================================================

class TestConfigFileLoading:
    """File-level config loading: graceful degradation on bad/missing files (F1+F2).

    _load_settings() is designed to never raise — it returns {} and logs on
    errors so the pipeline can always fall back to dataclass defaults.  These
    tests document and enforce that contract.
    """

    def test_malformed_yaml_returns_empty_dict_not_raises(self) -> None:
        """yaml.YAMLError during parse must return {} without propagating the exception."""
        import yaml
        from src.logic_layer._settings_loader import _load_settings
        from unittest.mock import patch
        with patch("yaml.safe_load", side_effect=yaml.YAMLError("simulated bad yaml")):
            result = _load_settings()
        assert isinstance(result, dict), "_load_settings() must always return a dict"
        assert result == {}, (
            "_load_settings() must return {} on YAML parse error; "
            f"got: {result!r}"
        )

    def test_missing_settings_file_returns_empty_dict_not_raises(self) -> None:
        """When settings.yaml does not exist, _load_settings() must return {}."""
        from src.logic_layer._settings_loader import _load_settings
        from unittest.mock import patch
        from pathlib import Path
        with patch.object(Path, "exists", return_value=False):
            result = _load_settings()
        assert isinstance(result, dict), "_load_settings() must always return a dict"
        assert result == {}, (
            "_load_settings() must return {} when settings.yaml is missing; "
            f"got: {result!r}"
        )


# =============================================================================
# Retrieval-funnel wiring (regression lock for TECHNICAL_ARCHITECTURE §11.16.4)
# =============================================================================

class TestRetrievalFunnelWiring:
    """Guard the §11.16.4 settings-wiring contract end-to-end.

    The §11.16.4 bug was a *silent* one: the evaluation pipeline hand-built
    ``RetrievalConfig`` and every unset knob fell back to the dataclass
    default, so ``vector_top_k`` / ``bm25_top_k`` ran at 10 instead of the
    documented 20 — half the retrieval funnel — with no error or warning.
    These tests make that class of regression *loud*: they assert the
    two-level contract (documented settings.yaml value + intentional
    dataclass fallback) and that the canonical factory resolves the real
    config to the documented width.

    Added 2026-06-09 (gap-analysis verification pass).
    """

    def test_settings_yaml_declares_funnel_width_20(self) -> None:
        """The shipped config/settings.yaml must declare both funnel knobs at 20.

        If someone reverts these to 10 (or deletes them), evaluation silently
        halves the candidate pool — exactly the §11.16.4 failure mode.
        """
        from src.logic_layer._settings_loader import _load_settings
        cfg = _load_settings()
        # Skip only if the repo's settings.yaml is genuinely absent (CI on a
        # config-less checkout); a present-but-wrong value must FAIL, not skip.
        if not cfg:
            pytest.skip("config/settings.yaml not present in this checkout")
        assert cfg.get("vector_store", {}).get("top_k_vectors") == 20, (
            "vector_store.top_k_vectors must be 20 in settings.yaml "
            "(§11.16.4 funnel-width contract)"
        )
        assert cfg.get("rag", {}).get("bm25_top_k") == 20, (
            "rag.bm25_top_k must be 20 in settings.yaml "
            "(§11.16.4 funnel-width contract)"
        )

    def test_dataclass_fallback_is_documented_ten(self) -> None:
        """RetrievalConfig() defaults stay 10 — the *intentional* config-less fallback.

        TECHNICAL_ARCHITECTURE §11.16.4 documents these as a fallback that the
        evaluation path never hits (it always loads settings.yaml). This test
        pins the value so the two-level contract stays coherent: change the
        default and you must also update §11.16.4.
        """
        from src.data_layer.hybrid_retriever import RetrievalConfig
        cfg = RetrievalConfig()
        assert cfg.vector_top_k == 10
        assert cfg.bm25_top_k == 10

    def test_factory_resolves_real_settings_to_width_20(self) -> None:
        """create_hybrid_retriever() applied to the real settings yields top_k=20.

        This is the actual §11.16.4 regression lock: it exercises the
        canonical factory's ``.get(...)`` resolution against the shipped
        settings.yaml. The heavy GLiNER/SpaCy query-NER extractor built in
        ``HybridRetriever.__init__`` is patched out so the test stays a fast
        unit test.
        """
        from unittest.mock import MagicMock, patch
        from src.logic_layer._settings_loader import _load_settings
        from src.data_layer import hybrid_retriever as hr

        cfg = _load_settings()
        if not cfg:
            pytest.skip("config/settings.yaml not present in this checkout")

        # hybrid_store only needs to exist; the factory reads
        # graph_store.HUB_MENTION_CAP defensively and tolerates failure.
        fake_store = MagicMock()
        fake_embeddings = MagicMock()

        with patch.object(hr, "ImprovedQueryEntityExtractor", return_value=MagicMock()):
            retriever = hr.create_hybrid_retriever(fake_store, fake_embeddings, cfg)

        assert retriever.config.vector_top_k == 20, (
            "factory must resolve vector_store.top_k_vectors=20 from settings.yaml"
        )
        assert retriever.config.bm25_top_k == 20, (
            "factory must resolve rag.bm25_top_k=20 from settings.yaml"
        )
        assert retriever.config.final_top_k == 20, (
            "final_top_k mirrors top_k_vectors (§3.5 single-source-of-truth)"
        )
