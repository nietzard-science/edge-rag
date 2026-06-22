"""
Shared configuration dataclass for the logic layer.

Internal module -- not part of the public API. Holds ControllerConfig so
that both navigator.py and controller.py can import it without creating a
circular dependency (controller.py -> navigator.py -> controller.py).

External consumers should import ControllerConfig via the package surface:
    from src.logic_layer import ControllerConfig
or via the controller module:
    from src.logic_layer.controller import ControllerConfig

Last reviewed: 2026-05-27 (audit pass, project version 5.4).
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict

__all__ = ["ControllerConfig"]


@dataclass
class ControllerConfig:
    """
    Configuration for the AgenticController pipeline.

    EMERGENCY FALLBACKS ONLY
    ------------------------
    The defaults below are *emergency fallbacks* exercised by test fixtures
    and by callers that construct ``ControllerConfig()`` without arguments.
    They intentionally lag behind ``config/settings.yaml`` -- updating them
    in lockstep with settings.yaml would silently shift behaviour for every
    bare-constructor caller (notably ``test_logic_layer.py``,
    ``test_verifier_semantic.py``, and ``test_config_robustness.py``,
    whose assertions reference the dataclass defaults verbatim).

    For paper-faithful values, callers must construct via
    ``ControllerConfig.from_yaml(settings)`` -- not the bare constructor.

    Known drift between dataclass defaults and current settings.yaml
    (as of 2026-05-25, project version 5.4):

        Field                            | dataclass | settings.yaml
        ---------------------------------+-----------+--------------
        max_verification_iterations      |     2     |       1
        relevance_threshold_factor       |   0.85    |      0.6
        max_context_chunks               |    10     |       8
        top_k_per_subquery               |    10     |      20
        max_chars_per_doc                |   500     |     800
        enable_contradiction_filter      |  False    |     True
        enable_reranker                  |  False    |     True

    All other fields are aligned. The drift list is part of the contract:
    a reviewer reading the dataclass defaults learns the test-fixture
    baseline, not the production-evaluation values.

    LLM Settings:
        model_name: Ollama model for S_V (e.g. "qwen2:1.5b").
        base_url: Ollama API endpoint.
        temperature: Sampling temperature (0.0 = fully deterministic).

    Pipeline Settings:
        max_verification_iterations: Maximum self-correction rounds.
            Reference: Madaan et al. (2023). "Self-Refine." NeurIPS 2023.

    Navigator Settings (pre-generative filtering, paper section 3.3):
        relevance_threshold_factor: Dynamic relevance threshold multiplier.
        redundancy_threshold: Jaccard deduplication threshold.
        max_context_chunks: Maximum chunks passed to S_V after filtering.
        rrf_k: RRF smoothing constant (Cormack et al., 2009. SIGIR).
        top_k_per_subquery: Retrieval results kept per sub-query.
        max_chars_per_doc: Per-chunk truncation limit for S_V prompt.
        corroboration_source_weight: RRF boost per additional unique source.
        corroboration_query_weight: RRF boost per additional sub-query hit.
        contradiction_overlap_threshold: Word-overlap threshold for numeric
            contradiction detection.
        contradiction_ratio_threshold: Minimum numeric ratio to flag conflict.
        contradiction_min_value: Minimum numeric value to consider for ratio.
        enable_entity_overlap_pruning / enable_entity_mention_filter /
            enable_context_shrinkage: per-filter ablation toggles (paper
            section 3.3 original-contribution filters).
        entity_mention_*: thresholds for the three-tier entity-mention
            filter (token-length floors, content-overlap floor, top-k RRF
            immunity, survivor floor).
        contradiction_number_context_window / contradiction_year_range_min /
            contradiction_year_range_max: number-classification parameters
            for the contradiction filter's year-vs-count disambiguation.
    """

    # LLM Settings -- emergency fallbacks; live values read from settings.yaml
    model_name: str = "qwen2:1.5b"           # settings.yaml: llm.model_name
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0

    # Pipeline Settings
    max_verification_iterations: int = 2      # settings.yaml: agent.max_verification_iterations (current: 1)

    # Navigator Settings
    relevance_threshold_factor: float = 0.85  # settings.yaml: navigator.relevance_threshold_factor (current: 0.6)
    redundancy_threshold: float = 0.8         # settings.yaml: navigator.redundancy_threshold
    max_context_chunks: int = 10              # settings.yaml: navigator.max_context_chunks (current: 8)
    rrf_k: int = 60                           # settings.yaml: navigator.rrf_k
    top_k_per_subquery: int = 10              # settings.yaml: navigator.top_k_per_subquery (current: 20)
    max_chars_per_doc: int = 500              # settings.yaml: llm.max_chars_per_doc (current: 800)
    corroboration_source_weight: float = 0.1  # settings.yaml: navigator.corroboration_source_weight
    corroboration_query_weight: float = 0.05  # settings.yaml: navigator.corroboration_query_weight
    enable_contradiction_filter: bool = False        # settings.yaml: navigator.enable_contradiction_filter (current: true)
    contradiction_overlap_threshold: float = 0.3   # settings.yaml: navigator.contradiction_overlap_threshold
    contradiction_ratio_threshold: float = 2.0     # settings.yaml: navigator.contradiction_ratio_threshold
    contradiction_min_value: float = 100.0         # settings.yaml: navigator.contradiction_min_value

    # Cross-encoder reranker (§11.7)
    # Default-disabled in the dataclass (22 MB model download on first use);
    # settings.yaml currently has enable_reranker: true for the paper runs.
    enable_reranker: bool = False        # settings.yaml: navigator.enable_reranker (current: true)
    reranker_model: str = (              # settings.yaml: navigator.reranker_model
        "BAAI/bge-reranker-base"         # 2026-05-27: upgraded from ms-marco-MiniLM-L-6-v2 (~22M) to bge-reranker-base (~278M)
    )
    reranker_top_k: int = 10            # settings.yaml: navigator.reranker_top_k
    # 2026-05-27: support for a larger cross-encoder (e.g. BAAI/bge-reranker-base
    # ~278M or bge-reranker-large ~560M). The default model is unchanged, so
    # existing behaviour is byte-identical until reranker_model is overridden.
    # reranker_max_length: BGE rerankers require an explicit max_length (the
    #   MiniLM default of 512 also works for the small model).
    # reranker_fp16: load the cross-encoder in half precision to halve the
    #   memory footprint of the larger models (~1.1GB -> ~550MB for large).
    #   Default False keeps fp32 for the small model (no accuracy change).
    reranker_max_length: int = 512      # settings.yaml: navigator.reranker_max_length
    reranker_fp16: bool = False         # settings.yaml: navigator.reranker_fp16

    # Pre-generative filter toggles (paper section 3.3 ablation rows).
    # These three filters are "original contributions" the paper reports
    # per-filter ablation results for; the flags let a reviewer regenerate
    # those rows by disabling one filter at a time.
    enable_entity_overlap_pruning: bool = True   # settings.yaml: navigator.enable_entity_overlap_pruning
    enable_entity_mention_filter: bool = True    # settings.yaml: navigator.enable_entity_mention_filter
    enable_context_shrinkage: bool = True        # settings.yaml: navigator.enable_context_shrinkage

    # Entity-mention filter thresholds. These govern which chunks reach S_V
    # and therefore affect EM/SF; chosen by inspection on the development
    # split (the paper's methodology reports the dev/test partition).
    entity_mention_single_token_min: int = 5        # min chars for a single-token entity match
    entity_mention_fallback_token_min: int = 8      # min chars for a per-token fallback (multi-word entity)
    entity_mention_specific_single_min: int = 8     # single token counts as "specific" iff >= this
    entity_mention_overlap_min_fraction: float = 0.5  # query-content overlap fraction for tier-1
    entity_mention_overlap_min_abs: int = 2         # absolute query-content overlap floor for tier-1
    entity_mention_rrf_immune_top_k: int = 2        # top-k RRF chunks never dropped by this filter
    entity_mention_survivor_floor: int = 5          # top up to this many chunks for full candidate sets

    # Contradiction filter number-classification parameters (only used when
    # enable_contradiction_filter is True).
    contradiction_number_context_window: int = 25   # +/- char window for year/count context words
    contradiction_year_range_min: int = 1000        # magnitude heuristic: year lower bound
    contradiction_year_range_max: int = 2100        # magnitude heuristic: year upper bound

    def __post_init__(self) -> None:
        if self.temperature != 0.0:
            warnings.warn(
                "ControllerConfig: temperature=%g -- use 0.0 for deterministic "
                "the paper's evaluation. Set llm.temperature in config/settings.yaml."
                % self.temperature,
                stacklevel=2,
            )

    @classmethod
    def from_yaml(cls, config: Dict[str, Any]) -> "ControllerConfig":
        """
        Build a ControllerConfig from a settings.yaml dict.

        Reads the ``navigator``, ``llm``, and ``agent`` blocks. The values
        returned by this classmethod ARE the paper-faithful configuration --
        the dataclass-default fallbacks documented in the class docstring
        intentionally drift from these values.

        Args:
            config: Full settings.yaml dict (or the relevant sub-dict).

        Returns:
            ControllerConfig populated from the provided settings dict.
        """
        nav = config.get("navigator", {})
        llm = config.get("llm", {})
        agent = config.get("agent", {})
        return cls(
            model_name=llm.get("model_name", "qwen2:1.5b"),
            base_url=llm.get("base_url", "http://localhost:11434"),
            temperature=llm.get("temperature", 0.0),
            max_verification_iterations=agent.get("max_verification_iterations", 2),
            relevance_threshold_factor=nav.get("relevance_threshold_factor", 0.85),
            redundancy_threshold=nav.get("redundancy_threshold", 0.8),
            max_context_chunks=nav.get("max_context_chunks", 10),
            rrf_k=nav.get("rrf_k", 60),
            top_k_per_subquery=nav.get("top_k_per_subquery", 10),
            max_chars_per_doc=llm.get("max_chars_per_doc", 500),
            corroboration_source_weight=nav.get("corroboration_source_weight", 0.1),
            corroboration_query_weight=nav.get("corroboration_query_weight", 0.05),
            enable_contradiction_filter=nav.get("enable_contradiction_filter", False),
            contradiction_overlap_threshold=nav.get("contradiction_overlap_threshold", 0.3),
            contradiction_ratio_threshold=nav.get("contradiction_ratio_threshold", 2.0),
            contradiction_min_value=nav.get("contradiction_min_value", 100.0),
            enable_reranker=nav.get("enable_reranker", False),
            reranker_model=nav.get(
                "reranker_model", "BAAI/bge-reranker-base"
            ),
            reranker_top_k=nav.get("reranker_top_k", 10),
            reranker_max_length=nav.get("reranker_max_length", 512),
            reranker_fp16=nav.get("reranker_fp16", False),
            enable_entity_overlap_pruning=nav.get("enable_entity_overlap_pruning", True),
            enable_entity_mention_filter=nav.get("enable_entity_mention_filter", True),
            enable_context_shrinkage=nav.get("enable_context_shrinkage", True),
            entity_mention_single_token_min=nav.get("entity_mention_single_token_min", 5),
            entity_mention_fallback_token_min=nav.get("entity_mention_fallback_token_min", 8),
            entity_mention_specific_single_min=nav.get("entity_mention_specific_single_min", 8),
            entity_mention_overlap_min_fraction=nav.get("entity_mention_overlap_min_fraction", 0.5),
            entity_mention_overlap_min_abs=nav.get("entity_mention_overlap_min_abs", 2),
            entity_mention_rrf_immune_top_k=nav.get("entity_mention_rrf_immune_top_k", 2),
            entity_mention_survivor_floor=nav.get("entity_mention_survivor_floor", 5),
            contradiction_number_context_window=nav.get("contradiction_number_context_window", 25),
            contradiction_year_range_min=nav.get("contradiction_year_range_min", 1000),
            contradiction_year_range_max=nav.get("contradiction_year_range_max", 2100),
        )
