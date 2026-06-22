"""
===============================================================================
S_V: Verifier — Pre-Generation Validation and Self-Correction
===============================================================================

Paper: "Edge-RAG: Empirical Characterization of When Knowledge-Graph Lanes Add Value in CPU-Only Hybrid Retrieval"
Artifact B: Agent-Based Query Processing (Logic Layer).

Role in the pipeline
--------------------
S_V is the third agent of the S_P → S_N → S_V pipeline. It runs three
optional pre-generation checks over the Navigator's filtered context,
formats a prompt for a quantised SLM (Ollama-hosted), generates an answer,
extracts atomic claims, verifies each claim against the graph store and
the retrieved context, and — if violations remain — re-prompts the LLM
with explicit violation feedback for up to ``max_iterations`` rounds.

Pre-generation validation (three independently toggleable checks)
-----------------------------------------------------------------
1. Entity-path validation     verifies retrieved chunks cover the query
                              entities (graph lookup when a KuzuDB store
                              is available; substring fallback otherwise).
2. Source credibility scoring weighted combination of cross-reference
                              corroboration, entity-mention density, and
                              retrieval-source provenance. Chunks below
                              ``min_credibility_score`` are filtered; at
                              least one chunk is always retained.
3. Contradiction detection    NLI cross-encoder on adjacent chunk pairs
                              (DEFAULT OFF — ablation-only, requires
                              ~270 MB Transformers model and contradicts
                              the edge-deployment constraint). The
                              Navigator runs an O(n) numeric-divergence
                              heuristic upstream that is on by default.

Generation
----------
The verifier caps the working context to ``max_docs`` chunks BY THE
NAVIGATOR'S RRF ORDER, then runs ``_reorder_by_question_relevance`` only
WITHIN that kept window to mitigate small-LLM positional bias (Liu et al.
2023, "Lost in the Middle", arXiv:2307.03172). The reorder cannot evict a
chunk that survived the RRF cap. Optional RECOMP-style context
distillation condenses the kept chunks into a structured fact list before
the answer prompt (Yu et al. 2024, NAACL).

Self-correction loop
--------------------
Up to ``max_iterations`` rounds (default 2 = baseline + one correction).
Each round extracts atomic claims, verifies each claim by entity presence
(conservative proxy, NOT logical entailment — see Kryscinski et al. 2020),
and on violation re-prompts the LLM with explicit feedback. Two bounded
single-shot retry paths fire opportunistically inside an iteration:
bridge-entity exclusion (when the LLM returns a known bridge entity as
the final answer) and format-mismatch retry (when the LLM returns a
yes/no to a wh-question). Each retry fires at most once per call and is
individually toggleable via ``enable_bridge_exclusion_retry`` /
``enable_format_validation_retry``.

Exports
-------
    SourceCredibility, PreValidationResult, VerifierConfig,
    ConfidenceLevel, VerificationResult                   — data classes
    PreGenerationValidator                                — three-check stage
    Verifier                                              — main entry point
    create_verifier(cfg=None, graph_store=None,
                    enable_pre_validation=False)          — factory

References (algorithm anchors)
------------------------------
    Madaan et al. (2023). Self-Refine: Iterative Refinement with
        Self-Feedback. NeurIPS 2023. arXiv:2303.17651.
    Bowman et al. (2015). A large annotated corpus for learning natural
        language inference (SNLI). EMNLP 2015. arXiv:1508.05326.
    Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using
        Siamese BERT-Networks. EMNLP 2019. arXiv:1908.10084.
    Kryscinski et al. (2020). Evaluating the Factual Consistency of
        Abstractive Text Summarization. EMNLP 2020. arXiv:1910.12840.
    Liu et al. (2023). Lost in the Middle: How Language Models Use Long
        Contexts. TACL. arXiv:2307.03172.
    Yu et al. (2024). RECOMP: Improving Retrieval-Augmented LMs with
        Context Compression and Selective Augmentation. NAACL.
    Spärck Jones (1972); Robertson (2004). IDF and length-normalised
        term overlap as the basis of the question-relevance reorder.

Dependencies
------------
    requests       (Ollama API)
    spacy          (optional — claim extraction, NER density)
    transformers   (optional — only when enable_contradiction_detection)
    stdlib otherwise (re, math, time, logging, typing, dataclasses, enum).

Review History:
    Last Reviewed: 2026-06-13
    Review Result: 0 CRITICAL, 4 IMPORTANT, 5 RECOMMENDED (all addressed)
    Reviewer: Code Review Prompt v2.1
    Next Review: After the next self-correction / retry-path change
    Changes applied: hoisted the 3 inline retry prompts to class constants
        (BRIDGE_EXCLUSION_RETRY_PROMPT / ANTI_ABSTENTION_RETRY_PROMPT /
        GROUNDING_RETRY_PROMPT) so export_prompts.py captures them; added
        _ensure_nlp + a spacy_model config field so the SpaCy model is
        settings-driven; best-answer selection now always prefers a substantive
        answer over a disclaimer; extracted the _bounded_retry helper that
        unifies the four retry paths' fire-once/guard/keep-best skeleton;
        narrowed two bare `except Exception`; removed dead bridge_idx locals;
        hoisted the dataclasses import; documented the single-thread NLI/NLP
        assumption and the credibility-filter informational-only default; fixed
        the _is_format_mismatch / _qa_to_hypothesis docstrings. No CRITICAL.
    ---
    Previous Review: 2026-06-01 (audit pass, project version 5.5)
    Previous Result: no itemized action list recorded
===============================================================================
"""

import logging
import math
import re
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import requests

logger = logging.getLogger(__name__)


from ._settings_loader import _load_settings
from ._text_utils import _PROPER_NOUN_RE


# =============================================================================
# OPTIONAL DEPENDENCIES
# =============================================================================

_DEFAULT_SPACY_MODEL = "en_core_web_sm"
# Name of the SpaCy model currently held in the module-global ``NLP`` (None if
# none is loaded). Used by ``_ensure_nlp`` to avoid reloading the same model.
_LOADED_SPACY_MODEL: Optional[str] = None

# SpaCy: used for sentence splitting in _extract_claims and for NER-based
# entity density in _compute_credibility.
try:
    import spacy
    _SPACY_IMPORTED = True
except ImportError:
    spacy = None        # type: ignore[assignment]
    _SPACY_IMPORTED = False
    logger.warning(
        "SpaCy not installed. Regex fallbacks will be used for claim "
        "extraction and credibility scoring. Install with: pip install spacy"
    )


def _load_spacy_model(model_name: str):
    """Load a SpaCy model, returning the pipeline or None on failure.

    Centralises the load so the import-time default and the config-driven
    reload (``_ensure_nlp``) share one code path. Never raises: a missing model
    logs a warning and returns None so the verifier degrades to its regex
    fallbacks rather than crashing.
    """
    if not _SPACY_IMPORTED:
        return None
    try:
        return spacy.load(model_name)
    except (OSError, IOError):
        logger.warning(
            "SpaCy model '%s' not found. Install with:\n"
            "  python -m spacy download %s\n"
            "Regex fallbacks will be used for claim extraction and "
            "credibility scoring.",
            model_name, model_name,
        )
        return None


def _ensure_nlp(model_name: Optional[str]) -> None:
    """Ensure the module-global ``NLP``/``SPACY_AVAILABLE`` reflect ``model_name``.

    Fixes the import-time config-bypass (review 2026-06-13, finding #2): the
    model was previously hard-loaded as ``en_core_web_sm`` at import, so the
    configured SpaCy model could never take effect. ``Verifier.__init__`` now
    calls this with ``config.spacy_model``; if it differs from what is already
    loaded, the model is (re)loaded and the globals are rebound. Mirrors the
    planner's ``_ensure_nlp``. No-op when the requested model is already loaded.
    """
    global NLP, SPACY_AVAILABLE, _LOADED_SPACY_MODEL
    target = model_name or _DEFAULT_SPACY_MODEL
    if target == _LOADED_SPACY_MODEL and NLP is not None:
        return
    loaded = _load_spacy_model(target)
    if loaded is not None:
        NLP = loaded
        SPACY_AVAILABLE = True
        _LOADED_SPACY_MODEL = target
        logger.info("SpaCy model '%s' loaded for claim extraction", target)
    elif NLP is None:
        # Only downgrade availability if nothing is loaded at all; keep a
        # previously-loaded model rather than dropping to regex on a bad reload.
        SPACY_AVAILABLE = False


# Import-time default load (preserves prior behaviour for callers that use the
# module — and the module-global NLP — without constructing a Verifier).
NLP = _load_spacy_model(_DEFAULT_SPACY_MODEL)
SPACY_AVAILABLE = NLP is not None
if SPACY_AVAILABLE:
    _LOADED_SPACY_MODEL = _DEFAULT_SPACY_MODEL
    logger.info("SpaCy model '%s' loaded for claim extraction", _DEFAULT_SPACY_MODEL)

# Transformers: used only when enable_contradiction_detection is True.
# The NLI model (~270 MB) is lazy-loaded on first use.
try:
    from transformers import pipeline as _hf_pipeline
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers available for NLI contradiction detection")
except ImportError:
    _hf_pipeline = None  # type: ignore[assignment]
    TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "Transformers not available; NLI contradiction detection disabled. "
        "Install with: pip install transformers"
    )


# =============================================================================
# DATA STRUCTURES
# =============================================================================


class ValidationStatus(Enum):
    """Status codes returned by the pre-generation validation stage."""

    PASSED = "passed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONTRADICTION_DETECTED = "contradiction_detected"
    LOW_CREDIBILITY = "low_credibility"


@dataclass
class SourceCredibility:
    """
    Credibility score for a single context chunk.

    Computes a weighted combination of three signals:

    - ``cross_references``: how often information in this chunk is corroborated
      by other chunks (weight: ``credibility_weight_cross_ref``, default 40 %).
    - ``entity_frequency``: named-entity mention density as a proxy for
      information richness (weight: ``credibility_weight_entity_freq``,
      default 30 %).
    - ``retrieval_provenance``: graph-retrieved chunks score higher than
      vector/BM25-only chunks (weight: ``credibility_weight_provenance``,
      default 30 %). Real provenance is supplied via
      ``chunk_is_graph_based`` from the Navigator; when absent, a constant
      baseline (0.5) is applied so the term degenerates to a uniform
      offset rather than crashing.

    Defense of the 40 / 30 / 30 weights:
        The weights are documented as a deliberate inspection-time choice
        rather than the output of a grid-search calibration.  Cross-reference
        corroboration receives the largest share because two independent
        chunks agreeing on a fact is a stronger correctness signal than
        either signal in isolation (Knowledge-Vault style multi-source
        fusion, Dong et al., 2014, KDD).  Entity-frequency and
        retrieval-provenance are weighted equally because they measure
        independent dimensions (information density vs. retrieval-path
        quality) and the paper does not claim one dominates the other.

        The total contribution of the credibility filter is bounded above
        by the chunk-eviction rate at ``min_credibility_score``; on the
        paper's HotpotQA evaluation this filter evicts < 10 % of chunks,
        so the weights' individual influence on final EM/F1 is small.
        The paper reports a single ablation row ("Verifier w/o credibility
        filter") rather than a weight-sweep, which is methodologically
        sufficient at this scale.

    The Navigator forwards per-chunk retrieval-source metadata via the
    ``chunk_is_graph_based`` parameter on ``PreGenerationValidator.validate``;
    when absent (e.g. legacy callers), the constant baseline is used.
    """

    text: str
    score: float = 0.5
    cross_references: int = 0
    entity_frequency: float = 0.0
    is_graph_based: bool = False

    def compute_score(
        self,
        weight_cross_ref: float = 0.4,
        weight_entity_freq: float = 0.3,
        weight_provenance: float = 0.3,
        cross_ref_max: float = 3.0,
        provenance_baseline: float = 0.5,
    ) -> float:
        """
        Compute the weighted credibility score and store it in ``self.score``.

        Parameters
        ----------
        weight_cross_ref, weight_entity_freq, weight_provenance :
            Signal weights (should sum to 1.0).
        cross_ref_max :
            Divisor for normalising ``cross_references`` to [0, 1].
        provenance_baseline :
            Score assigned to vector-only (non-graph) sources.

        Returns
        -------
        float
            Credibility score in [0, 1].
        """
        ref_score = min(1.0, self.cross_references / max(1.0, cross_ref_max))
        entity_score = self.entity_frequency
        provenance_score = 1.0 if self.is_graph_based else provenance_baseline
        self.score = (
            weight_cross_ref * ref_score
            + weight_entity_freq * entity_score
            + weight_provenance * provenance_score
        )
        return self.score


@dataclass
class PreValidationResult:
    """
    Result of the pre-generation validation stage.

    Attributes
    ----------
    status :
        Overall validation outcome.
    entity_path_valid :
        Whether retrieved chunks cover query entities.
    contradictions :
        Index-based contradiction pairs: ``(idx1, idx2, score)``.
    filtered_context :
        Context chunks that passed all validation filters.
    credibility_scores :
        Per-chunk credibility scores aligned with ``filtered_context``.
    validation_time_ms :
        Wall-clock time for the validation stage in milliseconds.
    details :
        Per-step diagnostic information.
    """

    status: ValidationStatus = ValidationStatus.PASSED
    entity_path_valid: bool = True
    contradictions: List[Tuple[int, int, float]] = field(default_factory=list)
    filtered_context: List[str] = field(default_factory=list)
    credibility_scores: List[float] = field(default_factory=list)
    validation_time_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifierConfig:
    """
    Configuration for the Verifier stage (S_V).

    All fields are loaded from config/settings.yaml via ``from_yaml()``.
    Hardcoded defaults serve as emergency fallbacks only and match the
    the paper's evaluation settings documented in settings.yaml.

    LLM settings
    ------------
    model_name : Ollama model name.
    base_url : Ollama API endpoint.
    temperature : Sampling temperature (0.0 = fully deterministic).
    max_tokens : Maximum answer tokens.
    timeout : HTTP timeout in seconds for a single Ollama call.

    Context settings (settings.yaml: llm.*)
    ----------------------------------------
    max_context_chars : Total character budget for the prompt context.
    max_docs : Maximum chunks forwarded to the LLM.
    max_chars_per_doc : Per-chunk truncation limit.

    Pre-validation settings (settings.yaml: verifier.*)
    -----------------------------------------------------
    enable_entity_path_validation, enable_contradiction_detection,
    enable_credibility_scoring : Feature flags for the three checks.
    contradiction_threshold : NLI confidence threshold.
    min_credibility_score : Chunks below this score are removed.
    entity_coverage_threshold : Minimum entity-coverage fraction for
        entity-path validation to pass.
    nli_model : HuggingFace model ID for NLI.
        Reference: Reimers & Gurevych (2019). arXiv:1908.10084.
    nli_max_input_chars : Per-chunk truncation before NLI inference.
    spacy_max_chars : Per-chunk truncation before SpaCy processing.

    Credibility scoring weights (settings.yaml: verifier.credibility_*)
    ----------------------------------------------------------------------
    Weights must sum to 1.0.

    Agentic loop settings (settings.yaml: agent.*)
    -----------------------------------------------
    max_iterations : Maximum self-correction rounds.
        1 = generation only (ablation baseline).
        2 = paper default (one correction round).
        Reference: Madaan et al. (2023). arXiv:2303.17651.
    stop_on_first_success : Exit when all claims are verified.

    Confidence thresholds (settings.yaml: verifier.confidence_*)
    --------------------------------------------------------------
    confidence_high_threshold : Verified-ratio >= this → HIGH.
    confidence_medium_threshold : Verified-ratio >= this → MEDIUM.

    Claim extraction limits (settings.yaml: verifier.*)
    -----------------------------------------------------
    min_claim_chars : Minimum characters for a sentence to count as a claim.
    max_entities_to_verify : Maximum entities checked per claim.
    max_key_phrases : Maximum key phrases per chunk in cross-reference scoring.
    """

    # LLM settings — emergency fallbacks; live values read from settings.yaml via from_yaml()
    model_name: str = "qwen2:1.5b"          # settings.yaml: llm.model_name
    base_url: str = "http://localhost:11434"
    # SpaCy model for claim extraction + NER-density credibility (settings.yaml:
    # ingestion.spacy_model). Honoured via _ensure_nlp in Verifier.__init__.
    spacy_model: str = "en_core_web_sm"
    temperature: float = 0.0
    max_tokens: int = 200
    timeout: int = 60

    # Context settings
    max_context_chars: int = 900
    max_docs: int = 5
    max_chars_per_doc: int = 500             # settings.yaml: llm.max_chars_per_doc

    # Pre-validation flags
    enable_entity_path_validation: bool = True
    enable_contradiction_detection: bool = False
    enable_credibility_scoring: bool = True

    # Credibility filtering mode (2026-05-28). When False (default, SAFE),
    # credibility scoring is INFORMATIONAL only — scores are computed and
    # logged but never evict a chunk. The downstream max_docs cap (RRF order)
    # owns set membership, so a top-RRF gold chunk with a low credibility
    # score is no longer silently dropped before the LLM sees it (this was a
    # silent gold-eviction path: a top-ranked answer chunk could score low on
    # the credibility heuristic and be evicted before generation). Set True to
    # restore the legacy drop-below-min_credibility behaviour for ablation.
    credibility_filter_drop: bool = False

    # Pre-validation parameters
    contradiction_threshold: float = 0.85
    min_credibility_score: float = 0.5
    entity_coverage_threshold: float = 0.5
    nli_model: str = "cross-encoder/nli-distilroberta-base"
    nli_max_input_chars: int = 200
    spacy_max_chars: int = 500

    # Credibility scoring weights
    credibility_weight_cross_ref: float = 0.4
    credibility_weight_entity_freq: float = 0.3
    credibility_weight_provenance: float = 0.3
    credibility_cross_ref_max: float = 3.0
    credibility_provenance_baseline: float = 0.5

    # Agentic loop
    max_iterations: int = 2

    # Confidence thresholds
    confidence_high_threshold: float = 0.8
    confidence_medium_threshold: float = 0.5

    # Claim extraction limits
    min_claim_chars: int = 15
    max_entities_to_verify: int = 5
    max_key_phrases: int = 10

    # Heuristic contradiction threshold (Finding 6)
    heuristic_contradiction_threshold: float = 0.5  # settings.yaml: verifier.heuristic_contradiction_threshold

    # Sentence-boundary fraction for context truncation (Finding 7)
    format_sentence_boundary_fraction: float = 0.7  # settings.yaml: verifier.format_sentence_boundary_fraction

    # Entity-density normalizers for credibility scoring (Finding 8)
    credibility_entity_freq_normalizer_spacy: float = 5.0   # settings.yaml: verifier.credibility_entity_freq_normalizer_spacy
    credibility_entity_freq_normalizer_regex: float = 10.0  # settings.yaml: verifier.credibility_entity_freq_normalizer_regex

    # Context-distillation / fact-extraction step.
    # When True, inserts one extra LLM call between the reorder and the
    # final answer call to condense the retrieved chunks into a structured
    # fact list. Targets Lost-in-the-Middle (Liu 2023) at SLM scale by
    # presenting the LLM a clean, ordered fact set instead of N noisy
    # chunks. Reference: Yu et al. 2024 RECOMP (NAACL).
    # NOTE on provenance: default ON was selected from a 50-question
    # development-split ablation diagnostic. The paper's methodology
    # section reports the dev-vs-test partition explicitly.
    # Cost: ~5-15s extra latency per query; observed EM gain +5-8pp on dev.
    enable_context_distillation: bool = True
    context_distillation_min_input_chars: int = 200   # skip below this length
    context_distillation_min_output_chars: int = 30   # treat shorter outputs as no-op

    # Calibrated format-mismatch retry. When True, detects wh-question →
    # yes/no answer mismatches and re-prompts the LLM with an explicit
    # format instruction. Bounded to one retry per query.
    # Provenance: same 50-question dev-split diagnostic — see note above.
    enable_format_validation_retry: bool = True

    # Bridge-entity exclusion retry. When True, detects when the LLM
    # returns one of the known bridge entities as the final answer and
    # re-prompts with an explicit exclusion list. Bounded to one retry
    # per query. Provenance: 50-question dev-split diagnostic identified
    # 3 / 9 row3 "bridge-as-final-answer" failures.
    enable_bridge_exclusion_retry: bool = True
    bridge_substring_min_length: int = 5   # min len for substring bridge-hit
    bridge_exclusion_top_k: int = 5        # entities listed in the retry prompt

    # Structured (slot-filling) Chain-of-Thought (2026-05-27). When True, the
    # answer prompts are replaced with step-by-step templates that constrain
    # the SLM to fill reasoning slots before a FINAL ANSWER marker. The final
    # answer is parsed out before claim verification, so claims stay grounded
    # in retrieved chunks, not the reasoning steps. Default False (opt-in).
    # Refs: Khot et al. 2023 (Decomposed Prompting, ICLR); Wei et al. 2022
    # (free-form CoT only helps >10B -> scaffolded slots for SLMs).
    enable_structured_cot: bool = False
    cot_max_tokens: int = 400              # multi-step output needs more room

    # ── Answer verification upgrades (2026-05-28) ─────────────────────────────
    # Evidence: verifier_failure_taxonomy.py on the n=50 no_cot sweep bucketed
    # the verifier-fault wrong answers as 10 grounded-hallucination
    # (answer is a context entity, but the WRONG one for the question) + 5
    # false-abstention (gold present, model said "I don't know"). These two
    # opt-in checks target those buckets. Both reuse the NLI cross-encoder
    # already configured for contradiction detection (nli_model) and the
    # bounded one-retry-per-query pattern of the existing retries — no extra
    # latency beyond at most one re-prompt when a check fires.

    # QA-conditioned NLI grounding gate. After the answer is finalised, the
    # (question, answer) pair is turned into a declarative hypothesis and
    # checked for entailment against the retrieved chunks (premise). A
    # wrong-but-grounded answer entails poorly (the context supports a
    # DIFFERENT answer to this question), so a sub-threshold entailment
    # triggers one bounded grounding retry. Plain answer-grounding would not
    # catch this (the wrong entity IS in the context); conditioning on the
    # question is what distinguishes it. Refs: Bowman et al. 2015 (NLI);
    # Chen et al. 2021 (QA-to-declarative for NLI verification).
    enable_nli_grounding_gate: bool = False
    nli_grounding_entail_threshold: float = 0.5   # entailment prob below -> retry
    nli_grounding_top_chunks: int = 4             # chunks scored as premise (max-pooled)

    # Anti-abstention retry. When the answer is an epistemic disclaimer
    # ("I don't know") BUT pre-validation found query entities present in the
    # context, re-prompt once with an explicit "the answer is in the context,
    # extract it" instruction. Targets the false-abstention bucket where the
    # SLM gives up despite the gold paragraph being present.
    enable_anti_abstention_retry: bool = False

    # Question-relevance reorder of the kept context window.
    # Mitigates small-LLM positional bias (Liu et al. 2023 "Lost in the
    # Middle"). The reorder cannot evict a chunk; the Navigator RRF
    # ranking owns set membership.
    enable_question_relevance_reorder: bool = True
    query_keyword_min_length: int = 4       # \b\w{N,}\b in tokeniser
    idf_min_candidates: int = 4             # below this, IDF degenerates
    distinctive_entity_min_length: int = 8  # single-token entity coverage floor
    length_norm_exponent: float = 0.5       # sqrt-length normaliser

    # Token-grounding check for claims with no extractable proper noun.
    short_claim_max_tokens: int = 6           # treated as "short factual claim"
    claim_content_token_min_length: int = 2   # min len for a non-numeric token

    @classmethod
    def from_yaml(cls, config: Dict[str, Any]) -> "VerifierConfig":
        """
        Build a VerifierConfig from a settings.yaml dict.

        Reads the ``llm``, ``agent``, and ``verifier`` blocks.  All
        defaults match the paper's evaluation settings in settings.yaml and
        serve as emergency fallbacks when a block is absent.  Follows the
        same pattern as PlannerConfig.from_yaml().

        Args:
            config: Full settings.yaml dict (or any compatible sub-dict).

        Returns:
            VerifierConfig populated from the provided settings dict.
        """
        llm = config.get("llm", {})
        agent = config.get("agent", {})
        v = config.get("verifier", {})
        ingestion = config.get("ingestion", {})
        return cls(
            model_name=llm.get("model_name", "qwen2:1.5b"),
            base_url=llm.get("base_url", "http://localhost:11434"),
            spacy_model=ingestion.get("spacy_model", "en_core_web_sm"),
            temperature=llm.get("temperature", 0.0),
            max_tokens=llm.get("max_tokens", 200),
            timeout=llm.get("timeout", 60),
            max_context_chars=llm.get("max_context_chars", 900),
            max_docs=llm.get("max_docs", 5),
            max_chars_per_doc=llm.get("max_chars_per_doc", 500),
            max_iterations=agent.get("max_verification_iterations", 2),
            enable_context_distillation=v.get("enable_context_distillation", True),
            context_distillation_min_input_chars=v.get(
                "context_distillation_min_input_chars", 200,
            ),
            context_distillation_min_output_chars=v.get(
                "context_distillation_min_output_chars", 30,
            ),
            enable_format_validation_retry=v.get("enable_format_validation_retry", True),
            enable_bridge_exclusion_retry=v.get("enable_bridge_exclusion_retry", True),
            bridge_substring_min_length=v.get("bridge_substring_min_length", 5),
            bridge_exclusion_top_k=v.get("bridge_exclusion_top_k", 5),
            enable_structured_cot=v.get("enable_structured_cot", False),
            cot_max_tokens=v.get("cot_max_tokens", 400),
            enable_nli_grounding_gate=v.get("enable_nli_grounding_gate", False),
            nli_grounding_entail_threshold=v.get("nli_grounding_entail_threshold", 0.5),
            nli_grounding_top_chunks=v.get("nli_grounding_top_chunks", 4),
            enable_anti_abstention_retry=v.get("enable_anti_abstention_retry", False),
            enable_question_relevance_reorder=v.get(
                "enable_question_relevance_reorder", True,
            ),
            query_keyword_min_length=v.get("query_keyword_min_length", 4),
            idf_min_candidates=v.get("idf_min_candidates", 4),
            distinctive_entity_min_length=v.get("distinctive_entity_min_length", 8),
            length_norm_exponent=v.get("length_norm_exponent", 0.5),
            short_claim_max_tokens=v.get("short_claim_max_tokens", 6),
            claim_content_token_min_length=v.get("claim_content_token_min_length", 2),
            enable_entity_path_validation=v.get("enable_entity_path_validation", True),
            enable_contradiction_detection=v.get("enable_contradiction_detection", False),
            enable_credibility_scoring=v.get("enable_credibility_scoring", True),
            credibility_filter_drop=v.get("credibility_filter_drop", False),
            contradiction_threshold=v.get("contradiction_threshold", 0.85),
            min_credibility_score=v.get("min_credibility_score", 0.5),
            entity_coverage_threshold=v.get("entity_coverage_threshold", 0.5),
            nli_model=v.get("nli_model", "cross-encoder/nli-distilroberta-base"),
            nli_max_input_chars=v.get("nli_max_input_chars", 200),
            spacy_max_chars=v.get("spacy_max_chars", 500),
            credibility_weight_cross_ref=v.get("credibility_weight_cross_ref", 0.4),
            credibility_weight_entity_freq=v.get("credibility_weight_entity_freq", 0.3),
            credibility_weight_provenance=v.get("credibility_weight_provenance", 0.3),
            credibility_cross_ref_max=v.get("credibility_cross_ref_max", 3.0),
            credibility_provenance_baseline=v.get("credibility_provenance_baseline", 0.5),
            confidence_high_threshold=v.get("confidence_high_threshold", 0.8),
            confidence_medium_threshold=v.get("confidence_medium_threshold", 0.5),
            min_claim_chars=v.get("min_claim_chars", 15),
            max_entities_to_verify=v.get("max_entities_to_verify", 5),
            max_key_phrases=v.get("max_key_phrases", 10),
            heuristic_contradiction_threshold=v.get("heuristic_contradiction_threshold", 0.5),
            format_sentence_boundary_fraction=v.get("format_sentence_boundary_fraction", 0.7),
            credibility_entity_freq_normalizer_spacy=v.get("credibility_entity_freq_normalizer_spacy", 5.0),
            credibility_entity_freq_normalizer_regex=v.get("credibility_entity_freq_normalizer_regex", 10.0),
        )


class ConfidenceLevel(Enum):
    """Confidence level derived from the fraction of verified claims."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class VerificationResult:
    """
    Result of the verification stage.

    Attributes
    ----------
    answer :
        Generated (and potentially self-corrected) answer string.
    iterations :
        Number of self-correction iterations executed.
    verified_claims :
        Claims whose entities were found in graph or context.
    violated_claims :
        Claims that could not be verified.
    all_verified :
        True when all extracted claims passed verification.
    pre_validation :
        Output of the pre-generation validation stage.
    timing_ms :
        Total wall-clock time in milliseconds.
    iteration_history :
        Per-iteration diagnostics (answer, claims, latency, error flag).
    retry_fired :
        Per-call flags recording whether the bounded retry paths fired
        (``bridge_exclusion``, ``format_mismatch``). Visible to reviewers
        running ablations so they can attribute EM gains to a specific
        retry path.
    confidence_high_threshold :
        Verified-ratio threshold for HIGH confidence (stored with result
        for reproducibility independent of the config in scope at read time).
    confidence_medium_threshold :
        Verified-ratio threshold for MEDIUM confidence.
    """

    answer: str
    iterations: int
    verified_claims: List[str] = field(default_factory=list)
    violated_claims: List[str] = field(default_factory=list)
    all_verified: bool = False
    pre_validation: Optional[PreValidationResult] = None
    timing_ms: float = 0.0
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)
    retry_fired: Dict[str, bool] = field(default_factory=dict)
    confidence_high_threshold: float = 0.8
    confidence_medium_threshold: float = 0.5

    @property
    def confidence(self) -> ConfidenceLevel:
        """
        Confidence level based on the fraction of verified claims.

        Returns LOW when no claims were extracted (e.g., one-word answers
        that contain no verifiable entities).
        """
        total = len(self.verified_claims) + len(self.violated_claims)
        if total == 0:
            return ConfidenceLevel.LOW
        ratio = len(self.verified_claims) / total
        if ratio >= self.confidence_high_threshold:
            return ConfidenceLevel.HIGH
        if ratio >= self.confidence_medium_threshold:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW


# =============================================================================
# PRE-GENERATION VALIDATOR
# =============================================================================


class PreGenerationValidator:
    """
    Pre-generation validation stage for S_V.

    DEFAULT pipeline (two checks):

    1. **Entity-Path Validation** — verifies that retrieved chunks cover all
       query entities.  Uses ``find_chunks_by_entity_multihop`` when a
       KuzuDB graph store is available; falls back to substring matching.

    2. **Source Credibility Scoring** — weighted combination of
       cross-reference corroboration, entity-mention density, and retrieval
       provenance.  **Informational-only by default** (review 2026-06-13,
       finding #7): with ``credibility_filter_drop=False`` (the default) scores
       are computed and logged but NEVER evict a chunk — set membership stays
       owned by the Navigator RRF order + the ``max_docs`` cap, so a top-RRF
       gold chunk is not silently dropped for a low heuristic score. Set
       ``credibility_filter_drop=True`` to restore the legacy drop-below-
       ``min_credibility_score`` behaviour (ablation only); only then are
       chunks filtered and at least one always retained.

    ABLATION-ONLY check (default OFF):

    3. **Contradiction Detection** — pairwise NLI check on adjacent chunk
       pairs (O(n); non-adjacent pairs are not checked).
       Reference: Bowman et al. (2015). arXiv:1508.05326;
       Reimers & Gurevych (2019). arXiv:1908.10084.
       The Navigator already runs a numeric-divergence contradiction filter
       on the same context; this Verifier-side check is research-mode only.
       Enable via ``enable_contradiction_detection: true``.
    """

    # Compiled at class level — reused across all instances.
    # Matches "<CapitalisedWord> was/is/has/had <number>" for the numeric
    # contradiction heuristic.
    _NUMBER_PATTERN = re.compile(
        r"(\b[A-Z][a-z]+\b)\s+(?:was|is|has|had)\s+(\d+(?:\.\d+)?)"
    )
    # Capitalised proper-noun sequences for key-phrase extraction.
    _PROPER_NOUN_PATTERN = re.compile(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
    )
    # Numeric tokens with optional unit word.
    _NUMERIC_PHRASE_PATTERN = re.compile(r"\d+(?:\.\d+)?(?:\s+\w+)?")

    def __init__(
        self,
        config: VerifierConfig,
        graph_store: Optional[Any] = None,
    ) -> None:
        """
        Parameters
        ----------
        config : VerifierConfig
        graph_store : KuzuGraphStore or compatible, optional
            When provided, entity-path validation uses graph lookups.
        """
        self.config = config
        self.graph_store = graph_store
        # Lazy-loaded; only instantiated when contradiction detection is
        # enabled and Transformers is available.
        self._nli_pipeline: Optional[Any] = None
        logger.info(
            "PreGenerationValidator initialised: entity_path=%s, "
            "contradiction=%s, credibility=%s",
            config.enable_entity_path_validation,
            config.enable_contradiction_detection,
            config.enable_credibility_scoring,
        )

    def validate(
        self,
        context: List[str],
        query: str,
        entities: Optional[List[str]] = None,
        hop_sequence: Optional[List[Dict[str, Any]]] = None,
        chunk_is_graph_based: Optional[List[bool]] = None,
    ) -> PreValidationResult:
        """
        Run all three pre-generation validation checks sequentially.

        Parameters
        ----------
        context :
            Context chunks from the Navigator.
        query :
            Original user query (used for logging).
        entities :
            Query entities from the Planner (used by entity-path check).
        hop_sequence :
            Hop plan from the Planner.  Reserved for future graph-path
            planning; not used in the current implementation.
        chunk_is_graph_based :
            Per-chunk retrieval-provenance flag (parallel to ``context``).
            True for chunks retrieved via the KuzuDB graph path. Used by the
            credibility scorer to give graph-corroborated chunks a higher
            provenance score. ``None`` disables the provenance signal —
            the constant baseline is used instead.

        Returns
        -------
        PreValidationResult
        """
        start_time = time.time()
        result = PreValidationResult()
        # Surface the input chunk count so downstream JSONL instrumentation
        # can distinguish "pre-validation fired but inert" from
        # "pre-validation dropped chunks".
        result.details["input_chunk_count"] = len(context) if context else 0

        if not context:
            result.status = ValidationStatus.INSUFFICIENT_EVIDENCE
            result.filtered_context = []
            result.details["error"] = "No context available"
            return result

        result.filtered_context = context.copy()

        # ── Check 1: Entity-Path Validation ──────────────────────────────────
        if self.config.enable_entity_path_validation and entities:
            path_valid, path_details = self._validate_entity_path(context, entities)
            result.entity_path_valid = path_valid
            result.details["entity_path"] = path_details
            if not path_valid:
                logger.warning(
                    "Entity-path validation failed for query: '%s'", query[:60]
                )
                # Continue with generation; the INSUFFICIENT_EVIDENCE prompt
                # instructs the LLM to qualify its answer accordingly.
                result.status = ValidationStatus.INSUFFICIENT_EVIDENCE

        # ── Check 2: Contradiction Detection ─────────────────────────────────
        if self.config.enable_contradiction_detection and len(context) > 1:
            contradictions = self._detect_contradictions(context)
            result.contradictions = contradictions
            if contradictions:
                logger.warning("%d contradiction(s) detected", len(contradictions))
                result.details["contradictions"] = [
                    {"chunk1_idx": c[0], "chunk2_idx": c[1], "score": c[2]}
                    for c in contradictions
                ]
                result.filtered_context = self._resolve_contradictions(
                    context, contradictions
                )
                if len(result.filtered_context) < len(context):
                    result.status = ValidationStatus.CONTRADICTION_DETECTED

        # ── Check 3: Source Credibility Scoring ───────────────────────────────
        if self.config.enable_credibility_scoring:
            # Remap the per-chunk provenance flag onto the (possibly
            # trimmed) filtered_context by exact-text lookup against the
            # original context. ``None`` entries fall back to the constant
            # baseline inside _compute_credibility so callers that pass no
            # provenance see no behavioural change.
            filtered_graph_flags: Optional[List[bool]] = None
            if chunk_is_graph_based is not None and len(chunk_is_graph_based) == len(context):
                _by_text = {c: g for c, g in zip(context, chunk_is_graph_based)}
                filtered_graph_flags = [
                    _by_text.get(c, False) for c in result.filtered_context
                ]

            credibility_scores = self._compute_credibility(
                result.filtered_context, context, entities=entities,
                chunk_is_graph_based=filtered_graph_flags,
            )
            result.credibility_scores = credibility_scores
            if self.config.credibility_filter_drop:
                # Legacy behaviour (opt-in): evict chunks below the credibility
                # floor. Retained for ablation comparison. This is a silent
                # gold-eviction path — a top-RRF answer chunk with a low
                # credibility score is removed before the LLM sees it.
                keep_indices = [
                    i for i, score in enumerate(credibility_scores)
                    if score >= self.config.min_credibility_score
                ]
                if keep_indices:
                    result.filtered_context = [result.filtered_context[i] for i in keep_indices]
                    if filtered_graph_flags is not None:
                        filtered_graph_flags = [filtered_graph_flags[i] for i in keep_indices]
                elif credibility_scores:
                    # Always retain the highest-credibility chunk so the
                    # generator is never given an empty context.
                    best_idx = credibility_scores.index(max(credibility_scores))
                    result.filtered_context = [result.filtered_context[best_idx]]
                    if filtered_graph_flags is not None:
                        filtered_graph_flags = [filtered_graph_flags[best_idx]]
                    result.status = ValidationStatus.LOW_CREDIBILITY
            else:
                # Default (SAFE): credibility is informational only — scores are
                # recorded but no chunk is evicted. Set membership stays owned
                # by the Navigator RRF order + the downstream max_docs cap, so a
                # low-credibility gold chunk is no longer dropped before the LLM.
                logger.debug(
                    "[PreValidation] credibility informational only "
                    "(min=%.2f, scores=%s) — no chunks evicted",
                    min(credibility_scores) if credibility_scores else 0.0,
                    ["%.2f" % s for s in credibility_scores],
                )
            # Expose the final per-chunk provenance flags for callers that want
            # to log retrieval-source statistics (e.g. ablation diagnostics).
            if filtered_graph_flags is not None:
                result.details["chunk_is_graph_based"] = filtered_graph_flags

        result.validation_time_ms = (time.time() - start_time) * 1000
        logger.info(
            "Pre-validation: status=%s, context=%d/%d, time=%.0fms",
            result.status.value,
            len(result.filtered_context),
            len(context),
            result.validation_time_ms,
        )
        return result

    def _validate_entity_path(
        self,
        context: List[str],
        entities: List[str],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check whether retrieved chunks cover all query entities.

        Prefers KuzuDB ``find_chunks_by_entity_multihop``; falls back to
        ``graph_search`` (HybridStore) and then to substring matching.

        Returns
        -------
        (is_valid, details)
            ``is_valid`` is True when the fraction of found entities meets
            ``config.entity_coverage_threshold``.
        """
        details: Dict[str, Any] = {
            "entities_found": [],
            "entities_missing": [],
            "path_exists": False,
        }

        if not self.graph_store:
            # Fallback: substring matching against concatenated context.
            context_text = " ".join(context).lower()
            for entity in entities:
                if entity.lower() in context_text:
                    details["entities_found"].append(entity)
                else:
                    details["entities_missing"].append(entity)
        else:
            for entity in entities:
                try:
                    found = False
                    if hasattr(self.graph_store, "find_chunks_by_entity_multihop"):
                        results = self.graph_store.find_chunks_by_entity_multihop(
                            entity_name=entity, max_results=1
                        )
                        found = bool(results)
                    elif hasattr(self.graph_store, "graph_search"):
                        results = self.graph_store.graph_search(
                            entities=[entity], top_k=1
                        )
                        found = bool(results)
                    if found:
                        details["entities_found"].append(entity)
                    else:
                        details["entities_missing"].append(entity)
                except (AttributeError, TypeError, RuntimeError) as exc:
                    logger.warning(
                        "Entity-path graph lookup failed for '%s': %s", entity, exc
                    )
                    details["entities_missing"].append(entity)

        coverage = len(details["entities_found"]) / max(1, len(entities))
        details["coverage"] = coverage
        details["path_exists"] = coverage >= self.config.entity_coverage_threshold
        return details["path_exists"], details

    def _detect_contradictions(
        self,
        context: List[str],
    ) -> List[Tuple[int, int, float]]:
        """
        Detect contradictions between consecutive chunk pairs.

        Uses an NLI cross-encoder when Transformers is available; otherwise
        falls back to the numeric-divergence heuristic.

        Note: only consecutive pairs (i, i+1) are checked — O(n) — to stay
        within edge CPU budget.  Non-adjacent contradictions are not detected.

        Reference (NLI): Bowman et al. (2015). arXiv:1508.05326.
        Reference (model): Reimers & Gurevych (2019). arXiv:1908.10084.

        Returns
        -------
        list of (idx1, idx2, score) tuples
            Index-based so downstream resolution addresses original chunks.
        """
        if TRANSFORMERS_AVAILABLE and self.config.enable_contradiction_detection:
            try:
                if self._nli_pipeline is None:
                    self._nli_pipeline = _hf_pipeline(
                        "text-classification",
                        model=self.config.nli_model,
                        device=-1,  # CPU; HuggingFace uses -1 (not "cpu")
                    )
                contradictions: List[Tuple[int, int, float]] = []
                for i in range(len(context) - 1):
                    c1 = context[i][: self.config.nli_max_input_chars]
                    c2 = context[i + 1][: self.config.nli_max_input_chars]
                    result = self._nli_pipeline(
                        "%s [SEP] %s" % (c1, c2), truncation=True
                    )
                    if (
                        result
                        and result[0]["label"] == "CONTRADICTION"
                        and result[0]["score"] >= self.config.contradiction_threshold
                    ):
                        contradictions.append((i, i + 1, result[0]["score"]))
                return contradictions
            except (RuntimeError, OSError, ValueError) as exc:
                logger.warning(
                    "NLI contradiction detection failed (%s); "
                    "falling back to heuristic detection.",
                    exc,
                )
                return self._heuristic_contradiction_detection(context)
        else:
            logger.warning(
                "Transformers unavailable — falling back to heuristic contradiction detection."
            )
            return self._heuristic_contradiction_detection(context)

    def _heuristic_contradiction_detection(
        self,
        context: List[str],
    ) -> List[Tuple[int, int, float]]:
        """
        Numeric-divergence heuristic for offline contradiction detection.

        Flags chunk pairs where the same capitalised entity is assigned
        substantially different numeric values (relative difference > 50 %).
        Score is set to the actual divergence magnitude (capped at 1.0)
        rather than a fixed constant.

        This is a conservative approximation: it misses semantic
        contradictions and can over-fire on entities with multiple numeric
        attributes.  Appropriate only as a last-resort fallback.

        Returns
        -------
        list of (idx1, idx2, score) tuples
        """
        contradictions: List[Tuple[int, int, float]] = []
        entity_values: Dict[str, List[Tuple[int, float]]] = {}
        for i, chunk in enumerate(context):
            for entity, value in self._NUMBER_PATTERN.findall(chunk):
                entity_values.setdefault(entity, []).append((i, float(value)))
        for entity, values in entity_values.items():
            for j in range(len(values)):
                for k in range(j + 1, len(values)):
                    v1, v2 = values[j][1], values[k][1]
                    if min(v1, v2) > 0:
                        diff = abs(v1 - v2) / max(v1, v2)
                        if diff > self.config.heuristic_contradiction_threshold:
                            contradictions.append(
                                (values[j][0], values[k][0], min(1.0, diff))
                            )
        return contradictions

    def _resolve_contradictions(
        self,
        context: List[str],
        contradictions: List[Tuple[int, int, float]],
    ) -> List[str]:
        """
        Remove the most-contradicted chunks from the context.

        Uses index-based counting so the lookup is immune to string
        truncation differences between detection (chunk[:nli_max_input_chars])
        and resolution (full-length chunk strings).

        Returns the original context unchanged if filtering would remove
        all chunks.
        """
        contradiction_counts: Dict[int, int] = {}
        for idx1, idx2, _ in contradictions:
            contradiction_counts[idx1] = contradiction_counts.get(idx1, 0) + 1
            contradiction_counts[idx2] = contradiction_counts.get(idx2, 0) + 1
        if not contradiction_counts:
            return context
        max_count = max(contradiction_counts.values())
        filtered = [
            chunk
            for i, chunk in enumerate(context)
            if contradiction_counts.get(i, 0) < max_count
        ]
        return filtered if filtered else context

    def _compute_credibility(
        self,
        filtered_context: List[str],
        original_context: List[str],
        entities: Optional[List[str]] = None,
        chunk_is_graph_based: Optional[List[bool]] = None,
    ) -> List[float]:
        """
        Compute credibility scores for each chunk in ``filtered_context``.

        Three signals:
        1. Cross-references: number of other chunks sharing a key phrase
           (corroboration proxy).
        2. Entity frequency: SpaCy NER density as an information-richness
           proxy.  Regex proper-noun count as fallback.
        3. Retrieval provenance: graph-retrieved chunks get the full
           weight (1.0); vector/BM25-only chunks get
           ``credibility_provenance_baseline``. Pass ``chunk_is_graph_based``
           (parallel to ``filtered_context``) to enable the real signal;
           ``None`` falls back to the constant baseline.

        ``entities`` adds token-level cross-reference matching: if a word
        (≥4 chars) from an entity name appears in both this chunk and another
        chunk, that counts as a cross-reference even when the full entity name
        is absent as a substring.  This handles surface-form mismatches such
        as 'Terrence "Uncle Terry" Richardson' which does not contain the
        substring 'Terry Richardson' but does contain 'Terry' and 'Richardson'.
        """
        # Pre-compute entity name tokens for token-level cross-reference.
        entity_tokens_lower: List[str] = []
        if entities:
            seen_tokens: Set[str] = set()
            for name in entities:
                for tok in name.split():
                    tok_l = tok.lower()
                    if len(tok_l) >= 4 and tok_l not in seen_tokens:
                        entity_tokens_lower.append(tok_l)
                        seen_tokens.add(tok_l)

        # Align the provenance flag with filtered_context. ``None`` means
        # the caller didn't supply provenance, so every chunk is treated
        # as non-graph (the constant baseline takes over).
        if chunk_is_graph_based is not None and len(chunk_is_graph_based) != len(filtered_context):
            logger.warning(
                "chunk_is_graph_based length mismatch (%d vs %d filtered chunks); "
                "ignoring provenance signal.",
                len(chunk_is_graph_based), len(filtered_context),
            )
            chunk_is_graph_based = None

        scores: List[float] = []
        for chunk_idx, chunk in enumerate(filtered_context):
            cred = SourceCredibility(text=chunk)
            key_phrases = self._extract_key_phrases(chunk)
            chunk_lower = chunk.lower()

            # Cross-reference: any other chunk that shares a key phrase OR
            # shares an entity name token with this chunk.
            for other in original_context:
                if other != chunk:
                    other_lower = other.lower()
                    phrase_matched = any(
                        phrase.lower() in other_lower for phrase in key_phrases
                    )
                    if phrase_matched:
                        cred.cross_references += 1
                    elif entity_tokens_lower:
                        # Token-level fallback: entity word present in both chunks.
                        chunk_has = any(t in chunk_lower for t in entity_tokens_lower)
                        other_has = any(t in other_lower for t in entity_tokens_lower)
                        if chunk_has and other_has:
                            cred.cross_references += 1

            # Self-relevance boost: if this chunk itself mentions a query entity
            # it is directly on-topic regardless of what other chunks say.
            # Without this, a chunk that is the unique Hop-2 bridge target
            # scores cross_references=0 (no other Hop-2 chunk corroborates the
            # answer entity, by construction) and falls below
            # min_credibility_score even though it IS the answer paragraph.
            if entity_tokens_lower and cred.cross_references == 0:
                if any(t in chunk_lower for t in entity_tokens_lower):
                    cred.cross_references = 1

            # Entity-frequency signal.
            if SPACY_AVAILABLE and NLP:
                doc = NLP(chunk[: self.config.spacy_max_chars])
                cred.entity_frequency = min(1.0, len(doc.ents) / self.config.credibility_entity_freq_normalizer_spacy)
            else:
                proper_count = len(self._PROPER_NOUN_PATTERN.findall(chunk))
                cred.entity_frequency = min(1.0, proper_count / self.config.credibility_entity_freq_normalizer_regex)

            # Use real retrieval-provenance when supplied; otherwise fall
            # back to ``is_graph_based=False`` so the constant baseline term
            # in ``compute_score`` is applied.
            if chunk_is_graph_based is not None:
                cred.is_graph_based = bool(chunk_is_graph_based[chunk_idx])
            else:
                cred.is_graph_based = False

            scores.append(
                cred.compute_score(
                    weight_cross_ref=self.config.credibility_weight_cross_ref,
                    weight_entity_freq=self.config.credibility_weight_entity_freq,
                    weight_provenance=self.config.credibility_weight_provenance,
                    cross_ref_max=self.config.credibility_cross_ref_max,
                    provenance_baseline=self.config.credibility_provenance_baseline,
                )
            )
        return scores

    def _extract_key_phrases(self, text: str) -> List[str]:
        """
        Extract named entities and numeric phrases for cross-reference scoring.

        Returns a deterministically sorted, deduplicated list of up to
        ``config.max_key_phrases`` items so results are reproducible
        regardless of set() insertion order.
        """
        phrases: List[str] = []
        if SPACY_AVAILABLE and NLP:
            doc = NLP(text[: self.config.spacy_max_chars])
            phrases.extend(ent.text for ent in doc.ents)
        phrases.extend(self._PROPER_NOUN_PATTERN.findall(text))
        phrases.extend(self._NUMERIC_PHRASE_PATTERN.findall(text))
        # Sort before deduplication for deterministic ordering (reproducibility).
        return sorted(set(phrases))[: self.config.max_key_phrases]


# =============================================================================
# MAIN VERIFIER CLASS
# =============================================================================


class Verifier:
    """
    S_V: Verifier with pre-generation validation and self-correction.

    Primary public interface: ``generate_and_verify()``.
    """

    # ── Prompt Templates ──────────────────────────────────────────────────────

    ANSWER_PROMPT = """You are a factual QA assistant. Answer based ONLY on the context below.

Rules:
- Give the shortest possible answer: a name, place, date, number, or yes/no.
- Do NOT explain or add sentences beyond the direct answer.
- If the answer is a person, place, or thing: reply with just that name.
- If the answer is a number or statistic (e.g. population, count, year): reply with just the number.
- If the answer is yes/no: reply with just "yes" or "no".
- The answer is almost always present in the context — read every chunk carefully before concluding otherwise. Reply "I don't know" ONLY if the answer is genuinely absent after a careful read.

Context:
{context}

Question: {query}

Answer (as short as possible):"""

    BRIDGE_PROMPT = """You are a factual QA assistant. Answer based ONLY on the context below.

This is a multi-step question. Use the following reasoning chain to find the answer:
{bridge_chain}

Rules:
- Give the shortest possible answer: a name, place, date, number, or yes/no.
- Do NOT explain or add sentences beyond the direct answer.
- If the answer is a number or statistic (e.g. population, count, year): reply with just the number.
- The answer is almost always present in the context — read every chunk carefully before concluding otherwise. Reply "I don't know" ONLY if the answer is genuinely absent after a careful read.

Context:
{context}

Question: {query}

Answer (as short as possible):"""

    COMPARISON_PROMPT = """You are a factual QA assistant. Answer based ONLY on the context below.

The question compares two people or things. Follow these steps:
1. Find the relevant fact for the FIRST person/thing in the context.
2. Find the relevant fact for the SECOND person/thing in the context.
3. Compare the two facts and give the answer.

Rules:
- For yes/no questions: reply with just "yes" or "no".
- For "which one" questions: reply with just the name.
- Do NOT explain beyond the direct answer.
- The answer is almost always present in the context — read both items' facts carefully before concluding otherwise. Reply "I don't know" ONLY if the answer is genuinely absent after a careful read.

Context:
{context}

Question: {query}

Answer (as short as possible):"""

    CORRECTION_PROMPT = """Your previous answer contained unverified claims.

Unverified claims:
{violations}

Context:
{context}

Question: {query}

Give the shortest correct answer (name, place, date, or yes/no only):"""

    INSUFFICIENT_EVIDENCE_PROMPT = """Based on the available context, I could not find sufficient evidence to fully answer your question.

Context:
{context}

Question: {query}

Please provide a partial answer based on the available evidence, clearly indicating what information is missing:"""

    # Tier 1.5 (2026-05-26): RECOMP-style context distillation prompt.
    # Run before the final answer prompt to condense ~8 noisy chunks into
    # a short structured fact list. Targets Lost-in-the-Middle (Liu 2023)
    # by giving the SLM a clean prompt window.
    # Reference: Yu et al. 2024 RECOMP (NAACL).
    DISTILL_PROMPT = """You are a fact-extraction assistant. Read the context below and extract ONLY the facts directly relevant to answering the question.

Output rules:
- Bullet list of short atomic facts, one per line, prefixed with "- ".
- Each fact must be one sentence containing concrete details (names, dates, places, titles, numbers).
- Include the question's named entities verbatim where they appear in the context.
- Do NOT add opinions, transitions, or background narrative not directly supported by the context.
- If a fact is not stated in the context, do not invent it.

Question: {query}

Context:
{context}

Relevant facts:"""

    # Tier 1.5 (2026-05-26): format-mismatch retry prompt.
    # Fired when the LLM produces a yes/no or abstention answer to a
    # wh-question. Bounded to one retry per query.
    FORMAT_RETRY_PROMPT = """Question: {query}

Context:
{context}

Your previous answer "{previous_answer}" does not match the format the question requires. The question asks for a specific name, place, date, title, or thing. Look in the context for that specific entity and provide it as the answer.

Final answer:"""

    # ── Structured (slot-filling) Chain-of-Thought templates (2026-05-27) ──
    # Used when config.enable_structured_cot is True. Each template constrains
    # the SLM to fill reasoning slots, then emit a `FINAL ANSWER:` marker that
    # _extract_final_answer_from_cot() parses out. Refs: Khot et al. 2023
    # (Decomposed Prompting, ICLR); Wei et al. 2022 (free-form CoT only helps
    # >10B, so small models get scaffolded slots, not open reasoning).
    ANSWER_PROMPT_COT = """You are a factual QA assistant. Answer using ONLY the context. Work through the steps, then give the final answer on its own line.

Context:
{context}

Question: {query}

Step 1 — Named entities in the question:
Step 2 — The exact fact(s) in the context that answer the question:
FINAL ANSWER (shortest form — a name, place, date, number, or yes/no):"""

    BRIDGE_PROMPT_COT = """You are a factual QA assistant. This is a multi-step question. Answer using ONLY the context. Work through the steps, then give the final answer on its own line.

{bridge_chain}

Context:
{context}

Question: {query}

Step 1 — Named entities in the question:
Step 2 — Bridge entity (the intermediate that links the question parts), from the context:
Step 3 — The specific fact about the bridge entity that the question asks for:
FINAL ANSWER (shortest form, and DIFFERENT from the bridge entity):"""

    COMPARISON_PROMPT_COT = """You are a factual QA assistant. This is a comparison question. Answer using ONLY the context. Work through the steps, then give the final answer on its own line.

Context:
{context}

Question: {query}

Step 1 — The two (or more) items being compared:
Step 2 — The relevant attribute of each item, from the context:
Step 3 — The comparison result:
FINAL ANSWER (shortest form — the winning item, or yes/no):"""

    # Marker the CoT parser anchors on. Case-insensitive at parse time.
    _COT_FINAL_MARKER = "final answer"

    # ── Bounded-retry prompts (hoisted from inline f-strings 2026-06-13) ──────
    # These three prompts drive the bridge-exclusion, anti-abstention, and
    # QA-grounding retries inside generate_and_verify. They were previously
    # inline f-strings, which made them invisible to scripts/export_prompts.py
    # (the B-2 reproducibility artifact). Hoisted to class constants so the
    # published prompt set is complete and every prompt the model receives is
    # in one place. `.format()` placeholders mirror the original f-string slots.
    BRIDGE_EXCLUSION_RETRY_PROMPT = """Question: {query}

Context:
{context}

IMPORTANT INSTRUCTION: The following entities are intermediate bridges in the reasoning chain, NOT the final answer the question asks for: {exclusion_list}.
Find the FINAL answer that the question asks for. The final answer must be DIFFERENT from the bridge entities listed above. Look in the context for the attribute, title, name, date, or fact that the question is actually asking about.

Final answer:"""

    ANTI_ABSTENTION_RETRY_PROMPT = """Question: {query}

Context:
{context}

The context above contains information about: {entity_list}. The answer to the question IS present in the context. Do NOT say you don't know. Read carefully and give the shortest specific answer (a name, place, date, number, or yes/no).

Answer:"""

    GROUNDING_RETRY_PROMPT = """Question: {query}

Context:
{context}

Your previous answer "{previous_answer}" may not be the specific thing the question asks for. Re-read the question and the context. The answer must be the exact entity, attribute, date, or number the question requests, and it must be stated in the context. Give only that answer.

Answer:"""

    # ── Class-Level Compiled Regex Constants ──────────────────────────────
    # Hoisted from per-call compilation in _verify_claim for performance.
    # The multi-word proper-noun matcher lives in _text_utils._PROPER_NOUN_RE
    # and is referenced directly at the call sites in this file — no alias
    # needed.

    # Single capitalised proper nouns of at least 3 characters.
    _SINGLE_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")
    # Quoted strings treated as entity mentions.
    _QUOTED_RE = re.compile(r'"([^"]+)"')
    # Sentence boundary splitter for regex fallback in _extract_claims and
    # the sentence-aware truncate.
    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

    # Three distinct stopword sets are maintained on this class because
    # they serve three different purposes:
    #   - _CLAIM_STOPWORDS     excluded from proper-noun ENTITY EXTRACTION
    #                          (Capitalised tokens that match the regex but
    #                          are never factual entities worth verifying:
    #                          sentence-starters, demonyms, yes/no).
    #   - _CLAIM_VERIFY_STOPWORDS  excluded from the TOKEN-GROUNDING check
    #                          for short / numeric claims (function words
    #                          that add no factual content and would
    #                          over-violate paraphrase variation).
    #   - _QR_STOPWORDS        excluded from the QUERY-RELEVANCE REORDER
    #                          tokeniser (function words that appear in
    #                          almost every question and carry no
    #                          discriminative signal).
    # Each set is intentionally cased to match the input it filters
    # (entities are Capitalised; claim tokens and query tokens are lower-
    # cased before lookup).

    _CLAIM_STOPWORDS: frozenset = frozenset({
        "The", "This", "That", "These", "Those",
        "However", "Therefore", "Furthermore", "Moreover", "Although",
        "American", "British", "European", "Australian", "Canadian",
        "Yes", "No",
    })

    _CLAIM_VERIFY_STOPWORDS: frozenset = frozenset({
        "the", "a", "an", "of", "in", "on", "at", "to", "for", "by", "with",
        "is", "was", "are", "were", "be", "been", "being", "has", "have", "had",
        "do", "does", "did", "and", "or", "but", "if", "so",
        "it", "its", "this", "that", "these", "those",
    })

    # Per-string truncation budget for iteration_history entries written
    # to the per-question JSONL log. Across 500 questions × 2 iterations
    # × ~400-char answers + claim lists, the raw history balloons the
    # JSONL by several MB and slows down jq/pandas analysis. 200 chars
    # per string is enough to diagnose failures (LLM error sentinels,
    # hallucination first line) while keeping the file flat-text and
    # grep-friendly.
    _HISTORY_STR_TRUNCATE_CHARS: int = 200

    @classmethod
    def _truncate_history_str(cls, s: str) -> str:
        """Truncate a string for storage in iteration_history.

        Keeps `iteration_history` bounded so a long-running self-correction
        loop cannot grow unbounded debug payloads."""
        if not isinstance(s, str):
            return s
        if len(s) <= cls._HISTORY_STR_TRUNCATE_CHARS:
            return s
        return s[: cls._HISTORY_STR_TRUNCATE_CHARS] + "...[truncated]"

    @classmethod
    def _truncate_history_list(cls, items: List[str]) -> List[str]:
        """Truncate every string in a list for iteration_history storage."""
        return [cls._truncate_history_str(x) for x in items]

    # Epistemic-disclaimer phrases that signal the LLM did NOT answer.
    # When an answer matches any of these, it must not be reported as
    # HIGH confidence with all_verified=True.
    # Provenance: hand-curated from observed Ollama outputs on the
    # development split. Extending this set is a code change; the paper
    # methodology section reports the dev-vs-test partition explicitly.
    _DISCLAIMER_PATTERNS: Tuple[str, ...] = (
        "i don't know",
        "i do not know",
        "i cannot determine",
        "i can't determine",
        "i cannot find",
        "i can't find",
        "i was unable to find",
        "unable to find",
        "no information",
        "no specific information",
        "not provided in the context",
        "not provided",
        "not mentioned",
        "not specified",
        "not stated",
        "not available",
        "is not in the context",
        "the context does not",
        "the context doesn't",
        "context does not contain",
        "context doesn't contain",
        "insufficient evidence",
        "insufficient information",
        "based on the available",
        "no answer can be",
        "cannot be determined",
        "however, there is no",
    )

    @classmethod
    def _is_disclaimer_answer(cls, answer: str) -> bool:
        """Return True if the answer is an epistemic disclaimer."""
        if not answer or answer.startswith("[Error:"):
            return True
        a = answer.lower()
        return any(p in a for p in cls._DISCLAIMER_PATTERNS)

    # Wh-question vocabulary for the format-mismatch validator.
    # Wh-questions require a specific entity / fact / number, NOT yes/no.
    _WH_PREFIXES: frozenset = frozenset({
        "what", "who", "whom", "where", "when", "which",
        "whose", "how", "why",
    })

    # Yes/No answer patterns that violate wh-question format.
    # Provenance: hand-curated from observed Ollama outputs on the
    # development split (same caveat as _DISCLAIMER_PATTERNS).
    _YN_ANSWERS: frozenset = frozenset({
        "yes", "no", "yes.", "no.", "yeah", "nope", "true", "false",
        "yes, it is.", "no, it isn't.", "yes, they are.", "no, they are not.",
        "yes, they were.", "no, they were not.",
    })

    @classmethod
    def _extract_final_answer_from_cot(cls, raw: str) -> str:
        """
        Pull the final answer out of a structured-CoT response.

        The CoT templates end each chain with a `FINAL ANSWER:` marker;
        everything after the LAST such marker is the answer. Falls back to
        the last non-empty line if the marker is absent (small LLMs sometimes
        drop it), then to the whole string. Only the first line of the
        captured tail is kept, so trailing step text never leaks into the
        answer.
        """
        if not raw:
            return raw
        lowered = raw.lower()
        idx = lowered.rfind(cls._COT_FINAL_MARKER)
        if idx != -1:
            tail = raw[idx + len(cls._COT_FINAL_MARKER):]
            # Strip a leading "(...)" hint clause and a leading colon/dash.
            tail = tail.lstrip()
            if tail.startswith("("):
                close = tail.find(")")
                if close != -1:
                    tail = tail[close + 1:]
            tail = tail.lstrip(":：-—  ").strip()
            first_line = tail.splitlines()[0].strip() if tail else ""
            if first_line:
                return first_line
        # Fallback: last non-empty line of the whole response.
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        return lines[-1] if lines else raw.strip()

    @classmethod
    def _is_format_mismatch(cls, query: str, answer: str) -> bool:
        """
        Return True if the answer's format does not match the question's
        expected output. Exactly ONE case is flagged here (kept narrow to bound
        the retry budget):

          - Wh-question receiving a yes/no answer (e.g. "What is the X?"
            answered "Yes."). Open-ended interrogatives require a specific
            entity, not a polarity.

        The complementary case — a wh-question receiving an abstention — is NOT
        handled here; it is covered by the anti-abstention retry and claim
        verification elsewhere in generate_and_verify. (Docstring corrected
        2026-06-13, finding #7: the abstention case was described but never
        implemented in this method.)

        Reference: prompt-format calibration in small LLMs (Yang et al.
        2024 "Alignment for Honesty"; narrowed to format mismatch).
        """
        if not query or not answer:
            return False
        # First token of the question (skip punctuation)
        q_norm = query.strip().lstrip("\"'(").lower()
        first = q_norm.split(maxsplit=1)[0] if q_norm else ""
        first = first.rstrip("?.,!")
        is_wh = first in cls._WH_PREFIXES
        # Normalised answer
        a_norm = answer.strip().lower().rstrip(".!?,;:").strip('"\'')
        is_yn = a_norm in cls._YN_ANSWERS or a_norm in {"yes", "no"}
        return bool(is_wh and is_yn)

    def __init__(
        self,
        config: Optional[VerifierConfig] = None,
        graph_store: Optional[Any] = None,
    ) -> None:
        """
        Parameters
        ----------
        config : VerifierConfig, optional
            Uses VerifierConfig() defaults when None.
        graph_store : KuzuGraphStore or compatible, optional
            Injected for claim verification and entity-path validation.
            Can be set or replaced later via ``set_graph_store()``.
        """
        self.config = config or VerifierConfig()
        # Honour the configured SpaCy model (settings.yaml → ingestion.spacy_model)
        # rather than the import-time default — review 2026-06-13, finding #2.
        # Reloads the module-global NLP only if the configured model differs.
        _ensure_nlp(self.config.spacy_model)
        self.graph_store = graph_store
        self.pre_validator = PreGenerationValidator(self.config, graph_store)
        # Lazy NLI text-classification pipeline for the QA-conditioned
        # grounding gate. Separate instance from the pre-validator's
        # contradiction pipeline so each owns its own lifecycle; both load the
        # same nli_model from HF cache, so the second load is cheap.
        #
        # THREAD-SAFETY (finding #9): this lazy pipeline, the pre-validator's
        # _nli_pipeline, and the module-global NLP are initialised on first use
        # without a lock. A single Verifier instance is therefore intended for
        # single-threaded use (the edge pipeline processes one query at a time);
        # concurrent calls could double-load a pipeline (benign, idempotent) or
        # race on the module NLP rebind. Construct one Verifier per thread for
        # concurrent use.
        self._grounding_nli_pipeline: Optional[Any] = None
        # Compile the query-keyword tokeniser once per instance from the
        # configured minimum length, so a subclass / config override is the
        # single source of truth.
        self._query_keyword_re: re.Pattern = re.compile(
            rf"\b\w{{{self.config.query_keyword_min_length},}}\b"
        )
        logger.info(
            "Verifier initialised: model=%s, max_iterations=%d, "
            "entity_path_validation=%s",
            self.config.model_name,
            self.config.max_iterations,
            self.config.enable_entity_path_validation,
        )

    def set_graph_store(self, graph_store: Any) -> None:
        """Inject or replace the graph store at runtime."""
        self.graph_store = graph_store
        self.pre_validator.graph_store = graph_store
        logger.info("Graph store connected to Verifier and PreGenerationValidator")

    # ── LLM Interaction ───────────────────────────────────────────────────────

    def _call_llm(
        self, prompt: str, max_tokens: Optional[int] = None
    ) -> Tuple[str, float]:
        """
        Call the Ollama generate endpoint.

        Args:
            prompt: The full prompt string.
            max_tokens: Optional override for the generation cap (num_predict).
                Defaults to ``self.config.max_tokens``. The structured-CoT path
                passes ``cot_max_tokens`` here so the multi-step output is not
                truncated before the FINAL ANSWER line.

        Returns ``(response_text, latency_ms)``.  On failure, returns an
        error-sentinel string beginning with ``"[Error:"`` so callers can
        detect failures without raising exceptions.
        """
        _num_predict = max_tokens if max_tokens is not None else self.config.max_tokens
        start = time.time()
        try:
            response = requests.post(
                "%s/api/generate" % self.config.base_url,
                json={
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": _num_predict,
                    },
                },
                timeout=self.config.timeout,
            )
            latency_ms = (time.time() - start) * 1000
            if response.status_code != 200:
                logger.error("Ollama API error: HTTP %d", response.status_code)
                return "[Error: API returned %d]" % response.status_code, latency_ms
            data = response.json()
            # Guard against Ollama error responses (e.g., model not found).
            if "error" in data:
                logger.error("Ollama error response: %s", data["error"])
                return "[Error: %s]" % str(data["error"])[:100], latency_ms
            return data.get("response", "").strip(), latency_ms
        except requests.exceptions.Timeout:
            latency_ms = (time.time() - start) * 1000
            logger.error("Ollama timeout after %ds", self.config.timeout)
            return "[Error: LLM timeout - try reducing context size]", latency_ms
        except requests.exceptions.ConnectionError:
            latency_ms = (time.time() - start) * 1000
            logger.error("Cannot connect to Ollama at %s", self.config.base_url)
            return "[Error: Cannot connect to Ollama - is it running?]", latency_ms
        except requests.exceptions.RequestException as exc:
            latency_ms = (time.time() - start) * 1000
            logger.error("LLM request failed: %s", exc)
            return "[Error: %s]" % str(exc)[:100], latency_ms

    # ── Context Relevance Reordering ──────────────────────────────────────────

    # Content words to exclude from question-keyword scoring.
    # These appear in almost every question and carry no discriminative signal.
    _QR_STOPWORDS: frozenset = frozenset({
        "what", "who", "where", "when", "which", "whom", "whose", "that",
        "this", "these", "those", "have", "been", "from", "with", "their",
        "they", "were", "there", "about", "also", "into", "more", "some",
        "does", "will", "would", "could", "should", "than", "then", "them",
    })

    def _reorder_by_question_relevance(
        self,
        query: str,
        context: List[str],
        entities: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Stable-sort context chunks so those sharing more content words with the
        query appear first in the LLM prompt.

        Rationale: small LLMs tend to latch onto the first plausible
        entity in the context (positional bias; Liu et al. 2023, "Lost in the
        Middle", arXiv:2307.03172). By positioning the most answer-relevant
        chunk first, the LLM is exposed to the fact it needs before encountering
        distractors.

        Contract (revised): this method reorders ONLY the chunks already
        selected for the prompt — it operates on PRESENTATION ORDER, never on
        SET MEMBERSHIP. The caller caps the context to ``max_docs`` by the
        Navigator's fused RRF ranking BEFORE calling this method, so the
        retrieval signal (Cormack et al. 2009) decides which chunks survive the
        cap and this lexical-overlap heuristic cannot evict a high-RRF answer
        chunk that happens to be lexically sparse in the surface query terms
        (the failure class this contract prevents).

        Scoring combines three terms:
        1. IDF-weighted query-term overlap — a query term occurring in
           MANY candidate chunks (a generic category word like "magazines")
           carries little discriminative power; a rare term (the specific
           entity) is decisive. Classic inverse document frequency (Spärck
           Jones 1972; Robertson 2004), computed over the candidate set in
           hand. Applied only when there are >= ``idf_min_candidates``
           chunks — below that, document frequency is degenerate, so the
           score falls back to the length-normalised hit count.
        2. Length normalisation (word_count ** ``length_norm_exponent``,
           default sqrt) — short direct-answer chunks are not penalised
           against long topic chunks that accumulate hits from sheer length.
        3. Structural-coverage floor — a chunk that names a DISTINCTIVE
           query entity (multi-word, or a single token >=
           ``distinctive_entity_min_length`` chars) receives a score floor
           so a required entity's article (e.g. a comparison conjunct, or a
           bridge target) cannot be demoted below the cap by keyword
           sparsity. Restricted to distinctive entities so a common
           single-word name does not over-fire.

        Stable-sort descending — ties preserve original (Navigator RRF) order.
        """
        if len(context) <= 1:
            return context

        # Tokenise the query using the configured minimum keyword length.
        # The compiled regex is cached on the instance in __init__.
        query_tokens = {
            t for t in self._query_keyword_re.findall(query.lower())
            if t not in self._QR_STOPWORDS
        }
        if not query_tokens:
            return context

        # IDF over the candidate set, guarded by a minimum count below
        # which document frequency is statistically meaningless.
        if len(context) >= self.config.idf_min_candidates:
            n_docs = len(context)
            lowered_all = [c.lower() for c in context]
            idf = {
                t: math.log((n_docs + 1.0) / (1.0 + sum(1 for c in lowered_all if t in c)))
                for t in query_tokens
            }
        else:
            idf = {t: 1.0 for t in query_tokens}

        # Distinctive query entities that earn a coverage floor — multi-token
        # names, or single tokens at least ``distinctive_entity_min_length``
        # characters long. The floor protects a chunk that names a required
        # entity from being demoted by sparse keyword overlap.
        min_distinctive_len = self.config.distinctive_entity_min_length
        distinctive_entities = [
            e.lower() for e in (entities or [])
            if len(e.split()) >= 2 or len(e) >= min_distinctive_len
        ]
        # Floor = maximum achievable IDF mass, so an entity-bearing chunk
        # always outranks a chunk that merely shares generic terms.
        coverage_floor = sum(idf.values()) if idf else 1.0
        length_exponent = self.config.length_norm_exponent

        def _score(chunk: str) -> float:
            chunk_lower = chunk.lower()
            weighted = sum(idf[t] for t in query_tokens if t in chunk_lower)
            word_count = max(1, len(chunk_lower.split()))
            base = weighted / (word_count ** length_exponent)
            if distinctive_entities and any(e in chunk_lower for e in distinctive_entities):
                base += coverage_floor
            return base

        return sorted(context, key=_score, reverse=True)

    # ── Context Formatting ────────────────────────────────────────────────────

    def _truncate_sentence_aware(self, doc: str, budget: int, query: str) -> str:
        """Truncate ``doc`` to ``budget`` chars by keeping the most
        query-relevant SENTENCES (in original order), not the first N chars.

        Head-truncation silently drops an answer-bearing sentence that sits
        in the tail of a chunk (confirmed failure: a chunk whose defining
        fact was in its last sentence). Selecting by query overlap and
        re-emitting the kept sentences in their ORIGINAL order preserves
        local coherence (which matters for a small LLM) while ensuring the
        answer sentence survives.
        """
        sentences = self._SENTENCE_SPLIT_RE.split(doc)
        if len(sentences) <= 1:
            # No sentence structure to exploit — fall back to head truncation.
            cut = doc[:budget]
            sp = cut.rfind(" ")
            return (cut[:sp] + "...") if sp > 0 else cut
        q_tokens: Set[str] = set(self._query_keyword_re.findall(query.lower()))

        def _rel(s: str) -> int:
            sl = s.lower()
            return sum(1 for t in q_tokens if t in sl)

        # Rank sentence INDICES by relevance, take the highest-scoring ones
        # that fit the budget, then emit them in original order.
        order = sorted(range(len(sentences)), key=lambda i: -_rel(sentences[i]))
        keep: Set[int] = set()
        used = 0
        for idx in order:
            s_len = len(sentences[idx]) + 1
            if used + s_len > budget and keep:
                break
            keep.add(idx)
            used += s_len
        kept = [sentences[i] for i in range(len(sentences)) if i in keep]
        return " ".join(kept).strip()

    def _format_context(self, context: List[str], query: str = "") -> str:
        """
        Format context chunks into a single prompt string with size limits.

        Strategy:
        1. Take at most ``max_docs`` chunks.
        2. Truncate each chunk at ``max_chars_per_doc`` (sentence-aware,
           keeping the most query-relevant sentences in original order when a
           query is supplied; head-truncation otherwise).
        3. Stop adding chunks once ``max_context_chars`` is reached.
        """
        if not context:
            return "No context available."
        formatted_parts: List[str] = []
        total_chars = 0
        for i, doc in enumerate(context[: self.config.max_docs]):
            if len(doc) > self.config.max_chars_per_doc:
                if query:
                    truncated = self._truncate_sentence_aware(
                        doc, self.config.max_chars_per_doc, query,
                    )
                else:
                    truncated = doc[: self.config.max_chars_per_doc]
                    last_period = truncated.rfind(". ")
                    if last_period > self.config.max_chars_per_doc * self.config.format_sentence_boundary_fraction:
                        truncated = truncated[: last_period + 1]
                    else:
                        last_space = truncated.rfind(" ")
                        if last_space > 0:
                            truncated = truncated[:last_space] + "..."
            else:
                truncated = doc
            part = "[%d] %s" % (i + 1, truncated)
            if total_chars + len(part) > self.config.max_context_chars:
                logger.debug(
                    "Context budget reached at doc %d/%d", i + 1, len(context)
                )
                break
            formatted_parts.append(part)
            total_chars += len(part) + 2  # +2 for "\n\n" separator
        logger.debug(
            "Context formatted: %d docs, %d chars (~%d tokens)",
            len(formatted_parts),
            total_chars,
            total_chars // 4,
        )
        return "\n\n".join(formatted_parts)

    # ── Claim Extraction ──────────────────────────────────────────────────────

    def _extract_claims(self, answer: str) -> List[str]:
        """
        Split a generated answer into atomic factual claims.

        Uses SpaCy sentence segmentation when available; falls back to
        punctuation-based regex splitting.  Meta-statements (hedges, "I don't
        know", etc.) are filtered out because they are not verifiable claims.

        Reference: Kryscinski et al. (2020). "Evaluating the Factual
        Consistency of Abstractive Text Summarization." EMNLP 2020.
        arXiv:1910.12840. — Motivation for claim-level factual consistency
        checking as a quality proxy.
        """
        if answer.startswith("[Error:"):
            return []
        if SPACY_AVAILABLE and NLP:
            doc = NLP(answer)
            claims = [
                s.text.strip()
                for s in doc.sents
                if len(s.text.strip()) > self.config.min_claim_chars
            ]
        else:
            claims = self._SENTENCE_SPLIT_RE.split(answer)
            claims = [
                c.strip()
                for c in claims
                if len(c.strip()) > self.config.min_claim_chars
            ]
        meta_patterns = (
            "based on the context",
            "according to the",
            "i cannot answer",
            "i don't know",
            "not enough information",
            "the context does not",
            "the context doesn't",
            "error:",
            "insufficient evidence",
        )
        filtered = [
            c for c in claims if not any(p in c.lower() for p in meta_patterns)
        ]
        logger.debug("Extracted %d claims from answer", len(filtered))
        return filtered

    # ── Claim Verification ────────────────────────────────────────────────────

    def _verify_claim(
        self,
        claim: str,
        context: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """
        Verify a single claim against the graph store and/or context text.

        Strategy:
        1. Extract proper-noun entities via regex.
        2. Query KuzuDB (``find_chunks_by_entity_multihop``) or fall back to
           ``graph_search`` / ``get_entity_relations`` depending on interface.
        3. If graph store is absent or returns no results, fall back to
           substring matching in the retrieved context.
        4. Claims with no extractable entities are considered verified by
           default (nothing to contradict).

        This is a conservative entity-presence proxy, not logical entailment.
        For entailment-based verification see Kryscinski et al. (2020).
        arXiv:1910.12840.

        Returns
        -------
        (is_verified, reason_code) : (bool, str)
        """
        entities: List[str] = []
        entities.extend(_PROPER_NOUN_RE.findall(claim))
        entities.extend(self._SINGLE_PROPER_NOUN_RE.findall(claim))
        entities.extend(self._QUOTED_RE.findall(claim))
        entities = [e for e in entities if e not in self._CLAIM_STOPWORDS]

        if not entities:
            # Claims with no extractable proper noun must not auto-verify
            # when they are short factual answers ("1995", "9 million
            # inhabitants", "ice hockey") — the answer shape produced for
            # "how many" / "in what year" / "what sport" questions. Such a
            # claim is checked against the retrieved context whenever it
            # contains a numeric token OR is at most
            # ``short_claim_max_tokens`` tokens long. If it isn't grounded
            # in context, it is a violation rather than auto-verified.
            # Multi-clause narrative sentences with no proper noun still
            # auto-verify because they carry no falsifiable anchor.
            claim_lower = claim.lower().strip()
            tokens = claim_lower.split()
            has_number = bool(re.search(r"\d", claim_lower))
            is_short = len(tokens) <= self.config.short_claim_max_tokens

            if (has_number or is_short) and context:
                # Token-level grounding check: every non-stopword content
                # token of the claim must appear somewhere in the joined
                # context. Stopwords/articles/auxiliary verbs are ignored
                # so phrasing differences ("was founded in 1995" vs
                # "founded 1995") don't falsely violate.
                #
                # Numeric tokens are KEPT regardless of length because they
                # are the falsifiable signal the check exists to catch
                # (e.g. "9 million inhabitants" vs "1.5 million inhabitants"
                # — without keeping the digit, "million" and "inhabitants"
                # both match and the hallucination would slip through).
                # Strip trailing punctuation so "1995." still matches
                # "1995" in the context.
                context_text = " ".join(context).lower()
                content_tokens: List[str] = []
                min_token_len = self.config.claim_content_token_min_length
                for raw in tokens:
                    t = raw.strip(".,;:!?\"'()[]")
                    if not t:
                        continue
                    if t in self._CLAIM_VERIFY_STOPWORDS:
                        continue
                    is_numeric_token = any(ch.isdigit() for ch in t)
                    if is_numeric_token or len(t) >= min_token_len:
                        content_tokens.append(t)
                if not content_tokens:
                    return True, "no_content_tokens_to_verify"
                grounded = all(t in context_text for t in content_tokens)
                if grounded:
                    return True, "context_token_grounded"
                return False, "no_entities_and_tokens_ungrounded"

            return True, "no_entities_to_verify"

        # ── Graph store verification ──────────────────────────────────────────
        if self.graph_store:
            for entity in entities[: self.config.max_entities_to_verify]:
                try:
                    found = False
                    if hasattr(self.graph_store, "find_chunks_by_entity_multihop"):
                        results = self.graph_store.find_chunks_by_entity_multihop(
                            entity_name=entity, max_results=1
                        )
                        found = bool(results)
                    elif hasattr(self.graph_store, "graph_search"):
                        results = self.graph_store.graph_search(
                            entities=[entity], top_k=1
                        )
                        found = bool(results)
                    elif hasattr(self.graph_store, "get_entity_relations"):
                        results = self.graph_store.get_entity_relations(entity)
                        found = bool(results)
                    if found:
                        return True, "graph_verified_%s" % entity
                except (AttributeError, TypeError, RuntimeError) as exc:
                    logger.debug(
                        "Graph query failed for entity '%s': %s", entity, exc
                    )

        # ── Context substring verification ────────────────────────────────────
        if context:
            context_text = " ".join(context).lower()
            for entity in entities[: self.config.max_entities_to_verify]:
                if entity.lower() in context_text:
                    return True, "context_verified_%s" % entity

        return False, "entities_not_found"

    # ── Bridge Chain Builder ──────────────────────────────────────────────────

    @staticmethod
    def _build_bridge_chain(
        query: str,
        entities: List[str],
        bridge_entities: List[str],
        hop_sequence: Optional[List[Dict[str, Any]]],
        context: Optional[List[str]] = None,
    ) -> str:
        """
        Build a human-readable reasoning scaffold for multi-hop prompts.

        Safety guarantees:
        1. Never emits a literal sentinel like "THIS IS THE ANSWER" — small
           quantized models echo such strings verbatim. The final step uses
           a directive verb ("→ derive the final answer") instead.
        2. Only injects a bridge entity into a hop's substitution if the
           entity is actually present in the retrieved context. Prevents
           propagation of spurious/distractor bridge entities into the
           reasoning chain — i.e. an off-topic named entity appearing in
           the gold chunk via a tangential clause must not be promoted to
           the inferred hop result.
        3. If no bridge entity passes the grounding check, the hop is
           rendered as a directive ("→ identify the intermediate result")
           rather than left blank or pre-filled with a wrong value.
        """
        lines: List[str] = []
        # Lowercased context for cheap substring grounding-check.
        context_blob = " ".join(context or []).lower()

        def _grounded(entity: str) -> bool:
            """Entity is grounded if it appears in the retrieved context.

            Empty context → trust the upstream extractor (cannot ground-
            check, but emitting nothing is worse than emitting unverified)."""
            if not context_blob:
                return True
            return entity.lower() in context_blob

        if hop_sequence:
            hops = sorted(hop_sequence, key=lambda h: h.get("step_id", 0))
            last_step_id = max(h.get("step_id", 0) for h in hops)
            # NOTE (2026-05-20): bridge-entity injection into the scaffold is
            # disabled. "Entity appears in context" does not mean the entity
            # answers the sub-query; a 20-sample diagnostic showed it injecting
            # the wrong-era name, a random PERSON, or a useless synonym. Both
            # bridge and non-bridge intermediate hops therefore render the same
            # directive form. The dead `grounded_bridges`/`bridge_idx`/
            # `is_bridge` locals were removed 2026-06-13 (finding #6).
            for hop in hops:
                step = hop.get("step_id", 0) + 1
                sub_q = hop.get("sub_query", "")
                is_last = hop.get("step_id", 0) == last_step_id
                if is_last:
                    # Directive verb form — NOT a literal placeholder string
                    # the LLM could mistake for an answer.
                    lines.append(
                        "Step %d: %s → derive the final answer" % (step, sub_q)
                    )
                else:
                    lines.append(
                        "Step %d: %s → identify the intermediate result"
                        % (step, sub_q)
                    )
        else:
            # Fallback: generic chain from entity names + (grounded) bridges.
            grounded_bridges = [e for e in bridge_entities if _grounded(e)]
            anchor = entities[0] if entities else "the subject"
            for i, bridge in enumerate(grounded_bridges, start=1):
                lines.append("Step %d: find information about %s → %s" % (i, anchor, bridge))
                anchor = bridge
            lines.append(
                "Step %d: derive the final answer about %s"
                % (len(grounded_bridges) + 1, anchor)
            )

        return "\n".join(lines)

    # ── QA-conditioned NLI grounding gate ─────────────────────────────────────

    # Leading interrogative words handled by the QA→declarative transform.
    # Closed class; anything not starting with one of these falls through to
    # the generic "question-body + answer" concatenation.
    _WH_LEADING = ("who", "whom", "whose")

    @staticmethod
    def _qa_to_hypothesis(question: str, answer: str) -> str:
        """
        Turn a (question, answer) pair into a declarative NLI hypothesis.

        Lightweight heuristic — no parser, no LLM:
        - "Who/Whom <body>?" -> "<answer> <body>"  (the answer fills the
          person slot, e.g. "Who founded X?"+"<PERSON>" -> "<PERSON> founded X")
        - "Whose <body>?"    -> "<answer>'s <body>"
        - otherwise          -> "<question body> <answer>"

        The aim is a statement whose truth the retrieved context can entail or
        not. A wrong-but-grounded answer (an entity that appears in the context
        but is not the one the question asks for) produces a statement the
        context does NOT entail, which is the signal the gate keys on.

        Caveat (finding #7): this is a coarse heuristic. The "whose" branch in
        particular can yield mildly ungrammatical hypotheses ("Tolkien's son was
        he"); the NLI gate is opt-in (default OFF) and fail-safe (returns
        grounded=True when NLI is unavailable), so the impact of imperfect
        hypotheses is bounded to the ablation configuration that enables it.
        """
        a = (answer or "").strip().rstrip(".").strip()
        q = (question or "").strip().rstrip("?").strip()
        if not q:
            return a
        tokens = q.split()
        lead = tokens[0].lower() if tokens else ""
        body = " ".join(tokens[1:]).strip()
        if lead in ("who", "whom") and body:
            return f"{a} {body}"
        if lead == "whose" and body:
            return f"{a}'s {body}"
        return f"{q} {a}".strip()

    def _load_grounding_nli(self) -> Optional[Any]:
        """Lazy-load the NLI text-classification pipeline (CPU). None on failure."""
        if not TRANSFORMERS_AVAILABLE:
            return None
        if self._grounding_nli_pipeline is None:
            try:
                self._grounding_nli_pipeline = _hf_pipeline(
                    "text-classification",
                    model=self.config.nli_model,
                    top_k=None,        # return all 3 label scores, not just the top
                    device=-1,         # CPU; HuggingFace uses -1
                )
            except (RuntimeError, OSError, ValueError) as exc:
                logger.warning(
                    "[Verifier] NLI grounding model load failed (%s); "
                    "grounding gate disabled for this run.", exc,
                )
                self._grounding_nli_pipeline = None
        return self._grounding_nli_pipeline

    def _nli_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Return the entailment probability of (premise -> hypothesis), or -1.0
        when NLI is unavailable so callers can skip the gate rather than
        falsely flag.
        """
        pipe = self._load_grounding_nli()
        if pipe is None:
            return -1.0
        try:
            p = premise[: self.config.nli_max_input_chars]
            h = hypothesis[: self.config.nli_max_input_chars]
            result = pipe({"text": p, "text_pair": h}, truncation=True)
            # top_k=None -> list of {label, score} dicts for one input.
            scores = result[0] if result and isinstance(result[0], list) else result
            for entry in scores:
                if str(entry.get("label", "")).upper().startswith("ENTAIL"):
                    return float(entry.get("score", 0.0))
            return 0.0
        except (RuntimeError, OSError, ValueError, KeyError, IndexError) as exc:
            logger.debug("[Verifier] NLI entailment failed (%s)", exc)
            return -1.0

    def _answer_grounded(
        self, query: str, answer: str, context: List[str],
    ) -> Tuple[bool, float]:
        """
        QA-conditioned grounding check: is the (query, answer) hypothesis
        entailed by ANY of the top-N retrieved chunks?

        Returns (grounded, max_entailment). grounded is True (gate passes,
        no retry) when NLI is unavailable (max_entailment == -1.0) so the
        feature fails safe.
        """
        if not context:
            return True, -1.0
        hypothesis = self._qa_to_hypothesis(query, answer)
        best = -1.0
        for chunk in context[: self.config.nli_grounding_top_chunks]:
            score = self._nli_entailment(chunk, hypothesis)
            if score > best:
                best = score
        if best < 0.0:                       # NLI unavailable -> fail safe
            return True, best
        return best >= self.config.nli_grounding_entail_threshold, best

    # ── Bounded retry engine ──────────────────────────────────────────────────

    def _bounded_retry(
        self,
        name: str,
        already_fired: bool,
        should_fire: bool,
        prompt: str,
        current_answer: str,
        accept: Callable[[str], bool],
    ) -> Tuple[str, bool, float]:
        """Run one bounded re-prompt and conditionally accept its output.

        Unifies the four self-correction retry paths (bridge-exclusion,
        format-mismatch, anti-abstention, NLI-grounding) which previously
        repeated the same *fire-once → call LLM → no-regression guard → maybe
        swap* skeleton inline (review 2026-06-13, finding #4).

        Parameters
        ----------
        name : retry label for logging.
        already_fired : True if this retry already fired this query (the
            one-call-per-query budget); when True this is a no-op.
        should_fire : the per-retry trigger predicate (already evaluated).
        prompt : the fully-formatted retry prompt.
        current_answer : the answer to potentially replace.
        accept : predicate on the retry's answer. The shared no-regression
            guard (non-empty, not an "[Error:" sentinel, not a disclaimer) is
            applied FIRST; ``accept`` adds the retry-specific condition (e.g.
            the grounding retry's strictly-higher-entailment keep-best check).

        Returns
        -------
        (answer, fired, added_latency_ms)
            ``answer`` is the retry's output if accepted, else ``current_answer``
            unchanged. ``fired`` is True iff the LLM was actually called.
        """
        if already_fired or not should_fire:
            return current_answer, False, 0.0
        retry_answer, latency = self._call_llm(prompt)
        # Shared no-regression guard: a retry that errored or fell back to a
        # disclaimer must never replace a concrete prior answer. The
        # unconditional swap was the bug class that sank the first NLI-gate
        # experiment (§11.17.4 keep-best discipline).
        if (
            retry_answer
            and not retry_answer.startswith("[Error:")
            and not self._is_disclaimer_answer(retry_answer)
            and accept(retry_answer)
        ):
            logger.info("[Verifier] %s retry accepted: %s", name, retry_answer[:80])
            return retry_answer, True, latency
        logger.info("[Verifier] %s retry rejected — keeping original.", name)
        return current_answer, True, latency

    # ── Main Verification Loop ────────────────────────────────────────────────

    def generate_and_verify(
        self,
        query: str,
        context: List[str],
        entities: Optional[List[str]] = None,
        hop_sequence: Optional[List[Dict[str, Any]]] = None,
        query_type: Optional[str] = None,
        bridge_entities: Optional[List[str]] = None,
        chunk_is_graph_based: Optional[List[bool]] = None,
    ) -> VerificationResult:
        """
        Main entry point: pre-validation, generation, and self-correction.

        Algorithm
        ---------
        1. Run pre-generation validation (entity-path, contradiction,
           credibility filtering).
        2. Select the initial prompt based on validation status.
        3. For each iteration up to ``max_iterations``:
           a. Call the LLM.
           b. Skip empty or error responses (with best-answer fallback).
           c. Extract atomic claims from the answer.
           d. Verify each claim; return immediately if all pass.
           e. Otherwise re-prompt with CORRECTION_PROMPT listing violated
              claims.
        4. Return the best answer seen across all iterations.

        Reference: Madaan, A., et al. (2023). "Self-Refine: Iterative
        Refinement with Self-Feedback." NeurIPS 2023. arXiv:2303.17651.

        Parameters
        ----------
        query :
            Original user question.
        context :
            Retrieved chunks from the Navigator.
        entities :
            Query entities from the Planner (forwarded to pre-validation).
        hop_sequence :
            Hop plan from the Planner (reserved for future graph-path
            planning; passed through to pre-validation but not used there).

        Returns
        -------
        VerificationResult
        """
        if query is None:
            query = ""
        start_time = time.time()
        logger.info("[Verifier] query='%s'", query[:60])
        logger.info("[Verifier] context docs: %d", len(context))

        # Boundary normalisation (added 2026-05-15): the signature
        # documents hop_sequence as List[Dict[str, Any]], but
        # AgentPipeline.process() passes the Planner's List[HopStep]
        # dataclasses directly. The bridge-chain builder uses .get() on
        # each entry, which works on dicts but not on dataclasses.
        # Normalise at the boundary so both call paths work:
        # dataclass -> dict via dataclasses.asdict(); dicts pass through.
        if hop_sequence:
            try:
                hop_sequence = [
                    asdict(h) if is_dataclass(h) else h
                    for h in hop_sequence
                ]
            except (TypeError, ValueError) as exc:
                # Defensive: hop_sequence normalisation is a convenience
                # for callers that mix dataclasses and dicts. If the cast
                # fails, leave as-is — the bridge-chain builder uses .get()
                # which already handles dict-vs-attribute access.
                logger.debug(
                    "hop_sequence normalisation skipped (%s); leaving as-is.",
                    exc,
                )

        # ── Pre-generation validation ─────────────────────────────────────────
        pre_validation = self.pre_validator.validate(
            context=context,
            query=query,
            entities=entities,
            hop_sequence=hop_sequence,
            chunk_is_graph_based=chunk_is_graph_based,
        )
        working_context = pre_validation.filtered_context
        logger.info(
            "[Verifier] Pre-validation: %s, context %d/%d",
            pre_validation.status.value,
            len(working_context),
            len(context),
        )

        # ── Hard early-return on truly empty evidence ────────────────────────
        # Skip the LLM call ONLY when there is genuinely no context to read.
        #
        # 2026-05-28: the previous condition ALSO fired on INSUFFICIENT_EVIDENCE
        # + no entity-path match, returning "I cannot determine" WITHOUT calling
        # the LLM. That force-abstained whenever the planner's entities failed a
        # substring match against the chunks — a frequent false negative on
        # bridge questions where the answer IS present but the entity is not
        # verbatim (e.g. surface form vs. canonical name). It was a silent
        # false-abstention path; the old credibility-drop side-effect happened
        # to mask it in some cases. We now keep any non-empty context and let
        # the LLM try (via INSUFFICIENT_EVIDENCE_PROMPT when flagged). This only
        # ADDS generation attempts on context we already have, so it cannot turn
        # a correct answer wrong — it can only recover force-abstained ones.
        path_details = pre_validation.details.get("entity_path", {}) or {}
        entities_found = path_details.get("entities_found", []) or []
        if not working_context:
            logger.warning(
                "[Verifier] Hard early-return: no usable context "
                "(working_context=%d, entities_found=%d) — skipping LLM call.",
                len(working_context),
                len(entities_found),
            )
            total_time = (time.time() - start_time) * 1000
            return VerificationResult(
                answer="I cannot determine the answer from the provided context.",
                iterations=0,
                verified_claims=[],
                violated_claims=[],
                all_verified=False,
                pre_validation=pre_validation,
                timing_ms=total_time,
                iteration_history=[],
                retry_fired={
                    "bridge_exclusion": False, "format_mismatch": False,
                    "anti_abstention": False, "nli_grounding": False,
                },
                confidence_high_threshold=self.config.confidence_high_threshold,
                confidence_medium_threshold=self.config.confidence_medium_threshold,
            )

        # Contract: the Navigator's fused RRF ranking owns SET MEMBERSHIP —
        # which chunks survive the ``max_docs`` cap — while the question-
        # relevance reorder owns only PRESENTATION ORDER within that window,
        # to mitigate LLM positional bias (Liu et al. 2023, "Lost in the
        # Middle", arXiv:2307.03172). Selecting by lexical query-overlap
        # would let a distractor that merely echoes the question evict a
        # question-term-sparse answer chunk that the retriever had ranked
        # #1. Cap by RRF order first, then reorder only inside the kept
        # window — and only when the reorder is enabled by config.
        selected = working_context[: self.config.max_docs]
        if self.config.enable_question_relevance_reorder:
            working_context = self._reorder_by_question_relevance(
                query, selected, entities=entities,
            )
        else:
            working_context = selected
        formatted_context = self._format_context(working_context, query=query)

        # RECOMP-style context distillation (Yu et al. 2024, NAACL).
        # One extra LLM call condenses the reordered chunks into a short
        # fact list. The LLM-facing prompt sees the distilled facts, but
        # claim-verification continues to run against the original
        # working_context (so claim-grounding is unchanged). Provenance:
        # dev-split diagnostic — see VerifierConfig field note.
        distilled_prompt_context: Optional[str] = None
        if (
            self.config.enable_context_distillation
            and working_context
            and len(formatted_context) > self.config.context_distillation_min_input_chars
        ):
            try:
                distill_prompt = self.DISTILL_PROMPT.format(
                    query=query, context=formatted_context
                )
                distilled_text, distill_latency = self._call_llm(distill_prompt)
                if (
                    distilled_text
                    and not distilled_text.startswith("[Error:")
                    and len(distilled_text.strip()) >= self.config.context_distillation_min_output_chars
                ):
                    distilled_prompt_context = (
                        "Key facts extracted from the retrieved sources:\n"
                        f"{distilled_text.strip()}"
                    )
                    logger.info(
                        "[Verifier] Context distillation: %d chars -> %d "
                        "chars (latency=%.0fms)",
                        len(formatted_context),
                        len(distilled_prompt_context),
                        distill_latency,
                    )
                else:
                    logger.debug(
                        "[Verifier] Distillation skipped (empty/error output); "
                        "falling back to raw chunks"
                    )
            except (KeyError, ValueError, RuntimeError, AttributeError) as exc:
                # Context distillation is on the default path (enable_context_
                # distillation defaults True). Any failure in the distillation
                # LLM call / formatting must NOT break the main verification
                # path — fall back to the raw filtered chunks so the eval
                # continues with the documented context. (Narrowed from a bare
                # Exception 2026-06-13; _call_llm already returns an error
                # sentinel rather than raising.)
                logger.warning(
                    "[Verifier] Distillation failed (%s); using raw chunks",
                    exc,
                )

        # Prompt-facing context: distilled if available, else the original
        # formatted chunks. Claim verification ALWAYS uses working_context
        # so atomic claims must still be grounded in retrieved chunks.
        prompt_context = distilled_prompt_context or formatted_context

        best_answer: Optional[str] = None
        best_verified: List[str] = []
        best_violated: List[str] = []
        iteration_history: List[Dict[str, Any]] = []
        violated_claims: List[str] = []
        # Track whether each bounded retry path has already fired for this
        # query. Both retries are capped at exactly one call per
        # generate_and_verify invocation (one per query, not per
        # self-correction iteration). Exposed on VerificationResult.retry_fired.
        bridge_retried_this_call: bool = False
        format_retried_this_call: bool = False
        abstention_retried_this_call: bool = False
        grounding_retried_this_call: bool = False

        # ── Self-correction loop ──────────────────────────────────────────────
        for iteration in range(1, self.config.max_iterations + 1):
            iter_start = time.time()
            logger.info(
                "[Verifier] === Iteration %d/%d ===",
                iteration,
                self.config.max_iterations,
            )

            # Structured-CoT prompt selection (2026-05-27). When enabled, the
            # first-iteration answer prompts use the slot-filling COT variants.
            # The INSUFFICIENT_EVIDENCE prompt and the iteration-2+ CORRECTION
            # prompt are left as-is (CoT does not apply to a partial-answer or
            # claim-correction flow). used_cot_this_iter gates the final-answer
            # extraction below.
            use_cot = self.config.enable_structured_cot
            used_cot_this_iter = False
            if iteration == 1:
                if pre_validation.status == ValidationStatus.INSUFFICIENT_EVIDENCE:
                    prompt = self.INSUFFICIENT_EVIDENCE_PROMPT.format(
                        context=prompt_context, query=query
                    )
                elif query_type == "comparison":
                    prompt = (
                        self.COMPARISON_PROMPT_COT if use_cot else self.COMPARISON_PROMPT
                    ).format(context=prompt_context, query=query)
                    used_cot_this_iter = use_cot
                elif query_type in ("multi_hop", "bridge") and hop_sequence:
                    bridge_chain = self._build_bridge_chain(
                        query, entities or [], bridge_entities or [],
                        hop_sequence, context=context,
                    )
                    prompt = (
                        self.BRIDGE_PROMPT_COT if use_cot else self.BRIDGE_PROMPT
                    ).format(
                        bridge_chain=bridge_chain,
                        context=prompt_context,
                        query=query,
                    )
                    used_cot_this_iter = use_cot
                else:
                    prompt = (
                        self.ANSWER_PROMPT_COT if use_cot else self.ANSWER_PROMPT
                    ).format(context=prompt_context, query=query)
                    used_cot_this_iter = use_cot
            else:
                prompt = self.CORRECTION_PROMPT.format(
                    violations="\n".join("- %s" % v for v in violated_claims),
                    context=prompt_context,
                    query=query,
                )

            # CoT needs a larger generation budget so the FINAL ANSWER line is
            # not truncated by the multi-step output. Only pass the override on
            # the CoT path; the default path calls _call_llm(prompt) unchanged
            # so it stays signature-compatible with all existing callers/mocks.
            if used_cot_this_iter:
                answer, llm_latency = self._call_llm(
                    prompt, max_tokens=self.config.cot_max_tokens
                )
            else:
                answer, llm_latency = self._call_llm(prompt)
            logger.info("[Verifier] LLM response in %.0fms", llm_latency)

            # Parse the FINAL ANSWER out of a structured-CoT response BEFORE any
            # downstream processing (bridge-exclusion, format-retry, claim
            # extraction) so they all operate on the answer, not the reasoning
            # steps. Claim verification therefore still grounds the answer in
            # retrieved chunks, not the CoT scaffold.
            if used_cot_this_iter and answer and not answer.startswith("[Error:"):
                answer = self._extract_final_answer_from_cot(answer)

            # Guard: empty answers cannot be verified and must not be
            # returned as correct results.
            if not answer or answer.isspace():
                logger.warning(
                    "[Verifier] Empty answer in iteration %d; skipping.",
                    iteration,
                )
                iteration_history.append({
                    "iteration": iteration,
                    "answer": self._truncate_history_str(answer),
                    "claims": [],
                    "verified": [],
                    "violated": [],
                    "llm_latency_ms": llm_latency,
                    "error": True,
                })
                if best_answer:
                    break
                continue

            if answer.startswith("[Error:"):
                logger.warning("[Verifier] LLM error: %s", answer)
                iteration_history.append({
                    "iteration": iteration,
                    "answer": self._truncate_history_str(answer),
                    "claims": [],
                    "verified": [],
                    "violated": [],
                    "llm_latency_ms": llm_latency,
                    "error": True,
                })
                if best_answer:
                    break
                continue

            # Bridge-entity exclusion retry.
            # Detects when the LLM returns one of the resolved bridge
            # entities as the final answer (a common failure mode where the
            # SLM picks the intermediate hop result instead of the final
            # attribute). On bridge_hit, re-prompt with an explicit
            # exclusion instruction. Bounded retry: at most one call per
            # query. Refs: Dhuliawala et al. (2023) Chain-of-Verification,
            # narrowed to the bridge-entity failure mode.
            if (
                self.config.enable_bridge_exclusion_retry
                and bridge_entities
                and answer
                and not bridge_retried_this_call
            ):
                ans_norm = answer.strip().lower().rstrip(".!?,;:").strip('"\'')
                bridge_norms = [
                    b.strip().lower().rstrip(".") for b in bridge_entities if b
                ]
                bridge_substr_min = self.config.bridge_substring_min_length
                bridge_hit = any(
                    ans_norm == b
                    or (len(ans_norm) >= bridge_substr_min and ans_norm in b)
                    or (len(b) >= bridge_substr_min and b in ans_norm)
                    for b in bridge_norms
                )
                if bridge_hit:
                    exclusion_list = ", ".join(
                        f'"{b}"' for b in bridge_entities[: self.config.bridge_exclusion_top_k]
                    )
                    logger.info(
                        "[Verifier] Bridge-entity hit '%s' — retrying with "
                        "exclusion prompt (bridges=%s)",
                        answer[:60], bridge_entities[:3],
                    )
                    answer, fired, retry_latency = self._bounded_retry(
                        name="bridge-exclusion",
                        already_fired=bridge_retried_this_call,
                        should_fire=True,
                        prompt=self.BRIDGE_EXCLUSION_RETRY_PROMPT.format(
                            query=query, context=prompt_context,
                            exclusion_list=exclusion_list,
                        ),
                        current_answer=answer,
                        accept=lambda _a: True,  # shared guard is sufficient here
                    )
                    bridge_retried_this_call = bridge_retried_this_call or fired
                    llm_latency += retry_latency

            # Calibrated format-mismatch retry.
            # Detects wh-question → yes/no answer mismatches and re-prompts
            # with an explicit format instruction. Bounded to one retry
            # per query (same pattern as bridge-exclusion). Provenance:
            # dev-split diagnostic — see VerifierConfig field note.
            if (
                self.config.enable_format_validation_retry
                and answer
                and not format_retried_this_call
                and self._is_format_mismatch(query, answer)
            ):
                logger.info(
                    "[Verifier] Format mismatch detected (query=%.50s; "
                    "answer=%.40s) — retrying with format prompt",
                    query, answer,
                )
                answer, fired, fmt_latency = self._bounded_retry(
                    name="format-mismatch",
                    already_fired=format_retried_this_call,
                    should_fire=True,
                    prompt=self.FORMAT_RETRY_PROMPT.format(
                        query=query, context=prompt_context,
                        previous_answer=answer.strip(),
                    ),
                    current_answer=answer,
                    accept=lambda _a: True,
                )
                format_retried_this_call = format_retried_this_call or fired
                llm_latency += fmt_latency

            # Anti-abstention retry (2026-05-28). The model abstained
            # ("I don't know") but pre-validation found query entities present
            # in the context — the answer is very likely there. Re-prompt once
            # with an explicit "the answer IS in the context, extract it"
            # instruction. Targets the false-abstention failure bucket.
            # Bounded: one call per query.
            if (
                self.config.enable_anti_abstention_retry
                and answer
                and not abstention_retried_this_call
                and self._is_disclaimer_answer(answer)
                and entities_found
            ):
                ent_list = ", ".join(str(e) for e in entities_found[:5])
                logger.info(
                    "[Verifier] Abstention detected with entities present "
                    "(%s) — retrying.", entities_found[:3],
                )
                # The shared no-regression guard already rejects a retry that is
                # itself a disclaimer, so a still-abstaining retry leaves the
                # original in place — exactly the intended "escape abstention or
                # keep the disclaimer" semantics.
                answer, fired, anti_latency = self._bounded_retry(
                    name="anti-abstention",
                    already_fired=abstention_retried_this_call,
                    should_fire=True,
                    prompt=self.ANTI_ABSTENTION_RETRY_PROMPT.format(
                        query=query, context=prompt_context, entity_list=ent_list,
                    ),
                    current_answer=answer,
                    accept=lambda _a: True,
                )
                abstention_retried_this_call = abstention_retried_this_call or fired
                llm_latency += anti_latency

            # QA-conditioned NLI grounding gate (2026-05-28). Turn the
            # (question, answer) pair into a declarative hypothesis and check
            # whether the retrieved chunks entail it. A wrong-but-grounded
            # answer (the entity is in the context but answers a different
            # question) entails poorly, triggering one bounded grounding
            # retry. Skipped for yes/no answers (bare-polarity NLI is
            # uninformative) and when NLI is unavailable (fails safe inside
            # _answer_grounded). Targets the grounded-hallucination bucket.
            ans_norm_l = answer.strip().lower().rstrip(".!?").strip() if answer else ""
            if (
                self.config.enable_nli_grounding_gate
                and answer
                and not grounding_retried_this_call
                and not self._is_disclaimer_answer(answer)
                and ans_norm_l not in {"yes", "no"}
                and working_context
            ):
                grounded, entail = self._answer_grounded(
                    query, answer, working_context
                )
                if not grounded:
                    logger.info(
                        "[Verifier] Grounding gate: answer '%.40s' entail=%.2f "
                        "< %.2f — retrying.",
                        answer, entail,
                        self.config.nli_grounding_entail_threshold,
                    )
                    # KEEP-BEST acceptor (§11.17.4): the unconditional swap made
                    # the first grounding-gate experiment net-negative (replaced
                    # 6 correct answers with worse ones). Accept the retry only
                    # if it scores STRICTLY higher QA-conditioned entailment than
                    # the original — so a correct answer that merely fell below
                    # the trigger threshold is never replaced by a worse retry.
                    def _accept_if_more_entailed(candidate: str) -> bool:
                        _, retry_entail = self._answer_grounded(
                            query, candidate, working_context
                        )
                        return retry_entail > entail

                    answer, fired, ground_latency = self._bounded_retry(
                        name="nli-grounding",
                        already_fired=grounding_retried_this_call,
                        should_fire=True,
                        prompt=self.GROUNDING_RETRY_PROMPT.format(
                            query=query, context=prompt_context,
                            previous_answer=answer.strip(),
                        ),
                        current_answer=answer,
                        accept=_accept_if_more_entailed,
                    )
                    grounding_retried_this_call = grounding_retried_this_call or fired
                    llm_latency += ground_latency

            claims = self._extract_claims(answer)
            # Short-answer-as-claim fix (2026-05-28). HotpotQA gold answers are
            # usually a bare entity ("<PERSON>", a year, a place) shorter than
            # min_claim_chars, so _extract_claims returns []. An empty claim
            # list then maps to all_verified=True / HIGH confidence via the
            # (0,0) ratio — a correct terse answer was being reported as
            # spuriously HIGH and a wrong one never flagged. When the answer is
            # a real (non-disclaimer) response that produced no claims, treat
            # the whole answer as one claim so it is grounded against the
            # context like any other. Calibration only — never changes the
            # answer string.
            if (
                not claims
                and answer
                and not answer.startswith("[Error:")
                and not self._is_disclaimer_answer(answer)
            ):
                claims = [answer.strip()]
                logger.debug(
                    "[Verifier] No claims extracted from terse answer — "
                    "treating the answer itself as one claim."
                )
            logger.info("[Verifier] %d claims extracted", len(claims))

            verified_claims: List[str] = []
            violated_claims = []
            for claim in claims:
                is_ok, reason = self._verify_claim(claim, working_context)
                if is_ok:
                    verified_claims.append(claim)
                    logger.debug(
                        "[Verifier] VERIFIED '%s...' (%s)", claim[:50], reason
                    )
                else:
                    violated_claims.append(claim)
                    logger.debug(
                        "[Verifier] VIOLATED '%s...' (%s)", claim[:50], reason
                    )

            # ── Disclaimer override ──────────────────────────────────────
            # If the answer is an epistemic disclaimer, _extract_claims has
            # already stripped the meta-statement and the claim list is
            # often empty — which silently maps to all_verified=True / HIGH
            # confidence via the (0,0) ratio in VerificationResult.confidence.
            # Force a single violated claim so downstream sees this as a
            # non-answer with LOW confidence.
            if self._is_disclaimer_answer(answer):
                logger.warning(
                    "[Verifier] Disclaimer answer detected — forcing LOW confidence."
                )
                violated_claims = [answer.strip()[:200]]
                verified_claims = []
            logger.info(
                "[Verifier] Verification: %d verified, %d violated",
                len(verified_claims),
                len(violated_claims),
            )

            iter_time = (time.time() - iter_start) * 1000
            # Truncate stored strings (answer + claim/verified/violated
            # lists) to _HISTORY_STR_TRUNCATE_CHARS each, keeping the
            # per-question JSONL flat-text and grep-friendly across
            # 500-question runs.
            iteration_history.append({
                "iteration": iteration,
                "answer": self._truncate_history_str(answer),
                "claims": self._truncate_history_list(claims),
                "verified": self._truncate_history_list(verified_claims),
                "violated": self._truncate_history_list(violated_claims),
                "llm_latency_ms": llm_latency,
                "total_time_ms": iter_time,
                "error": False,
            })

            # Track the best answer across iterations.
            # Selection order (review 2026-06-13, finding #3):
            #   1. A substantive (non-disclaimer) answer always beats a
            #      disclaimer. A disclaimer is forced to exactly ONE violated
            #      claim (see the disclaimer override below), so a plain
            #      fewest-violations rule could let an "I don't know" (1
            #      violation) win over a real answer with 2+ violations and
            #      silently suppress a correct terse answer. Preferring the
            #      substantive answer keeps the abstention signal honest:
            #      abstention is surfaced via confidence/all_verified, not by
            #      hijacking the returned answer.
            #   2. Among answers of the same kind, fewer violations wins.
            cur_is_disclaimer = self._is_disclaimer_answer(answer)
            best_is_disclaimer = (
                best_answer is not None and self._is_disclaimer_answer(best_answer)
            )
            if best_answer is None:
                better = True
            elif best_is_disclaimer and not cur_is_disclaimer:
                better = True   # substantive answer always beats a disclaimer
            elif cur_is_disclaimer and not best_is_disclaimer:
                better = False  # never let a disclaimer displace a real answer
            else:
                better = len(violated_claims) < len(best_violated)
            if better:
                best_answer = answer
                best_verified = verified_claims
                best_violated = violated_claims

            if len(violated_claims) == 0:
                logger.info(
                    "[Verifier] All claims verified in iteration %d.", iteration
                )
                total_time = (time.time() - start_time) * 1000
                return VerificationResult(
                    answer=answer,
                    iterations=iteration,
                    verified_claims=verified_claims,
                    violated_claims=[],
                    all_verified=True,
                    pre_validation=pre_validation,
                    timing_ms=total_time,
                    iteration_history=iteration_history,
                    retry_fired={
                        "bridge_exclusion": bridge_retried_this_call,
                        "format_mismatch": format_retried_this_call,
                        "anti_abstention": abstention_retried_this_call,
                        "nli_grounding": grounding_retried_this_call,
                    },
                    confidence_high_threshold=self.config.confidence_high_threshold,
                    confidence_medium_threshold=self.config.confidence_medium_threshold,
                )
            logger.info(
                "[Verifier] %d unverified claim(s); attempting correction.",
                len(violated_claims),
            )

        # ── Max iterations reached ────────────────────────────────────────────
        total_time = (time.time() - start_time) * 1000
        logger.warning(
            "[Verifier] Max iterations reached. Best result: %d verified, %d violated.",
            len(best_verified),
            len(best_violated),
        )
        return VerificationResult(
            answer=best_answer if best_answer is not None
            else "[Error: No valid answer generated]",
            iterations=self.config.max_iterations,
            verified_claims=best_verified,
            violated_claims=best_violated,
            all_verified=False,
            pre_validation=pre_validation,
            timing_ms=total_time,
            iteration_history=iteration_history,
            retry_fired={
                "bridge_exclusion": bridge_retried_this_call,
                "format_mismatch": format_retried_this_call,
                "anti_abstention": abstention_retried_this_call,
                "nli_grounding": grounding_retried_this_call,
            },
            confidence_high_threshold=self.config.confidence_high_threshold,
            confidence_medium_threshold=self.config.confidence_medium_threshold,
        )

    def __call__(
        self,
        query: str,
        context: List[str],
        entities: Optional[List[str]] = None,
        hop_sequence: Optional[List[Dict[str, Any]]] = None,
        query_type: Optional[str] = None,
        bridge_entities: Optional[List[str]] = None,
        chunk_is_graph_based: Optional[List[bool]] = None,
    ) -> VerificationResult:
        """Callable interface — forwards all arguments to generate_and_verify."""
        return self.generate_and_verify(
            query=query,
            context=context,
            entities=entities,
            hop_sequence=hop_sequence,
            query_type=query_type,
            bridge_entities=bridge_entities,
            chunk_is_graph_based=chunk_is_graph_based,
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_verifier(
    cfg: Optional[Dict[str, Any]] = None,
    graph_store: Optional[Any] = None,
    enable_pre_validation: bool = False,
) -> Verifier:
    """
    Factory function for Verifier — reads all values from a settings.yaml dict.

    Delegates to ``VerifierConfig.from_yaml()`` for YAML parsing.  The
    ``enable_pre_validation`` flag overrides the ``verifier.enable_*`` flags
    in the settings dict (useful for one-liner test construction).

    Parameters
    ----------
    cfg : dict, optional
        Full settings.yaml dict.  Relevant keys:
        ``llm.*``, ``agent.max_verification_iterations``, ``verifier.*``.
        Pass ``{"agent": {"max_verification_iterations": 1}}`` to construct
        a single-iteration verifier for unit tests.
    graph_store : KuzuGraphStore or compatible, optional
    enable_pre_validation : bool
        When True, activates entity-path and credibility validation,
        overriding the settings dict values.

    Returns
    -------
    Verifier
    """
    if cfg is None:
        cfg = _load_settings()
    config = VerifierConfig.from_yaml(cfg)
    if enable_pre_validation:
        config.enable_entity_path_validation = True
        config.enable_credibility_scoring = True
    return Verifier(config, graph_store)


# =============================================================================
# SMOKE TEST  (python verifier.py)
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    print("=" * 70)
    print("S_V: Verifier smoke test")
    print("SpaCy available: %s" % SPACY_AVAILABLE)
    print("Transformers available: %s" % TRANSFORMERS_AVAILABLE)
    print("=" * 70)

    _test_context = [
        "Albert Einstein was a German-born theoretical physicist who developed"
        " the theory of relativity.",
        "Einstein received the Nobel Prize in Physics in 1921 for his"
        " explanation of the photoelectric effect.",
        "He published more than 300 scientific papers and became a symbol of"
        " genius.",
        "Einstein was born in Ulm, Germany, on March 14, 1879.",
        "He worked at the Swiss Patent Office while developing his groundbreaking"
        " theories.",
    ]
    _test_query = (
        "When was Einstein born and what did he receive the Nobel Prize for?"
    )

    print("\nQuery: %s" % _test_query)
    print("Context docs: %d" % len(_test_context))

    _test_cfg: Dict[str, Any] = {
        "llm": {
            "max_context_chars": 2000,
            "max_docs": 5,
            "max_chars_per_doc": 400,
        },
        "agent": {"max_verification_iterations": 3},
        "verifier": {
            "enable_entity_path_validation": True,
            "enable_credibility_scoring": True,
        },
    }
    _verifier = create_verifier(cfg=_test_cfg, enable_pre_validation=True)

    print("\n--- Pre-Generation Validation ---")
    _pre = _verifier.pre_validator.validate(
        context=_test_context,
        query=_test_query,
        entities=["Einstein", "Nobel Prize"],
    )
    print("Status: %s" % _pre.status.value)
    print("Entity-path valid: %s" % _pre.entity_path_valid)
    print("Contradictions: %d" % len(_pre.contradictions))
    print(
        "Filtered context: %d/%d"
        % (len(_pre.filtered_context), len(_test_context))
    )
    print(
        "Credibility scores: %s" % [("%.2f" % s) for s in _pre.credibility_scores]
    )
    print("Validation time: %.0fms" % _pre.validation_time_ms)

    print("\n--- Full Verification (requires Ollama) ---")
    try:
        _result = _verifier.generate_and_verify(
            query=_test_query,
            context=_test_context,
            entities=["Einstein", "Nobel Prize"],
        )
        print("Answer: %s" % _result.answer)
        print("Iterations: %d" % _result.iterations)
        print("All verified: %s" % _result.all_verified)
        print("Verified claims: %d" % len(_result.verified_claims))
        print("Violated claims: %d" % len(_result.violated_claims))
        print("Confidence: %s" % _result.confidence.value)
        print("Total time: %.0fms" % _result.timing_ms)
    except Exception as exc:  # noqa: BLE001
        # Module __main__ smoke test: must complete cleanly even without
        # Ollama running, so any exception (connection refused, missing
        # model, network unreachable) is reported and the script exits
        # zero. The non-LLM verifier logic was already exercised above.
        print("Ollama not available: %s" % exc)
        print("Verifier logic functional; LLM generation requires Ollama.")

    print("\n" + "=" * 70)
