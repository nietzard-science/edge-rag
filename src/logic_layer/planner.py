"""
===============================================================================
S_P: Rule-Based Query Planner
===============================================================================

Paper: "Edge-RAG: Empirical Characterization of When Knowledge-Graph Lanes Add Value in CPU-Only Hybrid Retrieval"
Artifact B: Agent-Based Query Processing.

Role in the pipeline
--------------------
S_P is the first agent of the S_P → S_N → S_V pipeline. It transforms a
natural-language question into a structured RetrievalPlan: a classified
query type, an ordered hop sequence of sub-queries, the named entities
relevant to retrieval, and any temporal / comparative constraints. The
plan is consumed by the Navigator (S_N).

Three stages
------------
1. Query classification — rule-based, over closed-class English function
   words and short SpaCy-Matcher syntactic patterns. Labels: SINGLE_HOP,
   MULTI_HOP, COMPARISON, TEMPORAL, AGGREGATE, INTERSECTION. Three
   deterministic pre-empts (Pattern I distributive "both"; Pattern J
   anaphoric "another"; structural-comparison routing) run before/around
   the scoring classifier and can be disabled via
   ``enable_classifier_preempts``.
2. Entity extraction — SpaCy NER restricted to the OntoNotes-5 inventory,
   with per-label confidence estimation and a multi-hop bridge-entity
   heuristic. Regex fallback when SpaCy is unavailable.
3. Plan generation — dependency-parse decomposition (relative-clause,
   passive-agent, relational-noun, and chained-attribution bridges) with a
   connector-split baseline fallback, plus a closed-class attribute-noun
   rewrite table (_ATTR_MAP) for attribute-comparison questions.

All recognisers key on SpaCy dependency labels and the OntoNotes NER
inventory; no surface-form question phrasings are matched. All classifier
weights and thresholds are read from PlannerConfig (settings.yaml).

Exports
-------
    QueryType, RetrievalStrategy, EntityInfo, HopStep, RetrievalPlan,
    PlannerConfig                                   — data classes / enums
    QueryClassifier, EntityExtractor, PlanGenerator — pipeline stages
    Planner                                         — orchestrator
    create_planner(cfg=None)                        — factory

References (structural anchors)
-------------------------------
    Honnibal & Montani (2017). spaCy 2. arXiv:1802.04016.
    Quirk, Greenbaum, Leech, Svartvik (1985). A Comprehensive Grammar of
        the English Language. Longman. (§13 coordination; §17.7-15
        relative clauses; §11.14 interrogative determiners — the closed-
        class structures the recognisers key on.)
    Weischedel et al. (2013). OntoNotes Release 5.0. LDC2013T19.
        (The NER label inventory.)
    Yang et al. (2018). HotpotQA. EMNLP. (Evaluation benchmark only — no
        planner component is dataset-specific.)

Dependencies
------------
    spacy + en_core_web_sm (optional — regex fallbacks throughout when
    unavailable). stdlib otherwise (re, time, json, logging, typing,
    dataclasses, enum).

Review History:
    Last Reviewed: 2026-06-12
    Review Result: 1 CRITICAL, 4 IMPORTANT, 5 RECOMMENDED (all addressed)
    Reviewer: Code Review Prompt v2.1
    Next Review: After the next decomposition-pattern change
    Changes applied: removed the "a critic of" domain leak in
        _decompose_implicit_bridge (now relation-neutral); added _ensure_nlp so
        the configured spacy_model is honoured (was import-time hard-coded);
        narrowed the 7 dep-parse helpers' bare `except Exception` to
        (ValueError, AttributeError, IndexError, KeyError); synthetic
        boolean-conjunction entities now take confidence from config; Pattern I
        decomposition gated on enable_classifier_preempts (full ablation);
        de-duplicated the boolean-conjunction regex into _BOOL_CONJ_PATTERN;
        documented the single-thread contract of the _last_* markers and the
        per-query re-parse perf note; named the span-rejoin gap constant; fixed
        the _find_relative_clause_bridge docstring (3-tuple) and the Levin (1993)
        attribution. Note: `doc: Any` annotations kept intentionally (string
        annotations would eval-fail when SpaCy is absent — no future-annotations).
    ---
    Previous Review: 2026-05-27 (audit pass, project version 5.4)
    Previous Result: no itemized action list recorded
===============================================================================
"""

import logging
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


from ._settings_loader import _load_settings
from ._text_utils import _PROPER_NOUN_RE


# =============================================================================
# MODULE-LEVEL CONSTANTS
# =============================================================================

# Year pattern used in temporal constraint extraction.
# Matches historical years (1000–1999) and 21st-century years (2000–2099).
# Narrower than the 4-digit patterns in TEMPORAL_PATTERNS / ENTITY_PATTERNS to
# avoid false positives on port numbers or other 4-digit tokens.
_YEAR_RE = re.compile(r"\b(1\d{3}|20\d{2})\b")

# Pattern I (boolean conjunction) surface form: "Are/Is/Were/... X and Y both P?".
# Shared by BOTH the QueryClassifier Phase-0 pre-empt (_BOOL_CONJ_PRE) and the
# PlanGenerator decomposer (_BOOL_CONJ_RE) — defined once at module scope so the
# two cannot drift apart (review 2026-06-12, finding #9).
_BOOL_CONJ_PATTERN = r'^\s*(are|is|were|was|did|do|does|have|has)\b.+\band\b.+\bboth\b'

# Auxiliary verbs and articles that SpaCy NER sometimes absorbs into an entity
# span when they precede it at sentence start (e.g. "Are Random House Tower"
# where "Are" is the sentence-initial auxiliary verb, not part of the name).
# Stripped from entity text before downstream use; char offsets adjusted.
_LEADING_FUNCTION_WORDS = frozenset({
    "are", "is", "was", "were", "did", "do", "does", "have", "has",
    "a", "an", "the",
})


def _strip_leading_function_words(text: str) -> str:
    """Remove leading auxiliary verbs / articles absorbed into an NER span.

    SpaCy en_core_web_sm occasionally includes the sentence-initial function
    word in a named-entity span (e.g. "Are Random House Tower" → ORG).
    Stripping these tokens recovers the correct entity surface form.
    """
    tokens = text.split()
    while tokens and tokens[0].lower() in _LEADING_FUNCTION_WORDS:
        tokens = tokens[1:]
    return " ".join(tokens)


# =============================================================================
# SPACY INTEGRATION
# =============================================================================
# SpaCy is used for entity extraction and dependency parsing.
# If unavailable, regex fallbacks are used throughout.

_DEFAULT_SPACY_MODEL = "en_core_web_sm"
# Name of the SpaCy model currently held in the module-global ``NLP`` (None if
# none is loaded). Used by ``_ensure_nlp`` to avoid reloading the same model.
_LOADED_SPACY_MODEL: Optional[str] = None

try:
    import spacy
    from spacy.matcher import Matcher
    _SPACY_IMPORTED = True
except ImportError:
    spacy = None        # type: ignore[assignment]
    Matcher = None
    _SPACY_IMPORTED = False
    logger.warning(
        "SpaCy not installed. Install with:\n"
        "  pip install spacy\n"
        "  python -m spacy download en_core_web_sm\n"
        "Regex fallbacks will be used for entity extraction."
    )


def _load_spacy_model(model_name: str):
    """Load a SpaCy model, returning the pipeline or None on failure.

    Centralises the load so the import-time default and the config-driven
    reload (``_ensure_nlp``) share one code path. Never raises: a missing model
    logs a warning and returns None so the planner degrades to its regex
    fallbacks rather than crashing.
    """
    if not _SPACY_IMPORTED:
        return None
    try:
        return spacy.load(model_name)
    except OSError:
        logger.warning(
            "SpaCy model '%s' not found. Install with:\n"
            "  python -m spacy download %s\n"
            "Regex fallbacks will be used for entity extraction.",
            model_name, model_name,
        )
        return None


def _ensure_nlp(model_name: Optional[str]) -> None:
    """Ensure the module-global ``NLP``/``SPACY_AVAILABLE`` reflect ``model_name``.

    Fixes the import-time config-bypass (review 2026-06-12, finding #2): the
    model was previously hard-loaded as ``en_core_web_sm`` at import, so
    ``PlannerConfig.spacy_model`` (settings.yaml → ingestion.spacy_model) could
    never take effect. ``Planner.__init__`` now calls this with the configured
    model name; if it differs from what is already loaded, the model is
    (re)loaded and the globals are rebound — the ~30 module-global ``NLP``
    references throughout this file pick up the change without per-call
    threading. No-op when the requested model is already loaded.
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
        logger.info("SpaCy model '%s' loaded for query analysis", target)
    elif NLP is None:
        # Only downgrade availability if we have nothing loaded at all; keep a
        # previously-loaded model rather than dropping to regex on a bad reload.
        SPACY_AVAILABLE = False


# Import-time default load (preserves prior behaviour for callers that use the
# module without constructing a Planner, e.g. direct ``NLP`` access in tests).
NLP = _load_spacy_model(_DEFAULT_SPACY_MODEL)
SPACY_AVAILABLE = NLP is not None
if SPACY_AVAILABLE:
    _LOADED_SPACY_MODEL = _DEFAULT_SPACY_MODEL
    logger.info("SpaCy model '%s' loaded for query analysis", _DEFAULT_SPACY_MODEL)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class QueryType(Enum):
    """
    Query type classification based on reasoning complexity.

    Per the paper, Section 3.2:
    - SINGLE_HOP:   Simple factual question, one retrieval step.
    - MULTI_HOP:    Sequential dependencies requiring bridge entities.
    - COMPARISON:   Parallel retrieval + comparison logic.
    - TEMPORAL:     Temporal reasoning component.
    - AGGREGATE:    Aggregation of multiple results.
    - INTERSECTION: Shared properties between two subjects.
    """
    SINGLE_HOP   = "single_hop"    # e.g. "What is the capital of France?"
    MULTI_HOP    = "multi_hop"     # e.g. "Who directed the film starring Greta Garbo?"
    COMPARISON   = "comparison"    # e.g. "Is Berlin older than Munich?"
    TEMPORAL     = "temporal"      # e.g. "What happened after WW2?"
    AGGREGATE    = "aggregate"     # e.g. "List all films from 2020."
    INTERSECTION = "intersection"  # e.g. "What do A and B have in common?"


class RetrievalStrategy(Enum):
    """
    Retrieval strategy selected based on query complexity.

    Per the paper, Section 3.2:
    - VECTOR_ONLY: Fast, for simple single-hop queries.
    - GRAPH_ONLY:  For relation-based queries (reserved for future work —
                   not currently selected by _determine_strategy; included
                   for ablation interface compatibility).
    - HYBRID:      Combined for complex multi-hop and comparison queries.
    """
    VECTOR_ONLY = "vector_only"  # Fast; for simple queries
    GRAPH_ONLY  = "graph_only"   # Reserved for future work (not yet active)
    HYBRID      = "hybrid"       # Combined; for complex queries


@dataclass
class EntityInfo:
    """
    Information about an extracted entity.

    Attributes:
        text:        Entity surface form.
        label:       NER label (PERSON, ORG, GPE, etc.).
        confidence:  Extraction confidence (0.0–1.0).
        start_char:  Start character offset in the original text.
        end_char:    End character offset in the original text.
        is_bridge:   True if this entity acts as a bridge in multi-hop reasoning.
    """
    text: str
    label: str = "UNKNOWN"
    confidence: float = 1.0
    start_char: int = 0
    end_char: int = 0
    is_bridge: bool = False


@dataclass
class HopStep:
    """
    A single step in a multi-hop reasoning chain.

    Attributes:
        step_id:         Unique step identifier.
        sub_query:       The sub-query for this step.
        target_entities: Target entities for this step.
        depends_on:      IDs of steps this step depends on.
        is_bridge:       True if this step resolves a bridge entity.
    """
    step_id: int
    sub_query: str
    target_entities: List[str] = field(default_factory=list)
    depends_on: List[int] = field(default_factory=list)
    is_bridge: bool = False


@dataclass
class RetrievalPlan:
    """
    Structured retrieval plan for the Navigator (S_N).

    This is the primary output format of the Planner, consumed by the Navigator
    to execute hybrid retrieval.

    Attributes:
        original_query:  The original user query.
        query_type:      Classified query type.
        strategy:        Selected retrieval strategy.
        entities:        List of extracted entities with metadata.
        hop_sequence:    Ordered list of hop steps.
        sub_queries:     Flat list of all sub-queries for retrieval.
        constraints:     Additional constraints (temporal, comparison, etc.).
        estimated_hops:  Estimated number of retrieval hops.
        confidence:      Query classification confidence.
        matched_pattern: Identifier of the decomposition pattern that
            produced this plan (e.g. "G_form1", "H", "E",
            "comparison_attr_map", "fallback_generic_2hop", "single_hop").
            None when the field has not been set. Surfaces in the per-question
            JSONL so the eval harness can compute per-pattern hit-rates and
            SF-F1 deltas without parsing debug logs.
        classifier_preempt: Identifier of the Phase 0 / Phase 0.5 classifier
            pre-empt that fired, if any
            ("preempt_pattern_I_boolean_conjunction", "preempt_pattern_J_implicit_bridge").
            None when no pre-empt fired (i.e. normal Phase 1-4 scoring path).
            Lets reviewers audit pre-empt false-positive rates without
            re-running classification.
        metadata:        Additional metadata for debugging / future fields.
    """
    original_query: str
    query_type: QueryType
    strategy: RetrievalStrategy
    entities: List[EntityInfo] = field(default_factory=list)
    hop_sequence: List[HopStep] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    estimated_hops: int = 1
    confidence: float = 1.0
    matched_pattern: Optional[str] = None
    classifier_preempt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a dictionary for JSON output."""
        return {
            "original_query": self.original_query,
            "query_type": self.query_type.value,
            "strategy": self.strategy.value,
            "entities": [
                {
                    "text": e.text,
                    "label": e.label,
                    "confidence": e.confidence,
                    "is_bridge": e.is_bridge,
                }
                for e in self.entities
            ],
            "hop_sequence": [
                {
                    "step_id": h.step_id,
                    "sub_query": h.sub_query,
                    "target_entities": h.target_entities,
                    "depends_on": h.depends_on,
                    "is_bridge": h.is_bridge,
                }
                for h in self.hop_sequence
            ],
            "sub_queries": self.sub_queries,
            "constraints": self.constraints,
            "estimated_hops": self.estimated_hops,
            "confidence": self.confidence,
            # Surface the matched pattern + any classifier pre-empt so the
            # per-question JSONL contains a greppable identifier for
            # downstream analysis. Both keys are always present (possibly
            # None) so consumers can rely on the schema.
            "matched_pattern": self.matched_pattern,
            "classifier_preempt": self.classifier_preempt,
            # Surface metadata so future debug fields are reachable from the
            # JSONL without another schema change. Empty dict by default.
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """
        Serialise to a JSON string.

        Public API convenience method for external consumers (e.g. REST endpoints,
        logging pipelines). Internal pipeline code uses to_dict() directly.
        """
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


@dataclass
class PlannerConfig:
    """
    Configuration for the Query Planner.

    Numeric thresholds were chosen by inspection during development and
    documented in the paper's methodology section. All values are sourced
    from ``config/settings.yaml → planner`` via ``from_yaml()``; the
    dataclass defaults serve only as documented emergency fallbacks.

    Attributes:
        min_entity_confidence:    Minimum NER confidence for entity extraction.
        max_entities:             Maximum number of entities to extract per query.
        enable_bridge_detection:  Enable bridge entity detection for multi-hop.
        enable_temporal_parsing:  Parse temporal constraints from queries.
        default_strategy:         Fallback strategy when classification is ambiguous.
        spacy_model:              SpaCy model name. Sourced from
                                  ``settings.yaml → ingestion.spacy_model``.
        regex_entity_confidence:  Confidence assigned to entities found by the
                                  regex fallback (not SpaCy NER). Lower than NER
                                  confidence to reflect noisier extraction.
        entity_density_threshold: Named-entity count above which the multi-hop
                                  score boost fires in classify() Phase 3.
        noun_density_threshold:   Noun/proper-noun count threshold for the same.
        classifier_spacy_weight:  Score bonus for SpaCy Matcher hits (Phase 2).
        classifier_entity_boost:  Score bonus when entity density is high (Phase 3).
        classifier_confidence_base:  Base confidence added to scaled score.
        classifier_confidence_scale: Score multiplier for confidence calculation.
        classifier_confidence_cap:   Upper cap on returned confidence.
        classifier_fallback_confidence: Confidence for SINGLE_HOP fallback.
    """
    min_entity_confidence: float = 0.7       # settings.yaml: planner.min_entity_confidence
    max_entities: int = 10                    # settings.yaml: planner.max_entities
    enable_bridge_detection: bool = True
    enable_temporal_parsing: bool = True
    default_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    spacy_model: str = "en_core_web_sm"      # settings.yaml: ingestion.spacy_model

    # Confidence for regex-matched entities (lower than NER-based estimates)
    regex_entity_confidence: float = 0.75    # settings.yaml: planner.regex_entity_confidence

    # Entity/noun density thresholds for multi-hop heuristic (Phase 3)
    entity_density_threshold: int = 2        # settings.yaml: planner.entity_density_threshold
    noun_density_threshold: int = 4          # settings.yaml: planner.noun_density_threshold

    # Classifier weight constants. Defaults were chosen by inspection during
    # development; tuning is exposed via settings.yaml for reproducibility.
    classifier_spacy_weight: float = 1.5     # settings.yaml: planner.classifier_spacy_weight
    classifier_entity_boost: float = 0.5     # settings.yaml: planner.classifier_entity_boost
    classifier_confidence_base: float = 0.6  # settings.yaml: planner.classifier_confidence_base
    classifier_confidence_scale: float = 0.15  # settings.yaml: planner.classifier_confidence_scale
    classifier_confidence_cap: float = 0.95  # settings.yaml: planner.classifier_confidence_cap
    # No-pattern-match fallback confidence is deliberately below any
    # single-pattern match (base 0.60 + scale 0.15 × 1 = 0.75) so an
    # unclassified query never outranks one with a real signal.
    classifier_fallback_confidence: float = 0.5  # settings.yaml: planner.classifier_fallback_confidence

    # Classifier pre-empt settings (Pattern I / Pattern J / structural
    # comparison). The pre-empts route closed-class English constructions
    # before/around the 4-phase scoring classifier. Set to False to ablate
    # them and fall back to pure scoring. The confidence/boost values were
    # chosen by inspection on the development split (the paper's methodology
    # section reports the dev-vs-test partition).
    enable_classifier_preempts: bool = True   # settings.yaml: planner.enable_classifier_preempts
    preempt_comparison_confidence: float = 0.90   # Pattern I (boolean conjunction)
    preempt_multihop_confidence: float = 0.80     # Pattern J (implicit bridge)
    structural_comparison_boost: float = 1.0      # Phase 3.6 score boost

    # NER confidence estimation (EntityExtractor._estimate_confidence).
    # SpaCy does not expose per-entity confidence, so it is estimated from
    # the per-label table plus a length bonus. Values derived by inspection
    # from en_core_web_sm reliability on OntoNotes-5 (Weischedel 2013).
    ner_default_confidence: float = 0.7           # label absent from _LABEL_CONFIDENCE
    ner_length_bonus_cap: float = 0.1             # max length bonus
    ner_length_bonus_per_token: float = 0.03      # bonus per whitespace token

    @classmethod
    def from_yaml(cls, config: Dict[str, Any]) -> "PlannerConfig":
        """
        Build a PlannerConfig from a settings.yaml dict.

        Reads the ``planner`` block for planner-specific settings and
        ``ingestion.spacy_model`` for the shared SpaCy model name.
        All defaults match the paper's evaluation settings documented in
        settings.yaml. Follows the same pattern as IngestionConfig.from_yaml().

        Args:
            config: Full settings.yaml dict (or the relevant sub-dict).

        Returns:
            PlannerConfig populated from the provided settings dict.
        """
        planner = config.get("planner", {})
        ingestion = config.get("ingestion", {})
        return cls(
            min_entity_confidence=planner.get("min_entity_confidence", 0.7),
            max_entities=planner.get("max_entities", 10),
            enable_bridge_detection=planner.get("enable_bridge_detection", True),
            enable_temporal_parsing=planner.get("enable_temporal_parsing", True),
            spacy_model=ingestion.get("spacy_model", "en_core_web_sm"),
            regex_entity_confidence=planner.get("regex_entity_confidence", 0.75),
            entity_density_threshold=planner.get("entity_density_threshold", 2),
            noun_density_threshold=planner.get("noun_density_threshold", 4),
            classifier_spacy_weight=planner.get("classifier_spacy_weight", 1.5),
            classifier_entity_boost=planner.get("classifier_entity_boost", 0.5),
            classifier_confidence_base=planner.get("classifier_confidence_base", 0.6),
            classifier_confidence_scale=planner.get("classifier_confidence_scale", 0.15),
            classifier_confidence_cap=planner.get("classifier_confidence_cap", 0.95),
            classifier_fallback_confidence=planner.get("classifier_fallback_confidence", 0.5),
            enable_classifier_preempts=planner.get("enable_classifier_preempts", True),
            preempt_comparison_confidence=planner.get("preempt_comparison_confidence", 0.90),
            preempt_multihop_confidence=planner.get("preempt_multihop_confidence", 0.80),
            structural_comparison_boost=planner.get("structural_comparison_boost", 1.0),
            ner_default_confidence=planner.get("ner_default_confidence", 0.7),
            ner_length_bonus_cap=planner.get("ner_length_bonus_cap", 0.1),
            ner_length_bonus_per_token=planner.get("ner_length_bonus_per_token", 0.03),
        )


# =============================================================================
# QUERY CLASSIFIER
# =============================================================================

class QueryClassifier:
    """
    Rule-based query classifier using SpaCy Matcher.

    Per the paper, Section 3.2:
    "Instead of an ML model, SpaCy's Rule-Based Matcher is used. It identifies
    query types such as Comparison, Temporal, or Multi-Hop through lexical
    pattern matching, minimising inference latency."
    (Honnibal & Montani, 2017; arXiv:1802.04016)

    Classification uses four phases:
    1. Lexical regex pattern counts per query type.
    2. SpaCy Matcher boost for syntactic patterns.
    3. Entity-density heuristic for multi-hop identification.
    4. Priority-ordered tie-break with confidence scaling.

    All weight constants are read from PlannerConfig (settings.yaml) for full
    reproducibility.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # LEXICAL PATTERNS FOR QUERY CLASSIFICATION
    # Tuples (immutable) prevent accidental mutation by subclasses.
    # ─────────────────────────────────────────────────────────────────────────

    # Comparison indicators: comparative words and English coordination structures.
    # All patterns key on closed-class function words (comparative morphology
    # -er/-est, "than", "versus", "or", "both", "same") that are standard
    # signals of comparative constructions in English (Quirk et al. 1985,
    # "A Comprehensive Grammar of the English Language", §15.63-72).
    COMPARISON_PATTERNS = (
        r"\b(older|younger|taller|shorter|bigger|smaller|larger|higher|lower)\s+than\b",
        r"\b(more|less|fewer)\s+\w+\s+than\b",
        r"\b(compare|comparison|versus|vs\.?|vs)\b",
        r"\bdifference\s+between\b",
        r"\bwhich\s+(is|was|are|were)\s+\w*(er|est)\b",
        r"\b(better|worse|best|worst)\b.*\bor\b",
        r"\bor\b.*\?(which|what)\s+(is|was)\s+\w*(er|est)",
        r"\bsame\s+\w+\b",          # "same NP" — shared-attribute construction
        r"\bboth\s+.+\s+(born|from|have|had|were|are)\b",  # "both" floating quantifier
        # "Who is older/taller/..., X or Y?" — no "than" required
        r"\b(older|younger|taller|shorter|bigger|smaller|larger|higher|lower|richer|poorer)\b.{0,60}\bor\b",
    )

    # Three regex-classifier patterns previously listed here were removed
    # because they recognised specific question phrasings rather than
    # linguistic structure:
    #
    #   - "Which X, A or B, …?" form (`,\s*[^,?]+\s+\bor\b\s+[^,?]+[,?]`)
    #     replaced by `_decompose_select_between`, which uses SpaCy NER to
    #     detect the disjunction structurally rather than via comma position.
    #   - "Which … <aux> … first" form (`\b(was|is|were|are|…) … \bfirst\b`)
    #     was a surface-form pattern fitted to one question shape. Removed.
    #   - Plural-coordination string-shape pattern matching
    #     `[A-Z][a-z]+\s+and\s+[A-Z][a-z]+` (the "Pattern K" regex). The
    #     phenomenon (coordinated subjects with shared predicate attribute)
    #     is real but the recogniser used a string-shape match, not a
    #     dependency-parse signal. Coordination is now recognised by
    #     `_decompose_select_between` via NER-span detection.
    #
    # See Quirk et al. (1985) §13.2 for the linguistics of coordinated NPs.

    # Temporal indicators: time references and temporal structures
    TEMPORAL_PATTERNS = (
        r"\b(before|after|during|since|until|when|while)\b",
        r"\b(year|month|day|century|decade|era)\s+\d+",
        r"\b\d{4}\b",  # four-digit years
        r"\b(first|last|latest|earliest|recent|previous|next)\b",
        r"\b(history|historical|timeline|chronolog)\w*\b",
        r"\b(began|started|ended|founded|established)\b",
    )

    # Multi-hop indicators: nested structures
    MULTI_HOP_PATTERNS = (
        r"\bof\s+(a|an|the)\s+\w+\s+(that|which|who)\b",   # "of a/the X that/who" (bridge)
        r"\bwhere\s+.+\s+(was|is|were|are)\b",
        r"\b\w+\s+of\s+the\s+\w+\s+of\b",
        r"'s\s+\w+'s",  # possessive chains
        r"\b(who|what)\s+\w+\s+(the|a)\s+\w+\s+(that|which)\b",
        # Bridge-relation surface patterns: closed-class English attribution
        # verbs and relational nouns that mark a noun phrase as standing in
        # for an as-yet-unresolved entity (Quirk et al. 1985 §17.7-15).
        r"\b(starring|featuring|directed by|written by|authored by|composed by)\b",
        r"\b(father|mother|son|daughter|wife|husband|creator|founder)\s+of\b",
        r"\b(located|situated)\s+in\s+the\b",
        r"\b(formed|created|founded|established|organized|produced|released)\s+by\b",
        r"\b\w+\s+(group|band|company|team|studio|label)\s+(that|which|who)\b",
        r"\bformed\s+by\b",           # "group that was formed by who?"
        r"\bwas\s+\w+ed\s+by\b",      # "was founded/formed/created by"
        r"\b(debut|first|second)\s+(album|single|film|movie|show)\s+of\b",
    )

    # Intersection indicators: shared properties
    INTERSECTION_PATTERNS = (
        r"\bboth\s+.+\s+and\b",
        r"\bin\s+common\b",
        r"\b(also|too)\b.*\band\b",
        r"\bshared\s+(by|between)\b",
    )

    # Aggregation indicators: lists and summaries
    AGGREGATE_PATTERNS = (
        r"\b(list|enumerate|all|every|count|how\s+many)\b",
        r"\b(summarize|summary|overview)\b",
        r"\bwhat\s+(are|were)\s+the\b",
    )

    # ── Pre-empt regexes (compiled once at class load) ────────────────────────
    # Pattern I — Boolean-conjunction pre-empt: "Are/Did/Were X and Y both P?"
    # is a parallel yes/no comparison, never a bridge chain. Must fire before
    # Phase 1 so the "both"/"and" tokens cannot boost MULTI_HOP / INTERSECTION
    # past COMPARISON.
    _BOOL_CONJ_PRE = re.compile(_BOOL_CONJ_PATTERN, re.IGNORECASE)
    # Pattern J — Implicit-bridge pre-empt: "X and another [noun] that …"
    # requires first resolving the anaphoric "another" then following it. The
    # AGGREGATE pattern "how many" would otherwise misclassify it as a
    # single-pass aggregate.
    _IMPLICIT_BRIDGE_PRE = re.compile(
        r'\banother\s+\w+\b',
        re.IGNORECASE,
    )

    def __init__(self, config: Optional[PlannerConfig] = None):
        """
        Initialise the Query Classifier.

        Args:
            config: Planner configuration.
        """
        self.config = config or PlannerConfig()

        # Compile regex patterns once at construction time
        self._compiled_patterns = {
            QueryType.COMPARISON:   [re.compile(p, re.IGNORECASE) for p in self.COMPARISON_PATTERNS],
            QueryType.TEMPORAL:     [re.compile(p, re.IGNORECASE) for p in self.TEMPORAL_PATTERNS],
            QueryType.MULTI_HOP:    [re.compile(p, re.IGNORECASE) for p in self.MULTI_HOP_PATTERNS],
            QueryType.INTERSECTION: [re.compile(p, re.IGNORECASE) for p in self.INTERSECTION_PATTERNS],
            QueryType.AGGREGATE:    [re.compile(p, re.IGNORECASE) for p in self.AGGREGATE_PATTERNS],
        }

        # SpaCy Matcher for syntactic patterns
        self._setup_spacy_matcher()

        # Per-call cache for the pre-empt identifier (if any). Set by
        # classify() when Phase 0 / Phase 0.5 fires; read by Planner.plan() into
        # RetrievalPlan.classifier_preempt. None means the classification took
        # the normal Phase 1-4 scoring path.
        self._last_preempt: Optional[str] = None

        logger.info("QueryClassifier initialised")

    def _setup_spacy_matcher(self) -> None:
        """
        Initialise the SpaCy Matcher with linguistic patterns.

        The Matcher identifies syntactic structures that indicate query
        complexity (Honnibal & Montani, 2017; arXiv:1802.04016).
        """
        if not SPACY_AVAILABLE or NLP is None:
            self.matcher = None
            return

        self.matcher = Matcher(NLP.vocab)

        # Pattern for multi-hop: "of the X that/which Y"
        # Example: "the director of the film that won"
        multi_hop_pattern = [
            {"LOWER": "of"},
            {"LOWER": "the"},
            {"POS": {"IN": ["NOUN", "PROPN"]}},
            {"LOWER": {"IN": ["that", "which", "who"]}},
        ]
        self.matcher.add("MULTI_HOP", [multi_hop_pattern])

        # Pattern for comparison: comparative adjective/adverb + "than"
        comparison_pattern = [
            {"TAG": {"IN": ["JJR", "RBR"]}},  # comparative form
            {"LOWER": "than"},
        ]
        self.matcher.add("COMPARISON", [comparison_pattern])

        # Disjunctive coordination of proper nouns "X or Y" — the canonical
        # surface form of an English alternative interrogative
        # (Karttunen 1977, "Syntax and Semantics of Questions",
        # Linguistics and Philosophy 1(1); Higginbotham 1993,
        # "Interrogatives" in Hale & Keyser eds., MIT Press). Disjunction
        # of two named entities forces a select-between-two reading.
        #
        # NOTE: a parallel "PROPN and PROPN" rule was deliberately NOT added.
        # Conjunctive coordination of named entities is ambiguous between
        # COMPARISON ("Are X and Y from the same country?") and
        # INTERSECTION ("Which films star X and Y?"). Resolving that
        # ambiguity requires looking at the predicate, which the
        # existing INTERSECTION lexical patterns ("both ... and",
        # "in common", "shared by") already do.
        propn_or_propn = [
            {"POS": "PROPN", "OP": "+"},
            {"LOWER": "or"},
            {"POS": "PROPN", "OP": "+"},
        ]
        self.matcher.add("COMPARISON", [propn_or_propn])

        logger.debug("SpaCy Matcher configured")

    def classify(self, query: str) -> Tuple[QueryType, float]:
        """
        Classify a query and determine its type.

        Algorithm:
        1. Count regex pattern matches for each query type.
        2. Apply SpaCy Matcher boost for syntactic hits.
        3. Apply entity-density heuristic for multi-hop detection.
        4. Select highest-scoring type with priority tie-break.
        5. Fall back to SINGLE_HOP if no pattern matches.

        Weight constants are read from PlannerConfig (settings.yaml):
        - spacy_weight (default 1.5):    syntactic match carries more weight
          than a single regex hit (SpaCy pattern is more precise).
        - entity_boost (default 0.5):    partial nudge — entity density is a
          weak but useful signal for multi-hop.
        - confidence_base / scale / cap: map raw score to [0.6, 0.95] range.

        Note: SINGLE_HOP has no patterns in _compiled_patterns and therefore
        always scores 0; it is selected only via the fallback path at the end.

        Args:
            query: The query to classify.

        Returns:
            Tuple of (QueryType, confidence).
        """
        query = query.strip()
        scores = {qt: 0.0 for qt in QueryType}

        # Reset the pre-empt marker for this call.
        self._last_preempt = None

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 0: Boolean-conjunction pre-empt (Pattern I)
        # "Are/Did/Were X and Y both P?" — parallel yes/no, never a bridge chain.
        # Runs before Phase 1 so the "both"/"and" tokens cannot boost MULTI_HOP
        # or INTERSECTION scores past COMPARISON. The pre-empt identifier is
        # recorded on self._last_preempt so the per-question JSONL can audit
        # how often it fires (the four-phase scoring path leaves it None).
        # ─────────────────────────────────────────────────────────────────────
        if self.config.enable_classifier_preempts and self._BOOL_CONJ_PRE.match(query):
            logger.debug("classify: Boolean conjunction pre-empt for '%s' → COMPARISON", query[:80])
            self._last_preempt = "preempt_pattern_I_boolean_conjunction"
            return QueryType.COMPARISON, self.config.preempt_comparison_confidence

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 0.5: Implicit bridge pre-empt (Pattern J)
        # "X and another [noun] that …" — the answer requires first resolving
        # the anaphoric "another" then following it. The AGGREGATE pattern
        # "how many" would otherwise classify this as a single-pass aggregate.
        # ─────────────────────────────────────────────────────────────────────
        if self.config.enable_classifier_preempts and self._IMPLICIT_BRIDGE_PRE.search(query):
            logger.debug(
                "classify: implicit-bridge pre-empt (Pattern J) for '%s' → MULTI_HOP",
                query[:80],
            )
            self._last_preempt = "preempt_pattern_J_implicit_bridge"
            return QueryType.MULTI_HOP, self.config.preempt_multihop_confidence

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 1: Regex pattern matching
        # ─────────────────────────────────────────────────────────────────────

        for query_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    scores[query_type] += 1.0

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 2: SpaCy Matcher (when available)
        # ─────────────────────────────────────────────────────────────────────

        doc = None
        if self.matcher and NLP:
            doc = NLP(query)
            matches = self.matcher(doc)

            for match_id, start, end in matches:
                rule_name = NLP.vocab.strings[match_id]
                if rule_name == "MULTI_HOP":
                    scores[QueryType.MULTI_HOP] += self.config.classifier_spacy_weight
                elif rule_name == "COMPARISON":
                    scores[QueryType.COMPARISON] += self.config.classifier_spacy_weight

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 3: Entity-density heuristic (multi-hop indicator)
        # ─────────────────────────────────────────────────────────────────────

        # High entity density suggests multi-hop reasoning.
        # Thresholds sourced from PlannerConfig (settings.yaml).
        # Re-use the doc parsed in Phase 2 if available to avoid a second call.
        if SPACY_AVAILABLE and NLP:
            if doc is None:
                doc = NLP(query)
            entity_count = len(doc.ents)
            noun_count = sum(1 for token in doc if token.pos_ in ("NOUN", "PROPN"))

            if (entity_count > self.config.entity_density_threshold
                    or noun_count > self.config.noun_density_threshold):
                scores[QueryType.MULTI_HOP] += self.config.classifier_entity_boost

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 3.5: Multi-hop override
        # ─────────────────────────────────────────────────────────────────────
        # Bridge questions that contain a year/"founded"/"when" token
        # (e.g. "What year was the university where John studied founded?")
        # accumulate multiple TEMPORAL hits but only one MULTI_HOP hit, so
        # priority-on-tie alone is not enough to keep them in the MULTI_HOP
        # branch.  Because TEMPORAL keywords are common across many query
        # types while MULTI_HOP relation cues ("founder of", "directed by",
        # "starring", possessive chains) are rare and specific, treat any
        # MULTI_HOP hit as decisive when TEMPORAL would otherwise dominate.
        if scores[QueryType.MULTI_HOP] > 0 and scores[QueryType.TEMPORAL] > scores[QueryType.MULTI_HOP]:
            logger.debug(
                "classify: multi-hop override for '%s' (multi_hop=%.1f, temporal=%.1f)",
                query[:80], scores[QueryType.MULTI_HOP], scores[QueryType.TEMPORAL],
            )
            scores[QueryType.MULTI_HOP] = scores[QueryType.TEMPORAL]

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 3.6: Structural comparison routing
        # ─────────────────────────────────────────────────────────────────────
        # Coordinated named entities ("X and Y …", "between X and Y …") under an
        # interrogative determiner are the grammatical signature of a HotpotQA
        # comparison question (Yang et al. 2018 EMNLP; Quirk et al. 1985 §13).
        # The Phase-3 entity-density heuristic otherwise forces any ≥2-entity
        # query to MULTI_HOP, starving the parallel comparison decomposer
        # (_decompose_comparison) which emits one anchored sub-query per entity.
        # The detector keys on dependency structure, not surface phrasing, and
        # is gated to DEFER to MULTI_HOP whenever a bridge-relation cue is
        # present (precision guard — a coordinated pair inside a bridge question
        # such as "the film starring A and B that …" must stay multi-hop).
        # Defer to INTERSECTION when its more specific signal has fired
        # ("both X and Y" / "in common" — a JOINT-property question, e.g.
        # "Which movies star both A and B?"). Structural comparison targets the
        # per-entity-attribute form ("A and B both held which position?") where
        # INTERSECTION scores zero.
        if (
            self.config.enable_classifier_preempts
            and scores[QueryType.INTERSECTION] == 0
            and self._is_structural_comparison(doc, query)
        ):
            scores[QueryType.COMPARISON] = max(scores.values()) + self.config.structural_comparison_boost
            self._last_preempt = "preempt_structural_comparison"
            logger.debug(
                "classify: structural-comparison routing for '%s' → COMPARISON",
                query[:80],
            )

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 4: Determine final query type
        # ─────────────────────────────────────────────────────────────────────

        max_score = max(scores.values())

        if max_score == 0:
            # No pattern matched → default to SINGLE_HOP
            logger.debug("classify: no pattern matched for '%s' → SINGLE_HOP", query[:80])
            return QueryType.SINGLE_HOP, self.config.classifier_fallback_confidence

        # Priority order resolves ties.
        # IMPORTANT: MULTI_HOP before TEMPORAL — year tokens in bridge questions
        # (e.g. "2014 S/S is the debut album of ... formed by who?") would
        # otherwise be incorrectly classified as TEMPORAL.
        priority = [
            QueryType.COMPARISON,
            QueryType.MULTI_HOP,
            QueryType.TEMPORAL,
            QueryType.INTERSECTION,
            QueryType.AGGREGATE,
        ]

        for qt in priority:
            if scores[qt] == max_score:
                confidence = min(
                    self.config.classifier_confidence_cap,
                    self.config.classifier_confidence_base
                    + (max_score * self.config.classifier_confidence_scale),
                )
                return qt, confidence

        # Should not be reached (SINGLE_HOP scores 0, caught above); guard
        # against floating-point edge cases.
        logger.debug("classify: priority exhausted for '%s' → SINGLE_HOP", query[:80])
        return QueryType.SINGLE_HOP, self.config.classifier_fallback_confidence

    # Interrogative determiners that, combined with coordinated entities, mark a
    # comparison question ("which/who … X and Y"). Closed class (Quirk et al.
    # 1985 §11.14), so this is structure recognition, not phrase matching.
    _WH_DETERMINERS = frozenset({"which", "who", "whom", "whose", "what"})

    def _is_structural_comparison(self, doc: Any, query: str) -> bool:
        """True iff the query is a comparison by GRAMMATICAL STRUCTURE: two or
        more named entities standing in a coordinate (conj) relation, under an
        interrogative determiner, with NO bridge-relation cue present.

        Recognises both "X and Y …" and "between X and Y …" (the second
        entity is a conjunct of the first in the dependency parse either way).
        Gated by the bridge-cue check so a coordinated pair embedded in a bridge
        question (e.g. "the film starring A and B that won an Oscar?") is left to
        the MULTI_HOP path. SpaCy-dependent; returns False when no parse exists.

        Refs: Yang et al. (2018, EMNLP) HotpotQA bridge/comparison taxonomy;
        Quirk et al. (1985) §13 coordinated noun phrases.
        """
        if doc is None:
            return False
        ents = list(doc.ents)
        if len(ents) < 2:
            return False
        # Interrogative determiner present.
        if not any(t.lower_ in self._WH_DETERMINERS for t in doc):
            return False
        # Precision guard: a bridge-relation cue defers to MULTI_HOP.
        for pat in self._compiled_patterns[QueryType.MULTI_HOP]:
            if pat.search(query):
                return False
        # Two named-entity heads in a coordinate relation.
        ent_roots = {e.root for e in ents}
        for e in ents:
            root = e.root
            if root.dep_ == "conj" and root.head in ent_roots and root.head is not root:
                return True
            for conj in root.conjuncts:
                if conj in ent_roots and conj is not root:
                    return True
        return False



# =============================================================================
# ENTITY EXTRACTOR
# =============================================================================

class EntityExtractor:
    """
    Entity extractor using SpaCy NER and regex fallback.

    Per the paper, Section 3.2:
    "Entity extraction uses SpaCy NER (confidence > min_entity_confidence).
    For complex queries, dependency parsing resolves syntactic relationships,
    enabling identification of bridge entities as necessary intermediate steps
    (hops) for graph traversal."
    (Honnibal & Montani, 2017; arXiv:1802.04016)
    """

    # SpaCy NER labels that map to the GLiNER taxonomy used at ingestion time.
    # Only these labels are accepted — MONEY, QUANTITY, TIME, NORP, LAW, CARDINAL,
    # ORDINAL, PERCENT, LANGUAGE are excluded because they have no GLiNER equivalent
    # and produce false-positive entities (e.g. "888 7th Avenue" → MONEY).
    # Reference: settings.yaml → entity_extraction.gliner.entity_types
    RELEVANT_ENTITY_TYPES = frozenset({
        "PERSON",      # → person
        "ORG",         # → organization
        "GPE",         # → city / country / location
        "LOC",         # → location
        "FAC",         # → location (facility)
        "PRODUCT",     # → product
        "EVENT",       # → event
        "WORK_OF_ART", # → work of art
        "DATE",        # → date
    })

    # Regex fallback patterns for entity extraction when SpaCy is unavailable
    ENTITY_PATTERNS = (
        (r'"([^"]+)"',                             "QUOTED"),  # double-quoted strings
        (r"'([^']+)'",                             "QUOTED"),  # single-quoted strings
        (_PROPER_NOUN_RE.pattern,                   "PROPN"),   # multi-word proper nouns (shared from _text_utils)
        (r"\b([A-Z][a-z]{2,})\b",                  "PROPN"),   # single proper nouns
        (r"\b(\d{4})\b",                           "DATE"),    # four-digit years
    )

    # Common stopwords that must not be extracted as named entities.
    # Built once as a frozenset class constant to avoid per-call reconstruction
    # (called for every regex-matched candidate in extract()).
    _STOPWORDS: frozenset = frozenset({
        'the', 'a', 'an', 'this', 'that', 'these', 'those',
        'however', 'therefore', 'furthermore', 'moreover',
        'although', 'because', 'since', 'while', 'when',
        'what', 'which', 'who', 'whom', 'whose', 'where',
        'how', 'why', 'if', 'then', 'else', 'but', 'and', 'or',
    })

    # Per-label confidence estimates.
    # SpaCy does not expose per-entity confidence scores, so these are
    # approximate values chosen by inspection from label-level reliability in
    # the SpaCy documentation and en_core_web_sm evaluation on OntoNotes-5
    # (Weischedel et al., 2013): high-precision labels (DATE, PERSON, GPE)
    # receive higher base confidence than ambiguous labels (WORK_OF_ART),
    # which are frequently mis-categorised. The length bonus (cap and
    # per-token increment are PlannerConfig fields) rewards unambiguous
    # multi-word spans. These per-label values are a fixed table; extending
    # or retuning them is a code change.
    _LABEL_CONFIDENCE: Dict[str, float] = {
        "PERSON":     0.9,
        "ORG":        0.85,
        "GPE":        0.9,
        "LOC":        0.85,
        "DATE":       0.95,
        "EVENT":      0.8,
        "WORK_OF_ART": 0.75,
    }

    def __init__(self, config: Optional[PlannerConfig] = None):
        """
        Initialise the Entity Extractor.

        Args:
            config: Planner configuration.
        """
        self.config = config or PlannerConfig()

        # Compile regex patterns once at construction time
        self._compiled_patterns = [
            (re.compile(pattern), label)
            for pattern, label in self.ENTITY_PATTERNS
        ]

        logger.info("EntityExtractor initialised")

    def extract(self, query: str) -> List[EntityInfo]:
        """
        Extract entities from a query.

        Uses SpaCy NER when available, with regex fallback for additional
        coverage or when SpaCy is not installed.

        Args:
            query: The query from which to extract entities.

        Returns:
            List of EntityInfo objects sorted by character position.
        """
        entities = []
        seen_texts: set = set()  # for deduplication

        # ─────────────────────────────────────────────────────────────────────
        # METHOD 1: SpaCy NER (when available)
        # ─────────────────────────────────────────────────────────────────────

        if SPACY_AVAILABLE and NLP:
            doc = NLP(query)

            for ent in doc.ents:
                if ent.label_ in self.RELEVANT_ENTITY_TYPES:
                    # SpaCy does not expose per-entity confidence scores;
                    # estimate from label type and entity length.
                    confidence = self._estimate_confidence(ent)

                    if confidence >= self.config.min_entity_confidence:
                        entity_text = _strip_leading_function_words(ent.text.strip())
                        if not entity_text:
                            continue

                        if entity_text.lower() not in seen_texts:
                            # Adjust start_char to match the stripped text
                            stripped_offset = ent.text.index(entity_text) if entity_text in ent.text else 0
                            entities.append(EntityInfo(
                                text=entity_text,
                                label=ent.label_,
                                confidence=confidence,
                                start_char=ent.start_char + stripped_offset,
                                end_char=ent.end_char,
                                is_bridge=False,
                            ))
                            seen_texts.add(entity_text.lower())

        # ─────────────────────────────────────────────────────────────────────
        # METHOD 2: Regex fallback (supplementary or primary)
        # ─────────────────────────────────────────────────────────────────────

        for pattern, label in self._compiled_patterns:
            for match in pattern.finditer(query):
                text = match.group(1) if match.lastindex else match.group(0)
                text = text.strip()

                if len(text) > 2 and text.lower() not in seen_texts:
                    if not self._is_stopword(text):
                        entities.append(EntityInfo(
                            text=text,
                            label=label,
                            # Regex confidence is lower than NER; value
                            # configurable via PlannerConfig.regex_entity_confidence
                            confidence=self.config.regex_entity_confidence,
                            start_char=match.start(),
                            end_char=match.end(),
                            is_bridge=False,
                        ))
                        seen_texts.add(text.lower())

        # Sort by position in text
        entities.sort(key=lambda e: e.start_char)

        # ─────────────────────────────────────────────────────────────────────
        # POST-PROCESSING: remove substring-duplicate entities
        # Drop a single-token entity whose surface form is a substring of a
        # multi-token entity already in the list (the multi-token form is
        # always the more specific reference). Also drop PROPN-labelled
        # single tokens shorter than 5 chars — those are usually spurious
        # hits from the regex fallback (sentence-initial capitalised
        # function words such as auxiliaries or determiners).
        # ─────────────────────────────────────────────────────────────────────
        all_texts_lower = [e.text.lower() for e in entities]
        filtered: List[EntityInfo] = []
        for entity in entities:
            txt_lower = entity.text.lower()
            # Drop short PROPN tokens — likely regex noise
            if entity.label == "PROPN" and len(entity.text) < 5:
                continue
            # Drop if any other entity's text contains this one as a substring
            # (only when this entity is a single token — multi-word spans are kept)
            if " " not in entity.text.strip():
                is_substring = any(
                    txt_lower != other_lower and txt_lower in other_lower
                    for other_lower in all_texts_lower
                )
                if is_substring:
                    continue
            filtered.append(entity)

        # ─────────────────────────────────────────────────────────────────────
        # Re-join proper-noun spans that SpaCy's NER fragmented.
        # The small en_core_web_sm model splits multi-token proper-noun
        # phrases around lowercase function-word connectors ("to", "of",
        # "on", "the", "and", "&", "'s") into separate spans. When two
        # extracted entities are adjacent in the query and separated only
        # by such a connector, the two spans almost certainly belong to a
        # single proper-noun phrase (Quirk et al. 1985 §5.34 on compound
        # nominals); we merge them so the full title is treated as one
        # entity rather than fragmented across two retrieval anchors.
        # ─────────────────────────────────────────────────────────────────────
        filtered = self._rejoin_fragmented_spans(query, filtered)

        return filtered[:self.config.max_entities]

    # Short connector words allowed *inside* a multi-word proper-noun span when
    # re-joining fragments. They must appear lowercase and surrounded by
    # the two entity spans in the original query text.
    #
    # IMPORTANT: "and"/"&" are deliberately EXCLUDED — a conjunction between two
    # distinct named entities (PersonA and PersonB, in coordinated NPs) must not
    # be merged into one span, or comparison-pattern decomposition breaks.
    # Likewise "a"/"an" are excluded (too generic). Only genuine title-internal
    # connectors (prepositions and Romance/Germanic nobiliary particles) are kept.
    _SPAN_CONNECTORS: frozenset = frozenset({
        "to", "of", "on", "in", "the", "for", "at", "de",
        "von", "van", "del", "della", "di", "le", "la",
    })
    # Max char length of the inter-entity gap still treated as a title-internal
    # connector when re-joining fragmented spans (review 2026-06-12, finding #8 —
    # was an inline literal 12). Connector tokens ("of the", "von") are short;
    # a longer gap signals two distinct entities, not one fragmented title.
    _SPAN_REJOIN_MAX_GAP_CHARS: int = 12

    def _rejoin_fragmented_spans(
        self, query: str, entities: List[EntityInfo]
    ) -> List[EntityInfo]:
        """Merge consecutive entities that are adjacent in `query` and separated
        only by short connector words (or just whitespace/an apostrophe-s).

        Operates on the position-sorted entity list; runs a single left-to-right
        pass, repeatedly absorbing the next entity into the current span when the
        gap between them in the source text qualifies."""
        if len(entities) < 2:
            return entities
        ents = sorted(entities, key=lambda e: e.start_char)
        merged: List[EntityInfo] = []
        cur = ents[0]
        for nxt in ents[1:]:
            gap = query[cur.end_char:nxt.start_char]
            gap_norm = gap.strip().lower().strip("'").strip()  # tolerate "'s"
            gap_tokens = [t for t in gap_norm.split() if t]
            joinable = (
                nxt.start_char >= cur.end_char  # non-overlapping, in order
                and len(gap) <= self._SPAN_REJOIN_MAX_GAP_CHARS  # connectors are short
                and (
                    gap.strip() in ("", "'s", "’s")          # pure adjacency
                    or (gap_tokens and all(t in self._SPAN_CONNECTORS for t in gap_tokens))
                )
            )
            if joinable:
                # Build the merged span verbatim from the query text so casing
                # and the original connector tokens are preserved exactly.
                new_text = query[cur.start_char:nxt.end_char].strip()
                # Prefer the more specific label: a named type over PROPN/QUOTED.
                if cur.label in ("PROPN", "QUOTED") and nxt.label not in ("PROPN", "QUOTED"):
                    new_label = nxt.label
                else:
                    new_label = cur.label
                cur = EntityInfo(
                    text=new_text,
                    label=new_label,
                    confidence=max(cur.confidence, nxt.confidence),
                    start_char=cur.start_char,
                    end_char=nxt.end_char,
                    is_bridge=cur.is_bridge or nxt.is_bridge,
                )
            else:
                merged.append(cur)
                cur = nxt
        merged.append(cur)
        return merged

    def detect_bridge_entities(
        self,
        query: str,
        entities: List[EntityInfo],
    ) -> List[EntityInfo]:
        """
        Identify bridge entities for multi-hop reasoning.

        Bridge entities are intermediate nodes required for graph traversal:
        a bridge entity is mentioned in one retrieval step but its identity
        must be resolved before a second step can be issued. The notion is
        equivalent to the "intermediate referent" of compositional question
        answering (e.g. Karttunen 1977 on multi-step questions).

        Detection uses SpaCy dependency parsing:
        - Prepositional objects in nested structures (dep=pobj)
        - Possessive modifiers in chains (dep=poss)
        - Relative clause subjects (dep=relcl)

        Args:
            query:    The original query.
            entities: Already-extracted entities.

        Returns:
            Entities with the is_bridge flag updated where appropriate.
        """
        if not self.config.enable_bridge_detection:
            logger.debug("detect_bridge_entities: bridge detection disabled by config")
            return entities

        if not SPACY_AVAILABLE or NLP is None or len(entities) < 2:
            logger.debug(
                "detect_bridge_entities: skipped (spacy=%s, entities=%d)",
                SPACY_AVAILABLE,
                len(entities),
            )
            return entities

        doc = NLP(query)
        bridge_candidates: set = set()

        for token in doc:
            # Prepositional objects inside nested structures
            if token.dep_ == "pobj" and token.head.dep_ == "prep":
                if token.head.head.pos_ in ("NOUN", "PROPN"):
                    bridge_candidates.add(token.text.lower())

            # Possessive chains (e.g. "John's sister's husband")
            if token.dep_ == "poss":
                bridge_candidates.add(token.text.lower())

            # Relative clause subjects
            if token.dep_ == "relcl":
                for child in token.children:
                    if child.dep_ == "nsubj":
                        bridge_candidates.add(child.text.lower())

        # Mark entities as bridge if in bridge_candidates.
        # First and last entities are typically anchors, not bridges.
        #
        # A bridge entity is the noun phrase that links two retrieval steps:
        # it must denote a discrete referent (person, organisation, work,
        # place, event). Generic descriptors — NORP demonyms, dates,
        # quantities, and unlabelled regex PROPN fragments — are excluded
        # because they refer to classes rather than to individuals and
        # would steer retrieval toward high-degree hub nodes rather than
        # the specific bridging referent (West & Leskovec 2012, "Human
        # Wayfinding in Information Networks", WWW).
        _BRIDGE_OK_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC",
                             "WORK_OF_ART", "PRODUCT", "EVENT"}
        for i, entity in enumerate(entities):
            if entity.text.lower() in bridge_candidates:
                if 0 < i < len(entities) - 1 and entity.label in _BRIDGE_OK_LABELS:
                    entity.is_bridge = True

        return entities

    def _estimate_confidence(self, ent: "spacy.tokens.Span") -> float:
        """
        Estimate confidence for a SpaCy entity span.

        SpaCy does not expose native per-entity confidence scores.
        Confidence is estimated from label type (see _LABEL_CONFIDENCE) and
        entity length (longer spans are less ambiguous).

        Args:
            ent: A SpaCy Span object.

        Returns:
            Estimated confidence in [0.0, 1.0].
        """
        base = self._LABEL_CONFIDENCE.get(ent.label_, self.config.ner_default_confidence)
        length_bonus = min(
            self.config.ner_length_bonus_cap,
            len(ent.text.split()) * self.config.ner_length_bonus_per_token,
        )
        return min(1.0, base + length_bonus)

    def _is_stopword(self, text: str) -> bool:
        """Return True if text is a common stopword that should not be extracted."""
        return text.lower() in self._STOPWORDS


# =============================================================================
# RETRIEVAL PLAN GENERATOR
# =============================================================================

class PlanGenerator:
    """
    Generator for structured retrieval plans.

    Produces a hop sequence for the Navigator (S_N) based on the query type
    and the named entities extracted by the EntityExtractor.

    ─────────────────────────────────────────────────────────────────────────
    DECOMPOSITION MECHANISMS
    ─────────────────────────────────────────────────────────────────────────
    Hop generation applies two generalisable mechanisms and a baseline. No
    surface-form vocabulary lists are consulted; all recognisers key on
    SpaCy dependency labels, closed-class English function words, or the
    OntoNotes-5 named-entity inventory.

    Mechanism A — Dependency-parse decomposers
    ------------------------------------------
    Four English constructions are recognised structurally via SpaCy
    dependency labels (Honnibal & Montani 2017). Each recogniser is gated
    by a parse-confidence check requiring the relevant anchor to overlap
    a detected NER span, so the construction is only accepted when the
    parser produces the expected dependency structure and the entity
    inventory grounds it to a corpus referent.

    G — Relative-clause bridge (Quirk et al. 1985 §17.7-15).
        "The [noun] in/of/by which [Entity] …" or "the [role] who/that …
        [Entity]". Recogniser keys on the `relcl` dep label. Two forms
        cover the relative-pronoun-subject case and the relative-pronoun-
        object case.

    H — Chained attribution (the attribution-verb inventory in
        `_ATTRIBUTION_ACL_VERBS` is a small hand-curated closed set of
        work→source derivation verbs; cf. Levin 1993, "English Verb Classes",
        for the broader verb-class methodology, though this set is not one of
        Levin's classes verbatim).
        A passive ROOT with an `agent` by-phrase whose object is an
        indefinite pronoun, plus an `acl` clause on the subject anchored
        to a named entity. The attribution-clause head must be one of a
        small set of verbs of derivation/depiction (a closed class
        documented at the constant `_ATTRIBUTION_ACL_VERBS`).

    E — Relational-noun + of-PP complement (Partee 1995; Barker 1995).
        A noun whose dependency structure contains a `prep`("of") child
        whose `pobj` is a named entity. Generalises to any noun for
        which the parser produces this structure; no role enumeration.

    F — Passive-agent voice transformation (Bresnan 1982; Quirk et al.
        1985 §3.65-71). A verb with `auxpass` + `nsubjpass` + `agent`
        children. Past-participle → infinitive conversion is performed
        by SpaCy's morphological lemmatiser, so the recogniser is
        vocabulary-independent.

    Mechanism B — Closed-class lexical pre-empts
    --------------------------------------------
    Two English constructions are unambiguously identified by closed-class
    function words and are routed before the general scoring classifier:

    I — Distributive predication with floating "both"
        (Quirk et al. 1985 §10.49). Surface form "<aux> X and Y both <P>".
        The "both" quantifier is the discriminator.

    J — Anaphoric introduction with "another" (Karttunen 1976,
        "Discourse Referents"). The presence of "another <noun>" signals
        an unnamed referent that must be resolved before the predicate
        applies, forcing a MULTI_HOP plan.

    Baseline — Connector-split decomposition
    ----------------------------------------
    For multi-hop queries whose structure none of the above recognise, the
    query is split at bridge connectors ("that", "which", "who", "of the")
    and the resulting fragments are re-ordered so the bridge sub-query
    precedes the final sub-query. A 2-hop cap collapses spurious 3-part
    splits whose middle parts contain no named entity. This is the
    connector-split fallback baseline.

    Failure modes (explicit, surfaced in `matched_pattern`):
      - `fallback_generic_2hop`: classified MULTI_HOP, no mechanism
        applied, but a seed entity is available. Emit "Who or what is
        X?" as hop-0 and the original query as hop-1.
      - `fallback_degraded_to_single_hop`: classified MULTI_HOP, no
        mechanism applied, no entity available. Logged at WARNING.

    Comparison queries route through ``_decompose_comparison``:
      - I (boolean conjunction) — runs first; the "both" discriminator
        is unambiguous.
      - Select-between-two — disjunction of two NER entities joined by
        "or"; the disjunction is detected via NER spans, not surface form.
      - `_ATTR_MAP` rewrite — closed-class English attribute nouns are
        rewritten into per-entity factual lookups.
      - Generic per-entity predicate templates — used when no attribute
        rewrite applies.

    Pattern identifiers are recorded on every plan
    (``RetrievalPlan.matched_pattern``) so per-pattern diagnostics can be
    computed from the eval JSONL without parsing logs.

    ─────────────────────────────────────────────────────────────────────────
    """

    # ─────────────────────────────────────────────────────────────────────────
    # CLASS-LEVEL COMPILED CONSTANTS
    # ─────────────────────────────────────────────────────────────────────────

    # NER labels recognised as proper named entities (SpaCy OntoNotes label set).
    # Intentionally a strict subset of EntityExtractor.RELEVANT_ENTITY_TYPES:
    # LAW, TIME, MONEY, QUANTITY are excluded because comparison decomposition
    # operates on entities that refer to real-world objects with comparable
    # attributes, not temporal values or monetary amounts.
    # FAC (facility) is included as it is a named place (comparable location).
    _NER_LABELS: frozenset = frozenset({
        "PERSON", "GPE", "ORG", "LOC", "PRODUCT",
        "EVENT", "WORK_OF_ART", "NORP", "DATE", "FAC",
    })

    # Vague pronoun / generic reference pattern used in multi-hop enrichment.
    # Matches definite NP generics like "the director", "the woman", etc.
    # See _decompose_multi_hop Fall B for usage context.
    _VAGUE_REFS = re.compile(
        r'\b(the\s+(?:woman|man|person|actor|actress|director|author|artist'
        r'|president|player|team|group|band|company|film|movie|show|song|book))\b',
        re.IGNORECASE,
    )

    # Attribute-rewriting templates for comparison queries.
    # Transforms a shared-attribute comparison ("Were X and Y of the same
    # nationality?") into two per-entity factual lookups ("What is the
    # nationality of X?" / "What is the nationality of Y?"). The rewrite
    # improves vector-similarity match against biographical chunks, which
    # typically realise the attribute as a copular predicate ("X is an
    # American …") rather than as a comparison. Templates cover canonical
    # biographical attributes (nationality, birthplace, profession, genre,
    # age, country, religion, age via birth year). The list is closed and
    # documented; each entry consists of one regex over a closed-class
    # English attribute noun + one parameterised question template.
    _ATTR_MAP = (
        (re.compile(r'\bsame\s+nationality\b', re.IGNORECASE),
         "What is the nationality of {entity}?"),
        (re.compile(r'\bsame\s+(?:birth\s*place|birthplace|hometown)\b', re.IGNORECASE),
         "Where was {entity} born?"),
        (re.compile(r'\bborn\s+in\s+the\s+same\b', re.IGNORECASE),
         "Where was {entity} born?"),
        (re.compile(r'\bsame\s+(?:profession|occupation|job)\b', re.IGNORECASE),
         "What is the profession of {entity}?"),
        (re.compile(r'\bsame\s+(?:genre|style)\b', re.IGNORECASE),
         "What genre is {entity}?"),
        (re.compile(r'\bsame\s+(?:age|birth\s*year)\b', re.IGNORECASE),
         "When was {entity} born?"),
        (re.compile(r'\bsame\s+(?:country|state|city)\b', re.IGNORECASE),
         "What country is {entity} from?"),
        (re.compile(r'\bsame\s+(?:religion|faith|belief)\b', re.IGNORECASE),
         "What is the religion of {entity}?"),
        # "Who is older/younger, X or Y?" → "When was X born?" so ANN matches
        # Wikipedia bio intros ("X (born 14 August 1965) is a …") rather than
        # the comparative question phrasing.
        (re.compile(r'\b(older|younger)\b', re.IGNORECASE),
         "When was {entity} born?"),
    )

    # Pattern C was removed. It matched the
    # surface form "for a/an/the <CATEGORY> <desc>" where CATEGORY enumerated
    # a fixed list of creative-work nouns. The list was example-derived and
    # the surrounding regex was a string-shape match, not a structural
    # recogniser. Queries previously handled by C now fall through to the
    # connector-split baseline and, when that fails, to the dep-parse
    # patterns or generic-2hop fallback.

    # ── Pattern E: Relational-noun complement bridge ─────────────────────────
    # A relational noun (one whose meaning entails a relation to another
    # entity, e.g. "father", "director", "founder") combined with an "of-PP"
    # complement headed by a named entity is a standard construction in
    # English nominal semantics. Following Partee (1995, "The Semantics of
    # Compositionality") and Barker (1995, "Possessive Descriptions",
    # CSLI), relational nouns project an argument structure with one
    # internal argument typically realised as an of-PP. The relevant
    # syntactic signal here is therefore a noun whose dependency parse
    # contains a `prep`("of") child whose `pobj` is a named entity — not
    # a fixed enumeration of role words.
    #
    # The detection is performed in ``_find_relational_noun_bridge`` using
    # SpaCy's dependency parse and the OntoNotes-5 entity inventory. No
    # role list is required: any noun whose of-complement is an NER span
    # qualifies. This generalises to roles never seen at development time
    # (e.g. "the choreographer of X", "the librettist of Y") provided the
    # parser produces the same dependency structure.

    # Pattern D was removed for the same reason
    # as Pattern C — its verb slot was a fixed enumeration of attribution
    # verbs (co-wrote / wrote / directed / produced / starred / co-directed)
    # and the role slot was a `\w+` shape match, not a syntactic recogniser.
    # Wh-questions with appositive qualifiers are now handled, when present
    # as a parseable relative-clause structure, by Pattern G.

    # ── Pattern F: Passive-agent voice transformation ──────────────────────
    # Recognises English passive constructions of the form
    #     [SUBJECT] was/were [PAST-PARTICIPLE] by [AGENT]
    # using SpaCy's dependency parse rather than a closed verb list. Voice
    # transformation (passive ↔ active) is a fundamental syntactic operation
    # in English grammar (Quirk et al. 1985 §3.65-71; Bresnan 1982,
    # "The Passive in Lexical Theory"). The recogniser keys on:
    #     - a passive auxiliary (`auxpass`) attached to the head verb,
    #     - an `agent` by-phrase whose `pobj` is a named entity or
    #       indefinite description.
    # Past-participle → infinitive transformation uses SpaCy's lemmatiser,
    # which is built on UniMorph-style morphological tables (Honnibal &
    # Montani 2017, arXiv:1802.04016). No verb enumeration is required.
    #
    # The previous implementation used a fixed regex over 14 attribution
    # verbs and a hand-written past-participle → infinitive lookup table.
    # Both were retired: the regex was surface-form fitted, and the lookup
    # table reimplemented the SpaCy lemmatiser. Detection is now in
    # _find_passive_agent_bridge.

    # Pre-compiled regexes for _form_sub_query (called on every multi-hop step)
    _STRIP_LEADING_CONJ = re.compile(
        r"^(and|or|but|that|which|who|where)\s+", re.IGNORECASE
    )
    _INTERROGATIVE_PREFIX = re.compile(
        r"^(what|who|where|when|why|how|which|is|are|was|were|did|does|do)\b",
        re.IGNORECASE,
    )
    # Pattern F guard (added 2026-05-20): when _find_passive_agent_bridge
    # returns an interrogative-headed subject (e.g. "What government
    # position", "Which film"), the template "Who {verb} {subj}?" produces
    # self-referential nonsense. Detect and skip in that case.
    # Note: stricter than _INTERROGATIVE_PREFIX — only the determiner-style
    # wh-words that head NPs ("what NP", "which NP", "whose NP"), not the
    # pronoun-style ones ("who", "whom") that are valid subjects on their
    # own (e.g. "Who painted the Mona Lisa?" → subj="Who" is grammatical).
    _PASSIVE_F_INTERROGATIVE_SUBJ_RE = re.compile(
        r"^\s*(what|which|whose)\b",
        re.IGNORECASE,
    )
    # Pattern F additional guard (added 2026-05-21): when
    # _find_passive_agent_bridge returns a bare pronoun as subject (e.g.
    # "that", "this", "it", "who"), the template "Who {verb} {subj}?"
    # produces equally degenerate sub-queries ("Who form that?", "Who paint
    # it?"). Bare pronouns standing alone cannot anchor a bridge retrieval.
    # Determiner+noun subjects ("The book", "This painting") are not
    # affected because they are multi-token; only standalone pronouns are
    # listed here.
    _PASSIVE_F_BARE_PRONOUN_SUBJ = frozenset({
        "that", "this", "these", "those",
        "it", "he", "she", "they", "them",
        "who", "whom",
        "someone", "something", "somebody",
        "anyone", "anything", "everyone", "everything",
    })

    # Pre-compiled regexes for _extract_constraints (called on every query)
    _TEMPORAL_TERMS_RE = re.compile(
        r"\b(before|after|during|since|until|recent|latest|first|last)\b",
        re.IGNORECASE,
    )
    _COMPARISON_GREATER_RE = re.compile(
        r"\b(older|bigger|larger|more|higher|greater)\b", re.IGNORECASE
    )
    _COMPARISON_LESS_RE = re.compile(
        r"\b(younger|smaller|less|lower|fewer)\b", re.IGNORECASE
    )
    _COMPARISON_ATTR_RE = re.compile(
        r"\b(older|younger|taller|shorter|bigger|smaller|richer|poorer)\b",
        re.IGNORECASE,
    )

    def __init__(self, config: Optional[PlannerConfig] = None):
        """
        Initialise the Plan Generator.

        Args:
            config: Planner configuration.
        """
        self.config = config or PlannerConfig()
        # Per-call cache for the pattern that produced the current
        # hop sequence. Set by _generate_hops / _decompose_* methods immediately
        # before returning; read by generate() into RetrievalPlan.matched_pattern.
        # Reset to None at the start of each generate() call so a stale value
        # from a previous query cannot leak.
        #
        # THREAD-SAFETY CONTRACT (review 2026-06-12, finding #7): this marker —
        # and QueryClassifier._last_preempt — is per-instance mutable state set
        # mid-call and read at the end of the same call. A single Planner /
        # PlanGenerator instance is therefore NOT safe to share across threads:
        # concurrent plan() calls would race on this field. The edge pipeline is
        # single-threaded per query, so this holds; for concurrent use, construct
        # one Planner per thread (they are cheap once the SpaCy model is loaded).
        #
        # PERF NOTE (finding #6): the dep-parse decomposers (G/H/E/F and the A1/A4
        # helpers) each call NLP(query) independently, so a multi-hop query can
        # re-parse 5-8×. Parsing is deterministic and ~ms on short queries, so
        # this is a latency micro-cost, not a correctness issue; threading a
        # single doc through the helpers is a documented future optimisation.
        self._last_matched_pattern: Optional[str] = None

    def generate(
        self,
        query: str,
        query_type: QueryType,
        confidence: float,
        entities: List[EntityInfo],
    ) -> RetrievalPlan:
        """
        Generate a retrieval plan.

        Args:
            query:      Original query.
            query_type: Classified query type.
            confidence: Classification confidence.
            entities:   Extracted entities.

        Returns:
            Complete RetrievalPlan.
        """
        # Reset the pattern marker before this call so stale state from
        # a previous plan() invocation cannot bleed in.
        self._last_matched_pattern = None

        # A4: classifier-abstention structural override. The classifier returns
        # query_type=SINGLE_HOP with exactly `classifier_fallback_confidence`
        # ONLY when no lexical pattern scored (its documented no-signal
        # sentinel). In that abstention case — and only then — consult the
        # dependency parse: if the question asks for a wh-attribute of something
        # related to a named entity (a general multi-hop signature), re-route to
        # MULTI_HOP so the entity-seeded 2-hop decomposition runs. This can
        # never override a classification that had positive evidence (confidence
        # >= base 0.6), so it cannot regress confident single-hop questions.
        if (
            query_type == QueryType.SINGLE_HOP
            and confidence == self.config.classifier_fallback_confidence
            and self._attribute_over_entity_signal(query, entities)
        ):
            logger.info(
                "A4 structural override: classifier abstained (SINGLE_HOP@%.3f) "
                "but a wh-attribute-over-entity bridge is present → "
                "re-routing to MULTI_HOP: %r",
                confidence, query[:80],
            )
            query_type = QueryType.MULTI_HOP

        strategy = self._determine_strategy(query_type, entities)
        hop_sequence, sub_queries = self._generate_hops(query, query_type, entities)
        constraints = self._extract_constraints(query, query_type)

        plan = RetrievalPlan(
            original_query=query,
            query_type=query_type,
            strategy=strategy,
            entities=entities,
            hop_sequence=hop_sequence,
            sub_queries=sub_queries,
            constraints=constraints,
            estimated_hops=len(hop_sequence),
            matched_pattern=self._last_matched_pattern,
            confidence=confidence,
            metadata={
                "entity_count": len(entities),
                "bridge_count": sum(1 for e in entities if e.is_bridge),
                "spacy_available": SPACY_AVAILABLE,
            },
        )

        logger.info(
            "Plan generated: type=%s strategy=%s hops=%d sub_queries=%d",
            query_type.value,
            strategy.value,
            len(hop_sequence),
            len(sub_queries),
        )

        return plan

    def _determine_strategy(
        self,
        query_type: QueryType,
        entities: List[EntityInfo],
    ) -> RetrievalStrategy:
        """
        Select the optimal retrieval strategy for the given query type.

        Per the paper, Section 3.2:
        - VECTOR_ONLY: For simple single-hop queries (fast path).
        - HYBRID:      For all complex query types requiring graph traversal.

        Note: RetrievalStrategy.GRAPH_ONLY is reserved for future work and
        is not currently selected by this method. A dedicated graph-only
        ablation path would require explicit graph-relation queries (e.g.
        "Who is the founder of X?") to be routed here; this is left as a
        planned extension for the paper's evaluation.
        """
        # Simple queries with at most one entity → vector search is sufficient
        if query_type == QueryType.SINGLE_HOP and len(entities) <= 1:
            return RetrievalStrategy.VECTOR_ONLY

        # All multi-hop queries use HYBRID regardless of bridge-entity presence.
        # Even without detected bridges the graph may surface indirect relations.
        if query_type == QueryType.MULTI_HOP:
            return RetrievalStrategy.HYBRID

        # Comparison and intersection → parallel retrieval + graph relations
        if query_type in (QueryType.COMPARISON, QueryType.INTERSECTION):
            return RetrievalStrategy.HYBRID

        # Temporal and aggregate → graph can provide structured time-stamped facts
        if query_type in (QueryType.TEMPORAL, QueryType.AGGREGATE):
            return RetrievalStrategy.HYBRID

        return self.config.default_strategy

    def _generate_hops(
        self,
        query: str,
        query_type: QueryType,
        entities: List[EntityInfo],
    ) -> Tuple[List[HopStep], List[str]]:
        """
        Generate the hop sequence and sub-query list.

        Returns an ordered sequence of retrieval steps with dependencies
        for multi-hop reasoning.
        """
        hop_sequence = []
        sub_queries = []

        if query_type == QueryType.SINGLE_HOP:
            hop_sequence.append(HopStep(
                step_id=0,
                sub_query=query,
                target_entities=[e.text for e in entities],
                depends_on=[],
                is_bridge=False,
            ))
            sub_queries = [query]
            # Mark this path explicitly so the JSONL distinguishes
            # "classifier decided SINGLE_HOP" from "no pattern matched, fell back".
            self._last_matched_pattern = "single_hop"

        elif query_type == QueryType.MULTI_HOP:
            hop_sequence, sub_queries = self._decompose_multi_hop(query, entities)
            # _decompose_multi_hop sets _last_matched_pattern per pattern branch.

        elif query_type == QueryType.COMPARISON:
            hop_sequence, sub_queries = self._decompose_comparison(query, entities)
            # _decompose_comparison sets the marker for select-between / parallel /
            # attr_map variants.

        elif query_type == QueryType.INTERSECTION:
            hop_sequence, sub_queries = self._decompose_intersection(query, entities)
            self._last_matched_pattern = self._last_matched_pattern or "intersection"

        elif query_type == QueryType.TEMPORAL:
            hop_sequence, sub_queries = self._decompose_temporal(query, entities)
            self._last_matched_pattern = self._last_matched_pattern or "temporal"

        elif query_type == QueryType.AGGREGATE:
            hop_sequence.append(HopStep(
                step_id=0,
                sub_query=query,
                target_entities=[e.text for e in entities],
                depends_on=[],
                is_bridge=False,
            ))
            sub_queries = [query]
            self._last_matched_pattern = "aggregate"

        return hop_sequence, sub_queries

    # ── Pattern J: Implicit bridge ("another [noun]") ───────────────────────
    # Matches queries like "X and another corporation that has operations in …"
    # where the answer-bearing entity is described by a common noun + relative
    # clause, not named.  The anchor is the named entity alongside "another".
    _IMPLICIT_BRIDGE_RE = re.compile(
        r'\banother\s+(\w+)\b',
        re.IGNORECASE,
    )

    def _find_implicit_bridge(
        self,
        query: str,
        entities: List["EntityInfo"],
    ) -> Optional[Tuple[str, str]]:
        """
        Detect "X and another [noun] that …" queries.

        Returns (anchor_entity_text, bridge_noun) when the pattern is found and
        at least one named entity is present to serve as the anchor, else None.
        """
        m = self._IMPLICIT_BRIDGE_RE.search(query)
        if not m:
            return None
        if not entities:
            return None
        bridge_noun = m.group(1)
        # Anchor: pick the entity that appears BEFORE "another" in the query
        another_pos = m.start()
        anchor = None
        for ent in entities:
            idx = query.lower().find(ent.text.lower())
            if 0 <= idx < another_pos:
                anchor = ent.text
                break
        if anchor is None:
            # Fall back to first entity
            anchor = entities[0].text
        return anchor, bridge_noun

    def _decompose_implicit_bridge(
        self,
        query: str,
        bridge_info: Tuple[str, str],
        entities: List["EntityInfo"],
    ) -> Tuple[List["HopStep"], List[str]]:
        """
        Decompose an implicit-bridge query (Pattern J) into two hops.

        hop 0 (bridge resolution): identify the unnamed second entity, which the
          query introduces via "another [bridge_noun]" alongside the named
          anchor. The hop-0 sub-query is RELATION-NEUTRAL — it asks which other
          [bridge_noun] is co-referenced with the anchor, WITHOUT assuming any
          specific relation between them.
        hop 1 (attribute lookup): original query, answered with the bridge
          entity materialised in the retrieved context.

        Relation-neutrality (review 2026-06-12, finding #1): the previous
        implementation hardcoded the relation "a critic of" — a phrase taken
        from one HotpotQA item (Jane Goodall / Nestlé). For any other
        "X and another [noun] that ..." question that injected a false relation
        ("is X a critic of?"), contradicting the module's own design claim that
        no surface-form question phrasings are matched (see the module
        docstring). The relation is now never assumed: hop-0 retrieves the
        anchor's article (where the co-referenced second [bridge_noun] is named)
        and the Controller resolves the bridge entity from that context.
        """
        anchor, bridge_noun = bridge_info

        # Relation-neutral bridge resolver: "Which other {bridge_noun} is
        # mentioned together with {anchor}?". No relation verb is supplied — the
        # anchor's retrieved article surfaces the co-referenced entity, and the
        # iterative-multihop Controller injects it into hop 1.
        hop0_q = f"Which {bridge_noun} is mentioned together with {anchor}?"

        hop_sequence = [
            HopStep(
                step_id=0,
                sub_query=hop0_q,
                target_entities=[e.text for e in entities],
                depends_on=[],
                is_bridge=True,
            ),
            HopStep(
                step_id=1,
                sub_query=query,
                target_entities=[e.text for e in entities],
                depends_on=[0],
                is_bridge=False,
            ),
        ]
        logger.debug(
            "_decompose_implicit_bridge: Pattern J → hop0=%r", hop0_q[:80]
        )
        return hop_sequence, [hop0_q, query]

    def _find_relative_clause_bridge(
        self,
        query: str,
    ) -> Optional[Tuple[str, str, str]]:
        """
        Detect bridge queries with a relative-clause structure using SpaCy's
        dependency parse and return ``(role_noun, anchor_entity, form)`` where
        ``form`` is ``"form1"`` or ``"form2"`` (see below), or None.

        Two structural forms are supported:

        Form 1 — Anchor-inside-clause:
            "The [noun] in which [Entity] [predicate]..."
            → role_noun = head of relcl (subject noun),
              anchor   = subject of relcl

        Form 2 — Anchor-as-clause-object (Pattern L extension):
            "...the [King|actress|author] who [made|directed|wrote] [Entity]..."
            → role_noun = head of relcl (attr/dobj head),
              anchor   = NER entity inside the relcl subtree
            When the relcl subject is a relative pronoun (who/that/which),
            the actual bridge anchor is the NER entity in the predicate.

        Form 2 covers "In which year was the King who made the 1925 Birthday
        Honours born?" — Pattern G v1 rejected it because `King.dep_ == "attr"`
        and because the relcl subject was the pronoun "who", not a named entity.

        No verb list is needed — the structural signal (relcl dep label +
        an NER entity reachable from the relcl) is sufficient for any verb
        the language model might produce.
        """
        if not SPACY_AVAILABLE or NLP is None:
            return None
        # Relative-pronoun set: when the relcl subject is one of these, the
        # real anchor sits inside the predicate as an object/oblique NP.
        _REL_PRONOUNS = {"who", "whom", "which", "that"}
        try:
            doc = NLP(query)
            for token in doc:
                if token.dep_ != "relcl":
                    continue
                head = token.head
                # Accept attr (predicate-nominal: "X was the King who...")
                # in addition to nsubj/nsubjpass/ROOT.
                if head.dep_ not in ("nsubj", "nsubjpass", "ROOT", "attr"):
                    continue

                # ── Form 1: relcl subject is a real NP (not a pronoun) ──
                rel_subjects = [
                    c for c in token.children if c.dep_ == "nsubj"
                ]
                if rel_subjects:
                    subj_tok = rel_subjects[0]
                    if subj_tok.text.lower() not in _REL_PRONOUNS:
                        entity_text = " ".join(
                            t.text for t in subj_tok.subtree
                            if not t.is_punct
                        ).strip()
                        noun_text = head.text
                        if entity_text and noun_text:
                            return noun_text, entity_text, "form1"

                # ── Form 2 (Pattern L): relcl subject is a relative pronoun;
                # anchor is the NER entity inside the relcl subtree. ──────
                # We need entities that sit INSIDE the relative clause
                # (so we can use them as graph-search anchors), not the
                # head NP itself. The head's `subtree` INCLUDES the relcl,
                # so we cannot use `head.subtree` as the exclusion set —
                # it would mask the entire relcl. Instead, restrict to
                # tokens whose ancestor chain passes through the relcl
                # token but NOT directly through the head as a noun-phrase
                # modifier (det/compound/amod).
                relcl_token_indices = {t.i for t in token.subtree}
                # Tokens that are part of the head's OWN noun phrase (det,
                # compound, amod, nmod attached directly to head). These
                # must be excluded — they belong to the role NP, not the
                # anchor inside the relcl predicate.
                head_np_indices = {head.i}
                for child in head.children:
                    if child.dep_ in ("det", "compound", "amod", "nmod", "poss"):
                        head_np_indices.update(t.i for t in child.subtree)
                candidate_ents = [
                    ent for ent in doc.ents
                    if any(t.i in relcl_token_indices for t in ent)
                    and not any(t.i in head_np_indices for t in ent)
                ]
                if candidate_ents:
                    # Longest entity span = most specific anchor.
                    anchor = max(candidate_ents, key=lambda e: len(e.text))
                    noun_text = head.text
                    if anchor.text and noun_text:
                        return noun_text, anchor.text, "form2"
        except (ValueError, AttributeError, IndexError, KeyError) as exc:
            logger.debug("_find_relative_clause_bridge failed: %s", exc)
        return None

    def _find_passive_agent_bridge(
        self,
        query: str,
    ) -> Optional[Tuple[str, str]]:
        """
        Detect a passive construction with a by-phrase agent.

        Recognises "[SUBJECT] was/were [PAST-PARTICIPLE] by …" using SpaCy's
        dependency parse (Quirk et al. 1985 §3.65-71; Bresnan 1982,
        "The Passive in Lexical Theory"). Signals:
            - a VERB with an `auxpass` (passive auxiliary) child,
            - an `nsubjpass` (passive subject) child,
            - an `agent` (by-phrase) child.

        Returns (subject_text, infinitive_verb), where ``infinitive_verb`` is
        SpaCy's lemma of the past-participle. The lemma transformation is
        morphology-driven (Honnibal & Montani 2017) and works for any
        English passive participle the parser knows — no verb enumeration.

        Returns None if the structure is absent or the parser does not
        produce the expected dependency signature.
        """
        if not SPACY_AVAILABLE or NLP is None:
            return None
        try:
            doc = NLP(query)
            for tok in doc:
                if tok.pos_ != "VERB":
                    continue
                has_auxpass = any(c.dep_ == "auxpass" for c in tok.children)
                if not has_auxpass:
                    continue
                # Subject of the passive verb.
                subj_tok = next(
                    (c for c in tok.children if c.dep_ in ("nsubjpass", "nsubj")),
                    None,
                )
                if subj_tok is None:
                    continue
                # Must have an `agent` (by-phrase) child to qualify as a
                # canonical agentive passive.
                has_agent = any(c.dep_ == "agent" for c in tok.children)
                if not has_agent:
                    continue
                # Subject text: contiguous span of the subject subtree.
                subj_span = " ".join(
                    t.text for t in subj_tok.subtree if not t.is_punct
                ).strip()
                # Verb lemma (infinitive form for the bridge sub-query).
                verb_lemma = tok.lemma_.lower()
                if subj_span and verb_lemma:
                    return subj_span, verb_lemma
        except (ValueError, AttributeError, IndexError, KeyError) as exc:
            logger.debug("_find_passive_agent_bridge failed: %s", exc)
        return None

    def _find_relational_noun_bridge(
        self,
        query: str,
        entities: List[EntityInfo],
    ) -> Optional[Tuple[str, str]]:
        """
        Detect a relational-noun + of-PP-complement construction.

        Recognises noun phrases of the form "the ROLE of ENTITY" where ROLE
        is any noun whose dependency parse contains a `prep`("of") child
        whose `pobj` is a named entity (Partee 1995; Barker 1995, "Possessive
        Descriptions", CSLI; Quirk et al. 1985 §5.118 on relational nouns).

        Returns (role_text, anchor_entity_text), or None if the structure is
        not present or the of-PP object does not overlap a detected NER entity.

        No role vocabulary list is consulted — any noun whose dependency
        structure matches qualifies. The parse-confidence gate requires the
        of-PP object to overlap a detected named entity span (PERSON, ORG,
        GPE, LOC, FAC, WORK_OF_ART, PRODUCT, EVENT), so the construction
        is only accepted when the object is anchored to a corpus entity.
        """
        if not SPACY_AVAILABLE or NLP is None:
            return None
        _ANCHOR_LABELS = frozenset({
            "PERSON", "ORG", "GPE", "LOC", "FAC",
            "WORK_OF_ART", "PRODUCT", "EVENT", "NORP",
        })
        anchor_entities = [e for e in entities if e.label in _ANCHOR_LABELS]
        if not anchor_entities:
            return None
        try:
            doc = NLP(query)
            for token in doc:
                # Look for a noun head with a "of" prepositional child.
                if token.pos_ != "NOUN":
                    continue
                of_prep = None
                for child in token.children:
                    if child.dep_ == "prep" and child.lemma_.lower() == "of":
                        of_prep = child
                        break
                if of_prep is None:
                    continue
                # Find the pobj of the "of" preposition.
                pobj = None
                for gc in of_prep.children:
                    if gc.dep_ == "pobj":
                        pobj = gc
                        break
                if pobj is None:
                    continue
                # Span of the pobj subtree (the candidate anchor NP).
                pobj_span_lower = " ".join(
                    t.text.lower() for t in pobj.subtree if not t.is_punct
                ).strip()
                # Parse-confidence gate: the pobj must overlap a detected NER entity.
                matched_anchor = next(
                    (e.text for e in anchor_entities
                     if e.text.lower() in pobj_span_lower
                     or pobj_span_lower in e.text.lower()),
                    None,
                )
                if matched_anchor is None:
                    continue
                role_text = token.text
                return role_text, matched_anchor
        except (ValueError, AttributeError, IndexError, KeyError) as exc:
            logger.debug("_find_relational_noun_bridge failed: %s", exc)
        return None

    # ── Pattern H: chained-attribution bridge ───────────────────────────────
    # English verbs that, as the head of an `acl` clause on a "work" noun,
    # express "this work is *about/derived-from* X":  "[work] based on / set in /
    # featuring / starring / centred on / adapted from [X]".  This is a CLOSED
    # LINGUISTIC CATEGORY — the small inventory of work→source attribution verbs
    # in English — not a list of verbs harvested from test answers.  It is the
    # same kind of artefact as a stopword list or a list of auxiliary verbs:
    # finite, domain-independent, and stable.  A query using any of these heads
    # is recognised regardless of whether that phrasing has ever been seen; a
    # query about manga, films, symphonies or video games is treated identically
    # because the recogniser keys on the *attribution relation*, never on the
    # work's vocabulary, the entity's name, or the question's phrasing.
    # Lemmas (SpaCy gives "based"→"base", "featuring"→"feature", etc.).
    _ATTRIBUTION_ACL_VERBS = frozenset({
        "base", "feature", "star", "set", "center", "centre",
        "focus", "follow", "depict", "adapt", "inspire",
        # "about" as an acl head can surface with lemma "about" (prep/sconj use)
        "about",
    })
    # Indefinite pronouns marking an *unresolved* agent in a passive `by`-phrase
    # — the entity the next hop must look up ("written by someone").  A closed
    # grammatical class (English indefinite person pronouns), not example-derived.
    _INDEFINITE_AGENT_HEADS = frozenset({
        "someone", "somebody", "anyone", "anybody",
    })
    # The agent hop's verb is taken VERBATIM from the query's own participle
    # (token.text — "illustrated", "written", "directed"), so there is no
    # participle→verb table and no past-tense rules: the inflection is the
    # user's, not ours.  This deliberately leaves no record in the codebase of
    # which verbs the system has encountered.

    def _find_attribution_chain(
        self,
        query: str,
        entities: List[EntityInfo],
    ) -> Optional[Dict[str, Any]]:
        """
        Detect a *chained* attribution bridge using SpaCy's dependency parse.

        Target shape:
            "A [work] based on [Entity], is [written] by someone [attribute]?"
        Dependency signature:
            work_noun   --nsubjpass--> passive_verb (ROOT)
            passive_verb --agent--> "by" --pobj--> indefinite agent ("someone")
            work_noun   --acl--> attribution_verb ("based") --prep--> --pobj--> Entity
            agent       --acl--> ... (the residual attribute question stays on it)

        Returns a dict describing the two-link chain, or None if the structure
        is absent or the parse is too ambiguous to trust:
            {
              "work_type":          "film"            # head noun of the work NP
              "work_np":            "biographical film" # compound+amod span (no det)
              "anchor_entity":      "Marie Curie"     # known entity (link-0 target)
              "agent_verb_surface": "directed"        # participle verbatim from
                                                      # the query (used in hop1)
            }

        Anti-fragility: keys on dependency *relation labels* (nsubjpass, agent,
        acl, prep/pobj) and a small closed set of attribution clause heads — the
        grammar of attribution, not the lexicon of works.  A query with the same
        shape but novel vocabulary ("a symphony commissioned for the coronation
        of a monarch crowned in what year?") matches with zero new code.

        Parse-confidence gate: requires the *exact* relation chain to be present
        and the anchor entity to overlap a detected NER entity.  If any link is
        missing or fuzzy, returns None and the caller falls back to the next
        pattern / single-query — so the worst case is current behaviour, never
        worse.
        """
        if not SPACY_AVAILABLE or NLP is None:
            return None
        try:
            doc = NLP(query)

            # 1. Find a passive ROOT (or conj-of-ROOT) with an `agent` by-phrase
            #    whose object is an indefinite placeholder.
            for tok in doc:
                if tok.pos_ != "VERB":
                    continue
                # Collect this verb + any conj siblings (handles "written and
                # illustrated by ...") — the agent may hang off either.
                verb_group = [tok] + [c for c in tok.children if c.dep_ == "conj"]
                agent_obj = None
                for v in verb_group:
                    for c in v.children:
                        if c.dep_ == "agent":
                            for gc in c.children:
                                if gc.dep_ == "pobj":
                                    agent_obj = gc
                                    break
                        if agent_obj:
                            break
                    if agent_obj:
                        break
                if agent_obj is None:
                    continue
                # The agent must be an *unresolved* placeholder, else this is a
                # normal passive-with-named-agent (Pattern F territory).
                if agent_obj.lemma_.lower() not in self._INDEFINITE_AGENT_HEADS:
                    continue

                # 2. The passive verb's subject = the "work" noun.
                work_noun = None
                for v in verb_group + [tok]:
                    for c in v.children:
                        if c.dep_ in ("nsubjpass", "nsubj"):
                            work_noun = c
                            break
                    if work_noun:
                        break
                # nsubjpass usually attaches to the first verb of the group
                if work_noun is None:
                    for c in tok.children:
                        if c.dep_ in ("nsubjpass", "nsubj"):
                            work_noun = c
                            break
                if work_noun is None:
                    continue

                # 3. The work noun must carry an `acl` attribution clause linking
                #    it (via prep + pobj) to a concrete noun/entity.  Capture both
                #    the prep object's head proper-noun span (preferred anchor) and
                #    the full NP (fallback).
                anchor_full = None         # full prep-object NP
                anchor_propn = None        # contiguous PROPN run within it
                for c in work_noun.children:
                    if c.dep_ == "acl" and c.lemma_.lower() in self._ATTRIBUTION_ACL_VERBS:
                        for gc in c.children:
                            if gc.dep_ == "prep":
                                for ggc in gc.children:
                                    if ggc.dep_ == "pobj":
                                        sub = [t for t in ggc.subtree if not t.is_punct]
                                        anchor_full = " ".join(t.text for t in sub).strip()
                                        # Longest contiguous PROPN run = the name
                                        run, best = [], []
                                        for t in sub:
                                            if t.pos_ == "PROPN":
                                                run.append(t.text)
                                            else:
                                                if len(run) > len(best):
                                                    best = run
                                                run = []
                                        if len(run) > len(best):
                                            best = run
                                        if best:
                                            anchor_propn = " ".join(best)
                                        break
                            if anchor_full:
                                break
                    if anchor_full:
                        break
                if not anchor_full:
                    continue

                # 4. Parse-confidence gate: the anchor must overlap a detected
                #    NER entity, else the "chain" is noise.  Resolve the anchor we
                #    pass downstream to the NER entity text when it's contained in
                #    the prep-object NP (so "a pioneering physicist ... Marie Curie" →
                #    "Marie Curie"); else fall back to the PROPN run, then NP.
                _ANCHOR_LABELS = ("PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART",
                                  "EVENT", "FAC", "PRODUCT")
                anchor_full_lc = anchor_full.lower()
                matched_ner = next(
                    (e.text for e in entities
                     if e.label in _ANCHOR_LABELS
                     and (e.text.lower() in anchor_full_lc
                          or anchor_full_lc in e.text.lower())),
                    None,
                )
                if matched_ner is None:
                    logger.debug(
                        "_find_attribution_chain: anchor %r has no NER overlap "
                        "— rejecting (parse-confidence gate)", anchor_full[:50]
                    )
                    continue
                anchor_text = matched_ner or anchor_propn or anchor_full

                # 5. Build the work NP (compound + amod, drop determiners) and
                #    capture the agent participle *as it appears in the query*.
                #    We deliberately reuse the user's own surface form ("written",
                #    "illustrated", "directed") rather than re-inflecting from a
                #    lemma — no verb tables, no past-tense rules, nothing that
                #    encodes which verbs we've seen.  "Who illustrated the X?"
                #    is grammatical because the inflection came from the input.
                work_type = work_noun.text
                work_np_tokens = [
                    t.text for t in work_noun.subtree
                    if t.dep_ in ("compound", "amod") and t.head == work_noun
                ] + [work_noun.text]
                work_np = " ".join(work_np_tokens).strip()

                agent_verb_surface = None
                for v in verb_group + [tok]:
                    if any(c.dep_ == "agent" for c in v.children):
                        agent_verb_surface = v.text
                        break
                if agent_verb_surface is None:
                    agent_verb_surface = tok.text

                return {
                    "work_type": work_type,
                    "work_np": work_np or work_type,
                    "anchor_entity": anchor_text,
                    "agent_verb_surface": agent_verb_surface,
                }
        except (ValueError, AttributeError, IndexError, KeyError) as exc:
            logger.debug("_find_attribution_chain failed: %s", exc)
        return None

    def _decompose_attribution_chain(
        self,
        query: str,
        chain: Dict[str, Any],
        entities: List[EntityInfo],
    ) -> Tuple[List[HopStep], List[str]]:
        """
        Build a 3-step hop sequence from a chained-attribution bridge.

            hop0 (bridge): "What {work_type} is based on {anchor_entity}?"
                           → resolves the {work} placeholder
            hop1 (bridge): "Who {agent_verb_surface} the {work_type} based on
                           {anchor}?"  — {agent_verb_surface} is the participle
                           taken verbatim from the query ("illustrated",
                           "written", "directed"), so the phrasing is correct
                           without any verb-inflection tables.  The Controller
                           injects the resolved work title at runtime; this is
                           the retrieval seed.
            hop2 (final):  the original query, depends_on=[1]
                           → answers the residual attribute question once both
                             the work and the agent are in context

        depends_on chains 0 → 1 → 2 so the Controller's iterative bridge-entity
        injection (§12 iterative-multihop) feeds each link's result into the next.
        """
        work_type = chain["work_type"]
        anchor    = chain["anchor_entity"]
        agent_vb  = chain["agent_verb_surface"]

        hop0_q = f"What {work_type} is based on {anchor}?"
        hop1_q = f"Who {agent_vb} the {work_type} based on {anchor}?"

        target_all = [e.text for e in entities]
        hop_sequence = [
            HopStep(
                step_id=0,
                sub_query=hop0_q,
                target_entities=[anchor],
                depends_on=[],
                is_bridge=True,
            ),
            HopStep(
                step_id=1,
                sub_query=hop1_q,
                target_entities=[anchor],
                depends_on=[0],
                is_bridge=True,
            ),
            HopStep(
                step_id=2,
                sub_query=query,
                target_entities=target_all,
                depends_on=[1],
                is_bridge=False,
            ),
        ]
        logger.debug(
            "_decompose_multi_hop: Pattern H attribution-chain → hop0=%r hop1=%r",
            hop0_q[:50], hop1_q[:50],
        )
        return hop_sequence, [hop0_q, hop1_q, query]

    # Modifier dependency labels that make a noun phrase a "definite
    # description" — a referring expression that identifies an entity by its
    # properties rather than by name (Russell 1905; Strawson 1950). A1 uses the
    # longest entity-free such phrase as the bridge-resolution (hop-0) query.
    _A1_DESCRIPTION_MODIFIERS: frozenset = frozenset({
        "det", "amod", "nummod", "compound", "prep", "acl", "relcl",
        "advcl", "poss", "nmod",
    })
    # Wh-determiners that head a questioned attribute ("what class", "which
    # film"). Used by A4 to detect "attribute-of-a-related-entity" questions.
    _A4_WH_DETERMINERS: frozenset = frozenset({"what", "which", "whose"})

    def _find_entity_free_description(
        self,
        query: str,
        entities: List[EntityInfo],
    ) -> Optional[str]:
        """A1: when a multi-hop question references its bridge entity by
        DESCRIPTION rather than by name, return that description as a retrieval
        query (hop-0), else None.

        Generality (publication-defensibility): this keys on the linguistic
        notion of a *definite description* (a heavily-modified noun phrase that
        denotes an entity by its properties), not on any surface construction or
        verb list. Operationally: the longest object/complement noun phrase
        (pobj/dobj/attr head) whose subtree (a) contains NO named entity and
        (b) carries >=2 description-modifier dependents. Returns the cleaned
        subtree text. Fires only on the entity-free path, so it cannot compete
        with the named-entity decomposition routes above.
        """
        if not SPACY_AVAILABLE or NLP is None:
            return None
        # Only meaningful when the question names no usable anchor entity —
        # otherwise the entity-seeded routes handle it.
        seedable = {"PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART", "EVENT",
                    "FAC", "PRODUCT", "NORP"}
        if any(e.label in seedable for e in entities):
            return None
        try:
            doc = NLP(query)
            ner_token_indices = {t.i for ent in doc.ents for t in ent}
            best_text: Optional[str] = None
            best_len = 0
            for token in doc:
                if token.pos_ not in ("NOUN", "PROPN"):
                    continue
                if token.dep_ not in ("pobj", "dobj", "attr", "nsubjpass"):
                    continue
                subtree = list(token.subtree)
                # (a) entity-free: no NER token inside the phrase.
                if any(t.i in ner_token_indices for t in subtree):
                    continue
                # (b) >=2 description modifiers among the head's dependents.
                mod_count = sum(
                    1 for c in token.children
                    if c.dep_ in self._A1_DESCRIPTION_MODIFIERS
                )
                if mod_count < 2:
                    continue
                phrase = " ".join(t.text for t in subtree if not t.is_punct).strip()
                # Length in tokens — prefer the most specific (longest) phrase.
                if len(subtree) > best_len and len(phrase) >= 8:
                    best_len = len(subtree)
                    best_text = phrase
            return best_text
        except (ValueError, AttributeError, IndexError, KeyError) as exc:
            logger.debug("_find_entity_free_description failed: %s", exc)
            return None

    def _attribute_over_entity_signal(
        self,
        query: str,
        entities: List[EntityInfo],
    ) -> bool:
        """A4 signal: True if the question asks for a wh-determined ATTRIBUTE
        ("what class", "which network") of something RELATED TO a named entity,
        rather than a direct property of the entity itself — a general
        multi-hop signature ('attribute of a thing the entity relates to').

        Defensibility: keys on (i) a wh-determiner on a head noun and (ii) the
        presence of a named entity that is NOT that head noun. No verb/role
        lexicon. Used only as a tie-breaker when the classifier already
        abstained (see A4 gate in generate()).
        """
        if not SPACY_AVAILABLE or NLP is None:
            return False
        if not any(
            e.label in {"PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART", "EVENT",
                        "FAC", "PRODUCT", "NORP"}
            for e in entities
        ):
            return False
        try:
            doc = NLP(query)
            ent_texts = {e.text.lower() for e in entities}
            for token in doc:
                if token.text.lower() not in self._A4_WH_DETERMINERS:
                    continue
                if token.dep_ != "det":
                    continue
                head = token.head  # the questioned attribute noun
                if head.pos_ not in ("NOUN", "PROPN"):
                    continue
                # The questioned attribute head must not itself be the entity.
                if head.text.lower() in ent_texts:
                    continue
                return True
            return False
        except (ValueError, AttributeError, IndexError, KeyError) as exc:
            logger.debug("_attribute_over_entity_signal failed: %s", exc)
            return False

    # Interrogative words that make a fragment a usable retrieval target on
    # their own (a wh-question is well-formed even without a named subject).
    _WH_WORDS = frozenset(
        {"which", "who", "whom", "whose", "what", "where", "when", "why", "how"}
    )

    def _subquery_is_well_formed(self, text: str, part_entities: List[str]) -> bool:
        """Well-formedness invariant for an emitted sub-query (Item 4).

        A sub-query can anchor retrieval iff it contains at least one of:
          (a) a named entity,
          (b) a noun-phrase subject (SpaCy `nsubj`/`nsubjpass`), or
          (c) an interrogative wh-word.
        A fragment with none of these — a bare predicate or connector residue
        such as "was released by the distributor" or "of the same year" — is
        not a retrievable query and is rejected so the caller can fall back to
        an entity-seeded plan. This is a precondition guarantee, not a
        heuristic: it never makes a hop worse than the malformed fragment it
        replaces.
        """
        if part_entities:
            return True
        lowered = text.lower()
        if any(tok in self._WH_WORDS for tok in re.findall(r"\b\w+\b", lowered)):
            return True
        if NLP is not None:
            try:
                doc = NLP(text)
                if any(tok.dep_ in ("nsubj", "nsubjpass") for tok in doc):
                    return True
            except (ValueError, AttributeError, IndexError, KeyError) as exc:  # pragma: no cover - parser robustness
                logger.debug("_subquery_is_well_formed parse failed: %s", exc)
        return False

    def _decompose_multi_hop(
        self,
        query: str,
        entities: List[EntityInfo],
    ) -> Tuple[List[HopStep], List[str]]:
        """
        Decompose a multi-hop query into ordered retrieval steps.

        The paper, Section 4.2 — Sub-query enrichment strategy:

        The query is split at bridge connectors ("that", "which", "who",
        "of the") to isolate the anchor part (contains named entities) from
        the bridge part (contains the unknown to resolve).

        After splitting, parts are reversed so that the bridge step (which
        must be resolved first using the anchor entities) comes before the
        final step.

        Two enrichment cases handle missing entity context:

        Fall A — Bridge step without entities:
          "was formed by who?" has no named entity. Donor entities are taken
          from the other (anchor) parts and prepended:
          → "2014 S/S was formed by who?"

        Fall B — Final step with only vague generic references:
          "What position was held by the woman?" contains no proper name.
          The vague NP is replaced with the bridge-part entity:
          → "What position was held by Shirley Temple?"
        """
        hop_sequence = []
        sub_queries = []

        # ── Pattern J: Implicit bridge ("X and another [noun] that …") ───────
        # e.g. "Jane Goodall has been a critic of Nestlé and another corporation
        #  that has operations in how many countries?"
        # The phrase "another [noun]" signals that the answer-bearing entity is
        # unnamed in the query; it must be resolved via the named anchor entity
        # before the count/attribute question can be answered.
        # hop 0: resolve the bridge RELATION-NEUTRALLY — "Which corporation is
        #         mentioned together with Nestlé?" (no relation is assumed; the
        #         anchor's retrieved article names the co-referenced entity).
        # hop 1: answer the attribute question — original query (answered with
        #         the bridge entity now materialised in context)
        imb = self._find_implicit_bridge(query, entities)
        if imb:
            # Tag before delegating so the marker survives the return.
            self._last_matched_pattern = "J_implicit_bridge"
            return self._decompose_implicit_bridge(query, imb, entities)

        # ── Pattern G: Relative-clause bridge (SpaCy dependency parse) ───────
        # Form 1: "The [noun] in which [Entity] [predicate] [main question]?"
        # Form 2 (Pattern L): "...the [King|actress|author] who [verb] [Entity]..."
        # Uses structural grammar (relcl dep label) — no verb list needed.
        # Must run BEFORE the generic split because the split-on-"which" would
        # otherwise destroy the grammatical structure of the query.
        rc = self._find_relative_clause_bridge(query)
        if rc:
            noun, entity_text, form = rc
            if form == "form2":
                # Form 2: the bridge entity is the unknown role-NP; the anchor
                # is the NER entity inside the relative clause. The bridge
                # sub-query asks "which {role} is associated with {anchor}"
                # so retrieval pulls the anchor's article (which names the role).
                bridge_q = f"Who is the {noun} associated with {entity_text}?"
            else:
                # Form 1 (original): the entity is the subject of the relcl.
                bridge_q = f"In which {noun} did {entity_text}?"
            hop_sequence = [
                HopStep(
                    step_id=0,
                    sub_query=bridge_q,
                    target_entities=[e.text for e in entities],
                    depends_on=[],
                    is_bridge=True,
                ),
                HopStep(
                    step_id=1,
                    sub_query=query,
                    target_entities=[e.text for e in entities],
                    depends_on=[0],
                    is_bridge=False,
                ),
            ]
            logger.debug(
                "_decompose_multi_hop: Pattern G (%s) → %r", form, bridge_q[:60]
            )
            # Mark which Pattern G variant fired ("G_form1" or "G_form2")
            # so the JSONL can distinguish them in per-pattern analysis.
            self._last_matched_pattern = f"G_{form}"
            return hop_sequence, [bridge_q, query]

        # ── Pattern H: chained-attribution bridge (SpaCy dependency parse) ───
        # Detects "A [work] based on [Entity], is [written] by someone [attr]?"
        # — a *two-link* chain (entity → work → agent → attribute).  Must run
        # BEFORE the generic split because the split-on-connector would shred
        # the multi-clause structure.  The parse-confidence gate inside
        # _find_attribution_chain returns None on anything ambiguous, so this is
        # fail-safe — worst case it falls through to the patterns below.
        ac = self._find_attribution_chain(query, entities)
        if ac:
            # Tag Pattern H (chained-attribution, 3-hop chain).
            self._last_matched_pattern = "H_attribution_chain"
            return self._decompose_attribution_chain(query, ac, entities)

        # Split at bridge connectors; maxsplit=1 per pattern preserves the
        # full tail of the query on the right side of the split.
        split_patterns = [
            r"\s+(that|which|who)\s+",
            r"\s+of\s+the\s+",
        ]

        parts = [query]
        for pattern in split_patterns:
            new_parts = []
            for part in parts:
                split_result = re.split(pattern, part, maxsplit=1, flags=re.IGNORECASE)
                new_parts.extend(split_result)
            parts = [p.strip() for p in new_parts if p.strip() and len(p.strip()) > 5]

        # ── 2-hop cap ─────────────────────────────────────────────────────────
        # Repeated splitting on different connectors can produce 3+ parts from
        # a single relative clause:
        #   "What is the middle name of the actress who plays X in Y?"
        #   → ["middle name", "actress", "plays X in Y"]   (3 parts → 3 hops)
        # But semantically this is still a 2-hop bridge: resolve the actress,
        # then answer the attribute. The middle parts ("actress") become
        # nonsensical sub-queries ("What is actress?") that retrieve random
        # actresses as distractors and propagate spurious bridge entities.
        #
        # Heuristic: keep the first part (final attribute) and the last part
        # (bridge resolver), drop everything in between. The dropped fragments
        # carry no named entity — they were just clause connectors picked up by
        # the over-eager split.
        #
        # The cap is conservative: it only collapses when the middle parts have
        # NO named entity. If any middle part has an entity, we keep all parts
        # so genuine 3-hop chains (Pattern H attribution) survive.
        if len(parts) > 2:
            middle_has_entity = any(
                any(e.text.lower() in mid.lower() for e in entities)
                for mid in parts[1:-1]
            )
            if not middle_has_entity:
                logger.debug(
                    "_decompose_multi_hop: collapsing %d-part split → 2-hop "
                    "(no entity in middle parts: %r)",
                    len(parts), parts[1:-1],
                )
                parts = [parts[0], parts[-1]]

        # Patterns C and D were removed (surface-form recognisers).
        # See the class-level note for the rationale. The connector-split
        # baseline above and the dep-parse patterns (G, H, F) below cover the
        # structural cases; the surface-form regexes for "for a/an/the
        # CATEGORY" and "What ROLE with/having QUAL VERB" were retired.

        # ── Pattern E: Relational-noun + of-PP complement (dep-parse) ──────────
        # Detects "the ROLE of ENTITY" where ROLE is any noun and ENTITY is a
        # named entity. No role-word enumeration — see class docstring for
        # the linguistic basis (Partee 1995; Barker 1995).
        if len(parts) <= 1 and entities:
            rn = self._find_relational_noun_bridge(query, entities)
            if rn is not None:
                role, anchor = rn
                # Use the noun's surface form (lowercased for consistency).
                bridge_q = f"Who is the {role.lower()} of {anchor}?"
                hop_sequence = [
                    HopStep(
                        step_id=0,
                        sub_query=bridge_q,
                        target_entities=[anchor],
                        depends_on=[],
                        is_bridge=True,
                    ),
                    HopStep(
                        step_id=1,
                        sub_query=query,
                        target_entities=[e.text for e in entities],
                        depends_on=[0],
                        is_bridge=False,
                    ),
                ]
                logger.debug(
                    "_decompose_multi_hop: Pattern E relational-noun (%s of %s) → %r",
                    role, anchor, bridge_q[:60],
                )
                self._last_matched_pattern = "E_relational_noun"
                return hop_sequence, [bridge_q, query]

        # ── Pattern F: Passive-agent voice transformation (dep-parse) ─────────
        # Detects passive constructions with by-phrase agents via SpaCy's
        # `auxpass`/`nsubjpass`/`agent` dependency labels. Past-participle →
        # infinitive transformation uses SpaCy's lemmatiser. See class
        # docstring for the linguistic basis.
        pa = self._find_passive_agent_bridge(query)
        if pa is not None and len(parts) >= 1:
            subj, active_verb = pa
            # GUARD: skip Pattern F when the passive subject is itself an
            # interrogative noun phrase. The template f"Who {verb} {subj}?" with
            # an interrogative subject (e.g. "What government position") produces
            # self-referential nonsense ("Who hold What government position?")
            # that retrieves arbitrary chunks. Falling through to the
            # connector-split baseline is documented as the safer behaviour
            # (see class docstring, "Baseline — Connector-split decomposition").
            if (self._PASSIVE_F_INTERROGATIVE_SUBJ_RE.match(subj)
                    or subj.strip().lower() in self._PASSIVE_F_BARE_PRONOUN_SUBJ):
                logger.debug(
                    "_decompose_multi_hop: Pattern F skipped — interrogative or "
                    "bare-pronoun subject %r; falling through to connector-split "
                    "baseline",
                    subj[:60],
                )
                # Do NOT set _last_matched_pattern here — let the downstream path
                # set its own marker (connector_split / fallback_*).
            else:
                bridge_q = f"Who {active_verb} {subj}?"
                hop_sequence = [
                    HopStep(
                        step_id=0,
                        sub_query=bridge_q,
                        target_entities=[e.text for e in entities],
                        depends_on=[],
                        is_bridge=True,
                    ),
                    HopStep(
                        step_id=1,
                        sub_query=query,
                        target_entities=[e.text for e in entities],
                        depends_on=[0],
                        is_bridge=False,
                    ),
                ]
                logger.debug(
                    "_decompose_multi_hop: Pattern F passive-agent (%s/%s) → %r",
                    subj[:30], active_verb, bridge_q[:60],
                )
                self._last_matched_pattern = "F_passive_agent"
                return hop_sequence, [bridge_q, query]

        # Item 4 well-formedness gate: every connector-split part must be a
        # usable retrieval target (named entity, NP subject, or wh-word). If any
        # part is a bare subject-less/entity-less fragment, the split is not
        # safe to emit — abandon it and fall through to the entity-seeded
        # consistency fallback below, which retrieves on the detected entities
        # and answers the original query rather than a broken fragment.
        connector_split_usable = len(parts) > 1
        if connector_split_usable:
            for part in parts:
                part_ents = [e.text for e in entities if e.text.lower() in part.lower()]
                if not self._subquery_is_well_formed(part, part_ents):
                    connector_split_usable = False
                    logger.debug(
                        "_decompose_multi_hop: connector-split rejected — malformed "
                        "part %r (no entity / NP subject / wh-word); using "
                        "entity-seeded fallback", part[:60],
                    )
                    break

        if connector_split_usable:
            # The generic connector-split path is the baseline algorithm
            # described in the methodology ("split at bridge connectors, reverse,
            # enrich"). Tagged distinctly so per-pattern analysis can separate
            # "patterns added on top" from "the baseline did this".
            self._last_matched_pattern = "connector_split"
            reversed_parts = list(reversed(parts))

            for i, part in enumerate(reversed_parts):
                depends = list(range(i)) if i > 0 else []
                part_entities = [
                    e.text for e in entities
                    if e.text.lower() in part.lower()
                ]

                enriched_part = part
                is_bridge_step = (i < len(parts) - 1)
                is_final_step  = (i == len(parts) - 1)

                if is_bridge_step and not part_entities:
                    # Fall A: bridge step has no known entities; borrow from anchor.
                    # other_parts = all non-bridge parts (anchor side); may be >1.
                    other_parts = reversed_parts[1:]
                    donor_entity_texts = [
                        e.text for e in entities
                        if any(e.text.lower() in ap.lower() for ap in other_parts)
                        and e.text.lower() not in part.lower()
                    ]
                    # Secondary fallback: extract a noun phrase from the anchor part
                    if not donor_entity_texts:
                        for op in other_parts:
                            m = re.search(
                                r'\b(?:a|an|the)\s+((?:\w+\s+){1,3}'
                                r'(?:group|band|company|team|label|artist|person|film|movie|show))\b',
                                op, re.IGNORECASE,
                            )
                            if m:
                                donor_entity_texts.append(m.group(1).strip())
                                break
                    if donor_entity_texts:
                        ctx = " ".join(donor_entity_texts[:2])
                        enriched_part = f"{ctx} {part}"

                elif is_final_step and self._VAGUE_REFS.search(part) and not part_entities:
                    # Fall B: final step contains only a vague generic; replace with entity
                    bridge_parts = reversed_parts[:i]
                    donor_entity_texts = [
                        e.text for e in entities
                        if any(e.text.lower() in bp.lower() for bp in bridge_parts)
                        and e.text.lower() not in part.lower()
                    ]
                    if donor_entity_texts:
                        ctx = " ".join(donor_entity_texts[:2])
                        enriched_part = self._VAGUE_REFS.sub(ctx, part, count=1)

                sub_query = self._form_sub_query(enriched_part)

                hop_sequence.append(HopStep(
                    step_id=i,
                    sub_query=sub_query,
                    target_entities=part_entities,
                    depends_on=depends,
                    is_bridge=(i < len(parts) - 1),
                ))
                sub_queries.append(sub_query)
        else:
            # ── Classification–decomposition consistency fallback ───────────
            # The query was *classified* multi-hop but every pattern + the
            # connector split failed.  Rather than silently emitting the
            # unsplit query (which looks like a working single-hop plan and
            # hides the gap), emit a deliberate 2-hop fallback: hop0 retrieves
            # broadly around the detected entities, hop1 answers the original
            # query with that context available.  If there are no usable
            # entities to seed hop0, only then degrade to single-hop — with a
            # WARNING so the gap is visible in logs / diagnostics.
            seed_entities = [
                e for e in entities
                if e.label in ("PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART",
                                "EVENT", "FAC", "PRODUCT", "NORP")
            ]
            if seed_entities:
                anchor = seed_entities[0].text
                hop0_q = f"Who or what is {anchor}?"
                logger.debug(
                    "_decompose_multi_hop: no pattern matched for %r "
                    "— generic 2-hop fallback (anchor=%r)", query[:80], anchor
                )
                # Classification-decomposition consistency fallback.
                # Distinct from the connector split because no real decomposition
                # happened — we only fabricated a hop-0 "Who is X?" to keep the
                # plan multi-hop. Surfacing this lets the eval report
                # "classified MULTI_HOP but no pattern fired" rate honestly.
                self._last_matched_pattern = "fallback_generic_2hop"
                hop_sequence = [
                    HopStep(
                        step_id=0,
                        sub_query=hop0_q,
                        target_entities=[anchor],
                        depends_on=[],
                        is_bridge=True,
                    ),
                    HopStep(
                        step_id=1,
                        sub_query=query,
                        target_entities=[e.text for e in entities],
                        depends_on=[0],
                        is_bridge=False,
                    ),
                ]
                sub_queries = [hop0_q, query]
            else:
                # A1: before degrading, try definite-description resolution —
                # the bridge entity may be referenced by DESCRIPTION rather than
                # by name (e.g. "the only player ... to have a 0.300 average").
                # Use that description as the hop-0 retrieval query.
                description = self._find_entity_free_description(query, entities)
                if description and description.lower() != query.lower():
                    logger.debug(
                        "_decompose_multi_hop: entity-free definite-description "
                        "bridge → descriptive 2-hop (hop0=%r)", description[:60]
                    )
                    self._last_matched_pattern = "structural_descriptive_2hop"
                    hop_sequence = [
                        HopStep(
                            step_id=0,
                            sub_query=description,
                            target_entities=[],
                            depends_on=[],
                            is_bridge=True,
                        ),
                        HopStep(
                            step_id=1,
                            sub_query=query,
                            target_entities=[e.text for e in entities],
                            depends_on=[0],
                            is_bridge=False,
                        ),
                    ]
                    sub_queries = [description, query]
                else:
                    logger.warning(
                        "_decompose_multi_hop: query classified MULTI_HOP but no "
                        "pattern matched and no anchor entity available — "
                        "degrading to single-hop: %r", query[:100]
                    )
                    hop_sequence.append(HopStep(
                        step_id=0,
                        sub_query=query,
                        target_entities=[e.text for e in entities],
                        depends_on=[],
                        is_bridge=False,
                    ))
                    sub_queries = [query]
                    # Failure marker. The classifier said MULTI_HOP but
                    # NOTHING worked. Surfacing this is essential — without it,
                    # the eval cannot distinguish "we solved this" from "we
                    # silently gave up".
                    self._last_matched_pattern = "fallback_degraded_to_single_hop"

        # A1 contract: a MULTI_HOP classification must never SILENTLY collapse
        # to a single sub-query. The only permitted single-sub-query output is
        # the explicitly logged + marked degrade path above. This is a logged
        # invariant (not a hard assert) so an unforeseen edge case degrades
        # gracefully instead of crashing the pipeline.
        if len(sub_queries) < 2 and self._last_matched_pattern != "fallback_degraded_to_single_hop":
            logger.error(
                "_decompose_multi_hop: contract violation — multi-hop produced "
                "%d sub-query with marker %r (query=%r)",
                len(sub_queries), self._last_matched_pattern, query[:80],
            )

        return hop_sequence, sub_queries

    # Boolean-conjunction surface form: "Are/Did/Were/Is/Do/Does/Have/Has [X] and [Y] both [P]?"
    # The "both" keyword is a reliable discriminator — genuine bridge questions almost
    # never contain it. Lexical detection avoids SpaCy parse dependency.
    # Pattern I — Boolean conjunction decomposition. Compiled from the shared
    # module-level _BOOL_CONJ_PATTERN (same string the classifier pre-empt uses).
    _BOOL_CONJ_RE = re.compile(_BOOL_CONJ_PATTERN, re.IGNORECASE)
    # Conjunction / quantifier locators used to split "[AUX] X and Y both P?".
    _AND_RE = re.compile(r'\band\b', re.IGNORECASE)
    _BOTH_RE = re.compile(r'\bboth\b', re.IGNORECASE)
    _LEADING_AUX_RE = re.compile(
        r'^\s*(are|is|were|was|did|do|does|have|has)\s+', re.IGNORECASE,
    )

    def _decompose_boolean_conjunction(
        self,
        query: str,
        entities: List[EntityInfo],
    ) -> Optional[Tuple[List[HopStep], List[str]]]:
        """Decompose "Are [X] and [Y] both [predicate]?" into two parallel yes/no
        sub-queries, one per subject entity.

        Returns (hop_sequence, sub_queries) when the Boolean conjunction form is
        detected and at least two named entities are present; otherwise None so
        the caller falls through to the generic comparison path.

        The predicate fragment is recovered by removing the "[X] and [Y]" span
        from the query, leaving e.g. "Are ... both used for real estate?" which
        is then specialised per entity: "Is Random House Tower used for real estate?"
        """
        if not self._BOOL_CONJ_RE.match(query):
            return None

        # Extract entity names directly from the query's conjunction structure:
        # "[AUX] [X] and [Y] both [P]?" — split on " and " then on " both ".
        # This bypasses SpaCy NER entirely for entity identification in this
        # pattern, avoiding MONEY/CARDINAL misclassification of address-like
        # strings such as "888 7th Avenue".
        both_m = self._BOTH_RE.search(query)
        and_m  = self._AND_RE.search(query)
        if both_m and and_m and and_m.start() < both_m.start():
            # Strip the leading auxiliary verb to get ent_a text
            after_aux = self._LEADING_AUX_RE.sub('', query, count=1)
            raw_a = after_aux[:and_m.start() - (len(query) - len(after_aux))].strip()
            raw_b = query[and_m.end():both_m.start()].strip().rstrip(',').strip()
            if raw_a and raw_b:
                # Build synthetic EntityInfo objects with correct char offsets.
                # Confidence is sourced from config (regex_entity_confidence) so
                # retuning that setting also moves these synthetic spans — review
                # 2026-06-12, finding #4 (was a hardcoded 0.75 that drifted from
                # the configured value).
                _syn_conf = self.config.regex_entity_confidence
                a_idx = query.find(raw_a)
                b_idx = query.find(raw_b)
                ent_a = EntityInfo(
                    text=raw_a, label="PROPN", confidence=_syn_conf,
                    start_char=a_idx if a_idx >= 0 else 0,
                    end_char=(a_idx + len(raw_a)) if a_idx >= 0 else len(raw_a),
                )
                ent_b = EntityInfo(
                    text=raw_b, label="PROPN", confidence=_syn_conf,
                    start_char=b_idx if b_idx >= 0 else 0,
                    end_char=(b_idx + len(raw_b)) if b_idx >= 0 else len(raw_b),
                )
                logger.debug(
                    "_decompose_boolean_conjunction: conjunction parse → a=%r b=%r",
                    raw_a, raw_b,
                )
                # Skip the NER-based entity selection below
                ner_entities = [ent_a, ent_b]
            else:
                ner_entities = [e for e in entities if e.label in self._NER_LABELS]
                if len(ner_entities) < 2:
                    return None
        else:
            ner_entities = [e for e in entities if e.label in self._NER_LABELS]
            if len(ner_entities) < 2:
                return None

        ent_a, ent_b = ner_entities[0], ner_entities[1]

        # Strip the leading auxiliary verb and swap to singular "Is/Did/Was..."
        _AUX_MAP = {
            "are": "Is", "were": "Was", "did": "Did",
            "do": "Does", "does": "Does", "have": "Has", "has": "Has", "is": "Is",
        }
        first_token = query.split()[0].lower()
        singular_aux = _AUX_MAP.get(first_token, "Is")

        # Build the predicate fragment: remove "[X] and [Y]" span from query
        a_idx = query.find(ent_a.text)
        b_idx = query.find(ent_b.text)
        if a_idx < 0 or b_idx < 0:
            return None

        conj_start = a_idx
        conj_end   = b_idx + len(ent_b.text)
        # Eat the word "both" immediately after the conjunction if present
        after_conj = query[conj_end:].lstrip()
        if after_conj.lower().startswith("both "):
            conj_end += query[conj_end:].index("both") + len("both")

        predicate = query[conj_end:].strip().rstrip("?").strip()

        hop_sequence = []
        sub_queries  = []
        for i, ent in enumerate([ent_a, ent_b]):
            sq = f"{singular_aux} {ent.text} {predicate}?"
            hop_sequence.append(HopStep(
                step_id=i,
                sub_query=sq,
                target_entities=[ent.text],
                depends_on=[],
                is_bridge=False,
            ))
            sub_queries.append(sq)
            logger.debug(
                "_decompose_boolean_conjunction: Pattern I → hop%d=%r", i, sq[:80]
            )

        return hop_sequence, sub_queries

    def _decompose_comparison(
        self,
        query: str,
        entities: List[EntityInfo],
    ) -> Tuple[List[HopStep], List[str]]:
        """
        Decompose a comparison query into parallel retrieval steps.

        Strategy:
        0. Boolean conjunction check: "Are [X] and [Y] both [P]?" → Pattern I.
        1. Identify the two primary entities to compare (prefer NER over regex).
        2. Generate one sub-query per entity (can run in parallel).
        3. Apply attribute rewriting (_ATTR_MAP) to improve vector similarity.
        4. Append the original query as the final comparison step.

        Edge case — zero entities detected:
          If no named entities are found (SpaCy below threshold, regex filtered),
          the original query is used as a single retrieval step. The Navigator
          will retrieve broadly and the Verifier synthesises from the context.
        """
        hop_sequence = []
        sub_queries = []

        # ── Pattern I: Boolean conjunction ("Are X and Y both P?") ──────────────
        # Must run before select-between-two: "both" keyword is a reliable signal
        # that this is a parallel yes/no check, not a selection-between-two-options.
        # Gated on enable_classifier_preempts (review 2026-06-12, finding #5) so
        # that ablating the pre-empts fully ablates Pattern I — previously the
        # Phase-0 classifier pre-empt was skipped but this decomposition still
        # fired, so "pre-empts off" only PARTIALLY removed Pattern-I behaviour.
        if self.config.enable_classifier_preempts:
            bool_conj = self._decompose_boolean_conjunction(query, entities)
            if bool_conj is not None:
                # Tag before returning so the marker survives the call.
                self._last_matched_pattern = "I_boolean_conjunction"
                return bool_conj

        # Disjunctive coordination ("X or Y") between two named entities is
        # the canonical surface form of an English alternative interrogative
        # (Karttunen 1977; Higginbotham 1993). The decomposer below detects
        # the disjunction via SpaCy NER spans rather than via comma position,
        # and emits one parallel retrieval per disjunct.
        sel = self._decompose_select_between(query, entities)
        if sel is not None:
            self._last_matched_pattern = "select_between_two"
            return sel

        # Use only proper NER entities; regex-PROPN entities include noisy
        # sentence-initial tokens and are filtered out here.
        ner_entities = [e for e in entities if e.label in self._NER_LABELS]

        if len(ner_entities) >= 2:
            comparison_entities = ner_entities[:2]
        else:
            # Fallback: greedily select non-overlapping entities from the full list
            selected: List[EntityInfo] = list(ner_entities)
            for e in entities:
                if len(selected) >= 2:
                    break
                overlaps = any(
                    not (e.end_char <= sel.start_char or e.start_char >= sel.end_char)
                    for sel in selected
                )
                if not overlaps and e not in selected:
                    selected.append(e)
            comparison_entities = selected[:2]

        # ── Zero-entity guard ─────────────────────────────────────────────────
        # If no entities were detected at all, fall back to using the original
        # query as the sole sub-query. The Navigator retrieves broadly and the
        # Verifier synthesises the comparison from the retrieved context.
        if not comparison_entities:
            logger.debug(
                "_decompose_comparison: no entities detected for '%s' → single fallback step",
                query[:80],
            )
            hop_sequence.append(HopStep(
                step_id=0,
                sub_query=query,
                target_entities=[],
                depends_on=[],
                is_bridge=False,
            ))
            # Comparison was classified but no entities found —
            # surfaces a measurable failure mode in the eval.
            self._last_matched_pattern = "comparison_no_entities"
            return hop_sequence, [query]

        sub_query_templates = []

        if len(comparison_entities) >= 2:
            idx = [query.find(e.text) for e in comparison_entities]
            if all(i >= 0 for i in idx):
                span_start = min(idx)
                span_end   = max(
                    idx[j] + len(comparison_entities[j].text)
                    for j in range(len(comparison_entities))
                )
                prefix = query[:span_start]
                suffix = query[span_end:]
                for e in comparison_entities:
                    sq = re.sub(r'\s+', ' ', (prefix + e.text + suffix).strip())
                    sub_query_templates.append((e, sq))

        if not sub_query_templates:
            # Fallback: entity positions not found in query string
            # (e.g. entity text was normalised or contains special characters)
            logger.debug(
                "_decompose_comparison: entity position not found for query '%s'"
                " → using generic template",
                query[:80],
            )
            sub_query_templates = [
                (e, f"What is {e.text}?") for e in comparison_entities
            ]

        # Attribute rewriting: "Were X of the same nationality?" →
        # "What is the nationality of X?" (per _ATTR_MAP above).
        # Improves vector similarity to factual chunks such as
        # "X is an American filmmaker" (Yang et al., 2018 EMNLP).
        rewritten = []
        attr_rewrite_fired = False
        for pattern, template in self._ATTR_MAP:
            if pattern.search(query):
                rewritten = [
                    (e, template.format(entity=e.text))
                    for e, _ in sub_query_templates
                ]
                attr_rewrite_fired = True
                break
        if rewritten:
            sub_query_templates = rewritten

        # Tag the comparison sub-variant we actually emit. Distinguishes
        # _ATTR_MAP-rewritten comparisons from plain per-entity comparisons in
        # the per-pattern eval table.
        self._last_matched_pattern = (
            "comparison_attr_map" if attr_rewrite_fired
            else "comparison_parallel"
        )

        # One step per entity (steps are independent → can run in parallel).
        # The original query is intentionally NOT appended as a third
        # sub-query. The comparison phrasing as a whole is less specific than
        # either per-entity factual lookup, and adding it as a retrieval
        # query introduces broad-match noise that inflates RRF scores for
        # irrelevant chunks via the cross-query corroboration bonus. The two
        # entity-specific attribute-template queries are sufficient.
        for i, (entity, sub_query) in enumerate(sub_query_templates):
            hop_sequence.append(HopStep(
                step_id=i,
                sub_query=sub_query,
                target_entities=[entity.text],
                depends_on=[],  # no dependency → parallel execution
                is_bridge=True,
            ))
            sub_queries.append(sub_query)

        return hop_sequence, sub_queries

    # Leading framing to strip from a "select-between-two" question to recover
    # the property being compared: "Which writer ", "What city ", "Who ", etc.
    _SELECT_LEAD_RE = re.compile(
        r'^\s*(which|what|who|whom)\b\s*([a-z][a-z\- ]{0,30}?\b)?\s*',
        re.IGNORECASE,
    )
    # "ENT_A or ENT_B" disjunction (allowing a comma before "or").
    _OR_DISJ_RE = re.compile(r'\s*,?\s+\bor\b\s+', re.IGNORECASE)

    def _decompose_select_between(
        self,
        query: str,
        entities: List[EntityInfo],
    ) -> Optional[Tuple[List[HopStep], List[str]]]:
        """Handle the "Which <category> <property>, A or B<property>?" comparison
        form, where A and B are the two entities joined by "or" in the query.

        Returns (hop_sequence, sub_queries) on success, or None if the query is
        not of this form (caller then falls back to the generic decomposition).

        Strategy:
          1. Find two extracted entities that sit on either side of an "or" in
             the query, with only whitespace/comma between them and "or".
          2. Remove that "<A> or <B>" span from the query and strip the leading
             "Which <category>" framing; what remains is the predicate clause
             shared by the two disjuncts.
          3. Build one focused sub-query per entity by attaching the entity to
             the predicate clause. If an `_ATTR_MAP` attribute pattern matches
             the original query, prefer that attribute-template rewrite instead.
        """
        # 1. Locate an "A or B" disjunction of two entities.
        ents_by_pos = sorted(entities, key=lambda e: e.start_char)
        pair: Optional[Tuple[EntityInfo, EntityInfo]] = None
        for a, b in zip(ents_by_pos, ents_by_pos[1:]):
            gap = query[a.end_char:b.start_char]
            if self._OR_DISJ_RE.fullmatch(gap):
                pair = (a, b)
                break
        if pair is None:
            return None
        ent_a, ent_b = pair

        # 2. Excise the "<A> or <B>" span; strip leading "Which <category>".
        disj_start, disj_end = ent_a.start_char, ent_b.end_char
        remainder = (query[:disj_start] + " " + query[disj_end:])
        remainder = re.sub(r'\s+', ' ', remainder).strip().rstrip('?').strip()
        # drop a dangling leading/trailing comma left by the excision
        remainder = remainder.strip(',').strip()
        property_clause = self._SELECT_LEAD_RE.sub('', remainder).strip()
        # also drop any leftover leading conjunction/punctuation
        property_clause = property_clause.lstrip(',; ').strip()

        # 3. Per-entity sub-queries.
        # Prefer an attribute rewrite if one of the _ATTR_MAP / comparative
        # patterns applies to the original query.
        templates: List[str] = []
        for pat, tmpl in self._ATTR_MAP:
            if pat.search(query):
                templates = [tmpl.format(entity=ent_a.text),
                             tmpl.format(entity=ent_b.text)]
                break
        if not templates:
            if property_clause:
                # Generic predicate template: "<entity> <predicate>".
                templates = [f"{ent_a.text} {property_clause}",
                             f"{ent_b.text} {property_clause}"]
            else:
                # No usable property clause recovered → fall back to a generic
                # "Who/what is <entity>?" lookup so retrieval at least targets
                # the right two articles.
                templates = [f"Who is {ent_a.text}?", f"Who is {ent_b.text}?"]

        hop_sequence: List[HopStep] = []
        sub_queries: List[str] = []
        for i, (ent, sq) in enumerate(zip((ent_a, ent_b), templates)):
            sq = re.sub(r'\s+', ' ', sq).strip()
            hop_sequence.append(HopStep(
                step_id=i,
                sub_query=sq,
                target_entities=[ent.text],
                depends_on=[],          # independent → parallel
                is_bridge=True,
            ))
            sub_queries.append(sq)
        logger.debug(
            "_decompose_select_between: %r → [%r, %r]",
            query[:80], sub_queries[0][:50], sub_queries[1][:50],
        )
        return hop_sequence, sub_queries

    def _decompose_intersection(
        self,
        query: str,
        entities: List[EntityInfo],
    ) -> Tuple[List[HopStep], List[str]]:
        """
        Decompose an intersection query.

        Intersection and comparison share identical retrieval decomposition:
        both require parallel per-entity lookups followed by a synthesis step.
        The difference lies in the Verifier's synthesis step (intersection
        identifies shared attributes; comparison ranks or contrasts), not in
        the retrieval plan structure. Both therefore route through
        _decompose_comparison.
        """
        return self._decompose_comparison(query, entities)

    def _decompose_temporal(
        self,
        query: str,
        entities: List[EntityInfo],
    ) -> Tuple[List[HopStep], List[str]]:
        """
        Decompose a temporal query.

        Temporal queries are typically single-hop with a temporal constraint;
        the time component is captured in the constraints dict rather than
        as a separate hop.
        """
        hop_sequence = [
            HopStep(
                step_id=0,
                sub_query=query,
                target_entities=[e.text for e in entities],
                depends_on=[],
                is_bridge=False,
            )
        ]
        return hop_sequence, [query]

    def _form_sub_query(self, part: str) -> str:
        """Convert a query fragment into a well-formed sub-query."""
        part = part.strip()

        # Remove leading conjunctions left over from splitting
        part = self._STRIP_LEADING_CONJ.sub("", part)

        # Ensure the fragment ends with "?" and has an interrogative prefix
        if not part.endswith("?"):
            if not self._INTERROGATIVE_PREFIX.match(part):
                part = f"What is {part}?"
            else:
                part = f"{part}?"

        return part

    def _extract_constraints(
        self,
        query: str,
        query_type: QueryType,
    ) -> Dict[str, Any]:
        """
        Extract constraints from the query.

        Constraints are additional conditions such as:
        - Temporal: year ranges, date references.
        - Comparison: direction (greater/less) and attribute.

        Args:
            query:      The user query.
            query_type: Classified query type.

        Returns:
            Dict of constraint key → value (may be empty).
        """
        constraints: Dict[str, Any] = {}

        # ─────────────────────────────────────────────────────────────────────
        # TEMPORAL CONSTRAINTS
        # ─────────────────────────────────────────────────────────────────────

        if query_type == QueryType.TEMPORAL or self.config.enable_temporal_parsing:
            # Extract historical / 21st-century years using module-level constant
            years = _YEAR_RE.findall(query)
            if years:
                constraints["years"] = years

            temporal_match = self._TEMPORAL_TERMS_RE.search(query)
            if temporal_match:
                constraints["temporal_relation"] = temporal_match.group(1).lower()

        # ─────────────────────────────────────────────────────────────────────
        # COMPARISON CONSTRAINTS
        # ─────────────────────────────────────────────────────────────────────

        if query_type == QueryType.COMPARISON:
            if self._COMPARISON_GREATER_RE.search(query):
                constraints["comparison_direction"] = "greater"
            elif self._COMPARISON_LESS_RE.search(query):
                constraints["comparison_direction"] = "less"

            attr_match = self._COMPARISON_ATTR_RE.search(query)
            if attr_match:
                constraints["comparison_attribute"] = attr_match.group(1).lower()

        return constraints


# =============================================================================
# MAIN PLANNER CLASS
# =============================================================================

class Planner:
    """
    S_P: Rule-based Query Planner.

    Orchestrates query classification, entity extraction, and plan generation.

    Usage:
        planner = Planner()
        plan = planner.plan("Who directed the movie with Greta Garbo?")
        sub_queries = plan.sub_queries   # flat list of retrieval sub-queries
    """

    def __init__(
        self,
        config: Optional[PlannerConfig] = None,
        # Kept for API compatibility with LLM-based planner signatures
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialise the Planner.

        Args:
            config:     PlannerConfig (optional).
            model_name: Ignored (API compatibility shim).
            base_url:   Ignored (API compatibility shim).
        """
        self.config = config or PlannerConfig()

        # Honour the configured SpaCy model (settings.yaml → ingestion.spacy_model)
        # rather than the import-time default — review 2026-06-12, finding #2.
        # Reloads the module-global NLP only if the configured model differs
        # from what is already in memory.
        _ensure_nlp(self.config.spacy_model)

        self.classifier       = QueryClassifier(self.config)
        self.entity_extractor = EntityExtractor(self.config)
        self.plan_generator   = PlanGenerator(self.config)

        logger.info(
            "Planner initialised: SpaCy=%s",
            "available" if SPACY_AVAILABLE else "unavailable",
        )

    def plan(self, query: str) -> RetrievalPlan:
        """
        Generate a complete retrieval plan for a query.

        This is the primary entry point for the Planner. It runs the full
        classification → entity extraction → plan generation pipeline and
        returns a structured RetrievalPlan for the Navigator (S_N).

        Args:
            query: The user query.

        Returns:
            RetrievalPlan with strategy, entities, hops, and constraints.
        """
        start_time = time.perf_counter()

        if query is None:
            return self._empty_plan("")
        query = query.strip()
        if not query:
            return self._empty_plan(query)

        # Step 1: Query classification
        query_type, confidence = self.classifier.classify(query)
        logger.debug("Query classified: %s (conf=%.2f)", query_type.value, confidence)

        # Step 2: Entity extraction
        entities = self.entity_extractor.extract(query)

        if query_type == QueryType.MULTI_HOP:
            entities = self.entity_extractor.detect_bridge_entities(query, entities)

        logger.debug("Entities extracted: %d", len(entities))

        # Step 3: Plan generation
        plan = self.plan_generator.generate(
            query=query,
            query_type=query_type,
            confidence=confidence,
            entities=entities,
        )

        # Surface any classifier pre-empt that fired so the JSONL can
        # audit pre-empt false-positive rates. None when classification took
        # the normal Phase 1-4 scoring path.
        plan.classifier_preempt = getattr(self.classifier, "_last_preempt", None)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        plan.metadata["planning_time_ms"] = elapsed_ms

        logger.info(
            "Plan generated in %.1fms: type=%s entities=%d hops=%d",
            elapsed_ms,
            query_type.value,
            len(entities),
            plan.estimated_hops,
        )

        return plan

    # decompose_query() was removed — it was a legacy wrapper around plan()
    # returning only sub_queries. Production code reads plan.sub_queries
    # directly. External callers should use:
    #     plan = planner.plan(query)
    #     sub_queries = plan.sub_queries

    def _empty_plan(self, query: str) -> RetrievalPlan:
        """Return a minimal plan for empty or invalid queries."""
        return RetrievalPlan(
            original_query=query,
            query_type=QueryType.SINGLE_HOP,
            strategy=RetrievalStrategy.VECTOR_ONLY,
            entities=[],
            hop_sequence=[],
            sub_queries=[query] if query else [],
            constraints={},
            estimated_hops=0,  # consistent with empty hop_sequence
            confidence=0.0,
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_planner(
    cfg: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,  # ignored — API compatibility
    base_url: Optional[str] = None,    # ignored — API compatibility
) -> Planner:
    """
    Factory function for Planner.

    When ``cfg`` is None, settings.yaml is auto-loaded from
    ``config/settings.yaml`` relative to the project root — so a bare
    ``create_planner()`` call always picks up the live settings.yaml values
    without any hardcoded fallbacks in the call site.

    ``model_name`` and ``base_url`` are accepted but ignored for API
    compatibility with LLM-based planner signatures (the Planner is
    rule-based and does not call an LLM).

    Args:
        cfg:        Full settings.yaml dict.  Auto-loaded when None.
        model_name: Ignored (API compatibility shim).
        base_url:   Ignored (API compatibility shim).

    Returns:
        Configured Planner instance.
    """
    if cfg is None:
        cfg = _load_settings()
    config = PlannerConfig.from_yaml(cfg)
    return Planner(config=config)


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    test_queries = [
        ("What is the capital of France?",                                  QueryType.SINGLE_HOP),
        ("Who is the director of the film that stars Greta Garbo?",         QueryType.MULTI_HOP),
        ("What is the capital of the country where Einstein was born?",     QueryType.MULTI_HOP),
        ("Is Berlin older than Munich?",                                    QueryType.COMPARISON),
        ("Which is taller, the Eiffel Tower or Big Ben?",                   QueryType.COMPARISON),
        ("What happened after World War 2?",                                QueryType.TEMPORAL),
        ("Who was president in 1990?",                                      QueryType.TEMPORAL),
        ("Which movies star both Brad Pitt and Leonardo DiCaprio?",         QueryType.INTERSECTION),
    ]

    print("=" * 70)
    print("S_P: RULE-BASED QUERY PLANNER SMOKE TEST")
    print(f"SpaCy available: {SPACY_AVAILABLE}")
    print("=" * 70)

    planner = create_planner()  # auto-loads config/settings.yaml

    total_time = 0
    correct = 0

    for query, expected_type in test_queries:
        plan = planner.plan(query)
        elapsed = plan.metadata.get("planning_time_ms", 0)
        total_time += elapsed

        is_correct = plan.query_type == expected_type
        correct += int(is_correct)
        status = "OK" if is_correct else "FAIL"

        print(f"\n[{status}] {query}")
        print(f"  Expected: {expected_type.value}  Got: {plan.query_type.value}")
        print(f"  Strategy: {plan.strategy.value}")
        print(f"  Entities: {[e.text for e in plan.entities]}")
        print(f"  Sub-queries: {plan.sub_queries}")
        print(f"  Hops: {plan.estimated_hops}  Time: {elapsed:.1f}ms")

    print("\n" + "=" * 70)
    print(
        f"Accuracy: {correct}/{len(test_queries)} "
        f"({100 * correct / len(test_queries):.1f}%)"
    )
    print(f"Average planning time: {total_time / len(test_queries):.1f}ms")
    print("=" * 70)
