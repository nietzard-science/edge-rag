"""
Hybrid Retriever with Reciprocal Rank Fusion (RRF).

Role in the pipeline
--------------------
Implements the Hybrid Retrieval component of Artifact A (Data Layer). It
is consumed by:
    src/logic_layer/navigator.py             — S_N agent, primary consumer
    src/thesis_evaluations/benchmark_datasets.py — evaluation harness
    src/pipeline/ingestion_pipeline.py       — end-to-end smoke tests

Construction: use the settings-reading factory
``create_hybrid_retriever(hybrid_store, embeddings, cfg)`` so every
``rag.* / vector_store.* / graph.*`` key in ``config/settings.yaml``
(``vector_top_k``, ``bm25_top_k``, ``rrf_k``, ``cross_source_boost``,
``enable_bm25``, ``enable_hop3``, ``hub_mention_cap``, and the per-source
RRF weights ``vector_weight / graph_weight / bm25_weight``) is honoured.
Hand-constructing ``RetrievalConfig`` with only a subset of fields
silently falls back to dataclass defaults — go through the factory.

Three retrieval lanes
---------------------
    query string
       │
       ▼
    ImprovedQueryEntityExtractor   (GLiNER → SpaCy → Regex fallback chain)
       │
       ▼
    HybridRetriever.retrieve()
       ├── VectorStoreAdapter.vector_search()   (LanceDB ANN, top-K)
       ├── _bm25_search()                       (rank_bm25, sparse lane)
       └── HybridStore.graph_search()           (KuzuDB 1-hop + 2-hop)
       │
       ▼
    RRFFusion.fuse()  (per-source-weighted RRF + cross-source boost)
       │
       ▼
    List[RetrievalResult]

RRF
---
    rrf(d) = Σ_lane  w_lane × ( 1 / (k + rank_lane(d)) )  +  BONUS

    BONUS = cross_source_boost / (k + 1)   when d appears in ≥2 lanes
    k     = 60                              (Cormack et al. 2009 default)

Lanes are independent — each contributes its rank-reciprocal share scaled
by the per-source weight. ``RetrievalMode.{VECTOR, GRAPH, HYBRID}`` gates
which lanes run. ``VECTOR`` fuses dense+BM25 when BM25 is enabled (the
"non-graph" lanes); ``HYBRID`` fuses all three.

Exports
-------
    RetrievalMode, RetrievalConfig, RetrievalResult, RetrievalMetrics
                                                             — data classes
    RRFFusion                                                — fusion helper
    ImprovedQueryEntityExtractor                             — query NER
    HybridRetriever                                          — orchestrator
    create_hybrid_retriever(hybrid_store, embeddings, cfg)   — primary factory

References (algorithm anchors)
------------------------------
    Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms
        Condorcet and individual rank learning methods. SIGIR.
        DOI:10.1145/1571941.1572114.
    Bruch et al. (2023). An Analysis of Fusion Functions for Hybrid
        Retrieval. ACM TOIS 41(4). (Per-source RRF weighting.)
    Robertson & Walker (1994); Robertson & Zaragoza (2009). BM25 / Okapi
        sparse retrieval.
    Zaratiana et al. (2023). GLiNER. arXiv:2311.08526.

Dependencies
------------
    src.data_layer.entity_extraction, src.data_layer.entity_types,
    src.data_layer.graph_quality, src.utils. numpy; rank_bm25 (optional —
    BM25 lane self-disables when absent); spaCy + GLiNER for query NER
    (with regex fallback). LanceDB + KuzuDB via the injected HybridStore.

Last reviewed: 2026-06-01 (audit pass, project version 5.5).
"""

import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

import numpy as np
from .entity_extraction import normalize_entity_name
from .entity_types import GLINER_LABEL_MAP, SPACY_LABEL_MAP
from .graph_quality import DEFAULT_STOPLIST, canonical_form
from src.utils import jaccard_similarity

logger = logging.getLogger(__name__)

# Public API consumed by:
#   - src/data_layer/__init__.py     (re-exports the 6 dataclasses + classes)
#   - src/thesis_evaluations/benchmark_datasets.py (uses create_hybrid_retriever)
#   - src/logic_layer/navigator.py   (HybridRetriever, RetrievalConfig)
# Module-internal helpers (_bm25_tokenize, _normalize_query_entity, the
# query-side span-boundary functions, _get_gliner_model) stay accessible by
# direct-name import for tests.
__all__ = [
    "HybridRetriever",
    "RetrievalConfig",
    "RetrievalMode",
    "RetrievalResult",
    "RetrievalMetrics",
    "ImprovedQueryEntityExtractor",
    "create_hybrid_retriever",
]

# ---------------------------------------------------------------------------
# Shared regex constants
# ---------------------------------------------------------------------------
# Why:   ``re.sub(r"\s+", " ", text)`` is used by the query-side span
#        normalisers below to collapse interior whitespace; compiling it
#        once at module load avoids per-call regex compilation.
_WHITESPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# BM25 tokenisation
# ---------------------------------------------------------------------------
# Why:    Naive ``text.lower().split()`` leaves punctuation glued to tokens —
#         a query token like "childers?" never matches the corpus token
#         "childers", silently losing the most discriminative query term
#         (the subject surname). We also drop a small set of English
#         function words so common tokens ("which", "was", "from", "the", ...)
#         don't dominate the BM25 score over proper-noun content terms.
# What:   Lowercase, match runs of [a-z0-9] (and one optional apostrophe-
#         suffixed segment for "don't"/"it's"), then filter the stopword set.
# Misses: ALL-CAPS acronyms tokenise OK; Unicode-letter words outside
#         [a-z0-9] (accented Latin, CJK) are skipped. Acceptable for the
#         English Wikipedia corpora the paper targets.
_BM25_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[''][a-z]+)?")
_BM25_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "in", "into", "is", "it", "its",
    "of", "on", "or", "she", "that", "the", "their", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "whom", "whose", "will", "with",
    "would", "they", "them", "you", "your", "we", "our", "i",
})


def _bm25_tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, drop English function words. Used for BOTH
    the BM25 corpus and the query so the token vocabularies line up."""
    toks = _BM25_TOKEN_RE.findall((text or "").lower())
    return [t for t in toks if t not in _BM25_STOPWORDS]


# ---------------------------------------------------------------------------
# Entity-name normalisation
# ---------------------------------------------------------------------------
# Why: text fingerprint length for chunk deduplication in RRFFusion. 80 chars
#      is long enough to discriminate near-duplicate chunks (collision rate
#      negligible for typical paragraph-sized texts) while staying short
#      enough to avoid hashing full chunk bodies on every fusion call.
_FP_LEN = 80

_QUERY_LABEL_MAP = GLINER_LABEL_MAP


def _normalize_query_entity(text: str, label: str) -> str:
    """
    Normalise query entity names using the shared normalize_entity_name() function
    from entity_extraction.py so ingestion-time and query-time normalisation are
    identical.
    """
    canonical_type = _QUERY_LABEL_MAP.get(label.lower(), label.upper())
    return normalize_entity_name(text, canonical_type)


# ---------------------------------------------------------------------------
# Query-side span-boundary normalisation (deterministic, query-only).
#
# GLiNER/SpaCy spans occasionally absorb a leading copular/auxiliary verb
# ("Are Granta"), an interior year ("National 1998 Maritime Museum"),
# or emit overlapping fragments of one hyphenated name. The helpers below
# repair the span *boundaries* before the entity becomes a graph anchor or
# filter token. They never touch the ingestion path — query-side only.
#
# Design constraints (publication-defensibility):
#  - Leading-function-word stripping fires only on CLOSED-CLASS auxiliary/
#    copula verbs (is/are/was/were/do/does/did) and correlative quantifiers
#    (both/either/neither). These never begin an English proper-noun entity
#    name, so the rule has no known false positives. Interrogative wh-words
#    are deliberately EXCLUDED because legitimate titles begin with them
#    ("Who Framed Roger Rabbit", "What Women Want"). Leading ARTICLES are
#    excluded here too — article handling is done type-aware by the shared
#    normalize_entity_name() (so 'The Who' is preserved, 'The Cold War' is
#    handled identically to ingestion).
#  - Interior-year stripping fires only when alphabetic tokens flank the
#    year — the linguistic signature of a temporal qualifier inserted into
#    an ORG/EVENT name. Leading/trailing years are preserved because they
#    are often part of a title ("2001: A Space Odyssey", "Live Aid 1985").
# ---------------------------------------------------------------------------
_LEADING_AUX_COPULA: frozenset = frozenset({
    "is", "are", "was", "were", "do", "does", "did",
})
# Correlative/distributive quantifiers that head comparison/intersection
# question stems ("Both X and Y …", "Either X or Y …", "Neither X nor Y …").
# These are question-level function words, never proper-noun-name components,
# so the NER tagger absorbing them into the first entity span ("Both Scientific
# American") is always an error. Deliberately EXCLUDES "all"/"each", which DO
# begin legitimate names ("All Saints", "Each Tear"), keeping the rule
# false-positive-free.
_LEADING_QUANTIFIER: frozenset = frozenset({"both", "either", "neither"})
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


def _strip_leading_function_word(text: str) -> str:
    """Drop a leading auxiliary/copula verb or correlative quantifier
    absorbed into the span by the NER tagger. Closed-class only (no wh-words,
    no articles, no all/each) — see module note. Collapses internal whitespace
    ('Are  Granta' → 'Granta'; 'Both Scientific American' → 'Scientific
    American')."""
    cleaned = _WHITESPACE_RE.sub(" ", text).strip()
    if not cleaned:
        return cleaned
    tokens = cleaned.split(" ")
    if len(tokens) <= 1:
        return cleaned
    first = tokens[0].lower()
    if first in _LEADING_AUX_COPULA or first in _LEADING_QUANTIFIER:
        tokens = tokens[1:]
    return " ".join(tokens).strip()


def _strip_embedded_year(text: str) -> Tuple[str, Optional[str]]:
    """If a standalone 4-digit year is INTERIOR to the span (alphabetic
    tokens on both sides), return (year_stripped_anchor, year). Otherwise
    (text, None). The year is returned separately so a caller can treat it as a
    temporal constraint. Leading/trailing years are preserved (often part of a
    title), and a span that is *only* a year is left intact."""
    m = _YEAR_RE.search(text)
    if not m:
        return text, None
    before = text[:m.start()].strip()
    after = text[m.end():].strip()
    # Require alphabetic content on BOTH sides → the year is an interior
    # qualifier, not a leading/trailing title component.
    if not (re.search(r"[A-Za-z]", before) and re.search(r"[A-Za-z]", after)):
        return text, None
    year = m.group(0)
    stripped = _WHITESPACE_RE.sub(" ", before + " " + after).strip()
    if len(stripped) < 3:
        return text, None
    return stripped, year


def _dedup_overlapping_spans(ents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop spans whose character range is contained within a longer span,
    keeping the maximal span (merges fragmented hyphenated names like
    'Hook'/'Handed Man' into the surviving 'Hook-Handed Man'). Spans without
    offsets are kept as-is. Original left-to-right order is restored."""
    kept: List[Dict[str, Any]] = []
    for e in sorted(ents, key=lambda x: (x.get("end", 0) - x.get("start", 0)), reverse=True):
        s, en = e.get("start"), e.get("end")
        if s is None or en is None:
            kept.append(e)
            continue
        contained = any(
            k.get("start") is not None
            and k.get("end") is not None
            and k["start"] <= s
            and en <= k["end"]
            for k in kept
        )
        if not contained:
            kept.append(e)
    kept.sort(key=lambda x: x.get("start", 0))
    return kept


# ---------------------------------------------------------------------------
# Module-level GLiNER cache — each model name loaded at most once per process
# ---------------------------------------------------------------------------
# Keyed by model_name so a caller requesting a different GLiNER variant gets
# that variant, not whichever model happened to load first. A failed load is
# cached as None under its name so the (slow) load is not retried per query.
_GLINER_MODEL_CACHE: dict = {}
_GLINER_CACHE_LOCK = threading.Lock()


def _get_gliner_model(model_name: str = "urchade/gliner_small-v2.1"):
    """
    Load GLiNER once per model name and cache it for the process lifetime.

    Uses double-checked locking so each model is loaded at most once even
    under concurrent first-call scenarios.

    Args:
        model_name: HuggingFace model identifier for GLiNER.

    Returns:
        Loaded GLiNER model, or None if loading fails.
    """
    if model_name not in _GLINER_MODEL_CACHE:
        with _GLINER_CACHE_LOCK:
            if model_name not in _GLINER_MODEL_CACHE:  # double-checked locking
                model = None
                try:
                    from gliner import GLiNER
                    model = GLiNER.from_pretrained(model_name)
                    logger.info("GLiNER model loaded and cached: %s", model_name)
                except (ImportError, OSError, RuntimeError) as e:
                    logger.warning(
                        "FALLBACK ACTIVE: GLiNER could not be loaded (%s)"
                        " -> SpaCy/Regex extraction will be used for query entities.",
                        e,
                    )
                except Exception as _net_err:  # noqa: BLE001
                    # transformers ≥ 4.45 calls huggingface_hub.model_info() in
                    # _patch_mistral_regex for every from_pretrained call, even
                    # when the model is already cached locally.  In offline /
                    # air-gapped environments this raises httpx.ConnectError or
                    # httpcore.ConnectError.  Retry with HF_HUB_OFFLINE=1 so
                    # the local HuggingFace cache is used without any network
                    # request.
                    logger.warning(
                        "GLiNER online load raised %s (%s); retrying offline.",
                        type(_net_err).__name__, _net_err,
                    )
                    _prev_hf = os.environ.get("HF_HUB_OFFLINE")
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    try:
                        from gliner import GLiNER
                        model = GLiNER.from_pretrained(model_name)
                        logger.info(
                            "GLiNER model loaded from local cache: %s", model_name
                        )
                    except Exception as e2:  # noqa: BLE001 -- HF Hub raises non-stdlib exception types
                        logger.warning(
                            "FALLBACK ACTIVE: GLiNER offline load also failed (%s)"
                            " -> SpaCy/Regex extraction will be used.",
                            e2,
                        )
                    finally:
                        if _prev_hf is None:
                            os.environ.pop("HF_HUB_OFFLINE", None)
                        else:
                            os.environ["HF_HUB_OFFLINE"] = _prev_hf
                _GLINER_MODEL_CACHE[model_name] = model
    return _GLINER_MODEL_CACHE[model_name]


# ============================================================================
# CONFIGURATION
# ============================================================================

class RetrievalMode(str, Enum):
    """Retrieval modes for ablation studies."""
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


@dataclass
class RetrievalConfig:
    """Configuration for the Hybrid Retriever."""

    # Retrieval mode
    mode: RetrievalMode = RetrievalMode.HYBRID

    # Vector retrieval
    vector_top_k: int = 10

    # Graph retrieval
    graph_top_k: int = 10
    max_hops: int = 2

    # RRF parameters
    rrf_k: int = 60  # standard RRF constant (Cormack et al. 2009)

    # Fusion
    final_top_k: int = 10
    cross_source_boost: float = 1.2  # extra RRF credit for chunks in both paths

    # Similarity threshold (applied before fusion)
    similarity_threshold: float = 0.3

    # SpaCy model used for query entity extraction fallback
    spacy_model: str = "en_core_web_sm"

    # Query-time NER settings (sourced from settings.yaml entity_extraction.gliner)
    query_ner_confidence: float = 0.15
    query_entity_types: Optional[List[str]] = None  # None -> use ExtractionConfig default

    # GLiNER model name (sourced from settings.yaml entity_extraction.gliner.model_name)
    gliner_model_name: str = "urchade/gliner_small-v2.1"

    # BM25 sparse retrieval (third RRF lane)
    enable_bm25: bool = True    # settings.yaml: rag.enable_bm25
    bm25_top_k: int = 10        # settings.yaml: rag.bm25_top_k

    # Per-source RRF weights. Each source's RRF contribution is multiplied
    # by its weight, so a chunk at rank r from source S contributes
    # ``weight_S / (k + r)`` instead of vanilla ``1 / (k + r)``. Default
    # 1.0/1.0/1.0 reproduces equal-weight rank-only fusion (no regression).
    # Tune per ablation via settings.yaml
    # ``rag.vector_weight / rag.graph_weight / rag.bm25_weight``.
    #
    # Refs:
    #   - Per-source weighting on top of RRF: Bruch et al. (2023), "An
    #     Analysis of Fusion Functions for Hybrid Retrieval", ACM TOIS 41(4)
    #     — weighted-rank fusion outperforms unweighted RRF when sources
    #     have different precision/recall profiles.
    #   - The RRF constant k=60 default follows Cormack, Clarke & Buettcher
    #     (2009), SIGIR.
    vector_weight: float = 1.0
    graph_weight: float = 1.0
    bm25_weight: float = 1.0

    # Hop-3 graph traversal (2-bridge chains). OFF by default — adds latency
    # (~200-1000ms) and is only useful for questions where the answer is 2
    # relations away. Opt in via settings.yaml ``graph.enable_hop3: true``
    # when running multi-hop-heavy ablations.
    enable_hop3: bool = False


@dataclass
class RetrievalResult:
    """A single retrieval result returned by HybridRetriever."""
    chunk_id: str
    text: str
    source_doc: str
    position: int

    # Scores
    rrf_score: float = 0.0
    vector_score: Optional[float] = None
    vector_rank: Optional[int] = None
    graph_score: Optional[float] = None
    graph_rank: Optional[int] = None

    # Metadata
    retrieval_method: str = "hybrid"  # "vector", "graph", "bm25", or "hybrid"
    hop_distance: Optional[int] = None
    matched_entities: List[str] = field(default_factory=list)

    # BM25 scores (populated when enable_bm25=True)
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to plain dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_doc": self.source_doc,
            "position": self.position,
            "rrf_score": self.rrf_score,
            "vector_score": self.vector_score,
            "vector_rank": self.vector_rank,
            "graph_score": self.graph_score,
            "graph_rank": self.graph_rank,
            "retrieval_method": self.retrieval_method,
            "hop_distance": self.hop_distance,
            "matched_entities": self.matched_entities,
            "bm25_score": self.bm25_score,
            "bm25_rank": self.bm25_rank,
        }


@dataclass
class RetrievalMetrics:
    """Performance metrics for a single retrieval call."""
    total_time_ms: float
    vector_time_ms: float
    graph_time_ms: float
    fusion_time_ms: float
    vector_results: int
    graph_results: int
    final_results: int
    query_entities: List[str]


# ============================================================================
# RRF FUSION
# ============================================================================

class RRFFusion:
    """
    Reciprocal Rank Fusion of vector and graph result lists.

    Reference:
        Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
        Reciprocal Rank Fusion outperforms Condorcet and individual rank
        learning methods. SIGIR '09, pp. 758-759.
        https://doi.org/10.1145/1571941.1572114
    """

    def __init__(
        self,
        k: int = 60,
        cross_source_boost: float = 1.2,
        vector_weight: float = 1.0,
        graph_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        """
        Args:
            k: RRF constant (default 60, empirically optimal).
            cross_source_boost: Additional RRF credit for chunks found in
                both vector and graph paths (interpreted as
                cross_source_boost / (k + 1) additive bonus).
            vector_weight / graph_weight / bm25_weight: per-source weights
                applied to each path's RRF contribution. Default 1.0/1.0/1.0
                reproduces vanilla RRF. Set unequal weights to bias the
                fusion toward one source for ablation studies.
        """
        self.k = k
        self.cross_source_boost = cross_source_boost
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.bm25_weight = bm25_weight

    def fuse(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        final_top_k: int = 10,
        bm25_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[RetrievalResult]:
        """
        Fuse vector and graph result lists using RRF with additive cross-source boost.

        Formula:
            RRF(d) = Σ  1 / (k + rank_i(d))  +  BONUS

        where BONUS = cross_source_boost / (k + 1) when d appears in both lists.

        Expected key names (from storage layer):
            vector_results items: "document_id", "text", "similarity",
                                  "metadata" -> {"source_file": ...}, "position"
            graph_results items:  "chunk_id", "text", "hops",
                                  "source_file", "matched_entity", "position"

        Reference:
            Cormack, Clarke & Buettcher (2009). SIGIR '09, pp. 758-759.
            https://doi.org/10.1145/1571941.1572114

        Args:
            vector_results: Ranked list from LanceDB vector search.
            graph_results:  Ranked list from KuzuDB entity-based graph search.
            final_top_k:    Maximum number of results to return.

        Returns:
            Sorted list of RetrievalResult objects (highest RRF score first).
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_data: Dict[str, Dict[str, Any]] = {}
        vector_ranks: Dict[str, int] = {}
        graph_ranks: Dict[str, int] = {}
        vector_scores: Dict[str, float] = {}
        graph_scores: Dict[str, float] = {}
        graph_metadata: Dict[str, Dict[str, Any]] = {}

        # ------------------------------------------------------------------
        # Deduplication: identical text fingerprints get one RRF slot only.
        # Problem: Ingestion duplicates or overlapping chunks can accumulate
        # disproportionate RRF credit if the same content appears multiple
        # times.  Fix: deduplicate on the first 80 characters of text before
        # rank assignment; only the earliest (best-ranked) copy is kept.
        # ------------------------------------------------------------------
        seen_fps: set[str] = set()
        deduped_vector: List[Dict[str, Any]] = []
        for r in vector_results:
            fp = r.get("text", "")[:_FP_LEN]
            if fp not in seen_fps:
                seen_fps.add(fp)
                deduped_vector.append(r)
        vector_results = deduped_vector

        # Vector ranks
        # VectorStoreAdapter.vector_search() returns dicts with keys:
        #   "document_id", "text", "similarity", "metadata" -> {"source_file": ...}
        # vector_weight scales this source's RRF contribution (default 1.0).
        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result.get("document_id", "")
            rrf_scores[chunk_id] += self.vector_weight * (1.0 / (self.k + rank))
            vector_ranks[chunk_id] = rank
            vector_scores[chunk_id] = result.get("similarity", 0.0)

            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result.get("text", ""),
                    "source_doc": result.get("metadata", {}).get("source_file", "unknown"),
                    "position": result.get("position", 0),
                }

        # Graph ranks
        # HybridStore.graph_search() returns dicts with keys:
        #   "chunk_id", "text", "hops", "source_file", "matched_entity",
        #   "triple_confidence", "position"
        # graph_weight scales this source's RRF contribution (default 1.0).
        for rank, result in enumerate(graph_results, start=1):
            chunk_id = result.get("chunk_id", "")
            rrf_scores[chunk_id] += self.graph_weight * (1.0 / (self.k + rank))
            graph_ranks[chunk_id] = rank
            hops = result.get("hops", 1)
            # Combine hop-distance with the triple-frequency confidence
            # from storage. Hop-0 (direct
            # mention) → 1.0 confidence, Hop-2 multiplies by the bridge
            # triple's corpus-support strength. This replaces the
            # pure-hop proxy that ignored REBEL's (constant) confidence.
            triple_conf = result.get("triple_confidence", 1.0)
            graph_scores[chunk_id] = (1.0 / (hops + 1)) * triple_conf

            graph_metadata[chunk_id] = {
                "hop_distance": hops,
                "matched_entities": [result.get("matched_entity", "")],
                "triple_confidence": triple_conf,
            }

            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result.get("text", ""),
                    "source_doc": result.get("source_file", "unknown"),
                    "position": result.get("position", 0),
                }

        # BM25 sparse ranks — uses same document_id key as vector results
        bm25_ranks: Dict[str, int] = {}
        bm25_scores_map: Dict[str, float] = {}

        # bm25_weight scales this source's RRF contribution (default 1.0).
        if bm25_results:
            for rank, result in enumerate(bm25_results, start=1):
                chunk_id = result.get("document_id", "")
                if not chunk_id:
                    continue
                rrf_scores[chunk_id] += self.bm25_weight * (1.0 / (self.k + rank))
                bm25_ranks[chunk_id] = rank
                bm25_scores_map[chunk_id] = result.get("similarity", 0.0)

                if chunk_id not in chunk_data:
                    chunk_data[chunk_id] = {
                        "chunk_id": chunk_id,
                        "text": result.get("text", ""),
                        "source_doc": result.get("metadata", {}).get("source_file", "unknown"),
                        "position": result.get("position", 0),
                    }

        # Cross-source boost: any chunk in 2+ retrieval lists gets a bonus.
        # Extends the original vector+graph pair to include BM25.
        all_sources = [
            set(vector_ranks.keys()),
            set(graph_ranks.keys()),
        ]
        if bm25_results:
            all_sources.append(set(bm25_ranks.keys()))

        boosted: set = set()
        for i, src_a in enumerate(all_sources):
            for src_b in all_sources[i + 1:]:
                for chunk_id in src_a & src_b:
                    if chunk_id not in boosted:
                        bonus = self.cross_source_boost / (self.k + 1)
                        rrf_scores[chunk_id] += bonus
                        boosted.add(chunk_id)

        # Sort by RRF score descending
        sorted_chunks = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:final_top_k]

        # Build RetrievalResult objects
        results: List[RetrievalResult] = []
        for chunk_id, rrf_score in sorted_chunks:
            data = chunk_data.get(chunk_id, {})
            gm = graph_metadata.get(chunk_id, {})

            in_vector = chunk_id in vector_ranks
            in_graph = chunk_id in graph_ranks
            in_bm25 = chunk_id in bm25_ranks

            sources = sum([in_vector, in_graph, in_bm25])
            if sources >= 2:
                method = "hybrid"
            elif in_vector:
                method = "vector"
            elif in_bm25:
                method = "bm25"
            else:
                method = "graph"

            results.append(RetrievalResult(
                chunk_id=chunk_id,
                text=data.get("text", ""),
                source_doc=data.get("source_doc", "unknown"),
                position=data.get("position", 0),
                rrf_score=rrf_score,
                vector_score=vector_scores.get(chunk_id),
                vector_rank=vector_ranks.get(chunk_id),
                graph_score=graph_scores.get(chunk_id),
                graph_rank=graph_ranks.get(chunk_id),
                retrieval_method=method,
                hop_distance=gm.get("hop_distance"),
                matched_entities=[e for e in gm.get("matched_entities", []) if e],
                bm25_score=bm25_scores_map.get(chunk_id),
                bm25_rank=bm25_ranks.get(chunk_id),
            ))

        return results


# ============================================================================
# IMPROVED QUERY ENTITY EXTRACTOR
# ============================================================================

class ImprovedQueryEntityExtractor:
    """
    Query entity extraction with GLiNER consistency.

    Uses the same GLiNER model as chunk-level entity extraction so that
    query entities and graph entities share the same label space, improving
    graph lookup hit rates.

    Preference order:
        1. GLiNER (preferred — consistent with ingestion-time extraction)
        2. SpaCy NER (fallback when GLiNER unavailable)
        3. Regex (last-resort fallback)
    """

    # -- Junk-entity filter -------------------------------------------------
    # GLiNER over-extracts on question text and produces non-entities like
    # "what year", "third year", "this person" that pollute the graph
    # lookup and Verifier entity-path validation.  Combined with the
    # ingestion-time DEFAULT_STOPLIST so that query-side filtering and
    # graph-side cleanup agree on what counts as an entity.
    _QUERY_JUNK_REGEXES: Tuple[re.Pattern, ...] = (
        re.compile(r"^(what|which|who|whose|whom|how|where|when|why)\b", re.IGNORECASE),
        re.compile(r"\b(year|years|month|months|day|days|date|dates|time|times)\b\s*$", re.IGNORECASE),
        re.compile(r"^(first|second|third|fourth|fifth|last|next|previous)\s+(year|month|day|time|one|kind|sort|type)$", re.IGNORECASE),
        re.compile(r"^(this|that|these|those|some|any|many|few|several)\s+\w+$", re.IGNORECASE),
        re.compile(r"^(how\s+many|how\s+much)\b", re.IGNORECASE),
        re.compile(r"^(is|was|were|are|do|does|did|has|have|had|can|could|should|would|may|might|will)\b", re.IGNORECASE),
    )
    # Single-token junk that survives canonical_form but is not stoplisted
    # (mostly question stems and generic determiners).
    _QUERY_JUNK_TOKENS: frozenset = frozenset({
        "year", "years", "month", "months", "day", "days", "date", "dates",
        "time", "times", "person", "people", "thing", "things", "place", "places",
        "name", "names", "kind", "kinds", "sort", "sorts", "type", "types",
        "one", "ones", "way", "ways", "case", "cases",
        "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
        "the", "a", "an",
    })

    # Measure / temporal nouns. A span composed only of digits, these
    # nouns, and generic quantity adjectives (with NO capitalised proper-noun
    # token) is a statistical descriptor, not a graph-anchorable entity
    # (e.g. "7 consecutive seasons", "25 laps", "1993"). It matches no graph
    # node and, as a filter token, retains noise — so it must be gated out.
    _MEASURE_NOUNS: frozenset = frozenset({
        "season", "seasons", "year", "years", "game", "games", "time", "times",
        "match", "matches", "point", "points", "goal", "goals", "run", "runs",
        "win", "wins", "loss", "losses", "day", "days", "week", "weeks",
        "month", "months", "lap", "laps", "metre", "metres", "meter", "meters",
        "mile", "miles", "kilometre", "kilometres", "inning", "innings",
        "round", "rounds", "set", "sets", "place", "places", "title", "titles",
    })
    _GENERIC_QUANT_ADJ: frozenset = frozenset({
        "consecutive", "total", "straight", "combined", "overall",
        "first", "second", "third", "last", "only", "more", "fewer",
        "most", "least", "career", "single", "double", "triple",
    })

    @classmethod
    def _is_temporal_measure_phrase(cls, text: str) -> bool:
        """True if `text` is a pure digit/measure/quantity phrase with no
        proper-noun token. Capitalisation is checked on the *original* tokens
        (proper nouns survive); membership is checked on canonical tokens."""
        toks = text.split()
        if not toks:
            return False
        # Any capitalised token (other than a leading determiner already handled
        # by the query-side span-boundary normaliser) signals a proper noun —
        # not a pure measure phrase.
        if any(t[:1].isupper() for t in toks):
            return False
        canon_toks = canonical_form(text).split()
        if not canon_toks:
            return False
        return all(
            t.isdigit() or t in cls._MEASURE_NOUNS or t in cls._GENERIC_QUANT_ADJ
            for t in canon_toks
        )

    @classmethod
    def _is_junk_entity(cls, text: str) -> bool:
        """Return True if `text` is a question phrase or stoplisted token.

        Combines four checks:
        1. Length / canonical_form normalisation (drops empty + single chars).
        2. canonical_form ∈ DEFAULT_STOPLIST (pronouns, demonyms — shared
           with ingestion-time graph cleanup, so query side and graph side
           agree).
        3. canonical_form is a generic question stem (e.g. "what year").
        4. Pure temporal/measure phrase with no proper-noun token
           (e.g. "7 consecutive seasons", "25 laps").
        """
        if not text or len(text.strip()) < 2:
            return True
        canon = canonical_form(text)
        if not canon or len(canon) < 2:
            return True
        if canon in DEFAULT_STOPLIST:
            return True
        if canon in cls._QUERY_JUNK_TOKENS:
            return True
        for rx in cls._QUERY_JUNK_REGEXES:
            if rx.search(canon):
                return True
        if cls._is_temporal_measure_phrase(text):
            return True
        return False

    @classmethod
    def _filter_entities(cls, entities: List[str]) -> List[str]:
        """De-duplicate and drop junk entities while preserving input order."""
        seen: set = set()
        out: List[str] = []
        for e in entities:
            if cls._is_junk_entity(e):
                logger.debug("Dropping junk query entity: %r", e)
                continue
            key = e.lower().strip()
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
        return out

    def __init__(
        self,
        gliner_model: Optional[Any] = None,
        spacy_model: str = "en_core_web_sm",
        entity_types: Optional[List[str]] = None,
        confidence_threshold: float = 0.15,
        gliner_model_name: str = "urchade/gliner_small-v2.1",
    ) -> None:
        """
        Args:
            gliner_model: Pre-loaded GLiNER model.  If None, the module-level
                cache is used (loading on demand).
            spacy_model: SpaCy model name for the fallback extractor.
            entity_types: List of GLiNER entity type labels.  Defaults to the
                standard paper set sourced from settings.yaml.
            confidence_threshold: Minimum GLiNER confidence for an entity to
                be accepted.
            gliner_model_name: HuggingFace model ID passed to _get_gliner_model()
                when no pre-loaded model is supplied.
        """
        self.gliner = gliner_model
        self.nlp = None
        self.confidence_threshold = confidence_threshold
        self._gliner_model_name = gliner_model_name
        self._load_spacy(spacy_model)

        if self.gliner is None:
            self._load_gliner()

        # Entity types come from settings.yaml (via RetrievalConfig.query_entity_types)
        self.entity_types = entity_types or [
            "person", "organization", "city", "country",
            "state", "location", "film", "movie", "album",
            "work of art", "landmark", "event", "award",
        ]

    def _load_gliner(self) -> None:
        """Use the module-level cached GLiNER — loaded at most once per process."""
        self.gliner = _get_gliner_model(self._gliner_model_name)

    def _load_spacy(self, model_name: str) -> None:
        """Load SpaCy as fallback NER backend."""
        try:
            import spacy
            self.nlp = spacy.load(model_name)
            logger.info("SpaCy loaded for query analysis: %s", model_name)
        except (ImportError, OSError) as e:
            logger.warning("SpaCy not available: %s", e)
            self.nlp = None

    def extract(self, query: str, confidence_threshold: Optional[float] = None) -> List[str]:
        """
        Extract named entities from a query string.

        Args:
            query: User query text.
            confidence_threshold: Override for the instance-level threshold
                (None = use self.confidence_threshold).

        Returns:
            List of normalised entity name strings.
        """
        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold

        # Temporal constraints stripped from entity spans during this call
        # (e.g. the "1998" in "National 1998 Maritime Museum"). Recorded
        # for any downstream consumer (Planner constraint hops); inert until
        # consumed.
        self._last_temporal_constraints: List[str] = []

        # Method 1: GLiNER (preferred for ingestion-query consistency)
        if self.gliner is not None:
            try:
                entities = self.gliner.predict_entities(
                    query,
                    self.entity_types,
                    threshold=threshold,
                )
                # B1b: merge/drop overlapping fragments of one span (offsets present).
                entities = _dedup_overlapping_spans(entities)
                raw = [
                    _normalize_query_entity(self._b1_string_norm(ent["text"]), ent["label"])
                    for ent in entities
                ]
                return self._filter_entities(raw)
            except (RuntimeError, ValueError) as e:
                logger.warning("GLiNER query extraction failed: %s", e)

        # Method 2: SpaCy NER (fallback — GLiNER unavailable or failed)
        if self.nlp is not None:
            logger.warning(
                "FALLBACK ACTIVE: GLiNER not available -> SpaCy extraction for query entities."
            )
            return self._filter_entities(
                [self._b1_string_norm(t) for t in self._spacy_extract(query)]
            )

        # Method 3: Regex (last resort — neither GLiNER nor SpaCy available)
        logger.warning(
            "FALLBACK ACTIVE: Neither GLiNER nor SpaCy available -> regex extraction."
            " Graph retrieval will be severely limited!"
        )
        return self._filter_entities(
            [self._b1_string_norm(t) for t in self._fallback_extract(query)]
        )

    def _b1_string_norm(self, text: str) -> str:
        """B1a + B1c applied to a single entity string (all extraction paths):
        strip a leading function word, then split off an embedded year (recorded
        as a temporal constraint). Boundary repair only — no gating here."""
        cleaned = _strip_leading_function_word(text)
        cleaned, year = _strip_embedded_year(cleaned)
        if year is not None:
            self._last_temporal_constraints.append(year)
        return cleaned

    # SpaCy label → canonical type; GPE→GPE matches entity_extraction.py ingestion path
    # so that entity IDs are consistent across query-time and ingestion-time extraction.
    _SPACY_TYPE_MAP: Dict[str, str] = SPACY_LABEL_MAP

    def _spacy_extract(self, query: str) -> List[str]:
        """SpaCy-based extraction with GLiNER-compatible type mapping."""
        doc = self.nlp(query)
        entities: List[str] = []

        for ent in doc.ents:
            if ent.label_ in self._SPACY_TYPE_MAP and len(ent.text) > 2:
                entities.append(ent.text)

        # Also capture proper nouns not covered by NER
        for token in doc:
            if token.pos_ == "PROPN" and token.text not in entities:
                if len(token.text) > 2:
                    entities.append(token.text)

        return entities

    def _fallback_extract(self, query: str) -> List[str]:
        """Regex-based last-resort extraction."""
        entities: List[str] = []

        # Capitalised words/phrases (skip sentence-initial question words)
        for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query):
            entity = match.group(1)
            if len(entity) > 2:
                entities.append(entity)

        # Quoted strings
        for match in re.finditer(r'"([^"]+)"', query):
            entities.append(match.group(1))

        return entities


# ============================================================================
# HYBRID RETRIEVER
# ============================================================================

class HybridRetriever:
    """
    Hybrid Retriever with RRF fusion.

    Orchestrates:
        1. Vector retrieval (LanceDB, ANN)
        2. Graph retrieval (KuzuDB, entity-based)
        3. RRF fusion

    Performance targets (paper section 2.6):
        - Vector retrieval: 20-40 ms
        - Graph retrieval:  10-30 ms
        - Total:            < 100 ms
    """

    def __init__(
        self,
        hybrid_store: Any,
        embeddings: Any,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        """
        Args:
            hybrid_store: HybridStore instance (LanceDB + KuzuDB).
            embeddings: Embedding model (e.g. BatchedOllamaEmbeddings).
            config: RetrievalConfig.  Defaults to RetrievalConfig() if None.
        """
        self.store = hybrid_store
        self.embeddings = embeddings
        self.config = config or RetrievalConfig()

        self.rrf_fusion = RRFFusion(
            k=self.config.rrf_k,
            cross_source_boost=self.config.cross_source_boost,
            vector_weight=self.config.vector_weight,
            graph_weight=self.config.graph_weight,
            bm25_weight=self.config.bm25_weight,
        )

        # Re-use GLiNER model from HybridStore's entity pipeline when available.
        # SpacyEntityPipeline has no ner_extractor — guard with AttributeError.
        gliner_model = None
        if hasattr(hybrid_store, "entity_pipeline") and hybrid_store.entity_pipeline is not None:
            try:
                gliner_model = hybrid_store.entity_pipeline.ner_extractor.model
            except AttributeError:
                pass   # SpacyEntityPipeline or other pipeline without GLiNER

        self.entity_extractor = ImprovedQueryEntityExtractor(
            gliner_model=gliner_model,
            spacy_model=self.config.spacy_model,
            entity_types=self.config.query_entity_types,
            confidence_threshold=self.config.query_ner_confidence,
            gliner_model_name=self.config.gliner_model_name,
        )
        logger.info("HybridRetriever initialised: mode=%s", self.config.mode)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        entity_hints: Optional[List[str]] = None,
    ) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """
        Execute hybrid retrieval for the given query.

        Args:
            query: User query string.
            top_k: Number of results to return (overrides config.final_top_k).
            entity_hints: Optional list of entity strings extracted upstream by
                the Planner (S_P).  When provided, these are used directly for
                graph search instead of re-running GLiNER on the (usually short)
                sub-query string.  GLiNER often fails on isolated 3–5 word
                sub-queries, so reusing the full-query entities from S_P
                significantly improves graph recall.

        Returns:
            Tuple of (results, metrics).
        """
        start_time = time.time()
        top_k = top_k if top_k is not None else self.config.final_top_k

        # 1. Query entity extraction (~3-5 ms)
        # Use caller-supplied hints when available; they come from Planner's
        # full-query analysis and are more reliable than re-running GLiNER on
        # a short sub-query fragment.
        # Normalise with the same article-stripping logic used at ingestion so
        # "The Cold War" → "Cold War" and graph lookups succeed.
        if entity_hints:
            query_entities = [
                _normalize_query_entity(e, "LOCATION") if e.startswith(("The ", "A ", "An ")) else e
                for e in entity_hints
            ]
            logger.debug("Using entity hints from S_P (normalised): %s", query_entities)
        else:
            query_entities = self.entity_extractor.extract(query)
            logger.debug("Query entities (GLiNER): %s", query_entities)

        # 2. Vector retrieval
        vector_start = time.time()
        vector_results: List[Dict[str, Any]] = []

        # Gate on vector_weight in addition to mode so a pure-sparse ("BM25
        # only") ablation lane is expressible: mode=VECTOR with vector_weight=0
        # runs BM25 (enabled below) but skips the dense ANN search entirely.
        # Default vector_weight=1.0 leaves the production path unchanged — this
        # only suppresses the dense lane when a caller explicitly zeroes its
        # weight (modality_ablation's bm25_only config).
        if (self.config.mode in [RetrievalMode.VECTOR, RetrievalMode.HYBRID]
                and self.config.vector_weight > 0):
            try:
                query_embedding = self._embed_query(query)
                vector_results = self.store.vector_search(
                    query_embedding,
                    top_k=self.config.vector_top_k,
                    threshold=self.config.similarity_threshold,
                )
            except (OSError, RuntimeError, ValueError, ConnectionError, AttributeError) as e:
                logger.error("Vector retrieval failed: %s", e)
                vector_results = []

        vector_time = (time.time() - vector_start) * 1000

        # 2b. BM25 sparse retrieval (third RRF lane)
        bm25_results: List[Dict[str, Any]] = []
        if self.config.enable_bm25 and self.config.mode in [RetrievalMode.VECTOR, RetrievalMode.HYBRID]:
            try:
                bm25_results = self._bm25_search(query, self.config.bm25_top_k)
                logger.debug("BM25: %d results", len(bm25_results))
            except (RuntimeError, ValueError, AttributeError, IndexError, ImportError) as _bm25_err:
                logger.debug("BM25 search failed: %s", _bm25_err)

        # 3. Graph retrieval
        graph_start = time.time()
        graph_results: List[Dict[str, Any]] = []

        if self.config.mode in [RetrievalMode.GRAPH, RetrievalMode.HYBRID]:
            if query_entities:
                try:
                    graph_results = self.store.graph_search(
                        entities=query_entities,
                        max_hops=self.config.max_hops,
                        top_k=self.config.graph_top_k,
                        enable_hop3=self.config.enable_hop3,
                    )
                except (ValueError, RuntimeError, OSError) as e:
                    logger.warning("Graph retrieval failed: %s", e)
                    graph_results = []

            # The previous _keyword_entity_search() substring scan with dual
            # injection has been removed. BM25 strictly subsumes the
            # substring match: any chunk
            # that contained the entity name verbatim is now surfaced by the
            # BM25 path instead, with a principled term-frequency score rather
            # than a synthetic 0.76/0.95 similarity.

        graph_time = (time.time() - graph_start) * 1000

        # 4. Fusion
        fusion_start = time.time()

        if self.config.mode == RetrievalMode.VECTOR:
            # VECTOR mode = the non-graph lanes. When BM25 is enabled it is a
            # genuine second (sparse) lane, so it must be fused with the dense
            # vector lane via RRF — NOT discarded. Passing an empty graph list
            # keeps the graph lane out. When BM25 is disabled (bm25_results
            # empty) this falls back to the pure dense-vector path, byte-
            # identical to the previous behaviour.
            if bm25_results:
                results = self.rrf_fusion.fuse(
                    vector_results,
                    [],
                    final_top_k=top_k,
                    bm25_results=bm25_results,
                )
            else:
                results = self._vector_only_results(vector_results, top_k)
        elif self.config.mode == RetrievalMode.GRAPH:
            results = self._graph_only_results(graph_results, top_k)
        else:
            results = self.rrf_fusion.fuse(
                vector_results,
                graph_results,
                final_top_k=top_k,
                bm25_results=bm25_results or None,
            )

        # Tripwire: if BM25 produced candidates but none survived into the
        # fused output (in a mode where BM25 should participate), the lane is
        # being silently dropped — exactly the regression this branch existed
        # to fix. Loud, non-fatal: it can legitimately fire when every BM25
        # chunk ranks below the cut, but a persistent fire signals a wiring bug.
        if (
            self.config.enable_bm25
            and bm25_results
            and self.config.mode in (RetrievalMode.VECTOR, RetrievalMode.HYBRID)
            and not any(getattr(r, "bm25_rank", None) is not None for r in results)
        ):
            logger.warning(
                "[BM25] %d BM25 candidates produced but none reached the fused "
                "top-%d — the sparse lane may be silently dropped; check the "
                "fusion path.",
                len(bm25_results), top_k,
            )

        fusion_time = (time.time() - fusion_start) * 1000
        total_time = (time.time() - start_time) * 1000

        metrics = RetrievalMetrics(
            total_time_ms=total_time,
            vector_time_ms=vector_time,
            graph_time_ms=graph_time,
            fusion_time_ms=fusion_time,
            vector_results=len(vector_results),
            graph_results=len(graph_results),
            final_results=len(results),
            query_entities=query_entities,
        )

        logger.info(
            "Retrieval complete: %d results, %.1f ms total"
            " (vector: %.1f ms, graph: %.1f ms)",
            len(results), total_time, vector_time, graph_time,
        )

        return results, metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_bm25_index(self) -> None:
        """Build BM25 index lazily from LanceDB DataFrame (shared with _keyword_df_cache)."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed — BM25 disabled. Run: pip install rank_bm25")
            self._bm25_index = None
            return

        vector_store = getattr(self.store, "vector_store", None)
        if vector_store is None or getattr(vector_store, "table", None) is None:
            self._bm25_index = None
            return

        try:
            if not hasattr(self, "_keyword_df_cache"):
                self._keyword_df_cache = vector_store.table.to_pandas()
            df = self._keyword_df_cache
            corpus = [_bm25_tokenize(str(t)) for t in df["text"].tolist()]
            self._bm25_corpus_rows = df.to_dict("records")
            # Why:    rank_bm25.BM25Okapi divides by corpus_size and by the
            #         average document length at construction. An empty corpus
            #         — or one whose documents all tokenise to nothing after
            #         stopword removal — yields avgdl=0 and a ZeroDivisionError
            #         (hit by a freshly-created or empty LanceDB table, e.g.
            #         integration tests that build storage but skip ingestion).
            # What:   skip the BM25 lane entirely when there is no usable corpus.
            # Misses: nothing — an empty corpus carries no sparse signal; the
            #         dense and graph lanes still run.
            if not any(corpus):
                logger.info("BM25 index skipped: empty corpus (%d rows)", len(corpus))
                self._bm25_index = None
                return
            self._bm25_index = BM25Okapi(corpus)
            logger.info("BM25 index built: %d chunks", len(corpus))
        except (RuntimeError, ValueError, AttributeError, IndexError,
                ImportError, ZeroDivisionError) as e:
            logger.warning("BM25 index build failed: %s", e)
            self._bm25_index = None

    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        BM25 sparse keyword search.

        Returns top-k chunks in vector-result dict format (document_id, text,
        similarity, metadata, position) so they can be passed directly into
        RRFFusion.fuse(bm25_results=...).  similarity is normalised to [0, 1].
        """
        if not hasattr(self, "_bm25_index"):
            self._build_bm25_index()

        if not self._bm25_index:
            return []

        tokens = _bm25_tokenize(query)
        if not tokens:
            return []
        raw_scores = self._bm25_index.get_scores(tokens)

        max_score = float(raw_scores.max()) if raw_scores.size else 0.0
        if max_score <= 0:
            return []

        top_indices = raw_scores.argsort()[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        import json as _json
        for idx in top_indices:
            score = float(raw_scores[idx])
            if score <= 0:
                break
            row = self._bm25_corpus_rows[idx]
            try:
                meta = _json.loads(row.get("metadata", "{}"))
            except (_json.JSONDecodeError, TypeError, ValueError):
                meta = {}
            results.append({
                "document_id": str(row.get("document_id", "")),
                "text": str(row.get("text", "")),
                "similarity": score / max_score,
                "metadata": {"source_file": row.get("source_file", meta.get("source_file", "unknown"))},
                "position": int(row.get("position", 0)),
            })

        return results

    def _embed_query(self, query: str) -> np.ndarray:
        """Generate a query embedding vector."""
        if hasattr(self.embeddings, "embed_query"):
            embedding = self.embeddings.embed_query(query)
        else:
            embedding = self.embeddings.embed_documents([query])[0]
        return np.array(embedding, dtype=np.float32)

    # _keyword_entity_search() has been removed. BM25 provides the same
    # entity-name surfacing capability with a principled scoring function,
    # and the own-doc-boost behaviour is replicated by BM25 because the
    # entity name typically appears in the source_file path AND in the
    # article text — both contribute to the BM25 score.

    def _vector_only_results(
        self,
        vector_results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        Convert raw vector search dicts to RetrievalResult list (VECTOR mode).

        Expected keys from VectorStoreAdapter:
            "document_id", "text", "similarity",
            "metadata" -> {"source_file": ...}, "position"
        """
        results: List[RetrievalResult] = []
        for rank, r in enumerate(vector_results[:top_k], start=1):
            results.append(RetrievalResult(
                chunk_id=r.get("document_id", ""),
                text=r.get("text", ""),
                source_doc=r.get("metadata", {}).get("source_file", "unknown"),
                position=r.get("position", 0),
                rrf_score=r.get("similarity", 0.0),
                vector_score=r.get("similarity"),
                vector_rank=rank,
                retrieval_method="vector",
            ))
        return results

    def _graph_only_results(
        self,
        graph_results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        Convert raw graph search dicts to RetrievalResult list (GRAPH mode).

        Expected keys from HybridStore.graph_search():
            "chunk_id", "text", "hops", "source_file",
            "matched_entity", "triple_confidence", "position"
        """
        results: List[RetrievalResult] = []
        for rank, r in enumerate(graph_results[:top_k], start=1):
            # graph_search emits "triple_confidence" (corpus-support score) —
            # not "confidence" — so read that key; the neutral 0.5 remains the
            # fallback for legacy dicts. Floor at 0.0 so the -1.0 error
            # sentinel is never surfaced as a score.
            conf = max(0.0, r.get("triple_confidence", r.get("confidence", 0.5)))
            results.append(RetrievalResult(
                chunk_id=r.get("chunk_id", ""),
                text=r.get("text", ""),
                source_doc=r.get("source_file", "unknown"),
                position=r.get("position", 0),
                rrf_score=conf,
                graph_score=conf,
                graph_rank=rank,
                retrieval_method="graph",
                hop_distance=r.get("hops", 1),
                matched_entities=[r.get("matched_entity", "")],
            ))
        return results


# ============================================================================
# Pre-generative filtering does NOT live in this module. The Navigator (S_N)
# is the single owner of the pre-generative filter chain; see
# src/logic_layer/navigator.py. HybridRetriever returns raw fused candidates
# and lets the Navigator apply downstream relevance / redundancy / entity-
# mention / context-shrinkage filters.
# ============================================================================


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_hybrid_retriever(
    hybrid_store: Any,
    embeddings: Any,
    cfg: Optional[Dict[str, Any]] = None,
) -> "HybridRetriever":
    """
    Factory for HybridRetriever. Reads all parameters from a settings dict
    (typically loaded from config/settings.yaml).

    Args:
        hybrid_store: Initialised HybridStore instance.
        embeddings: Embedding model.
        cfg: Full settings dictionary. Settings keys read (all 17 verified
            present in config/settings.yaml):

            rag.retrieval_mode       -> RetrievalMode
            rag.rrf_k                -> RRF constant (default 60)
            rag.cross_source_boost   -> cross-source bonus factor (default 1.2)
            rag.enable_bm25          -> toggle BM25 lane (default True)
            rag.bm25_top_k           -> BM25 candidates per query (default 10)
            rag.vector_weight        -> per-source RRF weight (default 1.0)
            rag.graph_weight         -> per-source RRF weight (default 1.0)
            rag.bm25_weight          -> per-source RRF weight (default 1.0)
            vector_store.top_k_vectors        -> vector_top_k / final_top_k
            vector_store.similarity_threshold -> dense filter floor (default 0.3)
            graph.top_k_entities     -> graph_top_k
            graph.max_hops           -> graph traversal depth (default 2)
            graph.enable_hop3        -> opt-in 2-bridge hop (default False)
            graph.hub_mention_cap    -> I-3 hub-entity exclusion threshold
                                       (mutates hybrid_store.graph_store.HUB_MENTION_CAP
                                       on this instance only)
            ingestion.spacy_model    -> SpaCy fallback model for query NER
            entity_extraction.gliner.confidence_threshold
                                     -> query_ner_confidence (default 0.15)
            entity_extraction.gliner.entity_types
                                     -> query_entity_types (None = use defaults)
            entity_extraction.gliner.model_name
                                     -> GLiNER model identifier

    Returns:
        Configured HybridRetriever instance.
    """
    cfg = cfg or {}
    rag_cfg = cfg.get("rag", {})
    vs_cfg = cfg.get("vector_store", {})
    graph_cfg = cfg.get("graph", {})
    ingestion_cfg = cfg.get("ingestion", {})
    gliner_cfg = cfg.get("entity_extraction", {}).get("gliner", {})

    config = RetrievalConfig(
        mode=RetrievalMode(rag_cfg.get("retrieval_mode", "hybrid")),
        vector_top_k=vs_cfg.get("top_k_vectors", 10),
        graph_top_k=graph_cfg.get("top_k_entities", 10),
        max_hops=graph_cfg.get("max_hops", 2),
        rrf_k=rag_cfg.get("rrf_k", 60),
        final_top_k=vs_cfg.get("top_k_vectors", 10),
        cross_source_boost=rag_cfg.get("cross_source_boost", 1.2),
        similarity_threshold=vs_cfg.get("similarity_threshold", 0.3),
        spacy_model=ingestion_cfg.get("spacy_model", "en_core_web_sm"),
        query_ner_confidence=gliner_cfg.get("confidence_threshold", 0.15),
        query_entity_types=gliner_cfg.get("entity_types", None),
        gliner_model_name=gliner_cfg.get("model_name", "urchade/gliner_small-v2.1"),
        enable_bm25=rag_cfg.get("enable_bm25", True),
        bm25_top_k=rag_cfg.get("bm25_top_k", 10),
        # Per-source RRF weights + Hop-3 toggle. Defaults preserve vanilla
        # equal-weight RRF and Hop-3-disabled behaviour.
        vector_weight=rag_cfg.get("vector_weight", 1.0),
        graph_weight=rag_cfg.get("graph_weight", 1.0),
        bm25_weight=rag_cfg.get("bm25_weight", 1.0),
        enable_hop3=graph_cfg.get("enable_hop3", False),
    )

    # I-3: propagate the hub-mention cap from settings.yaml into the live
    # graph store (no class-wide mutation; only this store instance).
    # Falls back to the class default if the key is absent.
    _hub_cap = graph_cfg.get("hub_mention_cap")
    if _hub_cap is not None:
        try:
            hybrid_store.graph_store.HUB_MENTION_CAP = int(_hub_cap)
        except (AttributeError, ValueError, TypeError):
            logger.debug("hub_mention_cap override failed; using class default")

    # Retrieval-time hub fan-out cap (graph.hub_fanout_cap) — same per-instance
    # override pattern as hub_mention_cap. Absent key => class default (5).
    _fanout_cap = graph_cfg.get("hub_fanout_cap")
    if _fanout_cap is not None:
        try:
            hybrid_store.graph_store.HUB_FANOUT_CAP = int(_fanout_cap)
        except (AttributeError, ValueError, TypeError):
            logger.debug("hub_fanout_cap override failed; using class default")

    # Per-call graph_search wall-clock budget (graph.search_budget_seconds).
    # Set on the HybridStore facade (which owns graph_search). Absent => 8.0s.
    _budget = graph_cfg.get("search_budget_seconds")
    if _budget is not None:
        try:
            hybrid_store.GRAPH_SEARCH_BUDGET_S = float(_budget)
        except (AttributeError, ValueError, TypeError):
            logger.debug("search_budget_seconds override failed; using class default")

    return HybridRetriever(hybrid_store, embeddings, config)


# No __main__ smoke entry point lives here — RRFFusion and HybridRetriever
# are exercised by the canonical test suite. Run:
#     python -X utf8 -m pytest test_system/test_data_layer.py
