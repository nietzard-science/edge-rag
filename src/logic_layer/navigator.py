"""
===============================================================================
Navigator (S_N) — Hybrid Retrieval, RRF Fusion & Pre-Generative Filtering
===============================================================================

Paper: "Edge-RAG: Empirical Characterization of When Knowledge-Graph Lanes Add Value in CPU-Only Hybrid Retrieval"
Artifact B: Agent-Based Query Processing — S_N Component.

Role in the pipeline
--------------------
S_N is the second agent of the S_P → S_N → S_V pipeline. It executes the
RetrievalPlan produced by S_P and delivers high-quality evidence chunks to
S_V. Configuration shared with AgentPipeline is defined in ControllerConfig
(src/logic_layer/_config.py).

1. Hybrid retrieval orchestration — vector (LanceDB) + graph (KuzuDB) +
   BM25 paths, selected from the RetrievalPlan strategy.
2. RRF fusion — Reciprocal Rank Fusion across sub-query result lists, with
   a cross-source corroboration boost (configurable weights). Each chunk is
   tagged with the sub-query where it ranked best, used by the reranker.
   Reference: Cormack et al. (2009). SIGIR. DOI:10.1145/1571941.1572114.
3. Optional cross-encoder reranking (Stage 2.5) — re-scores the top-k fused
   chunks with a (query, chunk) cross-encoder; off by default.
4. Pre-generative filtering — six sequential filters + a final fairness cap:
     a) Relevance      dynamic threshold (relevance_factor × max RRF)
     b) Redundancy     Jaccard deduplication
     c) Contradiction  numeric-divergence heuristic (off by default)
     d) Entity-overlap subset entity-set pruning            (original)
     e) Entity-mention three-tier query-entity relevance     (original)
     f) Context shrink sentence-level trim for edge CPU       (original)
   The three original-contribution filters (d, e, f) each have an
   ``enable_*`` toggle in ControllerConfig so the paper's section-3.3
   per-filter ablation rows can be regenerated. The final cap uses
   per-anchor fairness (_fair_cap_by_subquery) for parallel decompositions
   so no single anchor monopolises the budget; single-hop is a pure global
   top-k.

Exports
-------
    RetrieverProtocol, NavigatorResult, Navigator.
    ControllerConfig is re-exported from _config.py for import compatibility.

References (algorithm anchors)
------------------------------
    Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion. SIGIR.
    Radlinski, Kurup, Joachims (2008). Team-draft interleaving. CIKM.
        (Basis of the per-anchor fairness cap.)
    Yang et al. (2018). HotpotQA. EMNLP. (Coverage requirement for the
        parallel-decomposition fairness cap.)
    Reimers & Gurevych (2019). Sentence-BERT. arXiv:1908.10084.
        (Cross-encoder reranker.)

Dependencies
------------
    src.utils.jaccard_similarity; sentence-transformers (optional — only
    when enable_reranker). stdlib otherwise (re, time, logging, typing,
    dataclasses, collections). A RetrieverProtocol implementation is
    injected at runtime.

Last reviewed: 2026-05-27 (audit pass, project version 5.4).
===============================================================================
"""

import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from ._config import ControllerConfig
from .planner import RetrievalPlan
from src.utils import jaccard_similarity

# Module logger — defined before any module-level code that might log.
logger = logging.getLogger(__name__)


# =============================================================================
# RETRIEVER PROTOCOL
# =============================================================================

@runtime_checkable
class RetrieverProtocol(Protocol):
    """Interface contract for HybridRetriever used by Navigator."""
    def retrieve(
        self, query: str, entity_hints: Optional[List[str]] = None
    ) -> "Tuple[List[Any], Dict[str, Any]]":
        ...


# ControllerConfig is defined in _config.py and imported above.
# It is re-exported here so that existing imports of the form
#   from src.logic_layer.navigator import ControllerConfig
# continue to work without change.


@dataclass
class NavigatorResult:
    """
    Result produced by the Navigator (S_N).

    Attributes:
        filtered_context: context chunks after pre-gen filtering (aligned with scores)
        raw_context: unfiltered chunks from RRF fusion
        scores: RRF score per filtered_context chunk
        metadata: provenance and per-filter counts
    """
    filtered_context: List[str] = field(default_factory=list)
    raw_context: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# NAVIGATOR (S_N) IMPLEMENTATION
# =============================================================================

class Navigator:
    """
    S_N: Navigator with hybrid retrieval and pre-generative filtering.

    The Navigator executes the retrieval plan produced by S_P and delivers
    high-quality evidence to S_V for generation.

    Per the paper, section 3.3, the Navigator implements:

    1. HYBRID RETRIEVAL ORCHESTRATION
       - Vector retrieval (semantic search)
       - Graph retrieval (relation-based)
       - Strategy selected from the RetrievalPlan

    2. RRF FUSION
       - Reciprocal Rank Fusion across sub-query result lists
       - Cross-source corroboration boost

    3. PRE-GENERATIVE FILTERING
       a) Relevance filter: drop chunks below a dynamic threshold
       b) Redundancy filter: deduplicate chunks by lexical similarity
       c) Contradiction filter: numeric-heuristic contradiction removal
       d) Entity overlap pruning: drop subsumed entity sets
       e) Entity-mention filter: require query entity presence
       f) Context shrinkage: trim each chunk to relevant sentences
    """

    # ── Class-level compiled regexes (compiled once at class load) ────────────
    # Number token matcher for the contradiction filter (4-digit years first,
    # then any integer/decimal).
    _NUMBER_RE = re.compile(r"\b\d{4}\b|\b\d+(?:\.\d+)?\b")
    # Capitalised-token sentence-priority proxy for context shrinkage.
    _SHRINK_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}")
    # Sentence-boundary splitter (shared by context shrinkage).
    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
    # Capitalised proper-noun proxy for entity-overlap pruning. Intentional
    # variant of _text_utils._PROPER_NOUN_RE: [a-zA-Z] continuations (mixed
    # case like "MacDonald") and * (includes single-word proper nouns).
    _OVERLAP_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")

    # ── Closed lexical lists for contradiction-filter number classification ───
    # Year-context words: a number near one of these is treated as a year
    # regardless of magnitude ("year 3000").
    _YEAR_CONTEXT_WORDS = (
        "year", "born", "founded", "established", "released",
        "published", "elected", "incorporated", "ad ", "bc ", "ce ",
    )
    # Count-context words: a number near one of these is treated as a
    # count/population even inside the year range ("Population 2014").
    _COUNT_CONTEXT_WORDS = (
        "population", "people", "inhabitants", "residents",
        "members", "employees", "employs", "employ", "staff",
        "workers", "users", "subscribers",
        "votes", "voters", "seats", "kilometers", "km",
        "miles", "dollars", "euros", "pounds",
        "versus", "vs",
    )

    # Minimum content-token lengths for the entity-mention filter's
    # query-content overlap fallback and variant-name (all-tokens-co-occur)
    # match. Tokenisation minutiae, not metric-ablation knobs.
    _QUERY_CONTENT_MIN_LEN = 4
    _AND_PATTERN_TOKEN_MIN = 3

    def __init__(self, config: ControllerConfig):
        """
        Initialise Navigator.

        Args:
            config: ControllerConfig with navigator settings
        """
        self.config = config

        # Retriever is injected later via set_retriever()
        self.retriever: Optional[RetrieverProtocol] = None

        # Cross-encoder reranker — lazy-loaded on first use.
        # None = not yet attempted; False = load failed (skip future attempts)
        self._reranker: Optional[Any] = None

        logger.info(
            "Navigator initialized: relevance_factor=%s, redundancy_threshold=%s",
            config.relevance_threshold_factor,
            config.redundancy_threshold,
        )

    def set_retriever(
        self,
        retriever: RetrieverProtocol,
    ) -> None:
        """
        Attach a HybridRetriever to this Navigator.

        Args:
            retriever: HybridRetriever instance implementing RetrieverProtocol
        """
        self.retriever = retriever
        logger.info("HybridRetriever connected")

    def navigate(
        self,
        retrieval_plan: Optional[RetrievalPlan],
        sub_queries: List[str],
        entity_names: Optional[List[str]] = None,
    ) -> NavigatorResult:
        """
        Execute hybrid retrieval and pre-generative filtering.

        Algorithm:
        1. Retrieve for each sub-query via HybridRetriever
        2. Fuse results with RRF
        3. Apply the six pre-generative filters in sequence
        4. Return filtered context as a NavigatorResult

        Args:
            retrieval_plan: RetrievalPlan from S_P
            sub_queries: list of sub-queries to retrieve for
            entity_names: optional pre-extracted entity name strings; when provided
                these take precedence over retrieval_plan.entities so that entity
                mention filtering works correctly when RetrievalPlan is reconstructed
                from a serialized state dict (e.g. in AgenticController._navigator_node).

        Returns:
            NavigatorResult with filtered context
        """
        start_time = time.time()

        result = NavigatorResult()
        result.metadata["retrieval_plan"] = retrieval_plan.to_dict() if retrieval_plan else None

        if self.retriever is None:
            logger.warning("[Navigator] No retriever set — returning empty result")
            return result

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 1: HYBRID RETRIEVAL
        # ─────────────────────────────────────────────────────────────────────

        logger.info("[Navigator] Retrieval for %d sub-queries", len(sub_queries))

        all_results = []

        # Entity hints from S_P passed to retriever so GLiNER is not re-run
        # on short sub-query fragments (e.g. "What is the nationality of
        # [Person]?") where it frequently fails to recognise an entity name
        # that was confidently identified upstream from the full question.
        #
        # Priority: explicit entity_names arg > plan.entities > None (GLiNER re-runs).
        # Using plan entities as hints ensures all sub-queries share the same full
        # entity set, so the keyword entity fallback in HybridRetriever can inject
        # entity-specific chunks into every sub-query result list.  Without this,
        # a sub-query that names only one of two compared entities produces no
        # keyword hit for the second entity, lowering its cross-query RRF score
        # below the Relevance Filter threshold and dropping its supporting
        # paragraph from the final context.
        if entity_names is not None:
            hints = entity_names
        elif retrieval_plan is not None and getattr(retrieval_plan, "entities", None):
            hints = [e.text for e in retrieval_plan.entities if getattr(e, "text", None)]
        else:
            hints = None

        for sub_query in sub_queries:
            try:
                # HybridRetriever.retrieve() returns (results, metrics) tuple
                results, _metrics = self.retriever.retrieve(sub_query, entity_hints=hints)

                for res in results[:self.config.top_k_per_subquery]:
                    text = res.text if hasattr(res, "text") else str(res)
                    # Prefer rrf_score (already fused by HybridRetriever), then raw score.
                    # Sentinel 1.0 used only for unknown result types — this assigns equal
                    # weight to all fallback results so they can still be ranked by RRF.
                    score = (
                        res.rrf_score if hasattr(res, "rrf_score")
                        else res.score if hasattr(res, "score")
                        else 1.0
                    )

                    # Capture the retrieval method ("vector"/"graph"/
                    # "bm25"/"hybrid") separately from source_doc so the Verifier
                    # credibility score can use real graph-provenance signal
                    # instead of a constant baseline.
                    retrieval_method = (
                        res.retrieval_method
                        if hasattr(res, "retrieval_method")
                        else "unknown"
                    )
                    all_results.append({
                        "text": text,
                        "score": score,
                        "source": (
                            res.source_doc if hasattr(res, "source_doc")
                            else res.source if hasattr(res, "source")
                            else "unknown"
                        ),
                        "retrieval_method": retrieval_method,
                        "sub_query": sub_query,
                    })

            except Exception as e:
                # Broad catch is intentional: retriever errors (network, DB, model)
                # must not abort the pipeline; missing sub-query results degrade
                # gracefully — remaining sub-queries still contribute context.
                logger.error(
                    "[Navigator] Retrieval error for sub-query %r: %s", sub_query, e, exc_info=True
                )
                result.metadata["retrieval_errors"] = (
                    result.metadata.get("retrieval_errors", []) + [str(e)]
                )

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 2: RRF FUSION
        # ─────────────────────────────────────────────────────────────────────

        logger.info("[Navigator] RRF fusion of %d results", len(all_results))

        fused_results = self._rrf_fusion(all_results)

        result.raw_context = [r["text"] for r in fused_results]

        result.metadata["pre_filter_count"] = len(fused_results)
        result.metadata["fusion_time_ms"] = (time.time() - start_time) * 1000

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 2.5: CROSS-ENCODER RERANKING (optional)
        # ─────────────────────────────────────────────────────────────────────

        original_query = (
            retrieval_plan.original_query
            if retrieval_plan and hasattr(retrieval_plan, "original_query")
            else (sub_queries[0] if sub_queries else "")
        )
        fused_results = self._reranker_filter(
            fused_results, original_query, entity_hints=entity_names,
        )
        result.metadata["reranker_applied"] = self.config.enable_reranker

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 3: PRE-GENERATIVE FILTERING
        # ─────────────────────────────────────────────────────────────────────

        logger.info("[Navigator] Pre-generative filtering")

        filter_start = time.time()

        # Filter 1: Relevance filter.
        # With >1 parallel sub-query (comparison / intersection decomposition),
        # RRF runs over *disjoint* per-entity result sets, so the score
        # distribution is flat and low: no chunk gets the cross-query
        # corroboration boost except suffix-noise chunks that happen to match
        # the shared tail of every sub-query ("… was from England"). A global
        # `relevance_threshold_factor × max` then keys off that inflated noise
        # max and discards the genuine per-entity answer chunks (each present in
        # only one sub-query's list). So: relax the relevance filter to a no-op
        # when there are multiple parallel sub-queries — the entity-mention /
        # redundancy / cap stages still trim the context.
        n_parallel = len(sub_queries) if sub_queries else 1
        if n_parallel > 1:
            relevance_filtered = fused_results
            logger.debug(
                "[Navigator] Relevance filter: skipped (%d parallel sub-queries → "
                "flat RRF distribution)", n_parallel,
            )
        else:
            relevance_filtered = self._relevance_filter(fused_results)
        result.metadata["after_relevance_filter"] = len(relevance_filtered)

        # Filter 2: Redundancy filter (lexical deduplication)
        redundancy_filtered = self._redundancy_filter(relevance_filtered)
        result.metadata["after_redundancy_filter"] = len(redundancy_filtered)

        # Filter 3: Contradiction filter (numeric heuristic, disabled by default)
        if self.config.enable_contradiction_filter:
            contradiction_filtered = self._contradiction_filter(redundancy_filtered)
        else:
            contradiction_filtered = redundancy_filtered
        result.metadata["after_contradiction_filter"] = len(contradiction_filtered)

        # Filter 4: Entity overlap pruning (toggleable for ablation)
        if self.config.enable_entity_overlap_pruning:
            entity_pruned = self._entity_overlap_pruning(contradiction_filtered)
        else:
            entity_pruned = contradiction_filtered
        result.metadata["after_entity_overlap_pruning"] = len(entity_pruned)

        # Filter 5: Entity-Mention Filter — drop chunks with no query-entity reference.
        # entity_names param takes precedence (used when plan is reconstructed from state dict).
        # DATE/TIME/CARDINAL/ORDINAL labels are excluded: numeric/temporal tokens (e.g.
        # "1992") match irrelevant chunks and produce false positives.
        _SKIP_NER_LABELS = {"DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY"}
        if entity_names is not None:
            query_entity_names = entity_names
        else:
            query_entity_names = (
                [e.text for e in retrieval_plan.entities
                 if e.label.upper() not in _SKIP_NER_LABELS]
                if (retrieval_plan and retrieval_plan.entities)
                else []
            )
        if self.config.enable_entity_mention_filter:
            mention_filtered = self._entity_mention_filter(
                entity_pruned, query_entity_names, sub_queries=sub_queries
            )
        else:
            mention_filtered = entity_pruned
        result.metadata["after_entity_mention_filter"] = len(mention_filtered)

        # Cap at max_context_chunks. With multiple parallel sub-queries
        # (comparison / intersection), cap with per-anchor fairness so a
        # high-degree entity cannot monopolise the budget and crowd out the
        # second entity's gold paragraph; single-hop is unchanged (no-op).
        top_results = self._fair_cap_by_subquery(
            mention_filtered, self.config.max_context_chunks, n_parallel,
        )

        # Filter 6: Context shrinkage (edge optimization: fewer input tokens;
        # toggleable for ablation)
        if self.config.enable_context_shrinkage:
            shrunk_results = self._context_shrinkage(top_results)
        else:
            shrunk_results = top_results

        result.filtered_context = [r["text"] for r in shrunk_results]
        result.scores = [r["rrf_score"] for r in shrunk_results]

        # Surface retrieval provenance per filtered chunk so the Verifier's
        # credibility scorer can use a real graph-based signal (see
        # verifier._compute_credibility). A chunk is "graph-based" if at least
        # one of the retrieval paths that produced it was the graph path.
        result.metadata["chunk_retrieval_methods"] = [
            list(r.get("retrieval_methods", [])) for r in shrunk_results
        ]
        result.metadata["chunk_is_graph_based"] = [
            "graph" in (r.get("retrieval_methods", []) or [])
            for r in shrunk_results
        ]

        result.metadata["filter_time_ms"] = (time.time() - filter_start) * 1000
        result.metadata["total_time_ms"] = (time.time() - start_time) * 1000

        logger.info(
            "[Navigator] Done: %d chunks (from %d raw) in %.0f ms",
            len(result.filtered_context),
            len(all_results),
            result.metadata["total_time_ms"],
        )

        return result

    def _rrf_fusion(
        self,
        results: List[Dict[str, Any]],
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) of retrieval results.

        RRF Score = Σ 1 / (k + rank_i)

        where k is a smoothing constant (default 60) and rank_i is the rank
        of the chunk in the i-th result list.

        Reference: Cormack et al. (2009). ACM. DOI:10.1145/1571941.1572114

        Cross-source corroboration: chunks that appear in multiple sub-query
        result lists receive a multiplicative boost.

        Args:
            results: list of retrieval result dicts with keys text/score/source/sub_query
            k: RRF smoothing constant (None = read from self.config.rrf_k)

        Returns:
            fused and sorted list of result dicts with added rrf_score key
        """
        if k is None:
            k = self.config.rrf_k

        # Pass 1: group all results by text, collecting scores/sources/sub-queries.
        # Pass 2: build per-sub-query rankings and compute 1/(k+rank) contributions.
        # Two passes are needed because a text may appear in multiple sub-query lists
        # and we need the full source/sub-query sets before computing the boost.
        text_groups: Dict[str, Any] = {}
        for r in results:
            text = r["text"]
            if text not in text_groups:
                text_groups[text] = {
                    "text": text,
                    "scores": [],
                    "sources": set(),
                    "sub_queries": set(),
                    # Track the retrieval methods that produced this chunk
                    # (vector/graph/bm25). Used by the Verifier to set
                    # is_graph_based for credibility scoring.
                    "retrieval_methods": set(),
                }
            text_groups[text]["scores"].append(r["score"])
            text_groups[text]["sources"].add(r["source"])
            text_groups[text]["sub_queries"].add(r["sub_query"])
            method = r.get("retrieval_method", "unknown")
            if method and method != "unknown":
                text_groups[text]["retrieval_methods"].add(method)

        # Build per-sub-query rankings and accumulate RRF contributions
        sub_query_rankings: Dict[str, List[Any]] = {}
        for r in results:
            sq = r["sub_query"]
            if sq not in sub_query_rankings:
                sub_query_rankings[sq] = []
            sub_query_rankings[sq].append(r)

        for sq, sq_results in sub_query_rankings.items():
            sq_results.sort(key=lambda x: x["score"], reverse=True)
            for rank, r in enumerate(sq_results):
                text = r["text"]
                if "rrf_contributions" not in text_groups[text]:
                    text_groups[text]["rrf_contributions"] = []
                    text_groups[text]["_best_sub_query"] = sq      # sub-query with best rank
                    text_groups[text]["_best_rank"] = rank
                else:
                    # Track the sub-query where this chunk ranked highest (lowest rank index)
                    if rank < text_groups[text]["_best_rank"]:
                        text_groups[text]["_best_sub_query"] = sq
                        text_groups[text]["_best_rank"] = rank
                text_groups[text]["rrf_contributions"].append(1.0 / (k + rank))

        # Aggregate RRF scores with cross-source corroboration boost.
        # Boost formula: 1 + source_weight*(N_sources-1) + query_weight*(N_queries-1).
        # Weights are sourced from config (settings.yaml navigator.corroboration_*).
        fused = []
        for text, group in text_groups.items():
            rrf_score = sum(group.get("rrf_contributions", []))

            source_count = len(group["sources"])
            query_count = len(group["sub_queries"])
            corroboration_boost = (
                1.0
                + self.config.corroboration_source_weight * (source_count - 1)
                + self.config.corroboration_query_weight * (query_count - 1)
            )

            fused.append({
                "text": text,
                "rrf_score": rrf_score * corroboration_boost,
                "original_scores": group["scores"],
                "source_count": source_count,
                "query_count": query_count,
                # Sub-query where this chunk ranked best — used by the cross-encoder
                # reranker so bridge chunks are scored against the hop that retrieved
                # them, not the surface query.
                "_best_sub_query": group.get("_best_sub_query", ""),
                # Propagate retrieval-method provenance so the Verifier can flag
                # graph-retrieved chunks as more credible.
                "retrieval_methods": list(group.get("retrieval_methods", set())),
            })

        # Sort descending by RRF score
        fused.sort(key=lambda x: x["rrf_score"], reverse=True)

        return fused

    def _fair_cap_by_subquery(
        self,
        results: List[Dict[str, Any]],
        budget: int,
        n_parallel: int,
    ) -> List[Dict[str, Any]]:
        """Per-anchor fairness cap for parallel (comparison/intersection) plans.

        A purely global top-k lets a high-degree entity's many chunks monopolise
        the context budget and crowd out the second entity's (often single) gold
        paragraph — observed when one comparison conjunct dominates retrieval.
        With multiple parallel sub-queries this interleaves the per-sub-query
        rankings round-robin (strongest anchor first each round) so every anchor
        is represented before any one anchor's surplus fills the remaining slots.

        Coverage of each decomposed aspect is the defining requirement of a
        comparison question — both supporting paragraphs are needed (Yang et al.
        2018, EMNLP); balanced interleaving of result lists follows team-draft
        interleaving (Radlinski et al. 2008, CIKM). Chunks carry the
        ``_best_sub_query`` tag set by ``_rrf_fusion``.

        Single sub-query (or a single distinct anchor present) → identical to
        ``results[:budget]``; this method is a no-op outside parallel plans.
        """
        if n_parallel <= 1 or budget >= len(results):
            return results[:budget]
        # Group preserving the global rrf-descending order within each anchor.
        groups: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
        for r in results:
            groups.setdefault(r.get("_best_sub_query", ""), []).append(r)
        if len(groups) <= 1:
            return results[:budget]
        # Strongest anchor (by its best chunk) leads each round.
        ordered = sorted(groups.values(), key=lambda g: g[0]["rrf_score"], reverse=True)
        selected: List[Dict[str, Any]] = []
        cursors = [0] * len(ordered)
        while len(selected) < budget:
            progressed = False
            for gi, g in enumerate(ordered):
                if cursors[gi] < len(g):
                    selected.append(g[cursors[gi]])
                    cursors[gi] += 1
                    progressed = True
                    if len(selected) >= budget:
                        break
            if not progressed:
                break
        return selected

    def _relevance_filter(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Relevance filter: drop low-confidence candidates.

        Per the paper, section 3.3: chunks with RRF scores below a dynamic
        threshold (relevance_threshold_factor × max_score) are discarded.

        Args:
            results: fused result list (sorted by rrf_score descending)

        Returns:
            filtered result list
        """
        if not results:
            return results

        max_score = max(r["rrf_score"] for r in results)
        threshold = self.config.relevance_threshold_factor * max_score

        filtered = [r for r in results if r["rrf_score"] >= threshold]

        logger.debug(
            "[Navigator] Relevance filter: threshold=%.4f, kept %d/%d",
            threshold, len(filtered), len(results),
        )

        return filtered

    def _redundancy_filter(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Redundancy filter: deduplicate similar chunks.

        Per the paper, section 3.3: chunks with high lexical overlap
        (similarity > redundancy_threshold) are deduplicated; the chunk
        with the higher RRF score is retained.

        Args:
            results: relevance-filtered result list (sorted by rrf_score)

        Returns:
            deduplicated result list
        """
        if not results:
            return results

        # Results are already sorted by score, so earlier entries win ties
        filtered = []
        seen_texts = []  # kept for pairwise similarity comparison

        for r in results:
            text = r["text"]
            is_duplicate = False

            # Compare against all already-accepted chunks
            for seen in seen_texts:
                similarity = jaccard_similarity(text, seen)
                if similarity > self.config.redundancy_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append(r)
                seen_texts.append(text)

        logger.debug(
            "[Navigator] Redundancy filter: kept %d/%d unique",
            len(filtered), len(results),
        )

        return filtered

    def _contradiction_filter(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Contradiction Filter: remove chunks with contradictory numeric values.

        Two chunks are considered contradictory when they share high word overlap
        (topic similarity) but contain strongly differing numeric values (factual
        conflict). The chunk with the lower RRF score is dropped.

        Threshold rationale (all configurable via settings.yaml):
          overlap_threshold = 0.3: a 30% word-overlap ensures the chunks discuss
            the same topic before declaring a numeric conflict.
          ratio_threshold = 2.0: numbers that differ by more than 2x are likely
            factually conflicting (e.g., "born in 1940" vs. "born in 1970").
          min_value = 100: filters out trivial small-integer differences such as
            list indices, day-of-month values, or short counts.
        """
        if len(results) < 2:
            return results

        overlap_threshold = self.config.contradiction_overlap_threshold
        ratio_threshold = self.config.contradiction_ratio_threshold
        min_value = self.config.contradiction_min_value
        window_size = self.config.contradiction_number_context_window
        year_min = float(self.config.contradiction_year_range_min)
        year_max = float(self.config.contradiction_year_range_max)

        def _extract_numbers_with_context(text: str) -> List[Tuple[float, bool]]:
            """Return (number, is_year_in_context) for each number found.

            Disambiguation strategy (count-context words take priority over
            year-context words, which take priority over the magnitude
            heuristic [year_range_min, year_range_max]):
              1. If a count-context word appears near the number → count.
              2. Else, if a year-context word appears near → year.
              3. Else, fall back to the magnitude heuristic.
            So "Population 1000" is a count even though 1000 looks like a year.
            """
            text_lower = text.lower()
            out: List[Tuple[float, bool]] = []
            for m in self._NUMBER_RE.finditer(text):
                val = float(m.group(0))
                pos = m.start()
                # Look in a +/- window (chars) around the number for context.
                start = max(0, pos - window_size)
                end = min(len(text_lower), pos + len(m.group(0)) + window_size)
                window = text_lower[start:end]
                near_count = any(w in window for w in self._COUNT_CONTEXT_WORDS)
                near_year = any(w in window for w in self._YEAR_CONTEXT_WORDS)
                in_range = year_min <= val <= year_max and val == int(val)
                # Count context wins; year context next; magnitude last.
                if near_count and not near_year:
                    is_year = False
                elif near_year:
                    is_year = True
                else:
                    is_year = in_range
                out.append((val, is_year))
            return out

        all_numbers: List[List[Tuple[float, bool]]] = [
            _extract_numbers_with_context(r["text"]) for r in results
        ]
        all_words: List[set] = [set(r["text"].lower().split()) for r in results]

        contradicting: set = set()
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                nums_i = all_numbers[i]
                nums_j = all_numbers[j]
                if not nums_i or not nums_j:
                    continue

                words_i = all_words[i]
                words_j = all_words[j]
                overlap = len(words_i & words_j) / max(len(words_i | words_j), 1)

                if overlap > overlap_threshold:
                    found = any(
                        n1 > 0 and n2 > 0
                        and max(n1, n2) / min(n1, n2) > ratio_threshold
                        and min(n1, n2) > min_value
                        # Only flag as contradiction when both numbers are the
                        # same kind (both years, or both counts/populations).
                        # A year paired with a population figure (e.g. 1940 vs
                        # 276170) measures different things and is not a conflict.
                        and is_year_1 == is_year_2
                        for (n1, is_year_1) in nums_i
                        for (n2, is_year_2) in nums_j
                    )
                    if found:
                        lower_idx = (
                            i if results[i]["rrf_score"] < results[j]["rrf_score"] else j
                        )
                        contradicting.add(lower_idx)

        filtered = [r for idx, r in enumerate(results) if idx not in contradicting]

        if contradicting:
            logger.debug(
                "[Navigator] Contradiction filter: removed %d, kept %d/%d",
                len(contradicting), len(filtered), len(results),
            )

        if not filtered:
            logger.debug(
                "[Navigator] Contradiction filter: all chunks removed — returning all"
            )
            return results

        return filtered

    def _entity_overlap_pruning(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Entity Overlap Pruning: drop chunks whose named-entity set is fully
        covered by a higher-ranked chunk.

        Original contribution (paper section 3.3); see the per-filter
        ablation in the paper's results section for its EM contribution
        (toggle via config.enable_entity_overlap_pruning).
        If entities(Chunk_B) ⊆ entities(Chunk_A) and score(A) > score(B),
        Chunk_B is informationally redundant and is removed.

        Heuristic: capitalized multi-word phrases serve as named-entity proxies
        (avoids a dependency on a full NER model at filter time).
        """
        if len(results) < 2:
            return results

        def extract_entities(text: str) -> set:
            tokens = self._OVERLAP_ENTITY_RE.findall(text)
            return {t.lower() for t in tokens if len(t) > 2}

        entity_sets = [extract_entities(r["text"]) for r in results]

        kept = []
        pruned: set = set()

        for i, r_i in enumerate(results):
            if i in pruned:
                continue
            if not entity_sets[i]:
                kept.append(r_i)
                continue

            is_subset = any(
                j not in pruned and entity_sets[i].issubset(entity_sets[j])
                for j in range(i)  # higher-ranked chunk (index < i because list is score-sorted)
            )

            if not is_subset:
                kept.append(r_i)
            else:
                pruned.add(i)

        if pruned:
            logger.debug(
                "[Navigator] Entity overlap pruning: removed %d, kept %d/%d",
                len(pruned), len(kept), len(results),
            )

        return kept if kept else results

    # Content-word stopword set for the overlap fallback. Small and
    # query-oriented (function words + the WH/auxiliary scaffolding of a
    # question); proper-noun content is never on it.
    _QUERY_STOPWORDS: frozenset = frozenset({
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "did", "do",
        "does", "for", "from", "had", "has", "have", "he", "her", "his", "in",
        "into", "is", "it", "its", "of", "on", "or", "she", "that", "the",
        "their", "this", "to", "was", "were", "what", "when", "where", "which",
        "who", "whom", "whose", "will", "with", "would", "they", "them", "you",
        "your", "we", "our", "i", "than", "more", "most", "less", "fewer",
        "first", "used", "during",
    })
    _WORD_RE = re.compile(r"[a-z0-9]+(?:[''][a-z]+)?")

    def _entity_mention_filter(
        self,
        results: List[Dict[str, Any]],
        entity_names: List[str],
        sub_queries: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Entity-mention filter (re-ranking with a relevance floor).

        Original contribution (paper section 3.3); the ablation showed an EM
        improvement on bridge questions when the entity layer is reliable.

        The earlier design *hard-dropped* any chunk that mentioned none of the
        query's named entities. On HotpotQA that is unsafe: many questions name
        the answer by *description* ("which British first-generation jet-powered
        medium bomber …"), so the answer chunk contains the *description words*
        but not any of the noisy SpaCy entities ("Which British", "World War").
        That chunk would be retrieved at RRF rank #1 (near-verbatim dense match)
        and then *deleted by this filter*, leaving the Verifier with strictly
        worse context.

        New behaviour — three tiers, stable-sorted:
          tier 0  chunk mentions a *specific* query entity (multi-word phrase,
                  or a distinctive single token ≥8 chars)
          tier 1  chunk mentions only a *generic* entity (short token like
                  "England") OR has strong content-word overlap with the query
                  (covers "description-subject" chunks whose answer words are
                  present but whose only "entities" are SpaCy noise)
          tier 2  chunk does neither
        Within each tier the original RRF order is preserved. Tier-2 chunks are
        dropped. Safety: if everything would be dropped, return all (never an
        empty context). The content-word fallback (tier 1) is the floor that
        keeps the answer chunk for hidden-bridge questions — it does not depend
        on RRF rank, so a noise chunk that the retriever happened to rank highly
        is still dropped if it shares no query content words.

        When `sub_queries` is None (e.g. unit tests, or a reconstructed plan),
        the content-word fallback is inactive and the filter degrades to the
        classic entity-only behaviour.

        Regexes are pre-compiled before the chunk loop (Python's re cache is 512
        entries; many entities × tokens × chunks can overflow it).
        """
        # No usable query entities. Behaviour is unchanged (we cannot
        # entity-gate without entities), but it is not SILENT — a bare
        # `return results` would hide the case where NER yielded only a
        # non-anchorable phrase (e.g. a DATE) and unfiltered noise chunks
        # reached the verifier. Log + flag so it is observable; downstream
        # ranking (RRF / reranker) carries relevance instead.
        self._entity_filter_skipped = not bool(entity_names)
        if not entity_names:
            logger.warning(
                "[Navigator] entity-mention filter received no usable entities — "
                "relying on RRF/reranker ranking, no entity gating applied (%d chunks kept)",
                len(results),
            )
            return results

        # Token-length floors (config-driven). The per-token fallback bar is
        # higher for multi-word entities so distinctive surnames are kept and
        # common first names excluded.
        single_token_min = self.config.entity_mention_single_token_min
        fallback_token_min = self.config.entity_mention_fallback_token_min
        specific_single_min = self.config.entity_mention_specific_single_min

        compiled: List[Any] = []   # (phrase_lower, tokens, token_patterns, and_patterns, is_specific)
        for name in entity_names:
            tokens = name.split()
            min_tok = fallback_token_min if len(tokens) >= 2 else single_token_min
            token_patterns = [
                re.compile(r"\b" + re.escape(t.lower()) + r"\b")
                for t in tokens if len(t) >= min_tok
            ]
            # Variant-name (all-tokens-co-occur) match for multi-word entities.
            # Recovers chunks that open with a full legal name (e.g. a four-token
            # legal name for a two-token entity) where the contiguous phrase is
            # absent AND no single token reaches the distinctive-token bar, so
            # neither the full-phrase nor the single-token fallback fires.
            # Requiring ALL content tokens (>= _AND_PATTERN_TOKEN_MIN chars,
            # non-stopword) to co-occur as whole words keeps this high-precision.
            and_patterns = (
                [re.compile(r"\b" + re.escape(t.lower()) + r"\b")
                 for t in tokens
                 if len(t) >= self._AND_PATTERN_TOKEN_MIN and t.lower() not in self._QUERY_STOPWORDS]
                if len(tokens) >= 2 else []
            )
            is_specific = len(tokens) >= 2 or len(name) >= specific_single_min
            compiled.append((name.lower(), tokens, token_patterns, and_patterns, is_specific))

        # Content words of the query (overlap fallback): everything in the
        # sub-queries that isn't a stopword and isn't part of an entity string
        # (entities are handled by the exact-match path above).
        query_content: set = set()
        for q in (sub_queries or []):
            for w in self._WORD_RE.findall((q or "").lower()):
                if len(w) >= self._QUERY_CONTENT_MIN_LEN and w not in self._QUERY_STOPWORDS:
                    query_content.add(w)
        # remove entity tokens — those are the entity path's job
        for name_lower, *_ in compiled:
            for tok in self._WORD_RE.findall(name_lower):
                query_content.discard(tok)
        # Below this fraction of query content words present, "overlap" does
        # not count as topical confirmation.
        overlap_min_fraction = self.config.entity_mention_overlap_min_fraction
        overlap_min_abs = self.config.entity_mention_overlap_min_abs

        def tier_of(text: str) -> int:
            """0 = specific-entity match, 1 = generic-entity or strong content-word
            overlap, 2 = neither."""
            text_lower = text.lower()
            matched_generic = False
            for name_lower, tokens, token_patterns, and_patterns, is_specific in compiled:
                hit = False
                if len(tokens) >= 2:
                    if name_lower in text_lower:
                        hit = True
                    else:
                        for pat in token_patterns:
                            if pat.search(text_lower):
                                hit = True
                                break
                        # Variant-name match: all content tokens co-occur.
                        if not hit and and_patterns and all(
                            p.search(text_lower) for p in and_patterns
                        ):
                            hit = True
                else:
                    for pat in token_patterns:
                        if pat.search(text_lower):
                            hit = True
                            break
                if hit:
                    if is_specific:
                        return 0
                    matched_generic = True
            if matched_generic:
                return 1
            # content-word overlap fallback (covers "description subject" chunks)
            if query_content:
                chunk_words = set(self._WORD_RE.findall(text_lower))
                shared = len(query_content & chunk_words)
                if shared >= overlap_min_abs and shared >= overlap_min_fraction * len(query_content):
                    return 1
            return 2

        # Top-K RRF immunity — the top chunks by RRF score are never dropped by
        # the entity-mention filter, regardless of entity match. Rationale: a
        # chunk ranked in the top few by a three-path RRF (dense + sparse +
        # graph) is almost certainly topically relevant; if its entity happens
        # to be an implicit bridge target absent from the Planner's entity list
        # the filter would otherwise destroy the answer chunk. The threshold is
        # config-driven (entity_mention_rrf_immune_top_k); chunks beyond it are
        # filtered normally.
        rrf_immune_top_k = self.config.entity_mention_rrf_immune_top_k
        immune_indices: set = set(range(min(rrf_immune_top_k, len(results))))

        tiers = [tier_of(r["text"]) for r in results]
        kept = [
            (t, rank, r)
            for rank, (t, r) in enumerate(zip(tiers, results))
            if t < 2 or rank in immune_indices
        ]
        dropped = len(results) - len(kept)

        if not kept:
            logger.debug("[Navigator] Entity-mention filter: all chunks filtered — returning all")
            return results

        # Survivor floor. When the retriever supplied a full candidate set
        # (>= survivor_floor chunks) an over-specific entity match must not
        # leave the Verifier with too little context to tolerate a single
        # retrieval error. If matching kept fewer than the floor, top up with
        # the highest-RRF dropped chunks (results are in RRF rank order, so the
        # lowest not-yet-kept indices are the strongest). The floor engages
        # only for full candidate sets — small inputs are filtered normally
        # (it does not force-keep noise in a 2-3 chunk set).
        survivor_floor = self.config.entity_mention_survivor_floor
        if len(results) >= survivor_floor and len(kept) < survivor_floor:
            kept_ranks = {rank for _, rank, _ in kept}
            for rank, r in enumerate(results):
                if rank not in kept_ranks:
                    kept.append((tiers[rank], rank, r))
                    kept_ranks.add(rank)
                    if len(kept) >= survivor_floor:
                        break
            dropped = len(results) - len(kept)

        kept.sort(key=lambda x: (x[0], x[1]))   # tier asc, then original RRF rank
        out = [r for _, _, r in kept]
        if dropped:
            logger.debug(
                "[Navigator] Entity-mention filter: dropped %d, kept %d/%d (tiers 0/1/2: %d/%d/%d)",
                dropped, len(out), len(results),
                tiers.count(0), tiers.count(1), tiers.count(2),
            )
        return out

    def _reranker_filter(
        self,
        results: List[Dict[str, Any]],
        query: str,
        entity_hints: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Optional cross-encoder reranking after RRF fusion.

        Re-scores the top `reranker_top_k` results with a (query, chunk) cross-
        encoder and sorts them by cross-encoder score.  The remaining chunks
        (beyond top_k) keep their RRF order and are appended unchanged.

        Entity-hint conditioning: when `entity_hints` is provided (Planner-
        extracted entities + bridge entities resolved in earlier hops), the entity
        names are appended to the query side of the (query, chunk) pair as
        a soft hint: "<query> [ENTITIES: A, B, C]". The cross-encoder
        learns to up-rank chunks that lexically contain the hinted entities.
        On bridge questions this means the answer chunk (which mentions the
        bridge entity) is up-ranked over distractor chunks that mention only
        the surface query keywords.

        Refs:
          - Annotated re-ranking with entity hints: ColBERTv2 (Santhanam et
            al., 2022, NAACL).
          - Cross-encoder bi-text reranking: Reimers & Gurevych (2019),
            arXiv:1908.10084.

        Disabled when config.enable_reranker is False (default).
        self._reranker is set to False after the first failed load attempt
        so subsequent calls skip the model-load overhead.

        Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (22 MB, CPU, ~30 ms/pair).
        """
        if not self.config.enable_reranker or not results:
            return results

        if self._reranker is False:
            return results

        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                # 2026-05-27: support a larger cross-encoder via explicit
                # max_length + optional fp16. BGE rerankers (base ~278M /
                # large ~560M) need max_length set; fp16 halves their memory
                # footprint. The small ms-marco MiniLM default is unaffected
                # (max_length=512, fp16=False == prior behaviour).
                _ce_kwargs = {
                    "max_length": getattr(self.config, "reranker_max_length", 512),
                }
                if getattr(self.config, "reranker_fp16", False):
                    try:
                        import torch
                        _ce_kwargs["automodel_args"] = {"torch_dtype": torch.float16}
                    except ImportError:
                        logger.warning(
                            "[Navigator] reranker_fp16 requested but torch "
                            "unavailable; loading in fp32."
                        )
                self._reranker = CrossEncoder(
                    self.config.reranker_model, **_ce_kwargs
                )
                logger.info(
                    "[Navigator] Cross-encoder reranker loaded: %s "
                    "(max_length=%d, fp16=%s)",
                    self.config.reranker_model,
                    _ce_kwargs["max_length"],
                    getattr(self.config, "reranker_fp16", False),
                )
            except ImportError:
                logger.warning(
                    "[Navigator] sentence-transformers not installed"
                    " — reranker disabled. Run: pip install sentence-transformers"
                )
                self._reranker = False
                return results
            except Exception as exc:  # noqa: BLE001 — HF Hub raises non-stdlib types
                # transformers ≥ 4.45 pings huggingface_hub on every
                # from_pretrained even when the model is already cached
                # locally; in an offline / air-gapped venue this raises a
                # connect error. Retry with HF_HUB_OFFLINE=1 so the local HF
                # cache is used without any network request — same pattern as
                # the GLiNER loader in hybrid_retriever._get_gliner_model.
                logger.warning(
                    "[Navigator] Reranker online load raised %s (%s); "
                    "retrying offline from the local HF cache.",
                    type(exc).__name__, exc,
                )
                import os as _os
                _prev_hf = _os.environ.get("HF_HUB_OFFLINE")
                _os.environ["HF_HUB_OFFLINE"] = "1"
                try:
                    from sentence_transformers import CrossEncoder
                    self._reranker = CrossEncoder(
                        self.config.reranker_model, **_ce_kwargs
                    )
                    logger.info(
                        "[Navigator] Cross-encoder reranker loaded from local "
                        "cache: %s", self.config.reranker_model,
                    )
                except Exception as exc2:  # noqa: BLE001
                    logger.warning(
                        "[Navigator] Reranker offline load also failed (%s) — "
                        "reranker DISABLED for this session; fused RRF order "
                        "is used unre-ranked (differs from the paper "
                        "contract).", exc2,
                    )
                    self._reranker = False
                    return results
                finally:
                    if _prev_hf is None:
                        _os.environ.pop("HF_HUB_OFFLINE", None)
                    else:
                        _os.environ["HF_HUB_OFFLINE"] = _prev_hf

        candidates = results[: self.config.reranker_top_k]
        rest = results[self.config.reranker_top_k :]

        # Score each chunk against the sub-query where it ranked best.
        # For single-hop queries _best_sub_query == the original query, so there
        # is no regression. For bridge/multi-hop queries this ensures that a
        # second-hop answer chunk (e.g. author bio) is scored against the hop
        # sub-query that retrieved it, not the surface question.
        #
        # Append entity hints to the query side. The cross-encoder treats this
        # as "query talks about these entities", up-ranking chunks that
        # lexically contain them. Capped at top-5 entity hints to keep the
        # query short (cross-encoder context window).
        def _query_with_hints(base_q: str) -> str:
            if not entity_hints:
                return base_q
            useful = [
                e for e in entity_hints[:5]
                if e and len(e) >= 3 and e.lower() not in base_q.lower()
            ]
            if not useful:
                return base_q
            return f"{base_q} [ENTITIES: {', '.join(useful)}]"

        pairs = [
            (_query_with_hints(r.get("_best_sub_query") or query), r["text"])
            for r in candidates
        ]
        try:
            scores = self._reranker.predict(pairs)
            for r, s in zip(candidates, scores):
                r["reranker_score"] = float(s)
            reranked = sorted(
                candidates, key=lambda r: r.get("reranker_score", 0.0), reverse=True
            )
            logger.debug(
                "[Navigator] Reranker top chunk score=%.3f text=%s…",
                reranked[0].get("reranker_score", 0) if reranked else 0,
                reranked[0]["text"][:60] if reranked else "",
            )
            return reranked + rest
        except (RuntimeError, ValueError) as exc:
            logger.warning("[Navigator] Reranker inference failed: %s", exc)
            return results

    def _context_shrinkage(
        self,
        results: List[Dict[str, Any]],
        max_chars_per_chunk: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Context Shrinkage: trim each chunk to its most relevant sentences.

        Original contribution (paper section 3.3); see the per-filter
        ablation in the paper's results section for its S_V-latency reduction
        (toggle via config.enable_context_shrinkage). Entity-containing
        sentences carry the key facts.
        Edge optimization: smaller context = fewer input tokens = faster LLM
        inference on CPU. Directly reduces the LLM prompt size toward the 500-char
        budget set in llm.max_chars_per_doc.

        Strategy:
        1. Split into sentences (punctuation-based heuristic)
        2. Prioritize sentences containing named entities (capitalization proxy)
        3. Concatenate sentences until max_chars_per_chunk is reached
        """
        if max_chars_per_chunk is None:
            # Use config value (maps to llm.max_chars_per_doc in settings.yaml)
            max_chars_per_chunk = self.config.max_chars_per_doc

        if not results:
            return results

        def has_entity(s: str) -> bool:
            return bool(self._SHRINK_ENTITY_RE.search(s))

        shrunk = []
        for r in results:
            text = r["text"]
            if len(text) <= max_chars_per_chunk:
                shrunk.append(r)
                continue

            sentences = self._SENTENCE_SPLIT_RE.split(text)
            priority = [s for s in sentences if has_entity(s)]
            rest = [s for s in sentences if not has_entity(s)]

            result_text = ""
            for sent in priority + rest:
                if len(result_text) + len(sent) + 1 > max_chars_per_chunk:
                    break
                result_text = (result_text + " " + sent).strip()

            new_r = dict(r)
            new_r["text"] = result_text if result_text else text[:max_chars_per_chunk]
            shrunk.append(new_r)

        return shrunk


# =============================================================================
# MAIN (smoke test)
# =============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("=" * 70)
    print("NAVIGATOR SMOKE TEST")
    print("=" * 70)

    # ── Test 1: ControllerConfig.from_yaml ─────────────────────────────────
    print("\n--- ControllerConfig.from_yaml ---")
    sample_cfg = {
        "navigator": {"rrf_k": 30, "max_context_chunks": 5},
        "llm": {"model_name": "phi3", "max_chars_per_doc": 200},
        "agent": {"max_verification_iterations": 3},
    }
    cfg = ControllerConfig.from_yaml(sample_cfg)
    assert cfg.rrf_k == 30, f"expected 30, got {cfg.rrf_k}"
    assert cfg.max_context_chunks == 5
    assert cfg.max_chars_per_doc == 200
    assert cfg.max_verification_iterations == 3
    print("  ✓ from_yaml reads navigator/llm/agent blocks correctly")

    # ── Test 2: Navigator with mock retriever ──────────────────────────────
    print("\n--- Navigator with mock retriever ---")

    class _MockResult:
        def __init__(self, text: str, score: float, source: str):
            self.text = text
            self.rrf_score = score
            self.source = source

    class _MockRetriever:
        def retrieve(self, query: str, entity_hints=None):
            return (
                [
                    _MockResult("Paris is the capital of France.", 0.9, "doc_france"),
                    _MockResult("France is a country in Western Europe.", 0.7, "doc_france"),
                    _MockResult("The Eiffel Tower is in Paris.", 0.6, "doc_paris"),
                ],
                {"vector": 3, "graph": 0},
            )

    nav_cfg = ControllerConfig()
    nav = Navigator(nav_cfg)
    nav.set_retriever(_MockRetriever())

    from .planner import RetrievalPlan, QueryType, RetrievalStrategy

    plan = RetrievalPlan(
        original_query="What is the capital of France?",
        query_type=QueryType.SINGLE_HOP,
        strategy=RetrievalStrategy.HYBRID,
        sub_queries=["What is the capital of France?"],
    )

    nav_result = nav.navigate(
        retrieval_plan=plan,
        sub_queries=["What is the capital of France?"],
        entity_names=["France"],
    )

    assert isinstance(nav_result, NavigatorResult)
    assert len(nav_result.filtered_context) > 0, "Expected at least one filtered chunk"
    assert all("france" in c.lower() or "paris" in c.lower() for c in nav_result.filtered_context), \
        "Entity-mention filter should retain France/Paris chunks"
    print(f"  ✓ navigate() returned {len(nav_result.filtered_context)} chunk(s)")
    for c in nav_result.filtered_context:
        print(f"    · {c[:80]}")

    # _contradiction_filter is off by default (enable_contradiction_filter);
    # its smoke checks live in the dedicated test suite, not here.

    print("\n" + "=" * 70)
    print("All smoke tests passed.")
    print("Note: full pipeline test (Ollama) is in src/logic_layer/controller.py")
    print("=" * 70)
    sys.exit(0)
