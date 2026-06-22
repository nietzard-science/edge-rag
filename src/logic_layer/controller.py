"""
===============================================================================
AgenticController — static helpers for bridge-entity extraction and hop rewrite
===============================================================================

Paper: "Edge-RAG: Empirical Characterization of When Knowledge-Graph Lanes Add Value in CPU-Only Hybrid Retrieval"
Artifact B: Agent-Based Query Processing — utility helpers consumed by the
production pipeline (src/pipeline/agent_pipeline.py).

Role in the pipeline
--------------------
The production orchestrator of the three-agent pipeline (S_P → S_N → S_V)
is ``src.pipeline.agent_pipeline.AgentPipeline``. The class
``AgenticController`` in this module is a stateless namespace of utility
helpers used by ``AgentPipeline._iterative_navigate``; it does not
orchestrate a pipeline by itself.

For the full pipeline, use::

    from src.pipeline import AgentPipeline, create_full_pipeline
    pipeline = create_full_pipeline()
    result = pipeline.process("Your query")

Exports
-------
    AgenticController._extract_bridge_entities(chunks, exclude, query)
        Return up to ``TOP_K_BRIDGES`` candidate bridge entities for the
        next hop's sub-query rewrite. Two-pass heuristic:
          * Pass 0  Location-context regex (gated to GPE queries).
          * Passes 1 + 2 unified  Surname-anchor reconstruction +
            general proper-noun fallback, scored in a single pool by
            ``_score_bridge_candidate`` with a C4 abstention floor.
    AgenticController._rewrite_hop_query_with_bridges(sub_query, bridges)
        Inject resolved bridge entities into a Hop-N sub-query so the
        downstream retriever can locate the answer paragraph for the
        next hop.
    AgenticController._score_bridge_candidate(...)  — internal helper.
    AgenticController._detect_expected_type(query)  — internal helper.

References (algorithm anchors the helpers implement)
----------------------------------------------------
    Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion. SIGIR.

Dependencies
------------
    stdlib only (re, logging, typing).
    Internal: src.logic_layer._text_utils._PROPER_NOUN_RE.

Review History
--------------
    Last Reviewed: 2026-06-13
    Review Result: 0 CRITICAL, 0 IMPORTANT, 6 RECOMMENDED (all addressed)
    Reviewer: Code Review Prompt v2.1
    Next Review: after a change to the bridge-scoring signals or call sites
    ---
    Previous Review: 2026-05-26 (audit pass, v5.4)
    Changes Since: query-keyword tokenisation hoisted out of the per-candidate
        scoring loop (perf); C4 abstention now logs (was silent); reference
        note added on the corpus-derived role/surname heuristics; cosmetic
        regex tidy. No change to scoring behaviour or public signatures.
===============================================================================
"""

import logging
import re
from typing import ClassVar, Dict, List, NamedTuple, Optional, Set, Tuple

from ._text_utils import _PROPER_NOUN_RE

logger = logging.getLogger(__name__)


# =============================================================================
# INTERNAL TYPES
# =============================================================================

class _Proposal(NamedTuple):
    """A bridge-entity candidate paired with its source chunk and RRF rank."""
    text: str
    chunk: str
    rank: int


# =============================================================================
# AGENTIC CONTROLLER — static helpers only
# =============================================================================

class AgenticController:
    """
    Static-helper container for bridge-entity extraction and hop-query rewriting.

    This class is intentionally stateless. It exists as a namespace for class-
    level constants and ``@staticmethod`` / ``@classmethod`` helpers used by
    ``AgentPipeline._iterative_navigate``.

    Do NOT instantiate. Calling ``AgenticController()`` will succeed (no
    ``__init__`` is defined, so Python uses ``object.__init__``) but the
    resulting instance has no behaviour beyond the static helpers.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # HYPERPARAMETERS (class-level constants, ablatable via subclass override)
    # ─────────────────────────────────────────────────────────────────────────
    # These numerical weights govern the bridge-entity scoring function.
    # They were chosen by examining failure cases on the development split
    # and are exposed here as class-level constants so any of them can be
    # overridden in an ablation (subclass + override) without touching the
    # algorithm code itself.

    # Proximity decay: 1 / (1 + min_dist / PROXIMITY_LENGTH_SCALE). Smaller
    # value → steeper decay → tighter co-occurrence required between a
    # candidate and the query keywords inside the same chunk.
    PROXIMITY_LENGTH_SCALE: ClassVar[float] = 200.0  # characters

    # Position penalty: min(POSITION_PENALTY_MAX, cand_pos / POSITION_PENALTY_SCALE).
    # Penalises candidates that appear deep inside a chunk on the assumption
    # that the article topic is named near the beginning.
    POSITION_PENALTY_SCALE: ClassVar[float] = 600.0  # characters
    POSITION_PENALTY_MAX: ClassVar[float] = 0.5

    # Type-match weights. PERSON queries reward 2–3-token capitalised
    # candidates and penalise role/title nouns; GPE queries reward short
    # capitalised candidates.
    PERSON_TYPE_BONUS: ClassVar[float] = 0.5
    PERSON_ROLE_PENALTY: ClassVar[float] = -0.3
    GPE_TYPE_BONUS: ClassVar[float] = 0.3
    PERSON_TYPE_MIN_TOKENS: ClassVar[int] = 2
    PERSON_TYPE_MAX_TOKENS: ClassVar[int] = 3

    # Token-count preference: two-token names are the most frequent PERSON
    # surface form in English text; spans of ≥4 tokens are usually noisy
    # bracketed phrases rather than entity names.
    TWO_TOKEN_BONUS: ClassVar[float] = 0.10
    LONG_SPAN_PENALTY: ClassVar[float] = -0.20
    LONG_SPAN_THRESHOLD: ClassVar[int] = 4  # n_tokens ≥ this triggers the penalty

    # Surname-anchor pass: only fire on a known entity of 2–3 tokens whose
    # final token (the surname) is at least this many characters, to avoid
    # spurious one-letter or two-letter surname matches.
    SURNAME_MIN_LENGTH: ClassVar[int] = 6

    # Minimum length filter for proposed candidates (characters and tokens).
    MIN_PHRASE_LENGTH: ClassVar[int] = 4
    MIN_PLACE_LENGTH: ClassVar[int] = 4

    # Query-keyword filter: words shorter than this are dropped before
    # measuring proximity (cuts function words that the stoplist misses).
    QUERY_KEYWORD_MIN_LENGTH: ClassVar[int] = 3

    # Top-K cap applied to all return paths.
    TOP_K_BRIDGES: ClassVar[int] = 3

    # C4 abstention floor: if the best-scoring candidate is at or below this
    # value, return [] rather than mislead Hop-N retrieval with a confidently-
    # wrong bridge.
    ABSTAIN_BELOW_SCORE: ClassVar[float] = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # LEXICAL CONSTANTS (compiled once at class load)
    # ─────────────────────────────────────────────────────────────────────────

    # Query-keyword stopwords for relevance ranking. Non-content words removed
    # before measuring proximity between a candidate entity and the query
    # keywords in the source chunk.
    _QUERY_STOPWORDS: ClassVar[frozenset] = frozenset({
        "a", "an", "and", "are", "as", "at", "be", "by", "do", "does",
        "for", "from", "has", "have", "he", "her", "him", "his", "how",
        "in", "is", "it", "its", "of", "on", "or", "that", "the", "their",
        "they", "this", "to", "was", "were", "what", "when", "where",
        "which", "who", "whom", "whose", "why", "will", "with", "would",
        "you", "your", "i", "we", "us", "our", "she", "them",
    })

    # Expected entity-type hints from interrogative words.
    _QUERY_TYPE_HINTS: ClassVar[Dict[str, str]] = {
        "who": "PERSON", "whose": "PERSON", "whom": "PERSON",
        "where": "GPE",
        "when": "DATE",  # matched by digit pattern downstream, not name-style
    }

    # Generic stop tokens that must not appear at either end of a multi-token
    # entity span (closed-class English prepositions and articles). Used both
    # for substring-exclusion expansion and for surname-anchor sanity checks.
    _SPAN_STOP_TOKENS: ClassVar[frozenset] = frozenset({
        "in", "on", "of", "at", "by", "the", "a", "an",
    })

    # Tokens that, when present, mark a candidate as a role/title rather than
    # a person name. Ad-hoc closed list; sufficient for the corpora used in
    # this paper. Sourced from the manual error analysis of the dev split,
    # not from a published lexicon — extending this set is a code change.
    # (No published role/title gazetteer is used; this is a corpus-derived
    # heuristic and is documented as such for reproducibility. A learned
    # entity-type classifier would replace it in future work.)
    _ROLE_TOKENS: ClassVar[frozenset] = frozenset({
        "investigator", "director", "producer", "manager", "president",
        "chairman", "founder", "owner", "captain", "coach", "lawyer",
        "attorney", "officer", "secretary", "minister", "governor",
        "actor", "actress", "author", "writer", "composer",
        "pictures", "studios", "movie", "film", "company", "corporation",
        "investigations", "agency", "department", "division", "group",
        "industries", "limited", "incorporated", "associates",
        "private", "public",
    })

    # Set of leading tokens that must NEVER start a surname-anchor
    # reconstruction (would produce sentence-initial determiners or
    # prepositions parsed as a "first name").
    _SURNAME_ANCHOR_FORBIDDEN_LEADS: ClassVar[frozenset] = frozenset({
        "The", "A", "An", "This", "In", "Of",
    })

    # Location-context regex (Pass 0). Captures place names introduced by
    # location prepositions: "in the city of X", "capital of X", "in X".
    _LOCATION_CTX_RE: ClassVar[re.Pattern] = re.compile(
        r"(?:in\s+the\s+(?:city|town|village|capital|region|province|district)\s+of"
        r"|capital\s+of"
        r"|(?:in|at|near|of)\s+)"
        r"([A-Z][a-z]{2,}(?:[- ][A-Z][a-z]+)*)",
        re.UNICODE,
    )

    # Query-keyword tokeniser. Built once from QUERY_KEYWORD_MIN_LENGTH so the
    # constant is the single source of truth.
    _QUERY_KEYWORD_RE: ClassVar[re.Pattern] = re.compile(
        rf"\b\w{{{QUERY_KEYWORD_MIN_LENGTH},}}\b"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL HELPERS — each returns the contribution of a single signal so
    # ablations can disable any one without rewriting the scoring function.
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def _proximity_signal(
        cls, cand_pos: int, chunk_lower: str, query_keywords: List[str],
    ) -> float:
        """Distance-weighted query-keyword proximity. 0 if no keyword found."""
        min_dist = float("inf")
        for kw in query_keywords:
            kw_pos = chunk_lower.find(kw)
            if kw_pos >= 0:
                dist = abs(kw_pos - cand_pos)
                if dist < min_dist:
                    min_dist = dist
        if min_dist == float("inf"):
            return 0.0
        return 1.0 / (1.0 + min_dist / cls.PROXIMITY_LENGTH_SCALE)

    @classmethod
    def _type_signal(cls, candidate: str, expected_type: Optional[str]) -> float:
        """Type-match bonus or role-token penalty per ``expected_type``."""
        if expected_type == "PERSON":
            tokens = candidate.split()
            tokens_lower = [t.lower() for t in tokens]
            has_role_token = any(t in cls._ROLE_TOKENS for t in tokens_lower)
            if (
                cls.PERSON_TYPE_MIN_TOKENS <= len(tokens) <= cls.PERSON_TYPE_MAX_TOKENS
                and all(t[0].isupper() for t in tokens if t)
                and not has_role_token
            ):
                return cls.PERSON_TYPE_BONUS
            if has_role_token:
                return cls.PERSON_ROLE_PENALTY
            return 0.0
        if expected_type == "GPE":
            if len(candidate.split()) <= 2:
                return cls.GPE_TYPE_BONUS
        return 0.0

    @classmethod
    def _length_signal(cls, candidate: str) -> float:
        """Token-count preference (2-token bonus / long-span penalty)."""
        n_tokens = len(candidate.split())
        if n_tokens == 2:
            return cls.TWO_TOKEN_BONUS
        if n_tokens >= cls.LONG_SPAN_THRESHOLD:
            return cls.LONG_SPAN_PENALTY
        return 0.0

    @classmethod
    def _position_penalty(cls, cand_pos: int) -> float:
        """Capped linear penalty for candidates far from the chunk start."""
        return min(cls.POSITION_PENALTY_MAX, cand_pos / cls.POSITION_PENALTY_SCALE)

    @classmethod
    def _rank_prior(cls, chunk_rank: int) -> float:
        """Reciprocal chunk-rank prior — the same primitive RRF uses
        (Cormack et al. 2009, SIGIR)."""
        return 1.0 / (1.0 + max(0, chunk_rank))

    @classmethod
    def _query_keywords(cls, query: str) -> List[str]:
        """Lower-cased content keywords of ``query`` (stopwords removed).

        Hoisted out of ``_score_bridge_candidate`` so the regex + stopword
        filter runs once per ``_extract_bridge_entities`` call rather than
        once per candidate proposal (the candidate pool can be large when the
        general proper-noun generator fires on every chunk).
        """
        return [
            w.lower() for w in cls._QUERY_KEYWORD_RE.findall(query)
            if w.lower() not in cls._QUERY_STOPWORDS
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # BRIDGE-ENTITY SCORING
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def _score_bridge_candidate(
        cls,
        candidate: str,
        chunk: str,
        query: str,
        expected_type: Optional[str] = None,
        chunk_rank: int = 0,
        query_keywords: Optional[List[str]] = None,
    ) -> float:
        """
        Relevance score for a bridge-entity candidate.

        Combines four signals (each implemented as its own helper to keep
        the function ablatable without restructuring):

            +α  ``_proximity_signal``    co-occurrence with query keywords
            +β  ``_type_signal``         expected-type match / role-penalty
            +γ  ``_length_signal``       token-count preference
            -δ  ``_position_penalty``    distance from chunk start
            ×ε  ``_rank_prior``          reciprocal RRF chunk rank

        ``query_keywords`` may be passed pre-computed (via ``_query_keywords``)
        to avoid re-tokenising the query once per candidate; when ``None`` it
        is derived from ``query`` here so the function is still correct when
        called standalone.

        Returns a non-negative float. Higher = more likely the bridge; 0
        means "not found / no query proximity" (used by the C4 abstention
        floor in ``_extract_bridge_entities``).
        """
        chunk_lower = chunk.lower()
        cand_pos = chunk_lower.find(candidate.lower())
        if cand_pos < 0:
            return 0.0

        if query_keywords is None:
            query_keywords = cls._query_keywords(query)
        proximity = cls._proximity_signal(cand_pos, chunk_lower, query_keywords)

        local_score = proximity

        # Type and length features MODULATE a candidate only when it has
        # lexical proximity to the question. Absent proximity, a candidate
        # has no positive evidence of being the answer (it merely "looks
        # like" the right type or length), so it must not clear the C4
        # abstention floor on type/length alone.
        if proximity > 0.0:
            local_score += cls._type_signal(candidate, expected_type)
            local_score += cls._length_signal(candidate)

        local_score -= cls._position_penalty(cand_pos)

        return max(0.0, local_score) * cls._rank_prior(chunk_rank)

    @classmethod
    def _detect_expected_type(cls, query: str) -> Optional[str]:
        """
        Infer the expected bridge-entity type from interrogative words.

        Returns the first matching hint type, or ``None`` if the query has
        no recognised question word. Conservative — only fires on canonical
        question starters.

        Note: returns the *first* interrogative match in the query. A
        compound question that mixes interrogatives (e.g. mixing "where"
        and "who") receives the type associated with whichever word the
        scanner sees first, which may not always be the type of the
        intended bridge. Callers that need richer typing should call
        ``_score_bridge_candidate`` with ``expected_type=None`` and rely
        on proximity + length signals alone.
        """
        q = query.lower().strip()
        for prefix, etype in cls._QUERY_TYPE_HINTS.items():
            if q.startswith(prefix + " ") or f" {prefix} " in q:
                return etype
        # Special case: "the actress who"/"the director who" → PERSON
        if re.search(r"\bthe\s+\w+\s+who\b", q):
            return "PERSON"
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # BRIDGE-ENTITY EXTRACTION (Pass 0 then unified Pass 1 + 2)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_bridge_entities(
        chunks: List[str], exclude: List[str], query: str = "",
    ) -> List[str]:
        """
        Extract candidate bridge entity names from retrieved text chunks.

        Two-pass heuristic:

        Pass 0 — Location-context extraction (highest priority, GPE-only):
          Scans the top chunk for place names introduced by location
          prepositions ("in the city of X", "capital of X", "in X").
          Gated to queries whose expected type is GPE.

        Passes 1 + 2 — Unified candidate pool:
          A surname-anchor reconstructor (recovers family-name variants
          whose middle tokens contain non-ASCII diacritics) and a general
          proper-noun fallback (``_PROPER_NOUN_RE``) propose candidates
          into a single scoring pool. ``_score_bridge_candidate`` ranks
          them and the top ``TOP_K_BRIDGES`` are returned, subject to the
          C4 abstention floor.
        """
        cls = AgenticController

        exclude_lower: Set[str] = {e.lower() for e in exclude}
        seen: Set[str] = set()
        candidates: List[str] = []

        # ── Pass 0: location-context extraction (GPE queries only) ───────
        expected_type = cls._detect_expected_type(query)
        if chunks and expected_type == "GPE":
            for m in cls._LOCATION_CTX_RE.finditer(chunks[0]):
                place = m.group(1).strip()
                place_lower = place.lower()
                if (
                    place_lower not in exclude_lower
                    and place_lower not in seen
                    and len(place) >= cls.MIN_PLACE_LENGTH
                ):
                    seen.add(place_lower)
                    candidates.append(place)

        if candidates:
            return candidates[: cls.TOP_K_BRIDGES]

        # ── Passes 1 & 2 unified: one confidence-scored candidate pool ───
        # The former design ran a surname-anchor pass that returned early
        # on first match, so a low-precision reconstruction (e.g. a span
        # spuriously built from an exclude entity's surname) could preempt
        # a stronger general-proper-noun candidate that the second pass
        # would have found. Scoring both generators in a single pool removes
        # that priority-on-specificity short-circuit: candidates compete on
        # the same scoring function and the strongest wins regardless of
        # which generator proposed it.

        # Substring-aware exclusion: a compound exclude entity also excludes
        # its multi-token sub-phrases so a partial variant cannot be
        # proposed as a "new" bridge.
        excluded_subphrases: Set[str] = set(exclude_lower)
        for known in exclude:
            tokens = known.split()
            for i in range(len(tokens)):
                for j in range(i + 2, len(tokens) + 1):
                    if tokens[i].lower() in cls._SPAN_STOP_TOKENS:
                        continue
                    if tokens[j - 1].lower() in cls._SPAN_STOP_TOKENS:
                        continue
                    excluded_subphrases.add(" ".join(tokens[i:j]).lower())

        # Candidate generators → (text, source_chunk, chunk_rank).
        # Chunks arrive in RRF rank order, so the list index IS the rank.
        proposals: List[_Proposal] = []

        # Generator A — surname-anchor reconstruction.
        for known in exclude:
            tokens = known.split()
            if len(tokens) not in (
                cls.PERSON_TYPE_MIN_TOKENS, cls.PERSON_TYPE_MAX_TOKENS,
            ):
                continue
            if any(t.lower() in cls._SPAN_STOP_TOKENS for t in tokens):
                continue
            surname = tokens[-1]
            if len(surname) < cls.SURNAME_MIN_LENGTH:
                continue
            pat = re.compile(
                r"\b([A-Z][^\s,.()\[\]]+)\s+(?:[A-Z][^\s,.()\[\]]+\s+)?"
                + re.escape(surname)
                + r"\b",
                re.UNICODE,
            )
            for rank, chunk in enumerate(chunks):
                for m in pat.finditer(chunk):
                    first = m.group(1)
                    full = f"{first} {surname}"
                    if (
                        len(full) > cls.MIN_PHRASE_LENGTH
                        and first not in cls._SURNAME_ANCHOR_FORBIDDEN_LEADS
                        and ":" not in first
                    ):
                        proposals.append(_Proposal(full, chunk, rank))

        # Generator B — general proper-noun fallback.
        for rank, chunk in enumerate(chunks):
            for m in _PROPER_NOUN_RE.finditer(chunk):
                phrase = m.group(1)
                if len(phrase) > cls.MIN_PHRASE_LENGTH:
                    proposals.append(_Proposal(phrase, chunk, rank))

        # Score the merged pool with a single function; keep the best-
        # scoring instance of each distinct candidate. Query keywords are
        # tokenised ONCE here and reused for every candidate (was previously
        # recomputed per candidate inside _score_bridge_candidate).
        query_keywords = cls._query_keywords(query)
        best: Dict[str, Tuple[float, str]] = {}
        for prop in proposals:
            cl_ = prop.text.lower()
            if cl_ in excluded_subphrases or cl_ in seen:
                continue
            score = cls._score_bridge_candidate(
                prop.text, prop.chunk, query, expected_type,
                chunk_rank=prop.rank, query_keywords=query_keywords,
            )
            if cl_ not in best or score > best[cl_][0]:
                best[cl_] = (score, prop.text)

        scored = sorted(best.values(), key=lambda x: -x[0])

        # ── C4: abstention floor ─────────────────────────────────────────
        # A candidate at or below ``ABSTAIN_BELOW_SCORE`` was either not
        # found in its chunk or had no query proximity. Returning a
        # confidently-wrong bridge actively misdirects Hop-N retrieval
        # (and reranker hints), which is worse than returning none — the
        # next hop then falls back to its un-rewritten sub-query.
        if not scored or scored[0][0] <= cls.ABSTAIN_BELOW_SCORE:
            # Observable no-op: no candidate cleared the floor, so Hop-N runs
            # on its un-rewritten sub-query. Logged (not silent) to match the
            # house convention for non-gating fall-throughs (cf. navigator
            # ``_entity_filter_skipped``).
            logger.debug(
                "[BridgeExtract] abstained: %d candidate(s), best score "
                "%.3f ≤ floor %.3f → returning no bridge",
                len(scored),
                scored[0][0] if scored else 0.0,
                cls.ABSTAIN_BELOW_SCORE,
            )
            return []
        return [
            text for score, text in scored[: cls.TOP_K_BRIDGES]
            if score > cls.ABSTAIN_BELOW_SCORE
        ]

    @staticmethod
    def _rewrite_hop_query_with_bridges(
        sub_query: str, bridges: List[str],
    ) -> str:
        """
        Inject resolved bridge entities into a hop's sub-query.

        Iterative multi-hop retrieval relies on each hop being able to find
        its supporting paragraph in the index. Planner-generated sub-queries
        are written before retrieval runs, so a Hop-2 sub-query is usually
        under-specified — it asks an attribute question without naming the
        entity, because the entity is only known after Hop-1 has returned a
        result. Once Hop-1 resolves a bridge entity, the Hop-2 sub-query
        must be rewritten to include that entity name before retrieval
        runs, or the index lookup misses the answer paragraph entirely.

        Heuristic (conservative — only fires when SAFE):
          - If sub_query already mentions any of the bridge entities, do
            nothing (no double-injection).
          - Otherwise append ``" — about <bridge_1>, <bridge_2>, ..."`` to
            the sub-query, capped at ``TOP_K_BRIDGES`` bridges.
        """
        cls = AgenticController
        if not bridges or not sub_query:
            return sub_query
        sq_lower = sub_query.lower()
        new_bridges = [b for b in bridges if b.lower() not in sq_lower]
        if not new_bridges:
            return sub_query
        injection = ", ".join(new_bridges[: cls.TOP_K_BRIDGES])
        rewritten = f"{sub_query.rstrip(' ?.')} — about {injection}"
        logger.debug(
            "[BridgeRewrite] %r → %r",
            sub_query[:60], rewritten[:80],
        )
        return rewritten
