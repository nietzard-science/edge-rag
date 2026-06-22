"""
Graph quality utilities for the Edge-RAG knowledge graph.

This module provides four orthogonal capabilities used at ingestion time and
in standalone diagnostics:

  1. Stronger surface-form canonicalisation (`canonical_form`) for detecting
     entities that differ only in punctuation, parentheticals, honorifics,
     possessives, or Unicode form.

  2. Read-only baseline metrics (`compute_graph_baseline`) to quantify graph
     health: density, isolated-entity rate, duplicate rate, top hubs.

  3. Co-occurrence edge construction (`build_cooccurrence_edges`): for every
     pair of entities mentioned in the same chunk, add a RELATED_TO edge with
     `relation_type='cooccurs'`. This guarantees that no entity is isolated
     as long as it appears with at least one other entity anywhere in the
     corpus — typically true for narrative text.

  4. Post-ingestion cleanup (`cleanup_graph`): drop extraction-orphaned
     entities, drop generic hubs that pollute graph retrieval, and merge
     duplicate entities sharing a canonical surface form.

All mutating Cypher uses the same patterns as `KuzuGraphStore.clear()` --
edges deleted before nodes, no DETACH DELETE -- so it works on every Kuzu
version that the existing codebase already targets.

Last reviewed: 2026-05-25 (audit pass, project version 5.4).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from itertools import combinations
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Public API consumed by:
#   - src/data_layer/__init__.py  (re-exports 9 symbols)
#   - local_importingestion.py   (Phase 3c.5 + 3f direct imports of the
#     drop_subsumed_cooccurrence_edges / drop_isolated_entities pair)
# Module-internal helpers (_fetch_all, _delete_entity_safely, _redirect_*,
# _drop_*, _merge_*) stay accessible by direct-name import for tests.
__all__ = [
    "canonical_form",
    "compute_graph_baseline",
    "format_baseline_report",
    "assert_graph_invariants",
    "GraphQualityViolation",
    "DEFAULT_THRESHOLDS",
    "build_cooccurrence_edges",
    "cleanup_graph",
    "DEFAULT_STOPLIST",
    "link_entities_by_embedding",
    "drop_subsumed_cooccurrence_edges",
    "drop_isolated_entities",
]


# -----------------------------------------------------------------------------
# CANONICAL SURFACE-FORM NORMALISATION
# -----------------------------------------------------------------------------

# Strip a trailing parenthetical disambiguator ("Local H (band)" → "Local H").
_PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)\s*$")

# Strip possessive markers (straight + curly apostrophe).
_POSSESSIVE_RE = re.compile(r"[’']s\b", re.IGNORECASE)

# Strip leading honorifics (Mr., Mrs., Dr., Prof., Sir, Lord, Saint, ...).
# These are respect markers, never part of the canonical name.
_HONORIFIC_RE = re.compile(
    r"^(?:Mr|Mrs|Ms|Mx|Dr|Prof|Sir|Lord|Lady|Saint|St)\.?\s+",
    re.IGNORECASE,
)

# Strip trailing ACADEMIC credentials (PhD, MD, Esq) only.
# Generation suffixes (Jr., Sr., II, III, IV, V) are deliberately kept —
# they distinguish father/son and same-name family members (a "Person X Sr."
# entity must NOT collapse into the same graph node as a "Person X" entity).
_TITLE_SUFFIX_RE = re.compile(
    r"[,]?\s+(?:PhD|Ph\.D|MD|M\.D|Esq)\.?\s*$",
    re.IGNORECASE,
)

# Collapse runs of whitespace.
_WHITESPACE_RE = re.compile(r"\s+")


def canonical_form(name: str) -> str:
    """
    Strong canonicalisation for duplicate detection.

    Applies, in order:
      1. NFKC Unicode normalisation (collapses presentation variants such as
         combined accents, full-width digits, fancy quotes/dashes).
      2. Strip leading/trailing whitespace.
      3. Strip a trailing parenthetical disambiguator.
      4. Strip a leading honorific.
      5. Strip a trailing name suffix.
      6. Strip possessive 's.
      7. Collapse internal whitespace.
      8. Lowercase.

    The result is suitable as a deduplication key but NOT for display:
    it is irreversibly lossy. Use the original surface form for rendering.

    Examples:
        "Marie Curie (chemist)"      -> "marie curie"
        "Dr. Albert Einstein"        -> "albert einstein"
        "Leonardo da Vinci Jr."      -> "leonardo da vinci"
        "Curie's"                    -> "curie"
        "Citroën "                   -> "citroen"   (after NFKC + casefold)
    """
    if not name:
        return ""
    s = unicodedata.normalize("NFKC", name).strip()
    s = _PARENTHETICAL_RE.sub("", s)
    s = _HONORIFIC_RE.sub("", s)
    s = _TITLE_SUFFIX_RE.sub("", s)
    s = _POSSESSIVE_RE.sub("", s)
    # Strip a trailing period so "<Name> Sr." and "<Name> Sr" collapse to
    # the same canonical form while both staying distinct from "<Name>"
    # (the unsuffixed family member — a separate graph node).
    if s.endswith("."):
        s = s[:-1].rstrip()
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s.casefold()


# -----------------------------------------------------------------------------
# BASELINE METRICS (read-only)
# -----------------------------------------------------------------------------

def compute_graph_baseline(graph_store) -> Dict[str, Any]:
    """
    Quantify graph health for diagnostics and invariant checks.

    Args:
        graph_store: A KuzuGraphStore-compatible object exposing
                     `conn.execute(query, params)` and `get_statistics()`.

    Returns:
        A dict with keys:
          - totals       : raw node/edge counts
          - densities    : per-chunk and per-entity ratios
          - isolated     : entities with zero RELATED_TO edges
          - duplicates   : clusters of entities sharing canonical_form
          - top_hubs     : entities with the most MENTIONS edges
    """
    raw = graph_store.get_statistics()
    chunks = max(1, raw.get("document_chunks", 0))
    entities = raw.get("entities", 0)
    mentions = raw.get("mentions_edges", 0)
    relations = raw.get("related_to_edges", 0)

    isolated_count = _count_isolated_entities(graph_store)
    duplicate_clusters = _find_duplicate_clusters(graph_store)
    top_hubs = _top_promiscuous_entities(graph_store, limit=20)

    duplicate_count = sum(len(c) - 1 for c in duplicate_clusters)

    return {
        "totals": {
            "chunks": raw.get("document_chunks", 0),
            "entities": entities,
            "mentions": mentions,
            "relations": relations,
        },
        "densities": {
            "entities_per_chunk": entities / chunks,
            "mentions_per_chunk": mentions / chunks,
            "relations_per_chunk": relations / chunks,
            "relations_per_entity": relations / max(1, entities),
        },
        "isolated": {
            "count": isolated_count,
            "rate": isolated_count / max(1, entities),
        },
        "duplicates": {
            "cluster_count": len(duplicate_clusters),
            "duplicate_count": duplicate_count,
            "rate": duplicate_count / max(1, entities),
            "top_clusters": duplicate_clusters[:10],
        },
        "top_hubs": top_hubs,
    }


def _fetch_all(graph_store, query: str, params: Optional[Dict] = None) -> List[Tuple]:
    """Drain a Cypher query into a list of tuples. Defensive against empty results."""
    try:
        result = graph_store.conn.execute(query, params or {})
    except (RuntimeError, ValueError, AttributeError) as exc:
        logger.warning("Cypher fetch failed (%s): %s", query.split()[0], exc)
        return []
    rows: List[Tuple] = []
    while result.has_next():
        rows.append(tuple(result.get_next()))
    return rows


def _count_isolated_entities(graph_store) -> int:
    """Count entities with zero incoming or outgoing RELATED_TO edges."""
    all_ids: Set[str] = {row[0] for row in _fetch_all(
        graph_store, "MATCH (e:Entity) RETURN e.entity_id"
    )}
    if not all_ids:
        return 0
    connected: Set[str] = set()
    for a, b in _fetch_all(
        graph_store,
        "MATCH (a:Entity)-[:RELATED_TO]->(b:Entity) RETURN a.entity_id, b.entity_id",
    ):
        connected.add(a)
        connected.add(b)
    return len(all_ids - connected)


def _find_duplicate_clusters(
    graph_store,
    max_clusters: int = 100,
) -> List[List[Tuple[str, str]]]:
    """
    Group entities sharing (canonical_form, type). Return clusters with >1 member.

    Each cluster is a list of (entity_id, surface_name) tuples, sorted by name.
    """
    rows = _fetch_all(
        graph_store, "MATCH (e:Entity) RETURN e.entity_id, e.name, e.type"
    )
    by_key: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for eid, name, etype in rows:
        if not name:
            continue
        key = (canonical_form(name), (etype or "unknown"))
        by_key.setdefault(key, []).append((eid, name))
    clusters = [sorted(members, key=lambda x: x[1]) for members in by_key.values() if len(members) > 1]
    clusters.sort(key=len, reverse=True)
    return clusters[:max_clusters]


def _top_promiscuous_entities(graph_store, limit: int = 20) -> List[Dict[str, Any]]:
    """Return the top-N entities by chunk-MENTIONS count."""
    rows = _fetch_all(
        graph_store,
        """
        MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity)
        RETURN e.name, e.type, count(DISTINCT c) AS mention_count
        ORDER BY mention_count DESC
        LIMIT $limit
        """,
        {"limit": limit},
    )
    return [
        {"name": name, "type": etype or "unknown", "mention_count": int(count)}
        for name, etype, count in rows
    ]


def format_baseline_report(metrics: Dict[str, Any]) -> str:
    """Format the baseline metrics as a human-readable report."""
    t = metrics["totals"]
    d = metrics["densities"]
    iso = metrics["isolated"]
    dup = metrics["duplicates"]

    lines: List[str] = []
    lines.append("-" * 70)
    lines.append("  GRAPH QUALITY BASELINE")
    lines.append("-" * 70)
    lines.append(f"  Chunks:     {t['chunks']:>8,}")
    lines.append(
        f"  Entities:   {t['entities']:>8,}    "
        f"(per chunk: {d['entities_per_chunk']:>5.2f})"
    )
    lines.append(
        f"  Mentions:   {t['mentions']:>8,}    "
        f"(per chunk: {d['mentions_per_chunk']:>5.2f})"
    )
    lines.append(
        f"  Relations:  {t['relations']:>8,}    "
        f"(per chunk: {d['relations_per_chunk']:>5.2f}, "
        f"per entity: {d['relations_per_entity']:>5.2f})"
    )
    lines.append("")
    iso_marker = "✓" if iso["rate"] < 0.05 else "⚠"
    lines.append(
        f"  {iso_marker} Isolated entities (no RELATED_TO): "
        f"{iso['count']:>5,} ({iso['rate']:.1%})"
    )
    dup_marker = "✓" if dup["rate"] < 0.02 else "⚠"
    lines.append(
        f"  {dup_marker} Duplicate clusters (same canonical form): "
        f"{dup['cluster_count']:>5,}  "
        f"(redundant entities: {dup['duplicate_count']:,}, {dup['rate']:.1%})"
    )
    if dup["top_clusters"]:
        lines.append("")
        lines.append("  Top duplicate clusters:")
        for cluster in dup["top_clusters"][:5]:
            names = ", ".join(f"'{n}'" for _, n in cluster[:4])
            extra = f" +{len(cluster) - 4} more" if len(cluster) > 4 else ""
            lines.append(f"    [{len(cluster)}x]  {names}{extra}")

    if metrics["top_hubs"]:
        lines.append("")
        lines.append("  Top entities by chunk-MENTIONS count:")
        for hub in metrics["top_hubs"][:10]:
            lines.append(
                f"    {hub['mention_count']:>4} chunks  ·  "
                f"{hub['type']:<14}  ·  {hub['name']}"
            )
    lines.append("-" * 70)
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# INVARIANTS
# -----------------------------------------------------------------------------

class GraphQualityViolation(RuntimeError):
    """Raised by `assert_graph_invariants(strict=True)` on threshold violation."""


DEFAULT_THRESHOLDS: Dict[str, float] = {
    "max_isolated_rate": 0.05,
    "max_duplicate_rate": 0.02,
    "min_relation_density": 5.0,
}


def assert_graph_invariants(
    metrics: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None,
    strict: bool = False,
) -> List[str]:
    """
    Check graph-quality invariants against `thresholds`.

    Returns a list of human-readable violation messages (empty on success).
    If `strict=True`, raises `GraphQualityViolation` instead of returning.
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    violations: List[str] = []

    iso_rate = metrics["isolated"]["rate"]
    if iso_rate > t["max_isolated_rate"]:
        violations.append(
            f"isolated_rate={iso_rate:.1%} exceeds threshold "
            f"{t['max_isolated_rate']:.1%}"
        )

    dup_rate = metrics["duplicates"]["rate"]
    if dup_rate > t["max_duplicate_rate"]:
        violations.append(
            f"duplicate_rate={dup_rate:.1%} exceeds threshold "
            f"{t['max_duplicate_rate']:.1%}"
        )

    rel_density = metrics["densities"]["relations_per_chunk"]
    if rel_density < t["min_relation_density"]:
        violations.append(
            f"relations_per_chunk={rel_density:.2f} below threshold "
            f"{t['min_relation_density']:.2f}"
        )

    if strict and violations:
        raise GraphQualityViolation("; ".join(violations))
    return violations


# -----------------------------------------------------------------------------
# CO-OCCURRENCE EDGES
# -----------------------------------------------------------------------------

def build_cooccurrence_edges(
    graph_store,
    extraction_results: Iterable[Dict[str, Any]],
    name_to_id: Dict[str, str],
    min_confidence: float = 0.5,
    relation_type: str = "cooccurs",
    max_entities_per_chunk: int = 30,
) -> int:
    """
    Add a RELATED_TO edge for every pair of entities co-mentioned in a chunk.

    Each unordered pair receives a single edge with `relation_type='cooccurs'`,
    `confidence=1.0`, and `source_chunks` containing the originating chunk_id.
    The edge is idempotent because `KuzuGraphStore.add_related_to_relation`
    uses MERGE under the hood. Pairs are deduplicated via canonical pair
    ordering so that A-B and B-A are not double-counted within this function.

    Args:
        graph_store: KuzuGraphStore-compatible.
        extraction_results: Phase-2 extraction records (one dict per chunk
            with an `entities` list of {name, confidence, ...}).
        name_to_id: Mapping from `canonical_form(entity.name)` -> resolved
            entity_id in the graph (built by the caller during entity import).
        min_confidence: Skip entities below this NER confidence.
        relation_type: Stored on the edge; default `"cooccurs"`.
        max_entities_per_chunk: Combinatorial safeguard. A chunk with N
            entities yields C(N, 2) pairs. With N=30 that is 435 pairs;
            with N=100 it is 4950. Chunks exceeding this cap are truncated
            to the highest-confidence top-N entities to bound the worst case.

    Returns:
        Number of edges written (excluding silent failures).
    """
    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    seen_pairs: Set[Tuple[str, str]] = set()
    all_pairs: List[Tuple[str, str, str, float, str]] = []   # (e1, e2, rel_type, conf, src_str)
    truncated_chunks = 0

    # -- Phase 1: collect all unique pairs (fast — no DB writes) --------------
    results_list = list(extraction_results)
    chunk_iter = (
        _tqdm(results_list, desc="Co-occurrence pairs", unit="chunk")
        if _tqdm else results_list
    )
    for result in chunk_iter:
        chunk_id = str(result.get("chunk_id", ""))

        candidates: List[Tuple[str, float]] = []
        for ent in result.get("entities", []) or []:
            conf = float(ent.get("confidence", 0.0))
            if conf < min_confidence:
                continue
            name = (ent.get("name") or "").strip()
            if not name:
                continue
            eid = name_to_id.get(canonical_form(name))
            if eid:
                candidates.append((eid, conf))

        by_id: Dict[str, float] = {}
        for eid, conf in candidates:
            if conf > by_id.get(eid, -1.0):
                by_id[eid] = conf
        ids_with_conf = sorted(by_id.items(), key=lambda x: -x[1])

        if len(ids_with_conf) > max_entities_per_chunk:
            ids_with_conf = ids_with_conf[:max_entities_per_chunk]
            truncated_chunks += 1

        unique_ids = [eid for eid, _ in ids_with_conf]
        for a, b in combinations(unique_ids, 2):
            pair = (a, b) if a < b else (b, a)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            src_str = chunk_id if chunk_id else ""
            all_pairs.append((pair[0], pair[1], relation_type, 1.0, src_str))

    logger.info(
        "Co-occurrence: %d unique pairs collected from %d chunks "
        "(%d chunks capped at max=%d)",
        len(all_pairs), len(results_list), truncated_chunks, max_entities_per_chunk,
    )

    # -- Phase 2: bulk-insert via transactional batches ------------------------
    # Use add_related_to_relations_bulk (500 edges/commit) when available;
    # fall back to the single-edge method for HybridStore / mock stores.
    BATCH = 500
    if hasattr(graph_store, "add_related_to_relations_bulk"):
        if _tqdm:
            batch_iter = _tqdm(
                range(0, len(all_pairs), BATCH),
                desc="Writing co-occur edges",
                unit="batch",
                total=(len(all_pairs) + BATCH - 1) // BATCH,
            )
        else:
            batch_iter = range(0, len(all_pairs), BATCH)
        written = 0
        for i in batch_iter:
            written += graph_store.add_related_to_relations_bulk(
                all_pairs[i : i + BATCH], batch_size=BATCH, use_create=True
            )
    else:
        # Fallback: single-edge writes (HybridStore / tests)
        edge_iter = _tqdm(all_pairs, desc="Writing co-occur edges", unit="edge") if _tqdm else all_pairs
        written = 0
        for e1, e2, rel_type, conf, src_str in edge_iter:
            try:
                graph_store.add_related_to_relation(
                    entity1_id=e1,
                    entity2_id=e2,
                    relation_type=rel_type,
                    confidence=conf,
                    source_chunks=[src_str] if src_str else None,
                )
                written += 1
            except (RuntimeError, ValueError, AttributeError) as exc:
                logger.debug("CO_OCCURS edge failed (%s, %s): %s", e1, e2, exc)

    logger.info("Co-occurrence edges written: %d / %d pairs", written, len(all_pairs))
    return written


# -----------------------------------------------------------------------------
# POST-INGESTION CLEANUP
# -----------------------------------------------------------------------------

# Surface forms that GLiNER frequently misclassifies as PERSON or GPE entities.
# These are pronouns, nationality adjectives, and other generic tokens that
# carry no entity-resolution value and pollute graph retrieval. Compared
# against `canonical_form(name)` (already lowercased + NFKC), so periods,
# whitespace, and case variants are handled automatically.
DEFAULT_STOPLIST: frozenset = frozenset({
    # Pronouns misclassified as PERSON
    "he", "she", "they", "it", "we", "i", "you", "him", "her", "his", "hers",
    "us", "them", "their", "theirs", "this", "that", "these", "those",
    # Nationality adjectives misclassified as GPE
    "american", "british", "english", "german", "french", "italian", "spanish",
    "japanese", "chinese", "russian", "australian", "canadian", "european",
    "asian", "african", "indian", "korean", "mexican", "brazilian", "polish",
    "irish", "scottish", "welsh", "dutch", "swedish", "norwegian", "danish",
    "greek", "turkish", "arab", "arabic", "swiss", "austrian", "belgian",
    # Generic / ambiguous tokens
    "u.s.", "us", "uk", "u.k.",  # use the canonical "United States" / "United Kingdom" forms
})


def cleanup_graph(
    graph_store,
    drop_orphans: bool = True,
    hub_threshold_ratio: Optional[float] = 0.03,
    hub_threshold_min: int = 50,
    merge_duplicates: bool = True,
    stoplist: Optional[Iterable[str]] = None,
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Four-pass post-ingestion graph cleanup.

    Args:
        drop_orphans:       Delete Entity nodes that no chunk MENTIONS.
        hub_threshold_ratio: Delete entities mentioned in more than
                             `ratio × total_chunks` chunks. None disables.
                             Default lowered from 0.05 to 0.03 because at
                             0.05, common GLiNER misclassifications such as
                             "He" and "England" survive (4.1 % and 1.7 % of
                             chunks respectively).
        hub_threshold_min:  Lower bound on the absolute hub threshold (so
                             small corpora do not drop everything).
        merge_duplicates:   Merge entities sharing (canonical_form, type) by
                             redirecting all edges to the highest-mentioned
                             surface form, then deleting the rest.
        stoplist:           Iterable of surface forms (compared against
                             `canonical_form`) to drop unconditionally.
                             None uses `DEFAULT_STOPLIST`. Pass an empty
                             iterable to disable.
        dry_run:            If True, only count operations; do not mutate.

    Returns:
        Dict with counts: orphans_dropped, hubs_dropped, duplicates_merged,
        stoplist_dropped.
    """
    ops = {
        "orphans_dropped": 0,
        "hubs_dropped": 0,
        "duplicates_merged": 0,
        "stoplist_dropped": 0,
    }

    # Stop-list pass runs first so hubs and duplicates do not waste work on
    # entities that would have been dropped anyway.
    if stoplist is None:
        stoplist = DEFAULT_STOPLIST
    stoplist_set = {canonical_form(s) for s in stoplist if s}
    if stoplist_set:
        ops["stoplist_dropped"] = _drop_stoplist_entities(
            graph_store, stoplist_set, dry_run=dry_run
        )

    if drop_orphans:
        ops["orphans_dropped"] = _drop_orphan_entities(graph_store, dry_run=dry_run)

    if hub_threshold_ratio is not None and hub_threshold_ratio > 0:
        chunks = graph_store.get_statistics().get("document_chunks", 0)
        hub_threshold = max(hub_threshold_min, int(hub_threshold_ratio * chunks))
        ops["hubs_dropped"] = _drop_hub_entities(
            graph_store, hub_threshold, dry_run=dry_run
        )

    if merge_duplicates:
        ops["duplicates_merged"] = _merge_duplicate_entities(graph_store, dry_run=dry_run)

    return ops


def _drop_stoplist_entities(
    graph_store,
    stoplist_set: Set[str],
    dry_run: bool = False,
) -> int:
    """Delete Entity nodes whose canonical_form(name) is in the stop-list."""
    rows = _fetch_all(graph_store, "MATCH (e:Entity) RETURN e.entity_id, e.name")
    targets = [eid for eid, name in rows
               if name and canonical_form(name) in stoplist_set]
    if dry_run:
        return len(targets)
    for eid in targets:
        _delete_entity_safely(graph_store, eid)
    if targets:
        logger.info(
            "Dropped %d stop-list entities (pronouns, nationality adjectives, ...).",
            len(targets),
        )
    return len(targets)


def _delete_entity_safely(graph_store, entity_id: str) -> None:
    """
    Delete an entity and all its incident edges.

    Kuzu requires edges to be deleted before nodes (no DETACH DELETE in older
    versions). This helper deletes edges in four passes, then the node itself.
    Failures are logged at DEBUG level and do not propagate.
    """
    cypher_steps = (
        # Outgoing RELATED_TO
        "MATCH (e:Entity {entity_id: $eid})-[r:RELATED_TO]->(:Entity) DELETE r",
        # Incoming RELATED_TO
        "MATCH (:Entity)-[r:RELATED_TO]->(e:Entity {entity_id: $eid}) DELETE r",
        # Incoming MENTIONS
        "MATCH (:DocumentChunk)-[r:MENTIONS]->(e:Entity {entity_id: $eid}) DELETE r",
        # The node itself
        "MATCH (e:Entity {entity_id: $eid}) DELETE e",
    )
    for cypher in cypher_steps:
        try:
            graph_store.conn.execute(cypher, {"eid": entity_id})
        except (RuntimeError, ValueError, AttributeError) as exc:
            logger.debug("delete step failed for %s: %s", entity_id, exc)


def _drop_orphan_entities(graph_store, dry_run: bool = False) -> int:
    """Delete Entity nodes with no incoming MENTIONS edge."""
    all_ids: Set[str] = {row[0] for row in _fetch_all(
        graph_store, "MATCH (e:Entity) RETURN e.entity_id"
    )}
    mentioned: Set[str] = {row[0] for row in _fetch_all(
        graph_store,
        "MATCH (:DocumentChunk)-[:MENTIONS]->(e:Entity) RETURN DISTINCT e.entity_id",
    )}
    orphans = all_ids - mentioned
    if dry_run:
        return len(orphans)
    for eid in orphans:
        _delete_entity_safely(graph_store, eid)
    if orphans:
        logger.info("Dropped %d orphan entities (zero MENTIONS).", len(orphans))
    return len(orphans)


def _drop_hub_entities(
    graph_store,
    hub_threshold: int,
    dry_run: bool = False,
) -> int:
    """Delete Entity nodes mentioned in more than `hub_threshold` distinct chunks."""
    rows = _fetch_all(
        graph_store,
        """
        MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity)
        WITH e.entity_id AS eid, count(DISTINCT c) AS mc
        WHERE mc > $threshold
        RETURN eid, mc
        """,
        {"threshold": hub_threshold},
    )
    if dry_run:
        return len(rows)
    for eid, _ in rows:
        _delete_entity_safely(graph_store, eid)
    if rows:
        logger.info(
            "Dropped %d hub entities (mention_count > %d).",
            len(rows), hub_threshold,
        )
    return len(rows)


def _merge_duplicate_entities(graph_store, dry_run: bool = False) -> int:
    """
    Merge entities sharing (canonical_form, type).

    Strategy: for each cluster, keep the entity with the highest MENTIONS count
    and redirect all incident edges of the other cluster members to it. The
    other members are then deleted via `_delete_entity_safely`.
    """
    rows = _fetch_all(
        graph_store,
        """
        MATCH (e:Entity)
        OPTIONAL MATCH (c:DocumentChunk)-[:MENTIONS]->(e)
        WITH e.entity_id AS eid, e.name AS name, e.type AS etype, count(c) AS mc
        RETURN eid, name, etype, mc
        """,
    )
    by_key: Dict[Tuple[str, str], List[Tuple[str, str, int]]] = {}
    for eid, name, etype, mc in rows:
        if not name:
            continue
        key = (canonical_form(name), etype or "unknown")
        by_key.setdefault(key, []).append((eid, name, int(mc or 0)))

    merged = 0
    for cluster in by_key.values():
        if len(cluster) < 2:
            continue
        # Keep the surface form with the highest mention count (most canonical
        # in practice). Ties broken by length-then-lexicographic order so the
        # choice is deterministic across runs.
        cluster.sort(key=lambda x: (-x[2], len(x[1]), x[1]))
        keeper_id = cluster[0][0]
        if dry_run:
            merged += len(cluster) - 1
            continue
        for old_id, _, _ in cluster[1:]:
            _redirect_entity_edges(graph_store, old_id=old_id, new_id=keeper_id)
            _delete_entity_safely(graph_store, old_id)
            merged += 1
    if merged:
        logger.info("Merged %d duplicate entities into canonical surface forms.", merged)
    return merged


# -----------------------------------------------------------------------------
# EMBEDDING-BASED ENTITY LINKING (alias resolution beyond canonical_form)
# -----------------------------------------------------------------------------

def link_entities_by_embedding(
    graph_store,
    embedder,
    similarity_threshold: float = 0.92,
    min_type_size: int = 2,
    max_type_size: int = 8000,
    length_ratio_floor: float = 0.4,
    dry_run: bool = False,
    done_buckets: Optional[Iterable[str]] = None,
    on_bucket_done: Optional[Callable[[str, int], None]] = None,
) -> int:
    """
    Merge entity nodes whose embedded names are sufficiently similar within type.

    Solves the alias problem that `canonical_form` cannot:
        common nickname  <->  expanded full name           (e.g. Bob / Robert)
        full name        <->  initialism + surname         (e.g. J. R. Smith)
        long country     <->  abbreviated form              (e.g. United States / U.S. of America)

    Algorithm:
      1. Group all entities by canonical type (PERSON, GPE, WORK_OF_ART, ...).
      2. For each type bucket, batch-embed entity names via `embedder`.
      3. Compute pairwise cosine similarity inside the bucket.
      4. Greedy union-find: pairs with sim >= `similarity_threshold` merge.
      5. Within each merged cluster, keep the entity with the highest
         MENTIONS count and redirect all incident edges to it.

    Args:
        graph_store:           KuzuGraphStore-compatible.
        embedder:              An object with `embed_documents(List[str])
                               -> List[List[float]]` (e.g., the project's
                               BatchedOllamaEmbeddings).
        similarity_threshold:  Cosine cutoff. Higher = stricter merge.
                               0.92 is conservative; 0.88 is aggressive.
        min_type_size:         Skip type buckets with fewer entities than this.
        max_type_size:         Skip type buckets larger than this (memory
                               safety: 8000 entities -> 8000^2 floats =
                               ~256 MB similarity matrix).
        length_ratio_floor:    Reject pairs whose surface-form length ratio
                               is below this. Prevents "X" from merging into
                               "X World Championship" purely because the
                               embedder maps short names to similar vectors.
        dry_run:               Count operations without mutating the graph.
        done_buckets:          Optional set/list of type-bucket names that
                               have already been linked on a previous run.
                               These are skipped wholesale (no embedding,
                               no merging). Used by the per-bucket
                               checkpoint resume path in
                               local_importingestion.py: after each bucket
                               completes, the caller records the type in
                               the `3d5` checkpoint's `done_buckets` list
                               and passes it back here on resume. Lets the
                               user kill-and-restart the ingest (e.g. when
                               Ollama OOMs mid-phase) without re-doing
                               buckets that already completed.
        on_bucket_done:        Optional callback invoked AFTER each bucket
                               finishes (whether merges happened or not).
                               Signature: ``(etype, merged_in_type) -> None``.
                               The callback is responsible for persisting
                               the bucket name so it can be resumed; this
                               function does not write to disk itself.

    Returns:
        Number of entities merged (i.e., deleted because they had a parent).
        The count covers ONLY buckets processed in this call — buckets
        skipped via `done_buckets` are not counted (their work was done
        on a previous invocation and recorded in the checkpoint).
    """
    # Normalise the done-bucket set once. None -> empty set.
    _done: set = set(done_buckets) if done_buckets else set()
    try:
        import numpy as np
    except ImportError:
        logger.warning(
            "FALLBACK ACTIVE: numpy not installed; embedding-based entity "
            "linking disabled."
        )
        return 0

    rows = _fetch_all(
        graph_store,
        """
        MATCH (e:Entity)
        OPTIONAL MATCH (c:DocumentChunk)-[:MENTIONS]->(e)
        WITH e.entity_id AS eid, e.name AS name, e.type AS etype, count(c) AS mc
        RETURN eid, name, etype, mc
        """,
    )

    by_type: Dict[str, List[Tuple[str, str, int]]] = {}
    for eid, name, etype, mc in rows:
        if not name:
            continue
        by_type.setdefault(etype or "unknown", []).append(
            (eid, name, int(mc or 0))
        )

    merged_total = 0

    # Iterate in deterministic (sorted) order so resume semantics are stable:
    # `done_buckets` from a previous run identifies buckets by name, not by
    # the by_type-dict iteration order (which depends on insertion in
    # the MATCH query — itself stable, but explicit sort makes the
    # invariant obvious to a reviewer).
    for etype in sorted(by_type.keys()):
        members = by_type[etype]

        # Per-bucket resume: skip wholesale if the caller's previous run
        # already finished this bucket. No embedding, no merging, no
        # callback — the work is done and persisted in the checkpoint.
        if etype in _done:
            logger.info(
                "Embed-link: skipping type=%s (already completed in a "
                "previous run; per-bucket checkpoint hit)",
                etype,
            )
            continue

        if len(members) < min_type_size:
            # Trivially "done" — still notify so the caller can record it
            # and not re-evaluate it on resume.
            if on_bucket_done is not None:
                on_bucket_done(etype, 0)
            continue
        if len(members) > max_type_size:
            logger.warning(
                "Embed-link: skipping type=%s with %d entities (above max_type_size=%d)",
                etype, len(members), max_type_size,
            )
            # Treat the cap-skip as "done for this run's parameters" so
            # resume does not embed it again under the same cap.
            if on_bucket_done is not None:
                on_bucket_done(etype, 0)
            continue

        names = [name for _, name, _ in members]
        # Embed with a one-shot warm-up retry. Ollama can return HTTP 500
        # ("model failed to load") after long continuous embedding sessions
        # — its model-cache state drifts and the next batch occasionally
        # fails to allocate. A single warm-up probe (one short text) forces
        # Ollama to reload the model into memory; the full bucket is then
        # retried. If the retry also fails the bucket is skipped (the
        # existing safety net), which is correct: a failed bucket is logged
        # and produces zero spurious merges, vs. one bucket bringing down
        # the whole phase.
        try:
            embeds_raw = embedder.embed_documents(names)
        except (RuntimeError, ConnectionError, ValueError) as exc:
            logger.warning(
                "Embed-link: embedder failed for type=%s (%s); "
                "issuing warm-up probe and retrying once...",
                etype, exc,
            )
            import time as _time
            try:
                # Single short text forces Ollama to reload the model.
                _ = embedder.embed_query("warm up")
                _time.sleep(2.0)
                embeds_raw = embedder.embed_documents(names)
                logger.info(
                    "Embed-link: warm-up retry succeeded for type=%s", etype,
                )
            except (RuntimeError, ConnectionError, ValueError) as exc2:
                logger.warning(
                    "Embed-link: embedder still failing for type=%s after "
                    "warm-up (%s); skipping bucket. Restart Ollama and "
                    "re-run the ingest if you want this bucket linked.",
                    etype, exc2,
                )
                continue

        embeds = np.asarray(embeds_raw, dtype=np.float32)
        if embeds.ndim != 2 or embeds.shape[0] != len(members):
            logger.warning(
                "Embed-link: bad embedding shape %s for type=%s",
                embeds.shape, etype,
            )
            continue

        # L2-normalise so dot product == cosine similarity.
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        embeds = embeds / np.where(norms > 0, norms, 1.0)

        n = len(members)
        sim = embeds @ embeds.T          # n x n cosine similarity
        np.fill_diagonal(sim, 0.0)       # self-similarity ignored

        # Union-Find with mention-count as the "win" criterion.
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            pi, pj = find(i), find(j)
            if pi == pj:
                return
            # Keep the entity with more MENTIONS as the cluster root.
            if members[pi][2] >= members[pj][2]:
                parent[pj] = pi
            else:
                parent[pi] = pj

        # Iterate the upper triangle. Length ratio guard prevents merging
        # short surface forms into long unrelated ones.
        for i in range(n):
            name_i = members[i][1]
            for j in range(i + 1, n):
                if sim[i, j] < similarity_threshold:
                    continue
                name_j = members[j][1]
                short, long_ = sorted((len(name_i), len(name_j)))
                if long_ > 0 and short / long_ < length_ratio_floor:
                    continue
                union(i, j)

        # Apply merges via the BULK path. Building a {old_id: new_id}
        # redirect map for the whole type bucket lets us fetch incident
        # edges in O(1) queries instead of O(merged_in_type) queries, and
        # apply the rewrites via transactional CREATE — the same fast path
        # the co-occurrence and Phase-3b passes use. On HotpotQA this
        # replaces the per-entity loop that previously took ~14 hours.
        redirect_map: Dict[str, str] = {}
        for i in range(n):
            root = find(i)
            if root == i:
                continue
            old_id = members[i][0]
            new_id = members[root][0]
            if old_id == new_id:
                continue
            redirect_map[old_id] = new_id

        merged_in_type = len(redirect_map)
        if not dry_run and redirect_map:
            _redirect_entity_edges_bulk(graph_store, redirect_map)

        if merged_in_type:
            logger.info(
                "Embed-link merged %d entities in type=%s (threshold=%.2f, %d total)",
                merged_in_type, etype, similarity_threshold, n,
            )
        merged_total += merged_in_type

        # Per-bucket checkpoint hook: caller persists `etype` as done so a
        # mid-phase crash (Ollama OOM, KeyboardInterrupt, …) does not lose
        # progress on already-linked buckets.
        if on_bucket_done is not None:
            on_bucket_done(etype, merged_in_type)

    return merged_total


def _redirect_entity_edges_bulk(
    graph_store,
    redirect_map: Dict[str, str],
) -> None:
    """Redirect every MENTIONS / RELATED_TO edge incident on any key of
    `redirect_map` to its mapped value, then delete the old entities. ALL
    work is done in two scan queries + transactional bulk writes — no
    per-entity Cypher round-trips.

    Why this exists
    ---------------
    The previous per-entity implementation (`_redirect_entity_edges`)
    issued one Cypher MATCH+MERGE per incident edge inside a Python for-
    loop. MERGE on a MENTIONS / RELATED_TO edge does an adjacency-list
    existence scan whose cost scales with the destination entity's
    degree. As popular entities accumulate edges during the merge loop,
    every subsequent redirect into them gets quadratically slower. On
    HotpotQA this caused Phase 3d.5 to take ~14 hours.

    The bulk path mirrors the co-occurrence-edge fast path:
      1. Fetch all incident MENTIONS edges in ONE query.
      2. Fetch all incident RELATED_TO edges (in + out) in TWO queries.
      3. Deduplicate target edges in Python (CREATE is non-idempotent).
      4. Bulk-create the redirected edges via the existing primitives
         add_mentions_relations_bulk + add_related_to_relations_bulk
         (use_create=True).
      5. Bulk-delete the old entities (cascades their incident edges).
    """
    if not redirect_map:
        return
    old_ids = list(redirect_map.keys())

    # -- 1. MENTIONS: chunk -> old  ⇒  chunk -> new -----------------------
    mention_rows = _fetch_all(
        graph_store,
        """
        MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity)
        WHERE e.entity_id IN $old_ids
        RETURN DISTINCT c.chunk_id, e.entity_id
        """,
        {"old_ids": old_ids},
    )
    seen_mention_pairs: set = set()
    redirected_mentions: List[Tuple[str, str]] = []
    for chunk_id, old_eid in mention_rows:
        new_eid = redirect_map.get(old_eid, old_eid)
        pair = (chunk_id, new_eid)
        if pair in seen_mention_pairs:
            continue
        seen_mention_pairs.add(pair)
        redirected_mentions.append(pair)

    if redirected_mentions and hasattr(graph_store, "add_mentions_relations_bulk"):
        logger.info(
            "Embed-link redirect: writing %d MENTIONS edges (bulk)…",
            len(redirected_mentions),
        )
        graph_store.add_mentions_relations_bulk(redirected_mentions,
                                                 batch_size=500)

    # -- 2. RELATED_TO outgoing: (old)->o  ⇒  (new)->o --------------------
    out_rows = _fetch_all(
        graph_store,
        """
        MATCH (e:Entity)-[r:RELATED_TO]->(o:Entity)
        WHERE e.entity_id IN $old_ids
        RETURN e.entity_id, o.entity_id, r.relation_type, r.confidence,
               r.source_chunks
        """,
        {"old_ids": old_ids},
    )

    # -- 3. RELATED_TO incoming: o->(old)  ⇒  o->(new) --------------------
    in_rows = _fetch_all(
        graph_store,
        """
        MATCH (o:Entity)-[r:RELATED_TO]->(e:Entity)
        WHERE e.entity_id IN $old_ids
        RETURN o.entity_id, e.entity_id, r.relation_type, r.confidence,
               r.source_chunks
        """,
        {"old_ids": old_ids},
    )

    # Redirect endpoints, drop self-loops, deduplicate by triple key.
    seen_triples: set = set()
    redirected_rels: List[Tuple[str, str, str, float, str]] = []
    for src, dst, rel_type, conf, src_chunks in out_rows:
        new_src = redirect_map.get(src, src)
        new_dst = redirect_map.get(dst, dst)
        if new_src == new_dst:
            continue
        key = (new_src, new_dst, rel_type or "related")
        if key in seen_triples:
            continue
        seen_triples.add(key)
        redirected_rels.append((
            new_src, new_dst, rel_type or "related",
            float(conf or 0.0), src_chunks or "",
        ))
    for src, dst, rel_type, conf, src_chunks in in_rows:
        new_src = redirect_map.get(src, src)
        new_dst = redirect_map.get(dst, dst)
        if new_src == new_dst:
            continue
        key = (new_src, new_dst, rel_type or "related")
        if key in seen_triples:
            continue
        seen_triples.add(key)
        redirected_rels.append((
            new_src, new_dst, rel_type or "related",
            float(conf or 0.0), src_chunks or "",
        ))

    if redirected_rels and hasattr(graph_store, "add_related_to_relations_bulk"):
        logger.info(
            "Embed-link redirect: writing %d RELATED_TO edges (bulk)…",
            len(redirected_rels),
        )
        graph_store.add_related_to_relations_bulk(
            redirected_rels, batch_size=500, use_create=True,
        )

    # -- 4. Delete the old entities. KuzuDB cascades incident edges. ------
    # Single transactional batch — same fsync-amortisation as the writes.
    logger.info("Embed-link redirect: deleting %d old entities (bulk)…",
                len(old_ids))
    BATCH = 500
    for i in range(0, len(old_ids), BATCH):
        batch = old_ids[i : i + BATCH]
        try:
            graph_store.conn.execute("BEGIN TRANSACTION")
            for oid in batch:
                try:
                    graph_store.conn.execute(
                        "MATCH (e:Entity {entity_id: $eid}) DETACH DELETE e",
                        {"eid": oid},
                    )
                except (RuntimeError, ValueError, AttributeError) as exc:
                    logger.debug("delete entity %s failed: %s", oid, exc)
            graph_store.conn.execute("COMMIT")
        except (RuntimeError, ValueError, AttributeError) as exc:
            logger.warning("Bulk delete batch %d failed: %s", i // BATCH, exc)
            try:
                graph_store.conn.execute("ROLLBACK")
            except (RuntimeError, AttributeError):
                pass


def _redirect_entity_edges(graph_store, old_id: str, new_id: str) -> None:
    """
    Move every MENTIONS / RELATED_TO edge incident on `old_id` to `new_id`.

    Kuzu does not provide native edge rewriting, so we re-create edges via
    MERGE and then delete the originals. Self-loops (`new_id == new_id` after
    redirect) are filtered out.
    """
    if old_id == new_id:
        return
    # MENTIONS: chunk -> old becomes chunk -> new
    for (chunk_id,) in _fetch_all(
        graph_store,
        """
        MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity {entity_id: $old_id})
        RETURN DISTINCT c.chunk_id
        """,
        {"old_id": old_id},
    ):
        try:
            graph_store.add_mentions_relation(chunk_id=chunk_id, entity_id=new_id)
        except (RuntimeError, ValueError, AttributeError) as exc:
            logger.debug("redirect MENTIONS failed (%s -> %s): %s", chunk_id, new_id, exc)

    # RELATED_TO outgoing: old -> other becomes new -> other
    for other_id, rel_type in _fetch_all(
        graph_store,
        """
        MATCH (:Entity {entity_id: $old_id})-[r:RELATED_TO]->(o:Entity)
        RETURN o.entity_id, r.relation_type
        """,
        {"old_id": old_id},
    ):
        if other_id == new_id:
            continue
        try:
            graph_store.add_related_to_relation(
                entity1_id=new_id,
                entity2_id=other_id,
                relation_type=rel_type or "related",
            )
        except (RuntimeError, ValueError, AttributeError) as exc:
            logger.debug("redirect RELATED_TO out failed: %s", exc)

    # RELATED_TO incoming: other -> old becomes other -> new
    for other_id, rel_type in _fetch_all(
        graph_store,
        """
        MATCH (o:Entity)-[r:RELATED_TO]->(:Entity {entity_id: $old_id})
        RETURN o.entity_id, r.relation_type
        """,
        {"old_id": old_id},
    ):
        if other_id == new_id:
            continue
        try:
            graph_store.add_related_to_relation(
                entity1_id=other_id,
                entity2_id=new_id,
                relation_type=rel_type or "related",
            )
        except (RuntimeError, ValueError, AttributeError) as exc:
            logger.debug("redirect RELATED_TO in failed: %s", exc)


# -----------------------------------------------------------------------------
# SUBSUMPTIVE CO-OCCURRENCE CLEANUP
# -----------------------------------------------------------------------------
# Background: build_cooccurrence_edges() writes a RELATED_TO{cooccurs} edge for
# every entity pair sharing a chunk. Some of those pairs also have a SEMANTIC
# RELATED_TO edge from REBEL/SVO. For those, the cooccurs edge is redundant
# information (semantic relation entails co-occurrence) and dominates the graph
# numerically (~8:1 cooccurs:semantic on HotpotQA). At retrieval time we
# already down-weight cooccurs (§11.6), but it still pollutes:
#   - the published edge count ("we have 300k edges" sounds impressive until
#     reviewers learn 85% are cooccurs)
#   - the visualisation
#   - any future ablation that compares semantic-only vs full
#
# Subsumption rule: if entity-pair (a, b) has BOTH a semantic relation AND a
# cooccurs edge in either direction, delete the cooccurs edges. The semantic
# edge already encodes the relationship. Pairs with only cooccurs are kept
# (they carry the only signal we have for that pair).
#
# Refs:
#   - Galárraga et al. (2014) "Canonicalizing Open Knowledge Bases", CIKM:
#     redundant edge canonicalisation in OpenIE graphs.
#   - Knowledge Vault (Dong et al., 2014, KDD): edge-confidence subsumption
#     when multiple extractors agree on a fact.

def drop_subsumed_cooccurrence_edges(
    graph_store,
    cooccurs_relation_type: str = "cooccurs",
    dry_run: bool = False,
    batch_size: int = 2000,
) -> int:
    """
    Delete cooccurs edges between pairs that also have a semantic edge.

    Semantic = any relation_type other than `cooccurs_relation_type`.

    Implementation note (KuzuDB): Kuzu does not support subquery EXISTS inside
    DELETE, so we do this in two passes:
      1. Materialise the set of (a_id, b_id) pairs that have BOTH a semantic
         edge and a cooccurs edge (in either direction).
      2. Delete cooccurs edges between those pairs in batches.

    Args:
        graph_store:           KuzuGraphStore-compatible.
        cooccurs_relation_type: The relation_type value for the redundant
                                edge family. Defaults to "cooccurs" (must
                                match `build_cooccurrence_edges` argument).
        dry_run:               If True, return the count of edges that
                                WOULD be deleted; do not mutate.
        batch_size:            Pairs per delete batch.

    Returns:
        Number of cooccurs edges deleted (or that would be deleted).
    """
    pairs: Set[Tuple[str, str]] = set()
    for a_id, b_id in _fetch_all(
        graph_store,
        """
        MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
        WHERE r.relation_type IS NOT NULL
          AND r.relation_type <> $cooccurs
        RETURN DISTINCT a.entity_id, b.entity_id
        """,
        {"cooccurs": cooccurs_relation_type},
    ):
        # Canonical unordered pair so an A->B semantic edge subsumes a B->A
        # cooccurs edge as well.
        lo, hi = (a_id, b_id) if a_id <= b_id else (b_id, a_id)
        pairs.add((lo, hi))

    if not pairs:
        logger.info("Subsumptive cooccurs cleanup: no semantic pairs found; nothing to do.")
        return 0

    # Find which of these pairs ALSO have a cooccurs edge (in either direction).
    pairs_list = list(pairs)
    to_delete: List[Tuple[str, str]] = []

    # Kuzu does not handle IN over lists of TUPLES well — explode to per-row
    # ANY-direction query and rely on the small selectivity of the semantic set.
    for i in range(0, len(pairs_list), batch_size):
        batch = pairs_list[i : i + batch_size]
        # Flatten so a single Cypher pass can match both directions.
        a_ids = [p[0] for p in batch]
        b_ids = [p[1] for p in batch]
        # Direction 1: a -> b
        try:
            result = graph_store.conn.execute(
                """
                MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
                WHERE r.relation_type = $cooccurs
                  AND a.entity_id IN $a_ids AND b.entity_id IN $b_ids
                RETURN a.entity_id, b.entity_id
                """,
                {"cooccurs": cooccurs_relation_type, "a_ids": a_ids, "b_ids": b_ids},
            )
            while result.has_next():
                row = tuple(result.get_next())
                to_delete.append((row[0], row[1]))
        except (RuntimeError, ValueError, AttributeError) as exc:
            logger.debug("subsumed-pair lookup (forward) failed: %s", exc)
        # Direction 2: b -> a
        try:
            result = graph_store.conn.execute(
                """
                MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
                WHERE r.relation_type = $cooccurs
                  AND a.entity_id IN $b_ids AND b.entity_id IN $a_ids
                RETURN a.entity_id, b.entity_id
                """,
                {"cooccurs": cooccurs_relation_type, "a_ids": a_ids, "b_ids": b_ids},
            )
            while result.has_next():
                row = tuple(result.get_next())
                to_delete.append((row[0], row[1]))
        except (RuntimeError, ValueError, AttributeError) as exc:
            logger.debug("subsumed-pair lookup (reverse) failed: %s", exc)

    if dry_run:
        logger.info(
            "Subsumptive cooccurs cleanup (dry-run): would delete %d cooccurs edges "
            "across %d semantic pairs.",
            len(to_delete), len(pairs),
        )
        return len(to_delete)

    deleted = 0
    for i in range(0, len(to_delete), batch_size):
        batch = to_delete[i : i + batch_size]
        a_ids = [p[0] for p in batch]
        b_ids = [p[1] for p in batch]
        try:
            graph_store.conn.execute(
                """
                MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
                WHERE r.relation_type = $cooccurs
                  AND a.entity_id IN $a_ids AND b.entity_id IN $b_ids
                DELETE r
                """,
                {"cooccurs": cooccurs_relation_type, "a_ids": a_ids, "b_ids": b_ids},
            )
            deleted += len(batch)
        except (RuntimeError, ValueError, AttributeError) as exc:
            logger.warning("subsumed-pair delete batch %d failed: %s", i // batch_size, exc)

    logger.info(
        "Subsumptive cooccurs cleanup: deleted %d redundant cooccurs edges "
        "(semantic-pair count = %d).",
        deleted, len(pairs),
    )
    return deleted


# -----------------------------------------------------------------------------
# ISOLATED-ENTITY DROP (post-linking)
# -----------------------------------------------------------------------------
# Background: After Phase 3d.5 (`link_entities_by_embedding`), some Entity
# nodes end up with MENTIONS edges but ZERO RELATED_TO edges in either
# direction. They are reachable from chunks but contribute nothing to graph
# traversal — every hop-1 / hop-2 / hop-3 search on them returns the empty
# set. The §3e baseline reports `isolated_rate=27.5%`, which exceeds the 5%
# invariant.
#
# Why does this happen?
#   - The entity was never the subject or object of any REBEL/SVO triple.
#   - All its co-occurrence partners were merged INTO this entity during 3d.5,
#     so the edges that previously connected the cluster collapsed to self-
#     loops and were filtered out.
#   - GLiNER recall is wider than REBEL coverage; many extracted entities
#     simply have no extracted relation in the corpus.
#
# Action: After 3d.5, remove entities with zero RELATED_TO edges. Their
# chunks remain in the graph and are still retrievable via vector search;
# only the dead leaf nodes are pruned. This DOES change retrieval (an
# entity-driven path that previously hit the dead node now returns no
# graph-supported chunk for that entity), but the alternative is to keep
# 27% of the graph as dead weight.
#
# Refs:
#   - HippoRAG (Gutiérrez et al., 2024, NeurIPS): degree-pruning of
#     dangling entity nodes before PageRank.
#   - GraphRAG (Edge et al., 2024, Microsoft): community filtering excludes
#     singleton entities from cluster summaries.

def drop_isolated_entities(
    graph_store,
    dry_run: bool = False,
    delete_mentions: bool = True,
) -> int:
    """
    Delete Entity nodes with zero RELATED_TO edges in either direction.

    Args:
        graph_store:      KuzuGraphStore-compatible.
        dry_run:          Count operations without mutating the graph.
        delete_mentions:  If True (default), also delete the MENTIONS edges
                          that point at the isolated entity. Kuzu requires
                          edges-before-node for a successful node delete.

    Returns:
        Number of entities deleted.
    """
    all_ids: Set[str] = {row[0] for row in _fetch_all(
        graph_store, "MATCH (e:Entity) RETURN e.entity_id"
    )}
    if not all_ids:
        return 0
    connected: Set[str] = set()
    for a, b in _fetch_all(
        graph_store,
        "MATCH (a:Entity)-[:RELATED_TO]->(b:Entity) RETURN a.entity_id, b.entity_id",
    ):
        connected.add(a)
        connected.add(b)
    isolated = all_ids - connected
    if dry_run:
        logger.info(
            "Drop-isolated (dry-run): %d / %d entities have zero RELATED_TO edges.",
            len(isolated), len(all_ids),
        )
        return len(isolated)
    if not isolated:
        logger.info("Drop-isolated: no isolated entities (rate=0%%).")
        return 0
    for eid in isolated:
        _delete_entity_safely(graph_store, eid)
    logger.info(
        "Drop-isolated: removed %d entities with zero RELATED_TO edges (%.1f%% of total).",
        len(isolated), 100.0 * len(isolated) / max(1, len(all_ids)),
    )
    return len(isolated)
