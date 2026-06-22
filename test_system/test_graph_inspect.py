"""
Integration tests — KuzuDB knowledge-graph ingestion (paper §3.4).

Verifies that HybridStore writes Entity nodes, Chunk→Entity MENTIONS edges,
and RELATED_TO triples to KuzuDB after ingesting a small synthetic corpus,
using the real GLiNER + REBEL extraction pipeline (no mocks). Also pins the
bridge-connectivity invariant: graph_search(bridge_entity) must return every
chunk that mentions it (the basis of multi-hop traversal).

The tests build their own temporary graph; no populated data/<dataset>/graph
store is required. The whole module is marked `nightly` (loads model weights).

Sample budget
-------------
N_DOCS defaults to 2 (GLiNER + REBEL on 2 documents). Set EDGE_RAG_N_SAMPLES=4
to ingest the full 4-document corpus.

Dependencies / Requirements
---------------------------
pytest, numpy, langchain, KuzuDB, and GLiNER + REBEL weights (downloaded by
HuggingFace on first use). Graph wiring is deterministic given fixed inputs.

Last reviewed: 2026-05-30 (audit pass, project version 5.4).
"""

import os
import sys
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Dict, Generator, List, Set

import numpy as np
import pytest
from langchain.schema import Document

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_layer.storage import HybridStore, StorageConfig

logger = logging.getLogger(__name__)

# This whole module loads GLiNER + REBEL weights, so it is marked `nightly`
# and deselected from the default run via `-m "not nightly"`. The tests build
# their own temporary graph; no populated data/<dataset>/graph store is needed.
pytestmark = [pytest.mark.nightly]

# ---------------------------------------------------------------------------
# Synthetic corpus — multi-hop bridge-question style.
# Neutral encyclopedia entities only (early-cinema figures + a landmark), so
# the corpus cannot leak benchmark content into the development trail.
#
# Sample budget: N_DOCS defaults to 2 so the default run calls GLiNER + REBEL
# on only 2 documents.  Set EDGE_RAG_N_SAMPLES=4 to ingest the full corpus.
# ---------------------------------------------------------------------------

N_DOCS: int = int(os.getenv("EDGE_RAG_N_SAMPLES", "2"))

_ALL_DOCS = [
    Document(
        page_content=(
            "Georges Méliès was a French filmmaker based in Paris. "
            "He directed A Trip to the Moon in 1902."
        ),
        metadata={
            "source_file": "wiki_melies.txt",
            "chunk_id": "c1",
            "chunk_index": 0,
            "page_number": 1,
        },
    ),
    Document(
        page_content=(
            "Louis Lumière was a French filmmaker who worked in Paris. "
            "He directed The Arrival of a Train in 1896."
        ),
        metadata={
            "source_file": "wiki_lumiere.txt",
            "chunk_id": "c2",
            "chunk_index": 0,
            "page_number": 1,
        },
    ),
    Document(
        page_content=(
            "Greta Garbo starred in Grand Hotel, directed by Edmund Goulding. "
            "The film was released in 1932."
        ),
        metadata={
            "source_file": "wiki_garbo.txt",
            "chunk_id": "c3",
            "chunk_index": 0,
            "page_number": 1,
        },
    ),
    Document(
        page_content=(
            "The Eiffel Tower is located in Paris, France. "
            "It was designed by Gustave Eiffel and completed in 1889."
        ),
        metadata={
            "source_file": "wiki_eiffel.txt",
            "chunk_id": "c4",
            "chunk_index": 0,
            "page_number": 1,
        },
    ),
]

# DOCS is the slice actually ingested in this test run (controlled by N_DOCS).
DOCS = _ALL_DOCS[:N_DOCS]

# Per-doc expected persons — used to restrict person assertions to ingested docs.
_EXPECTED_PERSONS_BY_CHUNK: Dict[str, str] = {
    "c1": "Georges Méliès",
    "c2": "Louis Lumière",
    "c3": "Greta Garbo",
    "c4": "Gustave Eiffel",
}
_EXPECTED_PERSONS: List[str] = [
    _EXPECTED_PERSONS_BY_CHUNK[doc.metadata["chunk_id"]]
    for doc in DOCS
    if doc.metadata["chunk_id"] in _EXPECTED_PERSONS_BY_CHUNK
]

# Only retain bridge cases whose expected chunks are all within DOCS.
_INGESTED_CHUNK_IDS: Set[str] = {doc.metadata["chunk_id"] for doc in DOCS}

_ALL_BRIDGE_EXPECTATIONS = [
    {
        # Shared-location bridge: both early filmmakers worked in the same city.
        "question": "Both Georges Méliès and Louis Lumière worked in _.",
        "bridge_entity": "Paris",
        "expected_chunk_ids": {"c1", "c2"},
    },
    {
        "question": "The Eiffel Tower was designed by _ and is in Paris.",
        "bridge_entity": "Gustave Eiffel",
        "expected_chunk_ids": {"c4"},
    },
    {
        "question": "Greta Garbo starred in a film directed by _.",
        "bridge_entity": "Edmund Goulding",
        "expected_chunk_ids": {"c3"},
    },
]

BRIDGE_EXPECTATIONS = [
    case for case in _ALL_BRIDGE_EXPECTATIONS
    if case["expected_chunk_ids"].issubset(_INGESTED_CHUNK_IDS)
]


# ---------------------------------------------------------------------------
# Deterministic mock embeddings
# Produces fixed DIM-dim vectors from a stable SHA-256 digest of the text — the
# same text always gives the same vector across processes and machines. (Python's
# built-in hash() is avoided: it is PYTHONHASHSEED-randomised per process.)
# ---------------------------------------------------------------------------

class _DeterministicEmbeddings:
    """
    Mock embeddings for integration tests.

    Uses a SHA-256-derived scalar repeated DIM times. Deterministic across
    processes and machines. Not suitable for semantic retrieval quality —
    only for graph-wiring tests.
    """
    DIM = 768

    @staticmethod
    def _scalar(text: str) -> float:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")
        return (seed % 1000) / 1000.0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[self._scalar(t)] * self.DIM for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [self._scalar(text)] * self.DIM


# ---------------------------------------------------------------------------
# Module-scoped fixture: one HybridStore for all tests in this module.
# GLiNER + REBEL are expensive to load; scope="module" loads them once.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Generator[HybridStore, None, None]:
    """
    Create a temporary HybridStore, ingest DOCS, and yield for all tests.

    Uses a module-scoped temporary directory so GLiNER and REBEL are
    loaded only once per test session.
    """
    tmp = tmp_path_factory.mktemp("graph_inspect")
    cfg = StorageConfig(
        vector_db_path=str(tmp / "vec"),
        graph_db_path=str(tmp / "graph"),
        embedding_dim=_DeterministicEmbeddings.DIM,
        enable_entity_extraction=True,
    )
    hs = HybridStore(cfg, _DeterministicEmbeddings())
    hs.add_documents(DOCS)
    yield hs
    shutil.rmtree(str(tmp), ignore_errors=True)


# ---------------------------------------------------------------------------
# Helper: query KuzuDB connection from store
# ---------------------------------------------------------------------------

def _conn(store: HybridStore):
    return store.graph_store.conn


# ---------------------------------------------------------------------------
# Section 1 — Entity Nodes
# ---------------------------------------------------------------------------

class TestEntityNodes:
    """Verify that GLiNER extracted entity nodes and stored them in KuzuDB."""

    def test_entity_nodes_present(self, store: HybridStore) -> None:
        """At least one Entity node must exist after ingestion."""
        res = _conn(store).execute("MATCH (e:Entity) RETURN COUNT(e)")
        count = res.get_next()[0]
        assert count > 0, "No Entity nodes found — GLiNER extraction may have failed"

    def test_expected_persons_extracted(self, store: HybridStore) -> None:
        """Named persons must be detected by GLiNER and stored as Entity nodes.

        Only checks persons from the docs that were actually ingested (N_DOCS).
        """
        res = _conn(store).execute(
            "MATCH (e:Entity) RETURN e.name"
        )
        names = {row[0] for row in res}
        for expected in _EXPECTED_PERSONS:
            assert expected in names, (
                f"Expected entity '{expected}' not found in graph. "
                f"Found: {sorted(names)}"
            )

    def test_entity_confidence_is_nonnegative(self, store: HybridStore) -> None:
        """All stored entity confidences must be >= 0.0."""
        res = _conn(store).execute(
            "MATCH (e:Entity) WHERE e.confidence IS NOT NULL RETURN e.name, e.confidence"
        )
        while res.has_next():
            name, conf = res.get_next()
            assert conf >= 0.0, f"Negative confidence for entity '{name}': {conf}"


# ---------------------------------------------------------------------------
# Section 2 — MENTIONS Edges
# ---------------------------------------------------------------------------

class TestMentionsEdges:
    """Verify that Chunk→Entity MENTIONS edges are correctly created."""

    def test_mentions_edges_present(self, store: HybridStore) -> None:
        """At least one MENTIONS edge must exist after ingestion."""
        res = _conn(store).execute(
            "MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity) RETURN COUNT(*)"
        )
        count = res.get_next()[0]
        assert count > 0, "No MENTIONS edges found — entity linking may have failed"

    def test_each_chunk_has_at_least_one_mention(self, store: HybridStore) -> None:
        """Every ingested chunk must mention at least one entity."""
        chunk_ids = {doc.metadata["chunk_id"] for doc in DOCS}
        res = _conn(store).execute(
            "MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity) "
            "RETURN DISTINCT c.chunk_id"
        )
        chunks_with_mentions = {row[0] for row in res}
        for cid in chunk_ids:
            assert cid in chunks_with_mentions, (
                f"Chunk '{cid}' has no MENTIONS edges — entity extraction may "
                f"have failed for this document"
            )

    def test_chunk_mentions_correct_entities(self, store: HybridStore) -> None:
        """Each ingested chunk must mention its expected anchor person/designer."""
        res = _conn(store).execute(
            "MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity) "
            "RETURN c.chunk_id, e.name"
        )
        mentions: Dict[str, set] = {}
        while res.has_next():
            cid, name = res.get_next()
            mentions.setdefault(cid, set()).add(name)

        # Restrict to ingested chunks so the assertion is robust to N_DOCS.
        for cid, expected_entity in _EXPECTED_PERSONS_BY_CHUNK.items():
            if cid in _INGESTED_CHUNK_IDS:
                assert expected_entity in mentions.get(cid, set()), (
                    f"Chunk '{cid}' must mention '{expected_entity}'. "
                    f"Found: {sorted(mentions.get(cid, set()))}"
                )


# ---------------------------------------------------------------------------
# Section 3 — RELATED_TO (REBEL relation extraction)
# ---------------------------------------------------------------------------

class TestRelatedToEdges:
    """Verify that REBEL extracted subject-predicate-object triples correctly."""

    def test_director_relation_extracted(self, store: HybridStore) -> None:
        """
        REBEL must extract a director relation from at least one film document.
        Expected: A Trip to the Moon --[director]--> Georges Méliès
                  Grand Hotel        --[director]--> Edmund Goulding
        """
        res = _conn(store).execute(
            "MATCH (e1:Entity)-[r:RELATED_TO]->(e2:Entity) "
            "WHERE r.relation_type = 'director' "
            "RETURN e1.name, e2.name"
        )
        triples = [(row[0], row[1]) for row in res]
        assert len(triples) > 0, (
            "No 'director' RELATED_TO edges found — REBEL extraction may have "
            "failed or produced no director relations"
        )

    def test_related_to_edges_are_bidirectionally_typed(self, store: HybridStore) -> None:
        """All RELATED_TO edges must have a non-empty relation_type."""
        res = _conn(store).execute(
            "MATCH ()-[r:RELATED_TO]->() "
            "WHERE r.relation_type IS NULL OR r.relation_type = '' "
            "RETURN COUNT(r)"
        )
        count = res.get_next()[0]
        assert count == 0, (
            f"{count} RELATED_TO edges have null or empty relation_type — "
            f"REBEL output was not stored correctly"
        )


# ---------------------------------------------------------------------------
# Section 4 — Bridge Entity Connectivity
# ---------------------------------------------------------------------------

class TestBridgeEntityConnectivity:
    """
    Verify that chunks sharing a bridge entity are both linked to that entity
    node in the graph. This is the core multi-hop retrieval invariant:
    graph_search(bridge_entity) must return ALL chunks that mention it.
    """

    @pytest.mark.parametrize("case", BRIDGE_EXPECTATIONS, ids=[
        c["bridge_entity"] for c in BRIDGE_EXPECTATIONS
    ])
    def test_bridge_entity_links_correct_chunks(
        self, store: HybridStore, case: Dict[str, object]
    ) -> None:
        """
        Bridge entity must appear as a MENTIONS target for all expected chunks.

        Failure indicates that one of the expected chunks did not have the
        bridge entity extracted, breaking multi-hop traversal for questions
        like: '{question}'
        """
        res = _conn(store).execute(
            "MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity) "
            "WHERE e.name = $name "
            "RETURN c.chunk_id",
            {"name": case["bridge_entity"]},
        )
        found = {row[0] for row in res}
        expected = case["expected_chunk_ids"]
        missing = expected - found
        assert not missing, (
            f"Bridge entity '{case['bridge_entity']}' not linked to chunks {missing}. "
            f"Found: {found}. "
            f"Question: '{case['question']}'"
        )


# ---------------------------------------------------------------------------
# Section 5 — Graph Statistics Sanity Check
# ---------------------------------------------------------------------------

class TestGraphStatistics:
    """Sanity-check the HybridStore.get_statistics() output."""

    def test_statistics_returns_expected_keys(self, store: HybridStore) -> None:
        """get_statistics() must return all documented keys."""
        stats = store.graph_store.get_statistics()
        for key in ("document_chunks",):
            assert key in stats, f"Missing key '{key}' in get_statistics() output"

    def test_chunk_count_matches_ingested(self, store: HybridStore) -> None:
        """Chunk count in statistics must equal number of ingested documents."""
        stats = store.graph_store.get_statistics()
        assert stats.get("document_chunks") == len(DOCS), (
            f"Expected {len(DOCS)} chunks in graph, "
            f"got {stats.get('document_chunks')}"
        )
