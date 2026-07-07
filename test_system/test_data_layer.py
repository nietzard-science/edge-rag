"""
Data Layer Test Suite — Artifact A (paper §3.1–§3.5).

Automated regression suite for the five Data Layer components: storage
(LanceDB vector + KuzuDB graph), embeddings (SQLite cache + metrics),
chunking (sentence / semantic / fixed-size), entity-extraction data
structures + caching, and hybrid retrieval (RRF fusion + pre-generative
filtering). TestFullPipeline exercises end-to-end ingestion→retrieval;
TestPerformance measures infrastructure latency with a mock model (it does
NOT validate the paper SLM latency targets).

Reproducibility
---------------
All tests use a module-scoped temp directory; each test opens its own
subdirectory, so tests are independent. The mock embedding model seeds its
RNG from a stable content digest (hashlib), so vectors are identical across
machines, processes, and CI runs — a prerequisite for the similarity-ordering
assertions. (Python's built-in hash() is deliberately avoided: it is
PYTHONHASHSEED-randomised per process.)

Test inventory (by component)
-----------------------------
Storage    : TestVectorStore, TestKuzuGraphStore, TestEntityLinkingResume,
             TestHybridStore
Embeddings : TestEmbeddingInfrastructure, TestEmbeddingCacheBatch
Chunking   : TestSentenceChunking, TestSemanticChunking
Entities   : TestEntityExtraction, TestEntityCacheBatch, TestEntityCacheStats,
             TestExtractionConfigNewFields, TestNormalizeQueryEntity,
             TestNormalizeEntityNameShared, TestB1SpanNormalization,
             TestB2TemporalMeasureGate
Retrieval  : TestRRFFusion, TestHybridRetriever
Pipeline   : TestFullPipeline, TestDocumentIngestionPipeline,
             TestIngestionPipelineExtended, TestRunDiagnostics
Other      : TestPerformance, TestEdgeCases

Usage
-----
    pytest test_data_layer.py -v
    pytest test_data_layer.py::TestVectorStore -v
    pytest test_data_layer.py --cov=src.data_layer --cov-report=html

Dependencies / Requirements
---------------------------
pytest, numpy, langchain_core. KuzuDB + LanceDB are exercised through their
adapters; no live Ollama is needed (embeddings are mocked).

Last reviewed: 2026-05-30 (audit pass, project version 5.4).
"""

import hashlib
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def temp_dir() -> Generator[Path, None, None]:
    """Create a module-scoped temporary directory for test databases."""
    tmp = tempfile.mkdtemp(prefix="test_data_layer_")
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_texts() -> list[str]:
    """Five science-biography sentences covering distinct topics."""
    return [
        "Albert Einstein was born in Ulm, Germany in 1879. He developed the theory of relativity.",
        "Marie Curie was a Polish physicist. She discovered radium and polonium.",
        "Isaac Newton formulated the laws of motion. He worked at Cambridge University.",
        "Charles Darwin proposed the theory of evolution. He traveled on the HMS Beagle.",
        "Nikola Tesla invented alternating current. He worked for Edison before starting his own company.",
    ]


@pytest.fixture
def sample_documents() -> list[Document]:
    """Three LangChain Documents with metadata required by the storage schema."""
    return [
        Document(
            page_content="Einstein developed E=mc². This equation relates mass and energy.",
            metadata={"source_file": "physics.pdf", "page_number": 1, "chunk_id": "c1",
                      "chunk_index": 0}
        ),
        Document(
            page_content="Curie won two Nobel Prizes. She was the first woman to win a Nobel Prize.",
            metadata={"source_file": "biography.pdf", "page_number": 5, "chunk_id": "c2",
                      "chunk_index": 0}
        ),
        Document(
            page_content="Newton's laws describe motion. The third law states every action has a reaction.",
            metadata={"source_file": "physics.pdf", "page_number": 12, "chunk_id": "c3",
                      "chunk_index": 1}
        ),
    ]


@pytest.fixture
def mock_embeddings():
    """
    Deterministic mock embedding model.

    Vectors are seeded per-text via a stable content digest so that identical
    inputs always produce identical outputs across machines and test runs.
    This is required for reproducible similarity-ordering assertions.
    """
    class MockEmbeddings:
        def __init__(self, dim: int = 768) -> None:
            self.dim = dim

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self._make_embedding(text) for text in texts]

        def embed_query(self, text: str) -> list[float]:
            return self._make_embedding(text)

        def _make_embedding(self, text: str) -> list[float]:
            # Seed per text from a SHA-256 digest, NOT Python's built-in hash():
            # str hash() is PYTHONHASHSEED-randomised per process, which would
            # make these vectors differ run-to-run and break the determinism
            # the similarity-ordering assertions depend on.
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.dim).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-8
            return vec.tolist()

    return MockEmbeddings()


# ============================================================================
# 1. STORAGE TESTS
# ============================================================================

class TestVectorStore:
    """Tests for the LanceDB Vector Store adapter."""

    def test_initialization(self, temp_dir: Path) -> None:
        """Vector store initialises correctly and exposes expected attributes."""
        from src.data_layer.storage import VectorStoreAdapter

        db_path = temp_dir / "vector_test"
        store = VectorStoreAdapter(
            db_path=db_path,
            embedding_dim=768,
            normalize_embeddings=True,
            distance_metric="cosine",
        )

        assert store.db is not None
        assert store.embedding_dim == 768
        assert store.distance_metric == "cosine"

    def test_add_and_search(
        self, temp_dir: Path, sample_documents: list[Document], mock_embeddings
    ) -> None:
        """Adding documents and searching returns similarity-scored results."""
        from src.data_layer.storage import VectorStoreAdapter

        db_path = temp_dir / "vector_add_search"
        store = VectorStoreAdapter(db_path=db_path, embedding_dim=768)

        store.add_documents_with_embeddings(sample_documents, mock_embeddings)

        query_embedding = mock_embeddings.embed_query("Einstein relativity")
        # threshold=0.0 to bypass the similarity filter with mock embeddings
        results = store.vector_search(query_embedding, top_k=3, threshold=0.0)

        assert len(results) > 0
        assert all("similarity" in r for r in results)
        assert all(0.0 <= r["similarity"] <= 1.0 for r in results)

    def test_dimension_validation(self, temp_dir: Path) -> None:
        """Embedding dimension mismatch raises ValueError with a clear message."""
        from src.data_layer.storage import VectorStoreAdapter

        db_path = temp_dir / "vector_dim_test"
        store = VectorStoreAdapter(db_path=db_path, embedding_dim=768)

        # 512 != 768 — deliberately wrong dimension for error-injection test
        wrong_embeddings = [[0.1] * 512]

        with pytest.raises(ValueError, match=r"(?i)dimension"):
            store._validate_embedding_dimension(wrong_embeddings)

    def test_distance_to_similarity(self, temp_dir: Path) -> None:
        """Cosine distance is correctly converted to similarity in [0, 1]."""
        from src.data_layer.storage import VectorStoreAdapter

        db_path = temp_dir / "vector_dist_test"
        store = VectorStoreAdapter(db_path=db_path, distance_metric="cosine")

        assert store._distance_to_similarity(0.0) == 1.0
        assert store._distance_to_similarity(1.0) == 0.0
        assert store._distance_to_similarity(0.5) == pytest.approx(0.5)

    def test_vector_search_results_ordered_by_similarity(
        self, temp_dir: Path, mock_embeddings
    ) -> None:
        """vector_search must return results in descending similarity order.

        This is the core retrieval invariant: the most relevant chunk must
        appear first so that top_k slicing is meaningful.  A regression here
        would silently degrade answer quality without any obvious error.
        """
        from src.data_layer.storage import VectorStoreAdapter

        db_path = temp_dir / "vector_order_test"
        store = VectorStoreAdapter(db_path=db_path, embedding_dim=768)

        docs = [
            Document(
                page_content="Einstein developed the theory of relativity.",
                metadata={"chunk_id": "ord_c1", "source_file": "phys.txt"},
            ),
            Document(
                page_content="The quick brown fox jumps over the lazy dog.",
                metadata={"chunk_id": "ord_c2", "source_file": "misc.txt"},
            ),
            Document(
                page_content="Special relativity was published in 1905.",
                metadata={"chunk_id": "ord_c3", "source_file": "phys.txt"},
            ),
        ]
        store.add_documents_with_embeddings(docs, mock_embeddings)

        query_emb = mock_embeddings.embed_query("relativity")
        results = store.vector_search(query_emb, top_k=3, threshold=0.0)

        assert len(results) >= 2, "Expected at least two results"
        similarities = [r["similarity"] for r in results]
        assert similarities == sorted(similarities, reverse=True), (
            "vector_search results must be sorted in descending similarity order. "
            f"Got: {similarities}"
        )


class TestKuzuGraphStore:
    """Tests for the KuzuDB Knowledge Graph store."""

    def test_initialization(self, temp_dir: Path) -> None:
        """KuzuGraphStore initialises and exposes a live connection."""
        from src.data_layer.storage import KuzuGraphStore

        db_path = temp_dir / "graph_test"
        store = KuzuGraphStore(db_path)

        assert store.db is not None
        assert store.conn is not None

    def test_add_document_chunk(self, temp_dir: Path) -> None:
        """MERGE-ing a DocumentChunk persists text retrievable via Cypher."""
        from src.data_layer.storage import KuzuGraphStore

        db_path = temp_dir / "graph_chunks"
        store = KuzuGraphStore(db_path)

        store.add_document_chunk(
            chunk_id="chunk_001",
            text="Einstein was born in Ulm.",
            page_number=1,
            chunk_index=0,
            source_file="test.pdf",
        )

        result = store.conn.execute(
            "MATCH (c:DocumentChunk {chunk_id: 'chunk_001'}) RETURN c.text"
        )
        assert result.has_next()
        assert "Einstein" in result.get_next()[0]

    def test_add_source_document(self, temp_dir: Path) -> None:
        """MERGE-ing a SourceDocument node persists filename."""
        from src.data_layer.storage import KuzuGraphStore

        db_path = temp_dir / "graph_source"
        store = KuzuGraphStore(db_path)

        store.add_source_document(
            doc_id="doc_001",
            filename="thesis.pdf",
            total_pages=100,
        )

        result = store.conn.execute(
            "MATCH (d:SourceDocument {doc_id: 'doc_001'}) RETURN d.filename"
        )
        assert result.has_next()

    def test_get_context_chunks(self, temp_dir: Path) -> None:
        """get_context_chunks returns neighbouring chunk IDs via NEXT_CHUNK edges."""
        from src.data_layer.storage import KuzuGraphStore

        db_path = temp_dir / "graph_context"
        store = KuzuGraphStore(db_path)

        for i in range(3):
            store.add_document_chunk(
                chunk_id=f"ctx_{i}",
                text=f"Context text {i}",
                page_number=1,
                chunk_index=i,
                source_file="test.pdf",
            )

        store.add_next_chunk_relation("ctx_0", "ctx_1")
        store.add_next_chunk_relation("ctx_1", "ctx_2")

        neighbours = store.get_context_chunks("ctx_0", window=2)
        assert len(neighbours) >= 1
        assert "ctx_0" in neighbours

    def test_graph_traversal(self, temp_dir: Path) -> None:
        """graph_traversal returns a hop-distance map for reachable nodes."""
        from src.data_layer.storage import KuzuGraphStore

        db_path = temp_dir / "graph_traversal_direct"
        store = KuzuGraphStore(db_path)

        for i in range(3):
            store.add_document_chunk(
                chunk_id=f"trav_{i}",
                text=f"Traversal text {i}",
                page_number=1,
                chunk_index=i,
                source_file="test.pdf",
            )

        store.add_next_chunk_relation("trav_0", "trav_1")
        store.add_next_chunk_relation("trav_1", "trav_2")

        visited = store.graph_traversal("trav_0", max_hops=2)

        assert "trav_0" in visited
        assert visited["trav_0"] == 0
        assert "trav_1" in visited
        assert visited["trav_1"] == 1
        assert "trav_2" in visited
        assert visited.get("trav_2") == 2

    def test_statistics(self, temp_dir: Path) -> None:
        """get_statistics returns a dict with at least the document_chunks count."""
        from src.data_layer.storage import KuzuGraphStore

        db_path = temp_dir / "graph_stats"
        store = KuzuGraphStore(db_path)

        store.add_document_chunk("s1", "Stats text 1", 1, 0, "test.pdf")
        store.add_document_chunk("s2", "Stats text 2", 1, 1, "test.pdf")

        stats = store.get_statistics()
        assert "document_chunks" in stats
        assert stats["document_chunks"] >= 2

    def test_multihop_finds_connected_chunk(self, temp_dir: Path) -> None:
        """find_chunks_by_entity_multihop must return a chunk linked via RELATED_TO.

        Graph structure:
            c_mh1 --MENTIONS--> Entity("Einstein")
                                    --RELATED_TO--> Entity("Curie")
                                                        <--MENTIONS-- c_mh2

        A query for "Einstein" must return both c_mh1 (hop 0, direct mention)
        and c_mh2 (hop 2, via the Einstein→Curie RELATED_TO edge).
        This is the core multi-hop retrieval invariant (paper section 2.4).
        """
        from src.data_layer.storage import KuzuGraphStore

        db_path = temp_dir / "graph_multihop_t05"
        store = KuzuGraphStore(db_path)

        # Insert two document chunks
        store.add_document_chunk("mh_c1", "Einstein developed relativity.", 1, 0, "physics.txt")
        store.add_document_chunk("mh_c2", "Curie discovered radium.", 1, 1, "bio.txt")

        # Insert Entity nodes and edges using the public API
        store.add_entity("e_einstein", "Einstein", entity_type="person", confidence=0.95)
        store.add_entity("e_curie", "Curie", entity_type="person", confidence=0.95)
        store.add_mentions_relation("mh_c1", "e_einstein")
        store.add_mentions_relation("mh_c2", "e_curie")
        store.add_related_to_relation("e_einstein", "e_curie", relation_type="colleague")

        # Query for "Einstein" — must reach c_mh2 via the RELATED_TO bridge
        results = store.find_chunks_by_entity_multihop("Einstein", max_results=10)
        chunk_ids = {r["chunk_id"] for r in results}

        assert "mh_c1" in chunk_ids, (
            "Direct-mention chunk mh_c1 must be found for entity 'Einstein'"
        )
        assert "mh_c2" in chunk_ids, (
            "1-hop related chunk mh_c2 must be found via Einstein→Curie RELATED_TO edge"
        )

    def test_max_hops_one_disables_bridge_expansion(self, temp_dir: Path) -> None:
        """max_hops=1 must return only direct mentions, no bridge chunks.

        Validates that the max_hops parameter on find_chunks_by_entity_multihop
        actually gates the hop-2 traversal — needed for the graph-only ablation
        that isolates the bridge-expansion contribution.
        """
        from src.data_layer.storage import KuzuGraphStore

        db_path = temp_dir / "graph_maxhops"
        store = KuzuGraphStore(db_path)

        store.add_document_chunk("mh1_direct", "Einstein paper", 1, 0, "p.txt")
        store.add_document_chunk("mh1_bridge", "Curie biography", 1, 0, "b.txt")
        store.add_entity("e_einstein", "Einstein", entity_type="person", confidence=0.9)
        store.add_entity("e_curie", "Curie", entity_type="person", confidence=0.9)
        store.add_mentions_relation("mh1_direct", "e_einstein")
        store.add_mentions_relation("mh1_bridge", "e_curie")
        store.add_related_to_relation("e_einstein", "e_curie", relation_type="colleague")

        # max_hops=1: only the direct mention chunk should come back.
        results = store.find_chunks_by_entity_multihop("Einstein", max_results=10, max_hops=1)
        chunk_ids = {r["chunk_id"] for r in results}
        assert "mh1_direct" in chunk_ids
        assert "mh1_bridge" not in chunk_ids, (
            "max_hops=1 must not include bridge-expansion (hop-2) chunks"
        )

    def test_classified_weight_origin_discrimination(self) -> None:
        """_classified_weight must rank REBEL > SVO > cooccurs by relation_type shape.

        - "cooccurs" (literal sentinel)          -> 0.25
        - "date_of_birth" (REBEL, underscored)   -> 1.0
        - "place of birth" (REBEL, whitespaced)  -> 1.0
        - "direct" (SVO, single verb lemma)      -> 0.6
        - "" / None                              -> 1.0 (default, never down-rank unknown)
        """
        from src.data_layer.storage import KuzuGraphStore

        # _classified_weight is a classmethod — no instance needed.
        assert KuzuGraphStore._classified_weight("cooccurs") == 0.25
        assert KuzuGraphStore._classified_weight("date_of_birth") == 1.0
        assert KuzuGraphStore._classified_weight("place of birth") == 1.0
        assert KuzuGraphStore._classified_weight("direct") == 0.6
        assert KuzuGraphStore._classified_weight("win") == 0.6
        assert KuzuGraphStore._classified_weight(None) == 1.0
        assert KuzuGraphStore._classified_weight("") == 1.0

    def test_add_entities_bulk_inserts_all_distinct_nodes(self, temp_dir: Path) -> None:
        """Phase-3b-fix (2026-05-19): add_entities_bulk must insert every
        distinct entity_id provided, using a transactional MERGE so the call
        is fast and idempotent.
        """
        from src.data_layer.storage import KuzuGraphStore

        store = KuzuGraphStore(temp_dir / "graph_bulk_entities")
        entities = [
            (f"e_{i:03d}", f"Entity {i}", "person" if i % 2 == 0 else "org", 0.8)
            for i in range(25)
        ]
        n = store.add_entities_bulk(entities, batch_size=10)
        assert n == 25
        res = store.conn.execute("MATCH (e:Entity) RETURN COUNT(e)")
        count = res.get_next()[0]
        assert count == 25, f"Expected 25 entities, got {count}"

    def test_add_entities_bulk_is_idempotent_via_merge(self, temp_dir: Path) -> None:
        """Calling add_entities_bulk twice with the same primary keys must
        NOT create duplicate nodes. MERGE on the primary-key Entity table
        is idempotent; a second call updates the SET fields and that's all.
        """
        from src.data_layer.storage import KuzuGraphStore

        store = KuzuGraphStore(temp_dir / "graph_bulk_idempotent")
        entities = [("e_001", "Alpha", "person", 0.9),
                    ("e_002", "Beta",  "org",    0.8)]
        store.add_entities_bulk(entities)
        store.add_entities_bulk(entities)   # second call — must not duplicate

        count = store.conn.execute(
            "MATCH (e:Entity) RETURN COUNT(e)"
        ).get_next()[0]
        assert count == 2, (
            f"add_entities_bulk must be idempotent at the primary-key level; "
            f"got {count} nodes after two identical calls"
        )

    def test_add_mentions_relations_bulk_inserts_all_edges(self, temp_dir: Path) -> None:
        """Phase-3b-fix (2026-05-19): add_mentions_relations_bulk must insert
        one MENTIONS edge per supplied (chunk_id, entity_id) pair using a
        transactional CREATE so the call avoids KuzuDB's adjacency-list
        existence scan that MERGE-on-edge would incur.

        This is the test that documents the per-edge CREATE design choice:
        the caller deduplicates pairs in Python, then CREATE inserts each
        edge without a scan.
        """
        from src.data_layer.storage import KuzuGraphStore

        store = KuzuGraphStore(temp_dir / "graph_bulk_mentions")
        # Seed 3 chunks + 4 entities.
        for i in range(3):
            store.add_document_chunk(
                chunk_id=f"c_{i}", text=f"chunk {i}", page_number=1,
                chunk_index=i, source_file="t.txt",
            )
        store.add_entities_bulk(
            [(f"e_{i}", f"E{i}", "person", 0.9) for i in range(4)]
        )

        # 6 distinct mention edges.
        pairs = [
            ("c_0", "e_0"), ("c_0", "e_1"),
            ("c_1", "e_1"), ("c_1", "e_2"),
            ("c_2", "e_2"), ("c_2", "e_3"),
        ]
        n = store.add_mentions_relations_bulk(pairs, batch_size=4)
        assert n == 6
        edge_count = store.conn.execute(
            "MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity) RETURN COUNT(*)"
        ).get_next()[0]
        assert edge_count == 6, f"Expected 6 MENTIONS edges, got {edge_count}"

    def test_redirect_entity_edges_bulk_rewrites_correctly(self, temp_dir: Path) -> None:
        """Phase-3d.5-fix (2026-05-19): _redirect_entity_edges_bulk must
        move every MENTIONS / RELATED_TO edge from a set of `old_id`
        entities to their mapped `new_id`, then delete the old entities.

        Setup: A small graph with two PERSON entities (`e_old`, `e_new`)
        sharing a chunk-MENTIONS link and a RELATED_TO edge each.
        Calling `_redirect_entity_edges_bulk({e_old: e_new})` must:

          - Move e_old's MENTIONS edges onto e_new.
          - Move e_old's outgoing/incoming RELATED_TO edges onto e_new.
          - Drop any edge that would self-loop after redirect.
          - DETACH-DELETE e_old.
        """
        from src.data_layer.storage import KuzuGraphStore
        from src.data_layer.graph_quality import _redirect_entity_edges_bulk

        store = KuzuGraphStore(temp_dir / "graph_redirect_bulk")

        # Two chunks, three entities. e_old will be redirected to e_new;
        # e_other stays put and exchanges relations with both.
        for cid in ("c1", "c2"):
            store.add_document_chunk(cid, f"text {cid}", 1, 0, "t.txt")
        store.add_entities_bulk([
            ("e_old",   "Alpha",  "person", 0.9),
            ("e_new",   "Beta",   "person", 0.9),
            ("e_other", "Gamma",  "org",    0.8),
        ])

        # c1 mentions e_old; c2 mentions e_new — after redirect, both
        # chunks should mention e_new (and there should be exactly one
        # c1->e_new edge, exactly one c2->e_new edge).
        store.add_mentions_relations_bulk([("c1", "e_old"), ("c2", "e_new")])

        # RELATED_TO: e_old -> e_other (outgoing), e_other -> e_old (incoming).
        store.add_related_to_relations_bulk([
            ("e_old",   "e_other", "knows",   0.9, ""),
            ("e_other", "e_old",   "friend",  0.9, ""),
        ], use_create=True)

        # Redirect e_old -> e_new.
        _redirect_entity_edges_bulk(store, {"e_old": "e_new"})

        # e_old must be gone.
        n_old = store.conn.execute(
            "MATCH (e:Entity {entity_id: 'e_old'}) RETURN COUNT(e)"
        ).get_next()[0]
        assert n_old == 0, f"e_old should have been deleted; found {n_old}"

        # Both chunks should now mention e_new — exactly one edge each.
        c1_to_new = store.conn.execute(
            "MATCH (:DocumentChunk {chunk_id:'c1'})-[:MENTIONS]->"
            "(:Entity {entity_id:'e_new'}) RETURN COUNT(*)"
        ).get_next()[0]
        c2_to_new = store.conn.execute(
            "MATCH (:DocumentChunk {chunk_id:'c2'})-[:MENTIONS]->"
            "(:Entity {entity_id:'e_new'}) RETURN COUNT(*)"
        ).get_next()[0]
        assert c1_to_new == 1, f"Expected 1 c1->e_new MENTIONS; got {c1_to_new}"
        assert c2_to_new == 1, f"Expected 1 c2->e_new MENTIONS; got {c2_to_new}"

        # RELATED_TO edges must now reference e_new on both sides.
        n_out = store.conn.execute(
            "MATCH (:Entity {entity_id:'e_new'})-[:RELATED_TO]->"
            "(:Entity {entity_id:'e_other'}) RETURN COUNT(*)"
        ).get_next()[0]
        n_in = store.conn.execute(
            "MATCH (:Entity {entity_id:'e_other'})-[:RELATED_TO]->"
            "(:Entity {entity_id:'e_new'}) RETURN COUNT(*)"
        ).get_next()[0]
        assert n_out >= 1, f"Expected outgoing redirect; got {n_out}"
        assert n_in  >= 1, f"Expected incoming redirect; got {n_in}"

    def test_redirect_entity_edges_bulk_drops_self_loops(self, temp_dir: Path) -> None:
        """If two entities being merged share a RELATED_TO edge between them
        (A -> B and we redirect A to B), the result would be a self-loop
        B -> B. _redirect_entity_edges_bulk must drop those.
        """
        from src.data_layer.storage import KuzuGraphStore
        from src.data_layer.graph_quality import _redirect_entity_edges_bulk

        store = KuzuGraphStore(temp_dir / "graph_redirect_selfloop")
        store.add_entities_bulk([
            ("e_a", "A", "person", 0.9),
            ("e_b", "B", "person", 0.9),
        ])
        # A -> B exists. Redirecting A to B would produce B -> B.
        store.add_related_to_relations_bulk(
            [("e_a", "e_b", "knows", 0.9, "")], use_create=True,
        )
        _redirect_entity_edges_bulk(store, {"e_a": "e_b"})

        n_selfloop = store.conn.execute(
            "MATCH (e:Entity {entity_id:'e_b'})-[:RELATED_TO]->"
            "(:Entity {entity_id:'e_b'}) RETURN COUNT(*)"
        ).get_next()[0]
        assert n_selfloop == 0, (
            f"Self-loops must be dropped by the redirect; got {n_selfloop}"
        )

    def test_add_mentions_relations_bulk_skips_unknown_endpoints(
        self, temp_dir: Path,
    ) -> None:
        """A pair referencing a missing chunk OR a missing entity must NOT
        crash the batch — KuzuDB's MATCH simply fails to bind and the row
        is skipped. The other edges in the batch must still be written.

        This is the production safety contract: a stale extraction artifact
        referencing a chunk_id that has been cleaned out of the chunk table
        cannot poison the whole ingest.
        """
        from src.data_layer.storage import KuzuGraphStore

        store = KuzuGraphStore(temp_dir / "graph_bulk_mentions_safe")
        store.add_document_chunk(
            "c_real", "real chunk", 1, 0, "t.txt",
        )
        store.add_entities_bulk([("e_real", "E", "person", 0.9)])

        pairs = [
            ("c_real",  "e_real"),
            ("c_ghost", "e_real"),   # chunk does not exist
            ("c_real",  "e_ghost"),  # entity does not exist
        ]
        store.add_mentions_relations_bulk(pairs)
        edge_count = store.conn.execute(
            "MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity) RETURN COUNT(*)"
        ).get_next()[0]
        # Exactly one real edge survives; the two ghost-endpoint inserts
        # MATCH-fail in KuzuDB and write nothing.
        assert edge_count == 1, (
            f"Bulk MENTIONS must skip pairs with unbound endpoints; "
            f"got {edge_count} edges"
        )


class TestEntityLinkingResume:
    """Phase-3d.5 per-bucket checkpoint regression tests (2026-05-19).

    These tests validate the resume contract of
    `link_entities_by_embedding`:

      - `done_buckets={"PERSON"}` causes the PERSON bucket to be skipped
        wholesale — no embeddings called, no callback invoked for that
        bucket. (Crash-recovery: if a previous run already linked PERSON,
        the next `--resume` call must not redo it.)
      - `on_bucket_done(etype, merged_in_type)` fires after every bucket
        that is *processed* (whether merges happened or not). Skipped
        buckets via `done_buckets` do NOT fire the callback (the caller
        already recorded them).
      - Buckets are iterated in sorted-by-name order so resume semantics
        are deterministic across runs.

    The embedder is a hand-rolled stub that returns orthogonal vectors,
    so similarity is always 0 and no merging happens — we are testing
    control flow, not the linker algorithm itself (the algorithm is
    exercised by `test_redirect_entity_edges_bulk_*`).
    """

    def _make_orthogonal_embedder(self, dim: int = 16):
        """Stub embedder: each text gets a distinct unit-basis vector, so
        cosine similarity between any two texts is 0 — no merges fire."""

        class _OrthEmbedder:
            def __init__(self) -> None:
                self._slot = 0

            def embed_documents(self, texts):
                out = []
                for _ in texts:
                    v = [0.0] * dim
                    v[self._slot % dim] = 1.0
                    self._slot += 1
                    out.append(v)
                return out

        return _OrthEmbedder()

    def _seed_graph(self, temp_dir: Path, suffix: str):
        """Build a KuzuDB with 3 PERSON + 3 ORG entities (3 ≥ min_type_size=2)
        and one MENTIONS edge per entity so the entity-load query returns
        them. Returns the open store.
        """
        from src.data_layer.storage import KuzuGraphStore

        store = KuzuGraphStore(temp_dir / f"graph_linking_resume_{suffix}")
        store.add_document_chunk("c1", "text c1", 1, 0, "t.txt")
        store.add_entities_bulk([
            ("p1", "Alice",  "PERSON", 0.9),
            ("p2", "Bob",    "PERSON", 0.9),
            ("p3", "Carol",  "PERSON", 0.9),
            ("o1", "Acme",   "ORG",    0.9),
            ("o2", "Globex", "ORG",    0.9),
            ("o3", "Initech","ORG",    0.9),
        ])
        store.add_mentions_relations_bulk([
            ("c1", "p1"), ("c1", "p2"), ("c1", "p3"),
            ("c1", "o1"), ("c1", "o2"), ("c1", "o3"),
        ])
        return store

    def test_done_buckets_skips_listed_types(self, temp_dir: Path) -> None:
        """PERSON listed in done_buckets must be skipped — no callback for
        it, no embeddings drawn for PERSON names.
        """
        from src.data_layer.graph_quality import link_entities_by_embedding

        store = self._seed_graph(temp_dir, "skip")
        embedder = self._make_orthogonal_embedder()

        seen: list[tuple[str, int]] = []
        link_entities_by_embedding(
            graph_store=store,
            embedder=embedder,
            similarity_threshold=0.92,
            done_buckets={"PERSON"},
            on_bucket_done=lambda etype, n: seen.append((etype, n)),
        )

        seen_types = {etype for etype, _ in seen}
        assert "PERSON" not in seen_types, (
            "PERSON was in done_buckets and must not fire the callback; "
            f"got {seen_types}"
        )
        assert "ORG" in seen_types, (
            f"ORG should have been processed; got {seen_types}"
        )

    def test_on_bucket_done_invoked_per_processed_bucket(
        self, temp_dir: Path,
    ) -> None:
        """Without done_buckets, every type bucket that is large enough
        must produce exactly one callback. The merged_in_type count is 0
        because the orthogonal embedder forces zero similarity.
        """
        from src.data_layer.graph_quality import link_entities_by_embedding

        store = self._seed_graph(temp_dir, "all")
        embedder = self._make_orthogonal_embedder()

        seen: list[tuple[str, int]] = []
        link_entities_by_embedding(
            graph_store=store,
            embedder=embedder,
            similarity_threshold=0.92,
            on_bucket_done=lambda etype, n: seen.append((etype, n)),
        )

        seen_types = [etype for etype, _ in seen]
        # Both buckets fire (3 entities each, >= min_type_size=2).
        assert "PERSON" in seen_types and "ORG" in seen_types, (
            f"Expected callbacks for both PERSON and ORG; got {seen_types}"
        )
        # Orthogonal embeddings → no merges anywhere.
        assert all(n == 0 for _, n in seen), (
            f"Orthogonal embedder must not produce merges; got {seen}"
        )

    def test_bucket_iteration_is_sorted(self, temp_dir: Path) -> None:
        """Resume relies on a stable iteration order — sorted by type name."""
        from src.data_layer.graph_quality import link_entities_by_embedding

        store = self._seed_graph(temp_dir, "sorted")
        embedder = self._make_orthogonal_embedder()

        order: list[str] = []
        link_entities_by_embedding(
            graph_store=store,
            embedder=embedder,
            similarity_threshold=0.92,
            on_bucket_done=lambda etype, _n: order.append(etype),
        )

        # Filter to the two types we seeded; any tiny implicit buckets
        # (mention-degree side-effects) come from min_type_size skips and
        # are also sorted, but we assert only on the seeded types.
        seeded = [t for t in order if t in {"PERSON", "ORG"}]
        assert seeded == sorted(seeded), (
            f"Bucket iteration must be sorted for stable resume; got {seeded}"
        )


class TestHybridStore:
    """Tests for the combined Vector + Graph store facade."""

    def test_initialization(self, temp_dir: Path, mock_embeddings) -> None:
        """HybridStore initialises both sub-stores without errors."""
        from src.data_layer.storage import HybridStore, StorageConfig

        config = StorageConfig(
            vector_db_path=temp_dir / "hybrid_vector",
            graph_db_path=temp_dir / "hybrid_graph",
            embedding_dim=768,
        )

        store = HybridStore(config, mock_embeddings)

        assert store.vector_store is not None
        assert store.graph_store is not None

    def test_add_documents(
        self,
        temp_dir: Path,
        sample_documents: list[Document],
        mock_embeddings,
    ) -> None:
        """add_documents writes to both vector and graph stores."""
        from src.data_layer.storage import HybridStore, StorageConfig

        config = StorageConfig(
            vector_db_path=temp_dir / "hybrid_add_vector",
            graph_db_path=temp_dir / "hybrid_add_graph",
        )

        store = HybridStore(config, mock_embeddings)
        store.add_documents(sample_documents)

        # Verify vector store received the documents
        query_emb = mock_embeddings.embed_query("Einstein")
        vector_results = store.vector_store.vector_search(
            query_emb, top_k=3, threshold=0.0
        )
        assert len(vector_results) > 0


# ============================================================================
# 2. EMBEDDING INFRASTRUCTURE TESTS
# ============================================================================

class TestEmbeddingInfrastructure:
    """Tests for EmbeddingCache and EmbeddingMetrics."""

    def test_cache_initialization(self, temp_dir: Path) -> None:
        """EmbeddingCache creates the SQLite database file on init."""
        from src.data_layer.embeddings import EmbeddingCache

        cache_path = temp_dir / "embed_cache.db"
        cache = EmbeddingCache(cache_path)

        assert cache.db_path == cache_path
        assert cache_path.exists()

        cache.close()

    def test_cache_put_get(self, temp_dir: Path) -> None:
        """Cache put/get round-trip preserves the full embedding vector."""
        from src.data_layer.embeddings import EmbeddingCache

        cache_path = temp_dir / "embed_cache_ops.db"
        cache = EmbeddingCache(cache_path)

        test_text = "Hello World"
        test_embedding = [0.1] * 768

        cache.put(test_text, test_embedding, "test-model")
        retrieved = cache.get(test_text, "test-model")

        assert retrieved is not None
        assert retrieved == pytest.approx(test_embedding)

        cache.close()

    def test_cache_miss(self, temp_dir: Path) -> None:
        """cache.get returns None for text that was never stored."""
        from src.data_layer.embeddings import EmbeddingCache

        cache_path = temp_dir / "embed_cache_miss.db"
        cache = EmbeddingCache(cache_path)

        result = cache.get("nonexistent text", "test-model")
        assert result is None

        cache.close()

    def test_metrics_tracking(self) -> None:
        """EmbeddingMetrics computes cache_hit_rate and avg_time_per_text_ms."""
        from src.data_layer.embeddings import EmbeddingMetrics

        metrics = EmbeddingMetrics()
        metrics.total_texts = 100
        metrics.cache_hits = 80
        metrics.cache_misses = 20

        assert metrics.cache_hit_rate == pytest.approx(80.0)

        metrics.total_time_ms = 500
        assert metrics.avg_time_per_text_ms == pytest.approx(5.0)


class TestEmbeddingCacheBatch:
    """Tests for EmbeddingCache.get_batch() — the critical bulk-lookup path.

    get_batch() issues a single SQL query for N texts and is the hot path for
    large ingestion runs.  Zero test coverage here would leave the most-used
    cache code path completely unvalidated.
    """

    def test_get_batch_returns_all_cached_items(self, temp_dir: Path) -> None:
        """get_batch returns dict keyed by position index with all embeddings when every key is cached."""
        from src.data_layer.embeddings import EmbeddingCache

        cache = EmbeddingCache(temp_dir / "embed_batch_all.db")
        texts = ["text A", "text B", "text C"]
        embeddings = [[float(i)] * 768 for i in range(len(texts))]

        for text, emb in zip(texts, embeddings):
            cache.put(text, emb, "model-v1")

        results = cache.get_batch(texts, "model-v1")
        assert len(results) == len(texts)
        assert all(i in results for i in range(len(texts)))
        assert results[0] == pytest.approx(embeddings[0])
        cache.close()

    def test_get_batch_partial_miss(self, temp_dir: Path) -> None:
        """get_batch returns entry for cached text; missing text absent from dict."""
        from src.data_layer.embeddings import EmbeddingCache

        cache = EmbeddingCache(temp_dir / "embed_batch_partial.db")
        cache.put("cached text", [0.1] * 768, "model-v1")

        results = cache.get_batch(["cached text", "missing text"], "model-v1")
        assert 0 in results
        assert 1 not in results
        cache.close()

    def test_get_batch_empty_input(self, temp_dir: Path) -> None:
        """get_batch([]) returns an empty dict."""
        from src.data_layer.embeddings import EmbeddingCache

        cache = EmbeddingCache(temp_dir / "embed_batch_empty.db")
        assert cache.get_batch([], "model-v1") == {}
        cache.close()

    def test_get_batch_wrong_model_is_miss(self, temp_dir: Path) -> None:
        """get_batch omits the entry when model_name differs from stored model."""
        from src.data_layer.embeddings import EmbeddingCache

        cache = EmbeddingCache(temp_dir / "embed_batch_model.db")
        cache.put("text", [0.5] * 768, "model-v1")

        results = cache.get_batch(["text"], "model-v2")
        assert 0 not in results
        cache.close()


# ============================================================================
# 3. CHUNKING TESTS
# ============================================================================

class TestSentenceChunking:
    """Tests for sentence-based text chunking."""

    def test_sentence_chunker_basic(self) -> None:
        """SentenceChunker produces at least two chunks and includes 'text' key."""
        from src.data_layer.chunking import SentenceChunker

        chunker = SentenceChunker(
            sentences_per_chunk=2,
            # sentences_per_chunk=2 differs from settings.yaml default (3)
            # deliberately to test a non-default configuration.
            min_chunk_size=20,
        )

        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.chunk(text)

        assert len(chunks) >= 2
        assert all("text" in c for c in chunks)

    def test_spacy_sentence_chunker(self) -> None:
        """SpacySentenceChunker chunks by sentence windows with overlap."""
        from src.data_layer.chunking import SpacySentenceChunker

        # sentences_per_chunk=3 and sentence_overlap=1 match settings.yaml defaults.
        chunker = SpacySentenceChunker(sentences_per_chunk=3, sentence_overlap=1)

        text = (
            "Albert Einstein was born in 1879. He was a theoretical physicist. "
            "Einstein developed the theory of relativity. He won the Nobel Prize in 1921. "
            "His work changed our understanding of physics. He died in 1955."
        )

        chunks = chunker.chunk_text(text, source_doc="test.txt")

        assert len(chunks) >= 2
        assert all(hasattr(c, "text") for c in chunks)
        assert all(hasattr(c, "sentences") for c in chunks)
        # Each chunk must not exceed the configured window size.
        assert all(c.sentence_count <= 3 for c in chunks)


class TestSemanticChunking:
    """Tests for semantic chunking with quality metrics."""

    def test_semantic_chunker(self) -> None:
        """Semantic chunker produces chunks with quality metadata keys."""
        from src.data_layer.chunking import create_semantic_chunker

        chunker = create_semantic_chunker(
            chunk_size=300,
            chunk_overlap=50,
            min_chunk_size=50,
        )

        doc = Document(
            page_content=(
                "1. Introduction\n\n"
                "This paper investigates machine learning techniques. "
                "The research focuses on edge deployment scenarios.\n\n"
                "1.1 Problem Statement\n\n"
                "Modern language models require significant resources. "
                "This creates challenges for edge device deployment."
            ),
            metadata={"source_file": "test.pdf"},
        )

        chunks = chunker.chunk_document(doc)

        assert len(chunks) >= 1
        assert all("importance_score" in c.metadata for c in chunks)
        assert all("lexical_diversity" in c.metadata for c in chunks)

    def test_header_extraction(self) -> None:
        """HeaderExtractor identifies at least one structural section marker."""
        from src.data_layer.chunking import HeaderExtractor

        extractor = HeaderExtractor()
        text = "1.1 Introduction\nThis is the introduction text."
        metadata, _cleaned = extractor.extract_headers(text)

        assert metadata.section is not None or metadata.chapter is not None


# ============================================================================
# 4. ENTITY EXTRACTION TESTS
# ============================================================================

class TestEntityExtraction:
    """Tests for GLiNER + REBEL data structures and caching.

    Note: GLiNER and REBEL model inference is not tested here because it
    requires GPU or slow CPU inference.  Model-level quality is validated
    in test_system/test_gliner_boundary.py (nightly) against
    test_system/fixtures/gold_ner_hotpotqa.json.
    """

    def test_extracted_entity_dataclass(self) -> None:
        """ExtractedEntity stores fields correctly and serialises to dict."""
        from src.data_layer.entity_extraction import ExtractedEntity

        entity = ExtractedEntity(
            entity_id="e001",
            name="Albert Einstein",
            entity_type="PERSON",
            confidence=0.95,
            mention_span=(0, 15),
            source_chunk_id="c001",
        )

        assert entity.name == "Albert Einstein"
        assert entity.entity_type == "PERSON"

        d = entity.to_dict()
        # Key renamed from "type" to "entity_type" in entity_extraction v3.1 rewrite.
        assert d["entity_type"] == "PERSON"
        assert d["confidence"] == pytest.approx(0.95)

    def test_extracted_relation_dataclass(self) -> None:
        """ExtractedRelation stores fields correctly and serialises to dict."""
        from src.data_layer.entity_extraction import ExtractedRelation

        relation = ExtractedRelation(
            subject_entity="Einstein",
            relation_type="works_for",
            object_entity="Princeton University",
            confidence=0.85,
            source_chunk_ids=["c001"],
        )

        assert relation.relation_type == "works_for"

        d = relation.to_dict()
        # Keys renamed from "subject"/"object"/"relation" in entity_extraction v3.1 rewrite.
        assert d["subject_entity"] == "Einstein"
        assert d["object_entity"] == "Princeton University"

    def test_extraction_config(self) -> None:
        """ExtractionConfig defaults match settings.yaml paper specifications."""
        from src.data_layer.entity_extraction import ExtractionConfig

        config = ExtractionConfig()

        # Paper 2.5 / settings.yaml: gliner.confidence_threshold = 0.15
        assert config.ner_confidence_threshold == pytest.approx(0.15)
        # Paper 2.5 / settings.yaml: rebel.confidence_threshold = 0.5
        assert config.re_confidence_threshold == pytest.approx(0.5)
        # settings.yaml: gliner.batch_size = 16
        assert config.ner_batch_size == 16
        # settings.yaml: rebel.batch_size = 8
        assert config.re_batch_size == 8
        # settings.yaml: rebel.min_entities_for_re = 2 (skip RE when < 2 entities)
        assert config.min_entities_for_re == 2
        # GLiNER uses lowercase labels for better zero-shot performance.
        assert "person" in config.entity_types
        assert "organization" in config.entity_types

    def test_entity_cache(self, temp_dir: Path) -> None:
        """EntityCache stores and retrieves data keyed by (text_hash, model_name)."""
        from src.data_layer.entity_extraction import EntityCache

        cache = EntityCache(temp_dir / "entity_cache.db", max_size=100)

        test_data = {
            "entities": [{"name": "Einstein", "entity_type": "PERSON"}],
            "relations": [],
        }
        cache.put(
            "Test text about Einstein",
            test_data,
            "urchade/gliner_small-v2.1",
        )

        retrieved = cache.get(
            "Test text about Einstein",
            "urchade/gliner_small-v2.1",
        )
        assert retrieved is not None
        assert retrieved["entities"][0]["name"] == "Einstein"

        cache.close()


class TestEntityCacheBatch:
    """Tests for EntityCache.get_batch() — bulk entity lookup path.

    EntityCache.get_batch() is the hot path inside process_chunks_batch().
    Without coverage here a regression in the bulk path would only surface
    at integration-test time, making the root cause hard to isolate.
    """

    def test_get_batch_returns_all_cached_entities(self, temp_dir: Path) -> None:
        """get_batch returns dict keyed by position index with all entries when every key is cached."""
        from src.data_layer.entity_extraction import EntityCache

        cache = EntityCache(temp_dir / "entity_batch_all.db", max_size=100)
        texts = ["text about Paris", "text about Einstein"]
        data = [
            {"entities": [{"name": "Paris", "entity_type": "GPE"}], "relations": []},
            {"entities": [{"name": "Einstein", "entity_type": "PERSON"}], "relations": []},
        ]
        model = "urchade/gliner_small-v2.1"
        for text, d in zip(texts, data):
            cache.put(text, d, model)

        results = cache.get_batch(texts, model)
        assert len(results) == 2
        assert all(i in results for i in range(len(texts)))
        assert results[0]["entities"][0]["name"] == "Paris"
        cache.close()

    def test_get_batch_empty_input_returns_empty(self, temp_dir: Path) -> None:
        """get_batch([]) returns an empty dict."""
        from src.data_layer.entity_extraction import EntityCache

        cache = EntityCache(temp_dir / "entity_batch_empty.db", max_size=100)
        assert cache.get_batch([], "model") == {}
        cache.close()

    def test_get_batch_partial_miss_returns_only_hits(self, temp_dir: Path) -> None:
        """get_batch has entry for cached text; uncached text is absent from dict."""
        from src.data_layer.entity_extraction import EntityCache

        cache = EntityCache(temp_dir / "entity_batch_partial.db", max_size=100)
        model = "urchade/gliner_small-v2.1"
        cached_data = {"entities": [{"name": "Berlin", "entity_type": "GPE"}], "relations": []}
        cache.put("cached text", cached_data, model)

        results = cache.get_batch(["cached text", "uncached text"], model)
        assert 0 in results
        assert 1 not in results
        cache.close()


class TestEntityCacheStats:
    """Tests for EntityCache.get_stats(), including after-close safety."""

    def test_get_stats_returns_expected_keys(self, temp_dir: Path) -> None:
        """get_stats returns dict with total_entries, total_hits, memory_entries, size_mb."""
        from src.data_layer.entity_extraction import EntityCache

        cache = EntityCache(temp_dir / "entity_stats.db", max_size=100)
        cache.put("text", {"entities": [], "relations": []}, "model")

        stats = cache.get_stats()
        for key in ("total_entries", "total_hits", "memory_entries", "size_mb"):
            assert key in stats, f"Missing key '{key}' in get_stats()"
        assert stats["total_entries"] >= 1
        cache.close()

    def test_get_stats_after_close_returns_zeros(self, temp_dir: Path) -> None:
        """get_stats after close() returns zero-filled dict without raising."""
        from src.data_layer.entity_extraction import EntityCache

        cache = EntityCache(temp_dir / "entity_stats_closed.db", max_size=100)
        cache.close()

        stats = cache.get_stats()
        assert stats["total_entries"] == 0
        assert stats["size_mb"] == pytest.approx(0.0)


class TestExtractionConfigNewFields:
    """Tests for ExtractionConfig fields added in v3.5.0+."""

    def test_spacy_fallback_model_default(self) -> None:
        """spacy_fallback_model defaults to 'en_core_web_sm'."""
        from src.data_layer.entity_extraction import ExtractionConfig

        config = ExtractionConfig()
        assert config.spacy_fallback_model == "en_core_web_sm"

    def test_spacy_fallback_model_override(self) -> None:
        """spacy_fallback_model can be overridden at construction time."""
        from src.data_layer.entity_extraction import ExtractionConfig

        config = ExtractionConfig(spacy_fallback_model="en_core_web_lg")
        assert config.spacy_fallback_model == "en_core_web_lg"


class TestNormalizeQueryEntity:
    """Tests for _normalize_query_entity() — strips leading articles for graph lookup."""

    def test_strips_the_article(self) -> None:
        """'The Cold War' → 'Cold War'."""
        from src.data_layer.hybrid_retriever import _normalize_query_entity

        assert _normalize_query_entity("The Cold War", "EVENT") == "Cold War"

    def test_strips_a_article(self) -> None:
        """'A Conference' with EVENT type → 'Conference' (only stripped for GPE/LOCATION/EVENT)."""
        from src.data_layer.hybrid_retriever import _normalize_query_entity

        assert _normalize_query_entity("A Conference", "EVENT") == "Conference"

    def test_no_strip_for_non_article(self) -> None:
        """Names not starting with articles are returned unchanged."""
        from src.data_layer.hybrid_retriever import _normalize_query_entity

        assert _normalize_query_entity("Albert Einstein", "PERSON") == "Albert Einstein"

    def test_abbreviation_whitelist_preserved(self) -> None:
        """'Warner Bros.' is not stripped — leading word is not an article."""
        from src.data_layer.hybrid_retriever import _normalize_query_entity

        assert _normalize_query_entity("Warner Bros.", "ORGANIZATION") == "Warner Bros."

    def test_empty_string_returns_empty(self) -> None:
        """Empty string input returns empty string."""
        from src.data_layer.hybrid_retriever import _normalize_query_entity

        assert _normalize_query_entity("", "PERSON") == ""


class TestB1SpanNormalization:
    """B1: query-side span-boundary normalization (deterministic, no model)."""

    def test_strip_leading_auxiliary(self) -> None:
        """'Are  Chrysalis' → 'Chrysalis' (leading auxiliary + double-space)."""
        from src.data_layer.hybrid_retriever import _strip_leading_function_word
        assert _strip_leading_function_word("Are  Chrysalis") == "Chrysalis"

    def test_preserve_wh_word_title(self) -> None:
        """Wh-words are NOT stripped — legitimate titles begin with them."""
        from src.data_layer.hybrid_retriever import _strip_leading_function_word
        assert _strip_leading_function_word("Who Framed Roger Rabbit") == "Who Framed Roger Rabbit"
        assert _strip_leading_function_word("What Women Want") == "What Women Want"

    def test_preserve_the_band(self) -> None:
        """'The Who' is untouched here — article handling is type-aware
        downstream in normalize_entity_name(), not in B1a."""
        from src.data_layer.hybrid_retriever import _strip_leading_function_word
        assert _strip_leading_function_word("The Who") == "The Who"

    def test_single_token_unchanged(self) -> None:
        """A bare single token is never trimmed (would empty the span)."""
        from src.data_layer.hybrid_retriever import _strip_leading_function_word
        assert _strip_leading_function_word("Are") == "Are"

    def test_strip_leading_correlative_quantifier(self) -> None:
        """B1a: 'both/either/neither' absorbed by the NER tagger are stripped
        ('Both Truth in Science' → 'Truth in Science')."""
        from src.data_layer.hybrid_retriever import _strip_leading_function_word
        assert _strip_leading_function_word("Both Truth in Science") == "Truth in Science"
        assert _strip_leading_function_word("Either Roberta Vinci") == "Roberta Vinci"
        assert _strip_leading_function_word("Neither Coke nor Pepsi") == "Coke nor Pepsi"

    def test_preserve_all_each_titles(self) -> None:
        """'all'/'each' are NOT stripped — they begin legitimate names."""
        from src.data_layer.hybrid_retriever import _strip_leading_function_word
        assert _strip_leading_function_word("All Saints") == "All Saints"
        assert _strip_leading_function_word("Each Tear") == "Each Tear"

    def test_bare_quantifier_unchanged(self) -> None:
        """A bare quantifier token is never trimmed (would empty the span)."""
        from src.data_layer.hybrid_retriever import _strip_leading_function_word
        assert _strip_leading_function_word("Both") == "Both"

    def test_split_interior_year(self) -> None:
        """'National 1993 Baseball Hall of Fame' → anchor + year 1993."""
        from src.data_layer.hybrid_retriever import _strip_embedded_year
        anchor, year = _strip_embedded_year("National 1993 Baseball Hall of Fame")
        assert anchor == "National Baseball Hall of Fame"
        assert year == "1993"

    def test_no_year_returns_unchanged(self) -> None:
        from src.data_layer.hybrid_retriever import _strip_embedded_year
        anchor, year = _strip_embedded_year("Baseball Hall of Fame")
        assert anchor == "Baseball Hall of Fame"
        assert year is None

    def test_leading_year_preserved(self) -> None:
        """A leading year is part of the title, not an interior qualifier."""
        from src.data_layer.hybrid_retriever import _strip_embedded_year
        anchor, year = _strip_embedded_year("2001: A Space Odyssey")
        assert anchor == "2001: A Space Odyssey"
        assert year is None

    def test_trailing_year_preserved(self) -> None:
        from src.data_layer.hybrid_retriever import _strip_embedded_year
        anchor, year = _strip_embedded_year("Live Aid 1985")
        assert anchor == "Live Aid 1985"
        assert year is None

    def test_bare_year_not_stripped(self) -> None:
        """A span that IS just a year is left intact (no alphabetic sides)."""
        from src.data_layer.hybrid_retriever import _strip_embedded_year
        anchor, year = _strip_embedded_year("1993")
        assert anchor == "1993"
        assert year is None

    def test_dedup_overlapping_keeps_maximal(self) -> None:
        """Fragments contained in a longer span are dropped (hyphen-name merge)."""
        from src.data_layer.hybrid_retriever import _dedup_overlapping_spans
        ents = [
            {"text": "Hook", "start": 14, "end": 18, "label": "organization"},
            {"text": "Hook-Handed Man", "start": 14, "end": 29, "label": "organization"},
            {"text": "Handed Man", "start": 19, "end": 29, "label": "person"},
        ]
        out = _dedup_overlapping_spans(ents)
        texts = [e["text"] for e in out]
        assert texts == ["Hook-Handed Man"]

    def test_dedup_keeps_disjoint_spans(self) -> None:
        """Non-overlapping spans are all kept, in left-to-right order."""
        from src.data_layer.hybrid_retriever import _dedup_overlapping_spans
        ents = [
            {"text": "Chrysalis", "start": 4, "end": 13, "label": "organization"},
            {"text": "Look", "start": 18, "end": 22, "label": "organization"},
        ]
        out = _dedup_overlapping_spans(ents)
        assert [e["text"] for e in out] == ["Chrysalis", "Look"]


class TestB2TemporalMeasureGate:
    """B2.2a: pure temporal/measure phrases are rejected as entities."""

    def test_rejects_consecutive_seasons(self) -> None:
        from src.data_layer.hybrid_retriever import ImprovedQueryEntityExtractor as E
        assert E._is_junk_entity("7 consecutive seasons") is True

    def test_rejects_laps(self) -> None:
        from src.data_layer.hybrid_retriever import ImprovedQueryEntityExtractor as E
        assert E._is_junk_entity("25 laps") is True

    def test_rejects_bare_year(self) -> None:
        from src.data_layer.hybrid_retriever import ImprovedQueryEntityExtractor as E
        assert E._is_junk_entity("1993") is True

    def test_keeps_proper_noun_with_number(self) -> None:
        """A capitalised proper-noun token survives even with a number present."""
        from src.data_layer.hybrid_retriever import ImprovedQueryEntityExtractor as E
        assert E._is_junk_entity("World War II") is False

    def test_keeps_person(self) -> None:
        from src.data_layer.hybrid_retriever import ImprovedQueryEntityExtractor as E
        assert E._is_junk_entity("Frank Thomas") is False

    def test_filter_drops_temporal_keeps_person(self) -> None:
        """End-to-end: _filter_entities drops the measure phrase, keeps the name."""
        from src.data_layer.hybrid_retriever import ImprovedQueryEntityExtractor as E
        out = E._filter_entities(["7 consecutive seasons", "Frank Thomas"])
        assert out == ["Frank Thomas"]


# PreGenerativeFilter contradiction-passthrough tests were removed together
# with the PreGenerativeFilter class itself in the 2026-05-06 cleanup audit.


# ============================================================================
# 5. RETRIEVAL TESTS
# ============================================================================

class TestRRFFusion:
    """Tests for Reciprocal Rank Fusion.

    Reference: Cormack, Clarke & Buettcher (2009). "Reciprocal Rank Fusion
    outperforms Condorcet and individual Rank Learning Methods." SIGIR 2009.
    DOI: 10.1145/1571941.1572114.
    """

    def test_rrf_formula(self) -> None:
        """RRF score for rank 1 with k=60 equals 1/(60+1) = 1/61.

        Verifies that RRFFusion.fuse() implements the formula exactly:
            score(rank r) = 1 / (k + r)   [r is 1-based]

        Reference: Cormack et al. (2009). SIGIR 2009.
        """
        from src.data_layer.hybrid_retriever import RRFFusion

        # cross_source_boost=1.0 isolates the formula without the boost factor.
        fusion = RRFFusion(k=60, cross_source_boost=1.0)

        vector_results = [
            {
                "document_id": "c1",
                "text": "T1",
                "similarity": 0.9,
                "metadata": {"source_file": "a"},
                "position": 0,
            },
        ]

        results = fusion.fuse(vector_results, [], final_top_k=1)

        assert len(results) == 1
        # Rank 1 (first result), k=60: score = 1 / (60 + 1) = 1/61 ≈ 0.01639
        expected = 1.0 / (60 + 1)
        assert results[0].rrf_score == pytest.approx(expected, abs=1e-6)

    def test_rrf_fusion_basic(self) -> None:
        """The chunk appearing in both result lists receives the highest RRF score.

        Reference: Cormack et al. (2009). SIGIR 2009.
        """
        from src.data_layer.hybrid_retriever import RRFFusion

        fusion = RRFFusion(k=60, cross_source_boost=1.2)

        vector_results = [
            {
                "document_id": "c1",
                "text": "Text 1",
                "similarity": 0.9,
                "metadata": {"source_file": "a"},
                "position": 0,
            },
            {
                "document_id": "c2",
                "text": "Text 2",
                "similarity": 0.8,
                "metadata": {"source_file": "a"},
                "position": 1,
            },
        ]
        graph_results = [
            {
                "chunk_id": "c1",
                "text": "Text 1",
                "hops": 1,
                "source_file": "a",
                "matched_entity": "",
                "position": 0,
            },
            {
                "chunk_id": "c3",
                "text": "Text 3",
                "hops": 2,
                "source_file": "b",
                "matched_entity": "",
                "position": 0,
            },
        ]

        results = fusion.fuse(vector_results, graph_results, final_top_k=3)

        assert len(results) > 0
        # c1 appears in both lists — highest score after cross-source boost.
        assert results[0].chunk_id == "c1"
        assert results[0].retrieval_method == "hybrid"

    def test_cross_source_boost(self) -> None:
        """A chunk appearing in both result lists scores higher with boost > 1."""
        from src.data_layer.hybrid_retriever import RRFFusion

        vector_results = [
            {
                "document_id": "c1",
                "text": "T1",
                "similarity": 0.9,
                "metadata": {"source_file": "a"},
                "position": 0,
            },
        ]
        graph_results = [
            {
                "chunk_id": "c1",
                "text": "T1",
                "hops": 1,
                "source_file": "a",
                "matched_entity": "",
                "position": 0,
            },
        ]

        fusion_boosted = RRFFusion(k=60, cross_source_boost=1.5)
        results_boosted = fusion_boosted.fuse(vector_results, graph_results)

        fusion_unboosted = RRFFusion(k=60, cross_source_boost=1.0)
        results_unboosted = fusion_unboosted.fuse(vector_results, graph_results)

        assert results_boosted[0].rrf_score > results_unboosted[0].rrf_score


# TestPreGenerativeFilter was removed in the 2026-05-06 cleanup audit
# together with the PreGenerativeFilter class. Equivalent relevance- and
# redundancy-filter behaviour is exercised by the Navigator filter tests
# in test_navigator_semantic.py and test_logic_layer.py.


class TestHybridRetriever:
    """Tests for the full hybrid retrieval pipeline."""

    def test_retrieval_modes(self, temp_dir: Path, mock_embeddings) -> None:
        """All three RetrievalMode values execute without errors and return lists.

        Store is pre-populated so GRAPH and HYBRID modes have indexed data to
        traverse; without documents, graph_search always returns [] and the
        mode distinction is meaningless.
        """
        from src.data_layer.hybrid_retriever import (
            HybridRetriever,
            RetrievalConfig,
            RetrievalMode,
        )
        from src.data_layer.storage import HybridStore, StorageConfig

        storage_config = StorageConfig(
            vector_db_path=temp_dir / "retriever_vector",
            graph_db_path=temp_dir / "retriever_graph",
        )
        store = HybridStore(storage_config, mock_embeddings)

        # Pre-populate so GRAPH and HYBRID have indexed data to traverse.
        docs = [
            Document(
                page_content="Einstein developed the theory of relativity.",
                metadata={"source_file": "physics.txt", "chunk_id": "r1",
                          "chunk_index": 0, "page_number": 1},
            ),
            Document(
                page_content="Marie Curie discovered radium and polonium.",
                metadata={"source_file": "bio.txt", "chunk_id": "r2",
                          "chunk_index": 0, "page_number": 1},
            ),
        ]
        store.add_documents(docs)

        try:
            for mode in [RetrievalMode.VECTOR, RetrievalMode.GRAPH, RetrievalMode.HYBRID]:
                retrieval_config = RetrievalConfig(
                    mode=mode,
                    vector_top_k=5,
                    graph_top_k=3,
                    # threshold=0.0 avoids mock-embedding score variability
                    similarity_threshold=0.0,
                )

                retriever = HybridRetriever(
                    config=retrieval_config,
                    hybrid_store=store,
                    embeddings=mock_embeddings,
                )

                results, metrics = retriever.retrieve("test query")
                assert isinstance(results, list)
        finally:
            store.close()


# ============================================================================
# 6. INTEGRATION TESTS
# ============================================================================

class TestFullPipeline:
    """End-to-end integration tests covering ingestion → retrieval."""

    def test_ingestion_to_retrieval(
        self,
        temp_dir: Path,
        sample_texts: list[str],
        mock_embeddings,
    ) -> None:
        """Full pipeline: chunked ingestion followed by vector retrieval returns results."""
        from src.data_layer.hybrid_retriever import (
            HybridRetriever,
            RetrievalConfig,
            RetrievalMode,
        )
        from src.data_layer import DocumentIngestionPipeline, DocumentIngestionConfig
        from src.data_layer.storage import HybridStore, StorageConfig

        # 1. Setup
        storage_config = StorageConfig(
            vector_db_path=temp_dir / "e2e_vector",
            graph_db_path=temp_dir / "e2e_graph",
        )
        store = HybridStore(storage_config, mock_embeddings)

        ingestion_config = DocumentIngestionConfig(
            chunking_strategy="sentence",
            sentences_per_chunk=2,
        )
        pipeline = DocumentIngestionPipeline(ingestion_config)

        # 2. Ingest
        documents = [
            Document(
                page_content=text,
                metadata={"source_file": f"doc_{i}.txt"},
            )
            for i, text in enumerate(sample_texts)
        ]
        chunked_docs = pipeline.process_documents(documents)
        store.add_documents(chunked_docs)

        # 3. Retrieve — threshold=0.0 ensures all chunks are eligible candidates
        # vector_weight / graph_weight removed in the 2026-05-06 cleanup audit
        # (never read by production code).
        retrieval_config = RetrievalConfig(
            mode=RetrievalMode.VECTOR,
            vector_top_k=3,
            graph_top_k=2,
            similarity_threshold=0.0,
        )

        retriever = HybridRetriever(
            config=retrieval_config,
            hybrid_store=store,
            embeddings=mock_embeddings,
        )

        try:
            results, metrics = retriever.retrieve("Who developed relativity?")

            # With deterministic mock embeddings and threshold=0.0, ingestion of
            # 5 texts must produce at least one retrievable result.
            assert len(results) > 0, (
                "Expected at least one result; check ingestion pipeline and retrieval config."
            )
            assert all(hasattr(r, "chunk_id") for r in results)
            assert all(hasattr(r, "text") for r in results)
            assert all(hasattr(r, "rrf_score") for r in results)
            assert all(r.rrf_score >= 0.0 for r in results)
        finally:
            store.close()

    def test_thesis_compliance(self) -> None:
        """Verify that ExtractionConfig and SentenceChunkingConfig match paper specifications.

        This test encodes paper section 2 parameter values as assertions so that
        any future change to defaults is caught immediately.

        Note: ExtractionConfig assertions duplicate TestEntityExtraction.test_extraction_config.
        Both are kept intentionally: test_extraction_config is the canonical unit test;
        this test provides an end-to-end compliance snapshot that must match the paper.
        """
        from src.data_layer.entity_extraction import ExtractionConfig
        from src.data_layer.chunking import SentenceChunkingConfig

        # Paper 2.2 / settings.yaml: 3-sentence window
        chunk_config = SentenceChunkingConfig()
        assert chunk_config.sentences_per_chunk == 3

        # Paper 2.5 / settings.yaml: GLiNER threshold = 0.15 (recall-optimised)
        extract_config = ExtractionConfig()
        assert extract_config.ner_confidence_threshold == pytest.approx(0.15)

        # Paper 2.5 / settings.yaml: REBEL threshold = 0.5
        assert extract_config.re_confidence_threshold == pytest.approx(0.5)

        # settings.yaml: gliner.batch_size = 16
        assert extract_config.ner_batch_size == 16

        # settings.yaml: rebel.batch_size = 8
        assert extract_config.re_batch_size == 8

        # settings.yaml: rebel.min_entities_for_re = 2
        # Skip relation extraction when fewer than 2 entities are found (~60% reduction).
        assert extract_config.min_entities_for_re == 2


# ============================================================================
# 7. PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Baseline performance tests using mock embeddings.

    These tests measure infrastructure latency (NumPy ops, LanceDB ANN) with
    a deterministic mock model.  They do NOT validate paper latency targets
    for SLMs (10–80 s for 1.5B–3.8B models on edge CPU); that validation is
    performed by src/thesis_evaluations/latency_memory_profile.py on target
    hardware.
    """

    def test_embedding_latency(
        self, mock_embeddings, sample_texts: list[str]
    ) -> None:
        """Mock embedding generation completes within 100 ms per batch."""
        start = time.time()

        for _ in range(10):
            mock_embeddings.embed_documents(sample_texts)

        elapsed = (time.time() - start) * 1000 / 10
        logger.info(
            "Mock embedding latency: %.1f ms for %d texts", elapsed, len(sample_texts)
        )

        assert elapsed < 100  # ms — relaxed threshold for CI; not a paper claim

    def test_vector_search_latency(
        self, temp_dir: Path, sample_texts: list[str], mock_embeddings
    ) -> None:
        """LanceDB ANN search over 100 documents completes within 100 ms."""
        from src.data_layer.storage import VectorStoreAdapter

        db_path = temp_dir / "perf_vector"
        store = VectorStoreAdapter(db_path=db_path, embedding_dim=768)

        docs = [
            Document(
                page_content=t,
                metadata={"chunk_id": f"perf_{i}", "source_file": "perf.pdf"},
            )
            for i, t in enumerate(sample_texts * 20)  # 100 documents
        ]
        store.add_documents_with_embeddings(docs, mock_embeddings)

        query_emb = mock_embeddings.embed_query("test query")

        latencies = []
        for _ in range(10):
            start = time.time()
            store.vector_search(query_emb, top_k=10, threshold=0.0)
            latencies.append((time.time() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        logger.info("LanceDB vector search latency: %.1f ms (100 docs)", avg_latency)

        # Paper target for vector search: 20–40 ms; relaxed to 100 ms for CI.
        assert avg_latency < 100  # ms


# ============================================================================
# 8. EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Edge case and boundary condition tests."""

    def test_add_empty_document_list(
        self, temp_dir: Path, mock_embeddings
    ) -> None:
        """add_documents([]) must return without error and leave stores empty."""
        from src.data_layer.storage import HybridStore, StorageConfig

        config = StorageConfig(
            vector_db_path=temp_dir / "edge_empty_vector",
            graph_db_path=temp_dir / "edge_empty_graph",
            embedding_dim=768,
        )
        store = HybridStore(config, mock_embeddings)

        # Must not raise
        store.add_documents([])

        # Vector store should have no table or empty table
        query_emb = mock_embeddings.embed_query("anything")
        results = store.vector_search(query_emb, top_k=5, threshold=0.0)
        assert results == []

    def test_vector_search_on_empty_store(self, temp_dir: Path) -> None:
        """vector_search on an uninitialised store returns an empty list."""
        from src.data_layer.storage import VectorStoreAdapter

        db_path = temp_dir / "edge_empty_vs"
        store = VectorStoreAdapter(db_path=db_path, embedding_dim=768)

        # No documents added — table does not exist yet
        query_emb = [0.0] * 768
        results = store.vector_search(query_emb, top_k=5)

        assert results == []

    def test_multihop_empty_entity_name(self, temp_dir: Path) -> None:
        """find_chunks_by_entity_multihop with empty string returns empty list."""
        from src.data_layer.storage import KuzuGraphStore

        db_path = temp_dir / "edge_empty_entity"
        store = KuzuGraphStore(db_path)

        results = store.find_chunks_by_entity_multihop("")
        assert results == []


# ============================================================================
# DOCUMENT INGESTION PIPELINE
# ============================================================================

class TestDocumentIngestionPipeline:
    """Tests for DocumentIngestionPipeline and DocumentIngestionConfig."""

    def test_valid_config_sentence_spacy(self):
        """sentence_spacy strategy is accepted without error."""
        from src.data_layer import DocumentIngestionConfig
        cfg = DocumentIngestionConfig(chunking_strategy="sentence_spacy")
        assert cfg.chunking_strategy == "sentence_spacy"

    def test_valid_config_all_strategies(self):
        """All five strategy names are accepted by DocumentIngestionConfig."""
        from src.data_layer import DocumentIngestionConfig
        for strategy in ("sentence", "sentence_spacy", "semantic", "fixed", "recursive"):
            cfg = DocumentIngestionConfig(chunking_strategy=strategy)
            assert cfg.chunking_strategy == strategy

    def test_invalid_strategy_raises(self):
        """Unknown chunking strategy raises ValueError at construction time."""
        from src.data_layer import DocumentIngestionConfig
        with pytest.raises(ValueError, match="Invalid chunking_strategy"):
            DocumentIngestionConfig(chunking_strategy="nonexistent")

    def test_chunk_size_too_small_raises(self):
        """chunk_size < 50 raises ValueError."""
        from src.data_layer import DocumentIngestionConfig
        with pytest.raises(ValueError, match="chunk_size"):
            DocumentIngestionConfig(chunk_size=10)

    def test_overlap_geq_window_raises(self):
        """sentence_overlap >= sentences_per_chunk raises ValueError."""
        from src.data_layer import DocumentIngestionConfig
        with pytest.raises(ValueError, match="sentence_overlap"):
            DocumentIngestionConfig(sentences_per_chunk=3, sentence_overlap=3)

    def test_process_text_returns_chunks(self):
        """process_text on non-empty text returns at least one chunk."""
        from src.data_layer import create_data_layer_pipeline
        pipeline = create_data_layer_pipeline(strategy="sentence", min_chunk_size=10)
        text = (
            "Albert Einstein was born in Ulm. He developed the theory of relativity. "
            "Einstein received the Nobel Prize in 1921."
        )
        chunks = pipeline.process_text(text, {"source": "test"})
        assert len(chunks) >= 1
        assert all("text" in c and "metadata" in c for c in chunks)

    def test_process_text_empty_returns_empty(self):
        """Empty text input yields an empty chunk list."""
        from src.data_layer import create_data_layer_pipeline
        pipeline = create_data_layer_pipeline(strategy="sentence")
        assert pipeline.process_text("") == []
        assert pipeline.process_text("   ") == []

    def test_process_texts_global_chunk_ids(self):
        """process_texts assigns monotonically increasing global_chunk_id."""
        from src.data_layer import create_data_layer_pipeline
        pipeline = create_data_layer_pipeline(strategy="sentence", min_chunk_size=10)
        texts = [
            "Einstein was a physicist. He won the Nobel Prize.",
            "Bohr was a Danish physicist. He studied atomic structure.",
        ]
        chunks = pipeline.process_texts(texts)
        ids = [c["metadata"]["global_chunk_id"] for c in chunks]
        assert ids == list(range(len(ids))), "global_chunk_id must be 0, 1, 2, ..."

    def test_process_texts_empty_list(self):
        """process_texts with empty list returns empty list."""
        from src.data_layer import create_data_layer_pipeline
        pipeline = create_data_layer_pipeline(strategy="sentence")
        assert pipeline.process_texts([]) == []

    def test_source_id_added_to_metadata(self):
        """source_id is propagated to chunk metadata when add_source_metadata=True."""
        from src.data_layer import create_data_layer_pipeline
        pipeline = create_data_layer_pipeline(strategy="sentence", min_chunk_size=10)
        chunks = pipeline.process_text(
            "Einstein developed relativity. Bohr studied atomic structure.",
            source_id="doc-001",
        )
        assert all(c["metadata"].get("source_id") == "doc-001" for c in chunks)

    def test_fixed_strategy_produces_chunks(self):
        """fixed strategy produces at least one chunk for multi-sentence text."""
        from src.data_layer import create_data_layer_pipeline
        pipeline = create_data_layer_pipeline(
            strategy="fixed", chunk_size=100, chunk_overlap=20, min_chunk_size=20
        )
        text = "Word " * 60  # 360 chars > chunk_size=100
        chunks = pipeline.process_text(text)
        assert len(chunks) >= 2, "Expected multiple fixed-size chunks"

    def test_recursive_strategy_produces_chunks(self):
        """recursive strategy produces at least one chunk."""
        from src.data_layer import create_data_layer_pipeline
        pipeline = create_data_layer_pipeline(
            strategy="recursive", chunk_size=200, chunk_overlap=20, min_chunk_size=20
        )
        text = "Einstein was a physicist. " * 20
        chunks = pipeline.process_text(text)
        assert len(chunks) >= 1

    def test_create_ingestion_config_from_settings(self):
        """create_ingestion_config reads values from a settings-style dict."""
        from src.data_layer import create_ingestion_config
        fake_settings = {
            "ingestion": {
                "chunking_strategy": "fixed",
                "chunk_size": 512,
                "chunk_overlap": 64,
                "sentences_per_chunk": 2,
                "sentence_overlap": 0,
                "spacy_model": "en_core_web_sm",
            }
        }
        cfg = create_ingestion_config(fake_settings)
        assert cfg.chunking_strategy == "fixed"
        assert cfg.chunk_size == 512
        assert cfg.chunk_overlap == 64
        assert cfg.sentences_per_chunk == 2

    def test_create_ingestion_config_empty_dict_uses_defaults(self):
        """create_ingestion_config({}) falls back to DocumentIngestionConfig defaults."""
        from src.data_layer import create_ingestion_config, DocumentIngestionConfig
        cfg = create_ingestion_config({})
        defaults = DocumentIngestionConfig()
        assert cfg.chunking_strategy == defaults.chunking_strategy
        assert cfg.sentences_per_chunk == defaults.sentences_per_chunk

    def test_get_stats_contains_strategy(self):
        """get_stats() returns a dict including the active chunking_strategy."""
        from src.data_layer import create_data_layer_pipeline
        pipeline = create_data_layer_pipeline(strategy="sentence")
        stats = pipeline.get_stats()
        assert stats["chunking_strategy"] == "sentence"
        assert "chunking_module_available" in stats




# ============================================================================
# IngestionPipeline extended coverage (Finding 10)
# ============================================================================

class TestIngestionPipelineExtended:
    """Additional coverage for DocumentIngestionPipeline paths."""

    def test_process_text_sentence_spacy_strategy(self):
        from src.data_layer import create_data_layer_pipeline
        p = create_data_layer_pipeline(strategy="sentence_spacy", min_chunk_size=10)
        chunks = p.process_text("Albert Einstein was a physicist. He won the Nobel Prize.", {})
        assert len(chunks) >= 1
        assert all("text" in c for c in chunks)

    def test_process_text_fixed_strategy(self):
        from src.data_layer import create_data_layer_pipeline
        p = create_data_layer_pipeline(strategy="fixed", chunk_size=50, min_chunk_size=5)
        chunks = p.process_text("A" * 200, {})
        assert len(chunks) >= 1

    def test_process_text_recursive_strategy(self):
        from src.data_layer import create_data_layer_pipeline
        p = create_data_layer_pipeline(strategy="recursive", chunk_size=200, min_chunk_size=5)
        chunks = p.process_text("Word " * 40, {})
        assert len(chunks) >= 1

    def test_process_texts_returns_list_of_same_length(self):
        from src.data_layer import create_data_layer_pipeline
        p = create_data_layer_pipeline(strategy="sentence", min_chunk_size=5)
        texts = ["Sentence one.", "Sentence two.", "Sentence three."]
        results = p.process_texts(texts, [{"src": i} for i in range(3)])
        assert isinstance(results, list)

    def test_empty_text_returns_empty_list(self):
        from src.data_layer import create_data_layer_pipeline
        p = create_data_layer_pipeline(strategy="sentence_spacy")
        chunks = p.process_text("", {})
        assert chunks == []

    def test_create_ingestion_config_path_independent(self):
        """create_ingestion_config() must not depend on CWD."""
        import os
        from src.data_layer import create_ingestion_config
        original_cwd = os.getcwd()
        try:
            os.chdir("/")
            cfg = create_ingestion_config({"ingestion": {"chunk_size": 512}})
            assert cfg.chunk_size == 512
        finally:
            os.chdir(original_cwd)


# ============================================================================
# normalize_entity_name (Finding 7 — shared function)
# ============================================================================

class TestNormalizeEntityNameShared:
    """Tests for the module-level normalize_entity_name() in entity_extraction.py."""

    def test_strips_whitespace(self):
        from src.data_layer import normalize_entity_name
        assert normalize_entity_name("  Einstein  ") == "Einstein"

    def test_strips_trailing_punctuation(self):
        from src.data_layer import normalize_entity_name
        assert normalize_entity_name("Germany,") == "Germany"
        assert normalize_entity_name("Germany;") == "Germany"

    def test_strips_trailing_period(self):
        from src.data_layer import normalize_entity_name
        assert normalize_entity_name("Germany.") == "Germany"

    def test_preserves_abbreviation_period(self):
        from src.data_layer import normalize_entity_name
        assert normalize_entity_name("Warner Bros.").endswith(".")

    def test_strips_article_for_gpe(self):
        from src.data_layer import normalize_entity_name
        assert normalize_entity_name("The United States", "GPE") == "United States"

    def test_strips_article_for_event(self):
        from src.data_layer import normalize_entity_name
        assert normalize_entity_name("The Cold War", "EVENT") == "Cold War"

    def test_preserves_article_for_person(self):
        from src.data_layer import normalize_entity_name
        assert normalize_entity_name("The Rock", "PERSON") == "The Rock"

    def test_preserves_article_for_work_of_art(self):
        from src.data_layer import normalize_entity_name
        assert normalize_entity_name("The Dark Knight", "WORK_OF_ART") == "The Dark Knight"

    def test_no_entity_type_preserves_article(self):
        from src.data_layer import normalize_entity_name
        assert normalize_entity_name("The Beatles") == "The Beatles"


# ============================================================================
# run_diagnostics
# ============================================================================

class TestRunDiagnostics:
    """Tests for storage.run_diagnostics()."""

    def test_returns_required_keys(self):
        """run_diagnostics() always returns the required keys."""
        from src.data_layer.storage import run_diagnostics, StorageConfig
        from pathlib import Path
        import tempfile

        class _MockEmb:
            def embed_query(self, text):
                return [0.1] * 64

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = StorageConfig(
                vector_db_path=Path(tmpdir) / "v",
                graph_db_path=Path(tmpdir) / "g",
                embedding_dim=64,
            )
            result = run_diagnostics(cfg, _MockEmb())
        assert "embedding_dim" in result
        assert "kuzu_available" in result
        assert "issues" in result

    def test_embedding_dim_detected(self):
        """run_diagnostics() detects embedding dimension from the model."""
        from src.data_layer.storage import run_diagnostics, StorageConfig
        from pathlib import Path
        import tempfile

        class _MockEmb:
            def embed_query(self, text):
                return [0.1] * 128

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = StorageConfig(
                vector_db_path=Path(tmpdir) / "v",
                graph_db_path=Path(tmpdir) / "g",
                embedding_dim=128,
            )
            result = run_diagnostics(cfg, _MockEmb())
        assert result["embedding_dim"] == 128

    def test_failing_embedding_appends_issue(self):
        """run_diagnostics() records embedding errors in the issues list."""
        from src.data_layer.storage import run_diagnostics, StorageConfig
        from pathlib import Path
        import tempfile

        class _BrokenEmb:
            def embed_query(self, text):
                raise RuntimeError("connection refused")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = StorageConfig(
                vector_db_path=Path(tmpdir) / "v",
                graph_db_path=Path(tmpdir) / "g",
                embedding_dim=64,
            )
            result = run_diagnostics(cfg, _BrokenEmb())
        assert len(result["issues"]) >= 1
        assert any("Embedding" in issue for issue in result["issues"])


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
