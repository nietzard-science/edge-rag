# Test suite — evidence for the paper

**Last reviewed:** 2026-06-19 (doc-sync audit pass).
**Current total:** 721 passing tests under the default CI selection
(`-m "not nightly"`); 739 with the nightly model-loading subset
included. See REPRODUCE.md §Step 6 for both invocations.

The default CI run is `python -X utf8 -m pytest test_system/ -m "not nightly" -q`.
`nightly`-marked tests load model weights or live resources (GLiNER + REBEL in
`test_graph_inspect.py`, GLiNER inference in `test_gliner_boundary.py`) and are
deselected from that default; run them explicitly with `-m nightly`. Each test
file pins one or more design decisions documented in `TECHNICAL_ARCHITECTURE.md`.
The mapping below lets a reviewer locate the test evidence for any section of the paper / architecture doc.

| Test file | Doc section / design decision | What it pins |
|---|---|---|
| `test_chunking.py` | §3.2 Chunking | `SpacySentenceChunker` window/overlap; deterministic SHA-256 chunk IDs (T-E invariant); `ChunkQualityFilter` thresholds. |
| `test_embeddings.py` | §3.1 Embeddings | `embed_query` vs `embed_documents` dim + cosine-≥0.99 identity (T-B invariant); `ConnectionError → RuntimeError` wrap; SQLite cache hit semantics. |
| `test_gliner_boundary.py` *(nightly)* | §3.3 Entity & Relation Extraction | GLiNER span-boundary correctness on compound entities ("Eiffel Tower" stays one span — T-C invariant). |
| `test_data_layer.py` | §3.4 Storage; §3.5 Hybrid Retriever; §11.16.4 settings-wiring | `HybridStore` + `KuzuGraphStore` close/lock semantics (`close()` releases the KuzuDB exclusive lock so the next pipeline can open the same store); query-side NER normalization (`_strip_leading_function_word`, year-strip, span-dedup); `ImprovedQueryEntityExtractor._is_junk_entity` discriminativeness gate. Mock embeddings seed from a stable SHA-256 digest (cross-machine determinism). |
| `test_logic_layer.py` | §4.1–4.5 Logic Layer end-to-end | Configuration loaders; planner/navigator/verifier interface contracts; entity-mention filter safety fallback (§11.15 survivor floor); navigator filter ordering. |
| `test_planner_semantic.py` | §4.1 Planner; §11.8 pattern classification; §11.16.3 Phase-3.6 router | Query-type classification, Pattern E/F/G(+L)/H decomposition, Pattern I/J pre-empts, Phase-3.6 structural-comparison router (with bridge-cue precision guard), well-formedness invariant on emitted sub-queries. Source-hygiene guard checks `planner.py` against `fixtures/forbidden_source_terms.json`. |
| `test_navigator_semantic.py` | §4.2 Navigator; §11.13 IDF specificity; §11.15 survivor floor; §11.16.2 fair-cap | RRF fusion + cross-source boost; the 6-stage filter chain; `_fair_cap_by_subquery` per-anchor fairness merge for parallel decompositions (single-hop no-op assertion). |
| `test_verifier_semantic.py` | §4.3 Verifier; §11.13 IDF/structural-coverage; §11.16.1 cap-by-RRF-first | Pre-validation, entity-path validation, credibility scoring; `_reorder_by_question_relevance` IDF + length-norm + structural-coverage floor; **membership-invariant test for the cap-by-RRF-first contract** (a high-RRF answer chunk cannot be evicted by the lexical-overlap reorder); sentence-aware per-doc truncation (`_truncate_sentence_aware` keeps the answer-bearing sentence when the chunk exceeds `max_chars_per_doc`). |
| `test_pipeline.py` | §4.4 AgentPipeline | FIFO cache (T-D invariant: FIFO not LRU); lazy agent construction; per-stage timing surfaced on `PipelineResult`; `_close_pipeline` lock-release in ablation loop. |
| `test_thesis_matrix.py`, `test_thesis_matrix_ext.py` | §11 Design Decisions (capability matrix) | Coverage of §11.1–11.16 — every documented design decision has at least one test pinning the observable behaviour. |
| `test_missing_coverage.py` | §10.3 Test coverage (gap-filling) | Edge cases the higher-level tests miss; small surface-area guards. |
| `test_config_robustness.py` | §11.16.5 `_REQUIRED_SETTINGS` reproducibility guard | Config-loader robustness: missing keys produce a WARNING, not a crash; defaults match documented values; the required-settings validator runs. |
| `test_thesis_cleanup.py` | §4, §7 evaluation-tooling support | Infrastructure invariants for the evaluation tooling and controller/coreference contracts: ablation config parsing + store-path scoping, coreference optionality, the `EmbeddingMetrics` field surface, and `AgenticController`'s static-helper surface. |
| `test_bootstrap.py` | §7 Evaluation Layer (significance testing) | Paired-bootstrap CI/p-value semantics for the ablation tables. |
| `test_graph_inspect.py` *(nightly)* | §3.4 Graph quality | Verifies the HybridStore writes entity nodes, MENTIONS edges, and RELATED_TO triples to KuzuDB after ingesting a small synthetic corpus. Builds its own temporary graph; loads GLiNER + REBEL weights (hence `nightly`). |

## Markers (`pytest.ini`)
- `slow` — long-running unit tests (deselected by default; run with `-m slow`).
- `nightly` — tests that load model weights (GLiNER / REBEL); not run in the CI
  default. They build their own fixtures and need no populated store.
- `llm` — tests that hit a live Ollama daemon; deselected unless explicitly requested.
- `integration` — multi-component tests that don't fit unit scope.

## Developer utilities (not tests)
- `graph_inspect.py` — KuzuDB health-check CLI (node/edge counts, sample edges)
  after ingestion. Run: `python test_system/graph_inspect.py --dataset hotpotqa`.
- `graph_3d.py` — generates the Chapter-4 knowledge-graph figure to
  `docs/figures/` (PNG + optional interactive HTML).

## Test invariants (named guarantees pinned across files)
- **T-A** — `verifier.py`: answer with entity absent from context → violated_claims or LOW confidence.
- **T-B** — `embeddings.py`: `embed_query`/`embed_documents` same dim, cosine ≥ 0.99 for identical text.
- **T-C** — `entity_extraction`: compound spans ("Eiffel Tower") extracted as one entity.
- **T-D** — `agent_pipeline.AgentPipeline._cache` is FIFO (not LRU).
- **T-E** — `ingestion`: source_doc metadata isolated per `_chunk_document()` call; chunk IDs globally unique.

Each invariant has a deterministic test that fails loudly if the contract drifts.
