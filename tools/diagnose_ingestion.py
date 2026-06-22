"""
Ingestion diagnostic: trace chunks from source corpus through the stores
into retrieval rank.

For each question index this script follows the data path:
    chunks_export.json -> LanceDB (vector) + KuzuDB (graph)
                       -> vector retrieval rank + graph retrieval rank
                       -> per-entity "crowd-out" analysis

Diagnoses three common failure modes: a supporting article never
ingested; a supporting chunk ingested but mis-ranked by vector
retrieval; a supporting chunk ingested but crowded out by competing
chunks on the same entity in the graph. All stores are opened read-only.

Used as a verification + triage step after an ingestion run, before
launching a full evaluation (see REPRODUCE.md).

Exports
-------
- diagnose_question(...)   -- full trace for one question, prints report
- check_lancedb(...)       -- (found, rows) for a source_file in LanceDB
- check_kuzu(...)          -- (found, chunk_ids) for a source_file in KuzuDB
- check_graph_entities(...)-- entity names recorded for a chunk
- check_vector_rank(...)   -- 1-based rank of first hit from a source
- check_graph_rank(...)    -- 1-based rank of first hit from a source
- parse_indices(spec)      -- "11,12" / "0-19" / "5" -> List[int]
- main()                   -- CLI entry point

Dependencies / Requirements
---------------------------
- src.data_layer.storage.KuzuGraphStore         -- graph reader
- src.data_layer.storage.VectorStoreAdapter     -- vector reader
- src.data_layer.embeddings.BatchedOllamaEmbeddings  -- query encoding
- src.thesis_evaluations.benchmark_datasets.load_config_file
                                                -- settings.yaml access
- lancedb                                       -- vector store backend
- ollama server reachable at config.llm.base_url  (only when --vector is set)
- data/<dataset>/{questions.json, chunks_export.json,
                  graph/extraction_results.json, vector/, graph/}

Usage (single line; -X utf8 required on Windows / PowerShell):
    python -X utf8 diagnose_ingestion.py --indices 11,12
    python -X utf8 diagnose_ingestion.py --indices 0-19 --dataset hotpotqa
    python -X utf8 diagnose_ingestion.py --indices 11,12 --vector

Last reviewed: 2026-05-30 (audit pass, project version 5.4)
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Why: anchor data paths on the project root so the CLI works regardless
# of cwd, and put _PROJECT_ROOT on sys.path so `from src.*` imports
# resolve when this script is launched from a subdirectory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_DATA_ROOT = _PROJECT_ROOT / "data"
_CACHE_ROOT = _PROJECT_ROOT / "cache"


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# Why: nomic-embed-text emits 768-dim vectors. Hard-coded by the embedding
# model choice; surfaced as a constant so a model swap touches one line.
# Read from settings.yaml at runtime; this is the emergency fallback.
_NOMIC_EMBED_DIM = 768

# Why: rank cut-offs for the diagnostic. 20 for vector mirrors the
# default top_k_vectors in settings.yaml; 10 for graph reflects the
# tighter graph-traversal budget.
_VECTOR_RANK_TOP_K = 20
_GRAPH_RANK_TOP_K = 10
_CROWD_OUT_TOP_K = 20

# Why: vector retrieval threshold for the rank check is 0 because the
# diagnostic wants to see ALL candidates, not the production-filtered set.
_VECTOR_RANK_THRESHOLD = 0.0

# Why: print-truncation caps so one report line stays scannable.
_DISPLAY_CHUNKS_PER_SOURCE = 3
_DISPLAY_ANSWER_BEARING_CHUNKS = 5
_DISPLAY_GRAPH_CHUNK_IDS = 5
_DISPLAY_ENTITIES_PER_CHUNK = 8
_DISPLAY_CROWD_OUT_SOURCES = 8
_DISPLAY_CHUNK_TEXT_TRUNC = 160

# Why: cap on heuristic-extracted entity candidates per question -- the
# 6th-onward candidates are typically partial matches or stop-word
# capitalisations and add noise to the rank report.
_ENTITY_CANDIDATES_MAX = 5

# Why: rule widths for the printed report header and section separators.
_BAR_WIDTH = 72
_SECTION_RULE_WIDTH = 68

# ---------------------------------------------------------------------------
# Heuristic entity-extraction regexes (hoisted from diagnose_question)
# ---------------------------------------------------------------------------
# Why:    a quick entity candidate set for the rank check without invoking
#         the full GLiNER stack; matches the "obvious" surface forms a
#         human would copy out of the question text.
# What:   _QUOTED_RE matches anything inside double quotes; _TITLE_CASE_RE
#         matches >=2 consecutive title-cased tokens (a common English
#         noun-phrase shape for named entities).
# Misses: lower-cased proper nouns (e.g. "iPhone"), single-token
#         capitalisations (collapsed into background noise), non-Latin
#         scripts (Cyrillic / CJK), and entities split by an article
#         ("the United States" matches only "United States").

_QUOTED_RE = re.compile(r'"([^"]+)"')
_TITLE_CASE_RE = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')


# ---------------------------------------------------------------------------
# ANSI colour helpers (same family as diagnose_verbose.py)
# ---------------------------------------------------------------------------

def _esc(code: str) -> str:
    return f"\033[{code}m"


RESET = _esc("0")
BOLD = _esc("1")
DIM = _esc("2")
GREEN = _esc("32")
RED = _esc("31")
YELLOW = _esc("33")
CYAN = _esc("36")
BLUE = _esc("34")


def bold(s: str) -> str:
    return f"{BOLD}{s}{RESET}"


def dim(s: str) -> str:
    return f"{DIM}{s}{RESET}"


def green(s: str) -> str:
    return f"{GREEN}{s}{RESET}"


def red(s: str) -> str:
    return f"{RED}{s}{RESET}"


def yellow(s: str) -> str:
    return f"{YELLOW}{s}{RESET}"


def cyan(s: str) -> str:
    return f"{CYAN}{s}{RESET}"


def header(title: str) -> None:
    print()
    print("=" * _BAR_WIDTH)
    print(f"  {bold(title)}")
    print("=" * _BAR_WIDTH)


def section(title: str) -> None:
    print()
    print(f"  {bold(CYAN + title + RESET)}")
    print("  " + "-" * _SECTION_RULE_WIDTH)


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def warn(msg: str) -> None:
    print(f"  [!!] {msg}")


def fail(msg: str) -> None:
    print(f"  [XX] {msg}")


def info(msg: str) -> None:
    print(f"  ..  {msg}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_questions(dataset: str) -> List[Dict[str, Any]]:
    path = _DATA_ROOT / dataset / "questions.json"
    if not path.exists():
        print(red(f"questions.json not found: {path}"))
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_chunks_index(
    dataset: str,
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, List[Dict[str, Any]]],
    List[Dict[str, Any]],
]:
    """Returns (by_chunk_id, by_source_file, all_chunks)."""
    path = _DATA_ROOT / dataset / "chunks_export.json"
    if not path.exists():
        print(red(f"chunks_export.json not found: {path}"))
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    by_id: Dict[str, Dict[str, Any]] = {}
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for c in chunks:
        cid = str(c["metadata"].get("chunk_id", ""))
        by_id[cid] = c
        src = c["metadata"].get("source_file", "")
        by_source.setdefault(src, []).append(c)
    return by_id, by_source, chunks


def load_extractions_index(dataset: str) -> Dict[str, Dict[str, Any]]:
    path = _DATA_ROOT / dataset / "graph" / "extraction_results.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {str(r["chunk_id"]): r for r in data.get("results", [])}


# ---------------------------------------------------------------------------
# Store checks
# ---------------------------------------------------------------------------

def check_lancedb(
    dataset: str, source_file: str,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Returns (found, rows) for the given source_file in LanceDB."""
    try:
        import lancedb
        db = lancedb.connect(str(_DATA_ROOT / dataset / "vector"))
        tables = db.table_names()
        if not tables:
            return False, []
        tbl = db.open_table(tables[0])
        df = tbl.to_pandas()
        rows = df[df["source_file"] == source_file]
        return len(rows) > 0, rows.to_dict("records")
    except Exception as e:  # noqa: BLE001 -- diagnostic best-effort
        warn(f"LanceDB check failed: {e}")
        return False, []


def check_kuzu(
    source_file: str, graph_store: Optional[Any],
) -> Tuple[bool, List[str]]:
    """Returns (found, chunk_ids_in_graph) for the source_file."""
    if graph_store is None:
        return False, []
    try:
        res = graph_store.conn.execute(
            "MATCH (c:DocumentChunk {source_file: $sf}) RETURN c.chunk_id",
            {"sf": source_file},
        )
        cids = []
        while res.has_next():
            cids.append(str(res.get_next()[0]))
        return len(cids) > 0, cids
    except Exception as e:  # noqa: BLE001 -- diagnostic best-effort
        warn(f"KuzuDB check failed: {e}")
        return False, []


def check_graph_entities(
    chunk_id: str, extraction_index: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Return entity names stored for a chunk in extraction_results."""
    ext = extraction_index.get(str(chunk_id), {})
    return [e["name"] for e in ext.get("entities", [])]


# ---------------------------------------------------------------------------
# Retrieval rank check
# ---------------------------------------------------------------------------

def check_vector_rank(
    dataset: str, query: str, target_source: str,
    config: Dict[str, Any],
) -> Optional[int]:
    """Run a vector search for `query` and return the 1-based rank of the
    first chunk from `target_source`, or None when the target does not
    appear in the top-`_VECTOR_RANK_TOP_K` results.
    """
    try:
        from src.data_layer.storage import VectorStoreAdapter
        from src.data_layer.embeddings import BatchedOllamaEmbeddings

        emb_cfg = config.get("embeddings", {}) or {}
        embeddings = BatchedOllamaEmbeddings(
            model_name=emb_cfg.get("model_name", "nomic-embed-text"),
            base_url=emb_cfg.get("base_url", "http://localhost:11434"),
            cache_path=_CACHE_ROOT / f"{dataset}_embeddings.db",
        )
        vec = embeddings.embed_query(query)
        store = VectorStoreAdapter(
            db_path=str(_DATA_ROOT / dataset / "vector"),
            embedding_dim=int(emb_cfg.get("embedding_dim", _NOMIC_EMBED_DIM)),
        )
        results = store.vector_search(
            vec, top_k=_VECTOR_RANK_TOP_K, threshold=_VECTOR_RANK_THRESHOLD,
        )
        for rank, r in enumerate(results, 1):
            src = r.get("metadata", {}).get("source_file", r.get("source_file", ""))
            if src == target_source:
                return rank
        return None
    except Exception as e:  # noqa: BLE001 -- diagnostic best-effort
        warn(f"Vector rank check failed: {e}")
        return None


def check_graph_rank(
    graph_store: Optional[Any], entity_name: str, target_source: str,
) -> Optional[int]:
    """Run a graph search for `entity_name` and return the 1-based rank of
    the first result from `target_source`, or None when the target does
    not appear in the top-`_GRAPH_RANK_TOP_K` results.
    """
    if graph_store is None:
        return None
    try:
        results = graph_store.find_chunks_by_entity_multihop(
            entity_name, max_results=_GRAPH_RANK_TOP_K,
        )
        for rank, r in enumerate(results, 1):
            if r.get("source_file", "") == target_source:
                return rank
        return None
    except Exception as e:  # noqa: BLE001 -- diagnostic best-effort
        warn(f"Graph rank check failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-question report sections (decomposed from diagnose_question)
# ---------------------------------------------------------------------------

def _section_answer_bearing_chunks(
    gold: str, all_chunks: List[Dict[str, Any]],
) -> None:
    """Section 1: which chunks contain the gold answer (substring proxy)."""
    section("1. Answer-bearing chunks (text CONTAINS gold answer)")
    # Why: naive lower-cased substring containment is a coarse upper bound
    # on "the gold answer appears in this chunk". Used as triage signal,
    # not ground truth -- false positives are still useful diagnostic
    # output (they reveal where the model COULD have found the answer).
    gold_lower = gold.lower()
    answer_chunks = [c for c in all_chunks if gold_lower in c["text"].lower()]
    if answer_chunks:
        ok(f"{len(answer_chunks)} chunk(s) contain the gold answer")
        for c in answer_chunks[:_DISPLAY_ANSWER_BEARING_CHUNKS]:
            src = c["metadata"].get("source_file", "?")
            cid = c["metadata"].get("chunk_id", "?")
            info(f"chunk_id={cid}  source={src}")
            info(
                f"  text: "
                f"{c['text'][:_DISPLAY_CHUNK_TEXT_TRUNC].replace(chr(10), ' ')}"
            )
    else:
        fail(f"Gold answer '{gold}' not found in any chunk text")


def _section_per_source_presence(
    supporting_sources: List[str],
    by_source: Dict[str, List[Dict[str, Any]]],
    extraction_index: Dict[str, Dict[str, Any]],
    dataset: str,
    graph_store: Optional[Any],
) -> None:
    """Section 2: for each supporting article, confirm presence in chunks
    export, LanceDB, KuzuDB."""
    section("2. Supporting fact articles -> chunks in stores")

    for source in sorted(supporting_sources):
        print(f"\n  {bold(source)}")

        # 2a. In chunks_export.json?
        src_chunks = by_source.get(source, [])
        if src_chunks:
            ok(f"chunks_export.json: {len(src_chunks)} chunk(s)")
            for c in src_chunks[:_DISPLAY_CHUNKS_PER_SOURCE]:
                cid = c["metadata"].get("chunk_id", "?")
                ents = check_graph_entities(cid, extraction_index)
                info(f"  chunk_id={cid}  "
                     f"entities: {ents[:_DISPLAY_ENTITIES_PER_CHUNK]}")
        else:
            fail("NOT in chunks_export.json -- article was not ingested!")
            continue

        # 2b. In LanceDB?
        found_vec, vec_rows = check_lancedb(dataset, source)
        if found_vec:
            ok(f"LanceDB: {len(vec_rows)} row(s)")
        else:
            fail("NOT in LanceDB -- vector search cannot find this article")

        # 2c. In KuzuDB?
        found_graph, graph_cids = check_kuzu(source, graph_store)
        if found_graph:
            ok(f"KuzuDB: {len(graph_cids)} DocumentChunk node(s): "
               f"{graph_cids[:_DISPLAY_GRAPH_CHUNK_IDS]}")
        else:
            fail("NOT in KuzuDB -- graph search cannot find this article")


def _extract_entity_candidates(q_text: str) -> List[str]:
    """Heuristic entity extraction for the rank-check section."""
    candidates = _QUOTED_RE.findall(q_text)
    candidates += _TITLE_CASE_RE.findall(q_text)
    # Dedup preserving order, cap to keep the rank report scannable.
    return list(dict.fromkeys(candidates))[:_ENTITY_CANDIDATES_MAX]


def _section_retrieval_rank(
    q_text: str,
    supporting_sources: List[str],
    entity_candidates: List[str],
    dataset: str,
    graph_store: Optional[Any],
    config: Dict[str, Any],
    run_vector: bool,
) -> None:
    """Section 3: where do supporting chunks appear in retrieval results?"""
    section("3. Retrieval rank -- where do supporting chunks appear?")

    print(f"\n  {bold('Query entities (heuristic):')} {entity_candidates}")

    for source in sorted(supporting_sources):
        print(f"\n  {bold('Target:')} {source}")

        # Graph rank per entity
        for ent in entity_candidates:
            rank = check_graph_rank(graph_store, ent, source)
            if rank is not None:
                ok(f"Graph  entity={ent!r:30s} -> rank #{rank}")
            else:
                fail(f"Graph  entity={ent!r:30s} -> NOT in top-{_GRAPH_RANK_TOP_K}")

        # Vector rank
        if run_vector:
            rank = check_vector_rank(dataset, q_text, source, config)
            if rank is not None:
                ok(f"Vector query  -> rank #{rank} "
                   f"(out of {_VECTOR_RANK_TOP_K})")
            else:
                fail(f"Vector query  -> NOT in top-{_VECTOR_RANK_TOP_K}")
        else:
            info("Vector rank check skipped (use --vector to enable)")


def _section_crowd_out(
    entity_candidates: List[str],
    supporting_sources: List[str],
    graph_store: Optional[Any],
) -> None:
    """Section 4: how many chunks per source compete for each entity?"""
    section("4. Crowd-out analysis -- how many chunks compete for each entity?")

    if graph_store is None:
        warn("graph_store unavailable -- crowd-out analysis skipped.")
        return

    supporting_set = set(supporting_sources)

    for ent in entity_candidates:
        try:
            results = graph_store.find_chunks_by_entity_multihop(
                ent, max_results=_CROWD_OUT_TOP_K,
            )
            sources = [r.get("source_file", "?") for r in results]
            print(f"\n  {bold(f'Entity: {ent!r}')}")
            info(f"  {len(results)} graph results total")
            src_counts = Counter(sources)
            for src, cnt in src_counts.most_common(_DISPLAY_CROWD_OUT_SOURCES):
                marker = green("<- TARGET") if src in supporting_set else ""
                print(f"    {cnt}x  {src}  {marker}")
        except Exception as e:  # noqa: BLE001 -- diagnostic best-effort
            warn(f"  crowd-out check failed for {ent!r}: {e}")


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def diagnose_question(
    q_idx: int,
    question: Dict[str, Any],
    by_id: Dict[str, Dict[str, Any]],
    by_source: Dict[str, List[Dict[str, Any]]],
    all_chunks: List[Dict[str, Any]],
    extraction_index: Dict[str, Dict[str, Any]],
    graph_store: Optional[Any],
    dataset: str,
    run_vector: bool,
    config: Dict[str, Any],
) -> None:
    """Full ingestion -> retrieval trace for one question."""

    q_text = question["question"]
    gold = question["answer"]
    q_type = question.get("question_type", "?")
    facts = question.get("supporting_facts", [])

    header(f"idx={q_idx}  [{q_type}]")
    print(f"  {bold('Question:')}  {q_text}")
    print(f"  {bold('Gold:')}      {gold}")
    print(f"  {bold('Supporting facts:')} {facts}")

    # Collect supporting article source-keys from the dataset's
    # supporting_facts schema (each fact is [title, sent_idx]).
    supporting_sources: List[str] = []
    seen: set = set()
    for fact in facts:
        article_title = fact[0] if isinstance(fact, list) else fact.get("title", "")
        source_key = f"{dataset}_{article_title}"
        if source_key not in seen:
            seen.add(source_key)
            supporting_sources.append(source_key)

    _section_answer_bearing_chunks(gold, all_chunks)
    _section_per_source_presence(
        supporting_sources, by_source, extraction_index, dataset, graph_store,
    )
    entity_candidates = _extract_entity_candidates(q_text)
    _section_retrieval_rank(
        q_text, supporting_sources, entity_candidates,
        dataset, graph_store, config, run_vector,
    )
    _section_crowd_out(entity_candidates, supporting_sources, graph_store)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_indices(s: str) -> List[int]:
    """Parse '11,12' or '0-19' or '5' into a sorted, deduped list of ints."""
    indices: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.extend(range(int(lo), int(hi) + 1))
        else:
            indices.append(int(part))
    return sorted(set(indices))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingestion diagnostic: trace chunks through stores to retrieval rank",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--indices", "-i", required=True,
                        help="Question indices, e.g. '11,12' or '0-19'")
    parser.add_argument("--dataset", "-d", default="hotpotqa",
                        help="Dataset name (default: hotpotqa)")
    parser.add_argument("--vector", action="store_true",
                        help="Also run vector search rank check "
                             "(slow -- needs Ollama)")
    args = parser.parse_args()

    indices = parse_indices(args.indices)
    dataset = args.dataset

    print(bold(f"\nIngestion Diagnostic -- dataset={dataset}  indices={indices}"))
    print(dim("Loading data..."))

    # Why: read embedding settings (model name, base URL, dimension) from
    # settings.yaml so the diagnostic stays in sync with production when
    # the embedding model is changed centrally.
    from src.thesis_evaluations.benchmark_datasets import load_config_file
    config = load_config_file()

    questions = load_questions(dataset)
    by_id, by_source, all_chunks = load_chunks_index(dataset)
    extraction_index = load_extractions_index(dataset)

    print(dim(f"  {len(questions)} questions, {len(all_chunks)} chunks loaded"))

    # Open the graph store once. None on failure -- consumers guard.
    graph_store: Optional[Any]
    try:
        from src.data_layer.storage import KuzuGraphStore
        graph_store = KuzuGraphStore(str(_DATA_ROOT / dataset / "graph"))
        ok("KuzuDB connected")
    except Exception as e:  # noqa: BLE001 -- diagnostic best-effort
        fail(f"KuzuDB connection failed: {e}")
        graph_store = None

    for idx in indices:
        if idx >= len(questions):
            warn(f"idx={idx} out of range (max {len(questions)-1})")
            continue
        diagnose_question(
            q_idx=idx,
            question=questions[idx],
            by_id=by_id,
            by_source=by_source,
            all_chunks=all_chunks,
            extraction_index=extraction_index,
            graph_store=graph_store,
            dataset=dataset,
            run_vector=args.vector,
            config=config,
        )

    print()
    print("=" * _BAR_WIDTH)
    print(bold("  DIAGNOSTIC COMPLETE"))
    print("=" * _BAR_WIDTH)
    print()


if __name__ == "__main__":
    main()
