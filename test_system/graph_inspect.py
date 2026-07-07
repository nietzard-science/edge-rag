"""
Graph Inspection Utility — KuzuDB Health Check

Connects directly to the KuzuDB knowledge graph and prints node/edge counts
plus sample MENTIONS data. Intended as a fast sanity check after ingestion
to verify that entities, relations, and chunk linkages were stored correctly.

Architectural position: Development utility (not part of the production pipeline).
Consumed by: Developer / paper author for manual graph quality assessment.

Usage:
    python test_system/graph_inspect.py
    python test_system/graph_inspect.py --dataset hotpotqa
    python test_system/graph_inspect.py --db-path data/hotpotqa/graph/graph_KuzuDB

Last reviewed: 2026-05-30 (audit pass, project version 5.4).
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import kuzu

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

PROJECT_ROOT = Path(__file__).parent.parent


def _resolve_db_path(dataset: str, db_path_override: str | None) -> Path:
    """
    Resolve the KuzuDB directory path.

    Priority:
      1. Explicit --db-path argument
      2. Derived from --dataset: data/{dataset}/graph/graph_KuzuDB

    Args:
        dataset:          Dataset name (e.g. "hotpotqa").
        db_path_override: Explicit path from CLI, or None.

    Returns:
        Resolved absolute Path to the KuzuDB directory.

    Raises:
        FileNotFoundError: Path does not exist.
    """
    if db_path_override:
        path = Path(db_path_override)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    else:
        path = PROJECT_ROOT / "data" / dataset / "graph" / "graph_KuzuDB"

    if not path.exists():
        raise FileNotFoundError(
            f"KuzuDB not found at: {path}\n"
            f"  Run ingestion first: python local_importingestion.py --dataset {dataset}"
        )
    return path


def _section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


def inspect(db_path: Path) -> None:
    """
    Run the full graph inspection sequence.

    Prints:
      - Node counts per table (DocumentChunk, SourceDocument, Entity)
      - Edge counts per relation type
      - First 10 MENTIONS edges (chunk → entity sample)
      - First 10 RELATED_TO triples (entity → entity via REBEL)

    Args:
        db_path: Path to the KuzuDB directory.
    """
    logger.info("Connecting to KuzuDB: %s", db_path)
    db   = kuzu.Database(str(db_path))
    conn = kuzu.Connection(db)

    _section("NODE COUNTS")
    for table in ["DocumentChunk", "SourceDocument", "Entity"]:
        r = conn.execute(f"MATCH (n:{table}) RETURN COUNT(n)")
        count = r.get_next()[0]
        print(f"  {table:<20s}: {count:>8,}")

    _section("EDGE COUNTS")
    for rel in ["FROM_SOURCE", "NEXT_CHUNK", "MENTIONS", "RELATED_TO"]:
        r = conn.execute(f"MATCH ()-[r:{rel}]->() RETURN COUNT(r)")
        count = r.get_next()[0]
        print(f"  {rel:<20s}: {count:>8,}")

    _section("SAMPLE: MENTIONS (Chunk → Entity, first 10)")
    r = conn.execute("""
        MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity)
        RETURN c.chunk_id, e.name, e.type, e.confidence
        LIMIT 10
    """)
    while r.has_next():
        chunk_id, name, etype, conf = r.get_next()
        conf_str = f"{conf:.2f}" if conf is not None else "n/a"
        print(f"  [{chunk_id}]  {etype:<15s}  {name:<35s}  conf={conf_str}")

    _section("SAMPLE: RELATED_TO (Entity → Entity via REBEL, first 10)")
    r = conn.execute("""
        MATCH (e1:Entity)-[r:RELATED_TO]->(e2:Entity)
        RETURN e1.name, r.relation_type, e2.name
        LIMIT 10
    """)
    found = False
    while r.has_next():
        e1, rel_type, e2 = r.get_next()
        print(f"  '{e1}'  --[{rel_type}]-->  '{e2}'")
        found = True
    if not found:
        print("  (no RELATED_TO edges found — REBEL extraction may not have run)")

    print()
    logger.info("Inspection complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a KuzuDB knowledge graph after ingestion."
    )
    parser.add_argument(
        "--dataset", default="hotpotqa",
        help="Dataset name used to derive default DB path (default: hotpotqa)"
    )
    parser.add_argument(
        "--db-path", default=None,
        help="Explicit path to the KuzuDB directory (overrides --dataset)"
    )
    args = parser.parse_args()

    try:
        db_path = _resolve_db_path(args.dataset, args.db_path)
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)

    inspect(db_path)


if __name__ == "__main__":
    main()
