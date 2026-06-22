#!/usr/bin/env python3
"""
Standalone graph-quality baseline diagnostic.

Opens an existing KuzuDB graph store, computes the baseline metrics
defined in `src.data_layer.graph_quality`, prints a human-readable
report (or JSON via --json), and optionally enforces the default
invariants via --strict. The graph is opened read-only -- this script
never mutates it.

Used as a verification step after an ingestion run (see REPRODUCE.md):
healthy invariants are a prerequisite for any downstream evaluation
that depends on the graph lane.

Exports
-------
- main() -> int      -- CLI entry point; exit code 0 (OK), 1 (config /
                        import / runtime error), 2 (--strict + invariant
                        violations)

Dependencies / Requirements
---------------------------
- src.data_layer.storage.KuzuGraphStore        -- graph reader
- src.data_layer.graph_quality                  -- baseline computation +
                                                  invariant assertions +
                                                  report formatter

Usage (single line; -X utf8 required on Windows / PowerShell):
    python -X utf8 diagnose_graph_baseline.py --dataset hotpotqa
    python -X utf8 diagnose_graph_baseline.py --graph-path ./data/hotpotqa/graph
    python -X utf8 diagnose_graph_baseline.py --dataset hotpotqa --strict
    python -X utf8 diagnose_graph_baseline.py --dataset hotpotqa --json

Last reviewed: 2026-05-30 (audit pass, project version 5.4)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Why: anchor data paths on the project root so the CLI works regardless
# of cwd, and put _PROJECT_ROOT on sys.path so the `from src.*` imports
# below resolve when this script is launched from a subdirectory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_DATA_ROOT = _PROJECT_ROOT / "data"

# Why: rule width for the printed report header.
_BAR_WIDTH = 70


def _resolve_graph_path(args: argparse.Namespace) -> Path:
    """Resolve the KuzuDB directory from CLI args.

    Precedence: --graph-path overrides --dataset. Raises SystemExit when
    neither is given (mirrors argparse's behaviour for missing required
    inputs).
    """
    if args.graph_path:
        return Path(args.graph_path)
    if args.dataset:
        return _DATA_ROOT / args.dataset / "graph"
    raise SystemExit("error: either --dataset or --graph-path is required")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute and print the graph-quality baseline for a KuzuDB store.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name; resolves to <project-root>/data/<dataset>/graph",
    )
    parser.add_argument(
        "--graph-path",
        default=None,
        help="Explicit path to a KuzuDB directory (overrides --dataset)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print metrics as JSON instead of the human-readable report",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if any invariant is violated",
    )
    args = parser.parse_args()

    graph_path = _resolve_graph_path(args)
    if not graph_path.exists():
        print(f"error: graph not found: {graph_path}", file=sys.stderr)
        return 1

    # Why: late import keeps `--help` cheap (no heavy data-layer loading
    # just to print usage) and isolates any import failure behind the CLI
    # parse so the user sees a clean message rather than a top-level
    # traceback.
    try:
        from src.data_layer.storage import KuzuGraphStore
        from src.data_layer.graph_quality import (
            assert_graph_invariants,
            compute_graph_baseline,
            format_baseline_report,
        )
    except ImportError as exc:
        print(f"error: import failed: {exc}", file=sys.stderr)
        return 1

    # Why: the graph reader and baseline computation can raise on schema
    # mismatch or a corrupt store; surface as a clean CLI error rather
    # than a traceback so the script remains a friendly verification tool.
    try:
        store = KuzuGraphStore(str(graph_path))
        metrics = compute_graph_baseline(store)
    except Exception as exc:  # noqa: BLE001 -- best-effort CLI error path
        print(f"error: baseline computation failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(metrics, indent=2, default=str))
    else:
        print()
        print("=" * _BAR_WIDTH)
        print(f"  GRAPH QUALITY BASELINE  -  {graph_path}")
        print("=" * _BAR_WIDTH)
        print(format_baseline_report(metrics))

    violations = assert_graph_invariants(metrics, strict=False)
    if violations:
        print()
        print("  Invariant violations:")
        for v in violations:
            print(f"    - {v}")
        if args.strict:
            return 2
    else:
        print()
        print("  All invariants OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
