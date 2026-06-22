"""
pytest conftest.py for pipeline tests.

Inserts the project root (Entwicklungfolder) onto sys.path so that
``from src.pipeline.X import Y`` resolves regardless of the working
directory from which pytest is invoked.

Exports
-------
    PROJECT_ROOT : Path
        Resolved project root, asserted to be a directory at import time.

Dependencies
------------
    stdlib only (sys, pathlib).

Last reviewed: 2026-06-01 (audit pass, project version 5.5).
"""
import sys
from pathlib import Path

# Project root = Entwicklungfolder. Three levels up: conftest -> pipeline -> src -> root.
PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
assert PROJECT_ROOT.is_dir(), f"Expected project root at {PROJECT_ROOT}"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
