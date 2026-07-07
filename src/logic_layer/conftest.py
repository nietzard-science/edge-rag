"""
pytest conftest.py for logic_layer tests.

Adds the project root (Entwicklungfolder) to sys.path so that
'from src.logic_layer.X import Y' works regardless of the working directory
from which pytest is invoked.

Last reviewed: 2026-05-25 (audit pass, project version 5.4).
"""
import sys
from pathlib import Path

# Project root = Entwicklungfolder (3 levels above this file: conftest -> logic_layer -> src -> root)
# Insert at position 0 so project imports take precedence over any installed packages.
PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
assert PROJECT_ROOT.is_dir(), f"Expected project root at {PROJECT_ROOT}"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
