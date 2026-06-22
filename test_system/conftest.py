"""
pytest configuration for test_system/.

Configures two things for every test session under this directory:
  1. sys.path -- prepends the project root so `from src.data_layer...`
     and `from src.logic_layer...` resolve regardless of the directory
     pytest is invoked from.
  2. collect_ignore_glob -- a forward guard that excludes ad-hoc helper
     scripts (no pytest-collectible `test_*` functions) if one is ever
     dropped into this directory.

Model-loading integration tests (e.g. test_graph_inspect.py, which loads
GLiNER + REBEL weights) are marked `@pytest.mark.nightly` at the module
level and are deselected from the default run with `-m "not nightly"`;
they are not gated here. There is intentionally no data-path skip guard:
the integration tests build their own temporary graph and do not read the
populated data/<dataset>/graph store.

Exports (pytest-recognised names)
---------------------------------
PROJECT_ROOT        : Path to the repository root.
collect_ignore_glob : List[str] of filename globs to skip at collection.

Dependencies / Requirements
---------------------------
pytest.

Last reviewed: 2026-06-01 (audit pass, project version 5.5).
"""
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT: Path = Path(__file__).parent.parent
assert PROJECT_ROOT.is_dir(), f"Expected project root at {PROJECT_ROOT}"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Forward guard: ignore ad-hoc helper scripts that carry no pytest-collectible
# test functions, so pytest never emits a spurious "no tests ran" notice if
# such a script is dropped into this directory.
# ---------------------------------------------------------------------------
collect_ignore_glob: List[str] = [
    "diagnose_*.py",
]
