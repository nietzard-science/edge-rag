"""
Generate + verify the held-out dev band for train/dev/test discipline (P0-T2).

Why this script exists
----------------------
A reviewer's standard red flag is test-set tuning: hyperparameters chosen by
maximising the metric on the same questions the headline numbers are reported
on. The paper datasets each contain exactly 500 questions (the evaluated test
set), so there is no spare data for a separate tuned-on split. This script makes
the discipline explicit and reproducible instead:

  * It carves a deterministic **dev band** — the LAST ``--dev-size`` questions of
    each dataset's ``questions.json`` (default 100, i.e. indices [400, 500)) —
    and records their ids in ``data/splits/dev_band.json``.
  * That band is the *declared* sanity/dev region: any hyperparameter that is
    ever tuned by looking at aggregate metrics must be tuned on this band only.
    The provenance table in REPRODUCE.md documents that the shipped settings
    were NOT tuned this way (they come from literature defaults, measured corpus
    properties, and single-query diagnostic traces), so the band is currently a
    *reserved* hold-out rather than one that was consumed.

The split is by stored-file order, which is byte-stable for anyone who passes
the ``data/SHA256.txt`` verification, so the recorded ids reproduce exactly.

Outputs
-------
data/splits/dev_band.json   per-dataset {dev_ids, test_ids, dev_indices, ...}

Usage
-----
    python -X utf8 -m src.thesis_evaluations.make_dev_split --write
    python -X utf8 -m src.thesis_evaluations.make_dev_split --verify   # CI check

Last reviewed: 2026-06-05.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_DATA_ROOT = _PROJECT_ROOT / "data"
_SPLITS_DIR = _DATA_ROOT / "splits"
_SPLIT_PATH = _SPLITS_DIR / "dev_band.json"

# Datasets that carry per-question retrieval (StrategyQA is LLM-only by design,
# so it has no retrieval-tuned knobs and is excluded from the band).
_DATASETS = ("hotpotqa", "2wikimultihop")
_DEFAULT_DEV_SIZE = 100


def _load_ids(dataset: str) -> List[str]:
    """Return question ids in stored-file order for a dataset."""
    p = _DATA_ROOT / dataset / "questions.json"
    if not p.exists():
        raise FileNotFoundError(f"questions.json not found for {dataset}: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    ids = [q.get("id") for q in data]
    if any(i is None for i in ids):
        raise ValueError(f"{dataset}: some questions have no 'id' field")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{dataset}: duplicate question ids")
    return ids


def build_split(dev_size: int = _DEFAULT_DEV_SIZE) -> Dict[str, object]:
    """Build the split manifest dict (dev band = last ``dev_size`` per dataset)."""
    manifest: Dict[str, object] = {
        "_doc": (
            "Held-out dev/sanity band for train/dev/test discipline (P0-T2). "
            "dev_ids = the LAST dev_size questions of each dataset in stored "
            "order; reserved for any metric-driven hyperparameter tuning. "
            "Headline numbers are reported on the full 0..N set with settings "
            "that were NOT tuned on aggregate metrics (see REPRODUCE.md "
            "provenance table). test_ids = the complement (0..N-dev_size)."
        ),
        "dev_size": dev_size,
        "order": "stored-file order (byte-stable under data/SHA256.txt)",
        "datasets": {},
    }
    for ds in _DATASETS:
        ids = _load_ids(ds)
        n = len(ids)
        if dev_size >= n:
            raise ValueError(
                f"{ds}: dev_size={dev_size} >= dataset size {n}; no test "
                f"questions would remain"
            )
        split_at = n - dev_size
        manifest["datasets"][ds] = {
            "n_total": n,
            "test_indices": [0, split_at],     # [start, end) → questions[0:split_at]
            "dev_indices": [split_at, n],       # questions[split_at:n]
            "test_ids": ids[:split_at],
            "dev_ids": ids[split_at:],
        }
        logger.info("%s: n=%d → test[0:%d] (%d) + dev[%d:%d] (%d)",
                    ds, n, split_at, split_at, split_at, n, dev_size)
    return manifest


def write_split(dev_size: int = _DEFAULT_DEV_SIZE) -> Path:
    """Write the split manifest to data/splits/dev_band.json."""
    manifest = build_split(dev_size)
    _SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    _SPLIT_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    logger.info("Wrote split manifest: %s", _SPLIT_PATH)
    return _SPLIT_PATH


def verify_split() -> bool:
    """Re-derive the split and confirm it matches the committed manifest.

    Returns True on match. Used as a CI guard so a corpus change that would
    silently shift the dev band is caught. Disjointness (dev ∩ test == ∅) and
    completeness (dev ∪ test == all ids) are asserted for every dataset.
    """
    if not _SPLIT_PATH.exists():
        logger.error("No committed split at %s — run with --write first.", _SPLIT_PATH)
        return False
    committed = json.loads(_SPLIT_PATH.read_text(encoding="utf-8"))
    dev_size = committed.get("dev_size", _DEFAULT_DEV_SIZE)
    fresh = build_split(dev_size)

    ok = True
    for ds in _DATASETS:
        c = committed["datasets"].get(ds, {})
        f = fresh["datasets"][ds]
        dev_c, test_c = set(c.get("dev_ids", [])), set(c.get("test_ids", []))
        dev_f, test_f = set(f["dev_ids"]), set(f["test_ids"])
        if dev_c != dev_f or test_c != test_f:
            logger.error("%s: committed split != freshly derived split", ds)
            ok = False
        if dev_f & test_f:
            logger.error("%s: dev and test bands OVERLAP (%d ids)",
                         ds, len(dev_f & test_f))
            ok = False
        if dev_f | test_f != set(_load_ids(ds)):
            logger.error("%s: dev ∪ test != all ids", ds)
            ok = False
        logger.info("%s: OK — dev=%d test=%d disjoint+complete",
                    ds, len(dev_f), len(test_f))
    logger.info("Split verification: %s", "PASS" if ok else "FAIL")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true",
                        help="Generate and write data/splits/dev_band.json.")
    parser.add_argument("--verify", action="store_true",
                        help="Re-derive and check the committed manifest "
                             "(disjoint + complete + reproducible).")
    parser.add_argument("--dev-size", type=int, default=_DEFAULT_DEV_SIZE,
                        help=f"Dev-band size per dataset (default "
                             f"{_DEFAULT_DEV_SIZE}, the last N questions).")
    args = parser.parse_args()

    if not (args.write or args.verify):
        parser.print_help()
        return
    if args.write:
        write_split(args.dev_size)
    if args.verify:
        ok = verify_split()
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
