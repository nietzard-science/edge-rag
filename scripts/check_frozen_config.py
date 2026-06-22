"""Reproducibility guard: assert config/frozen_paper.yaml matches the live
config/settings.yaml schema.

The paper pins a frozen configuration (`frozen_paper.yaml`) for paper-release
reproduction. If a key is added to or removed from `settings.yaml` but not
mirrored in the frozen file (or vice-versa), the two drift silently — the same
silent-default bug class `_settings_loader._validate_settings` guards against at
runtime, here enforced at push time.

This checks the KEY SCHEMA (the set of dotted key paths), not the values: the
frozen file deliberately *may* hold different values than the live default for
some keys, but it must never have a different set of keys. A missing/extra key
is what breaks reproducibility.

Exit codes: 0 = schemas match; 1 = drift (keys printed); 2 = a file is missing
or unparseable.

Run:
    python -X utf8 scripts/check_frozen_config.py
    python -X utf8 scripts/check_frozen_config.py --values   # also diff values
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_LIVE = _ROOT / "config" / "settings.yaml"
_FROZEN = _ROOT / "config" / "frozen_paper.yaml"


def _key_paths(node, prefix: str = "") -> set[str]:
    """Set of dotted key paths for every key in a nested mapping."""
    out: set[str] = set()
    if isinstance(node, dict):
        for k, v in node.items():
            out.add(prefix + str(k))
            out |= _key_paths(v, prefix + str(k) + ".")
    return out


def _flat_values(node, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    if isinstance(node, dict):
        for k, v in node.items():
            out.update(_flat_values(v, prefix + str(k) + "."))
    else:
        out[prefix[:-1]] = node
    return out


def _load(path: Path):
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(2)
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        print(f"ERROR: {path} failed to parse: {exc}", file=sys.stderr)
        sys.exit(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", action="store_true",
                    help="also report value differences (informational; not a failure).")
    args = ap.parse_args()

    live = _load(_LIVE)
    frozen = _load(_FROZEN)

    live_keys = _key_paths(live)
    frozen_keys = _key_paths(frozen)

    only_live = sorted(live_keys - frozen_keys)
    only_frozen = sorted(frozen_keys - live_keys)

    if not only_live and not only_frozen:
        print(f"OK: frozen_paper.yaml schema matches settings.yaml "
              f"({len(live_keys)} keys).")
        rc = 0
    else:
        print("DRIFT: frozen_paper.yaml and settings.yaml have different key schemas.",
              file=sys.stderr)
        for k in only_live:
            print(f"  + in settings.yaml, MISSING from frozen_paper.yaml: {k}",
                  file=sys.stderr)
        for k in only_frozen:
            print(f"  - in frozen_paper.yaml, MISSING from settings.yaml: {k}",
                  file=sys.stderr)
        print("Fix: add the missing key(s) to the file that lacks them so both "
              "configs expose the same surface.", file=sys.stderr)
        rc = 1

    if args.values:
        lv, fv = _flat_values(live), _flat_values(frozen)
        shared = sorted(set(lv) & set(fv))
        diffs = [(k, lv[k], fv[k]) for k in shared if lv[k] != fv[k]]
        print(f"\nValue differences on shared keys (informational): {len(diffs)}")
        for k, a, b in diffs:
            print(f"  {k}: live={a!r}  frozen={b!r}")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
