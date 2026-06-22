"""
Shared text-processing utilities for the logic layer.

Internal module — not part of the public API. Holds small, dependency-free
NLP helpers that are used by more than one logic_layer module. Keeping these
out of ``_settings_loader.py`` and ``_config.py`` preserves single-responsibility
for those files (settings I/O and configuration dataclass respectively).

Exports
-------
    _PROPER_NOUN_RE : re.Pattern
        Compiled regex matching multi-word capitalised proper nouns. Used as
        a lightweight NER fallback wherever loading a second NER model at
        inference would be wasteful.

Last reviewed: 2026-05-26 (audit pass, project version 5.4).
"""

import re

__all__ = ["_PROPER_NOUN_RE"]

# ---------------------------------------------------------------------------
# Multi-word capitalised proper-noun proxy.
#
# Matches:  Two-or-more consecutive Capitalised tokens — the canonical
#           surface form of multi-word proper nouns in English.
#           E.g.: "Marie Curie", "Eiffel Tower", "New York", "Mount Everest".
# Misses:   ALL-CAPS acronyms (NATO), names with lowercase particles
#           ("Tower of London"), names with non-ASCII diacritics (any
#           accented Latin letter). These known gaps are acceptable for a
#           heuristic that avoids loading a second NER model at inference.
#
# Used by:
#   planner.py    — EntityExtractor.ENTITY_PATTERNS (regex NER fallback)
#   navigator.py  — _entity_overlap_pruning(), _extract_bridge_entities()
#   controller.py — _extract_bridge_entities()
#   verifier.py   — Verifier._MULTI_PROPER_NOUN_RE (claim verification)
# ---------------------------------------------------------------------------
_PROPER_NOUN_RE: re.Pattern = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
)
