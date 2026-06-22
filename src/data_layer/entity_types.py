"""
Canonical entity-type label maps -- the type-normalisation contract.

Edge-RAG mints entity IDs at two distinct points in the pipeline:

  - Ingestion time (`entity_extraction.GLiNERExtractor` and
    `SpaCyEntityExtractor`), when a document is chunked and stored.
  - Query time (`hybrid_retriever.ImprovedQueryEntityExtractor`), when
    a user question is parsed and its entities are matched against the
    graph.

Both phases call separate NER backends (GLiNER for ingestion, SpaCy as
a fallback and at query time) that emit overlapping but non-identical
label vocabularies. If the two phases assigned different canonical
types to the same surface form, a graph lookup would miss its target
even when the entity is present. This module is the single point of
agreement: every label-to-type translation in the pipeline goes
through one of the three maps below.

Exports
-------
GLINER_LABEL_MAP : Final[Dict[str, str]]
    Lowercase GLiNER prompts / query-time labels -> canonical type.
    Keys are a SUPERSET of the GLiNER prompts configured in
    config/settings.yaml; extra finer-grained keys (e.g. "director",
    "album") are tolerated so query-time sub-labels normalise to the
    same canonical type. Adding keys here is safe and idempotent;
    adding new GLiNER *prompts* to settings.yaml is a separate,
    methodologically significant decision.
SPACY_LABEL_MAP : Final[Dict[str, str]]
    Uppercase SpaCy NER labels -> canonical type. Preserves GPE as a
    type distinct from LOCATION because the graph store keys on this
    distinction. Used wherever an ID must collide bit-for-bit between
    ingestion time and query time.
SPACY_LABEL_MAP_FLAT : Final[Dict[str, str]]
    Uppercase SpaCy NER labels -> canonical type, with GPE collapsed to
    LOCATION and WORK_OF_ART / FAC added. Used only by
    SpaCyEntityExtractor, an alternative extractor that never reaches
    the graph store, so the GPE/LOCATION distinction is not required.

Canonical type universe
------------------------
Over the GLiNER prompts configured in settings.yaml the reachable
image is eight types: PERSON, ORGANIZATION, LOCATION, GPE, DATE, EVENT,
WORK_OF_ART, PRODUCT (the OntoNotes core set). The SpaCy maps add no
new types (FAC -> LOCATION; WORK_OF_ART already present). The map also
defines a TECHNOLOGY key, reachable only if a "technology" prompt is
explicitly configured; it is outside the default eight-type universe.

The three maps are module-level constants (annotated Final) and are
treated as read-only by every consumer; all references resolve a label
via `.get()` / membership only and never mutate the table.

Last reviewed: 2026-06-01 (audit pass, project version 5.5).
"""

from typing import Dict, Final

__all__ = [
    "GLINER_LABEL_MAP",
    "SPACY_LABEL_MAP",
    "SPACY_LABEL_MAP_FLAT",
]

# ---------------------------------------------------------------------------
# GLiNER label map -- lowercase GLiNER output / query label -> canonical type.
# Single source of agreement for the GLiNER (ingestion) and query-time
# extractors; all keys are lower-case.
# ---------------------------------------------------------------------------
GLINER_LABEL_MAP: Final[Dict[str, str]] = {
    # People
    "person":       "PERSON",
    "director":     "PERSON",
    "actor":        "PERSON",
    "politician":   "PERSON",
    "scientist":    "PERSON",
    "athlete":      "PERSON",
    # Organisations
    "organization": "ORGANIZATION",
    "company":      "ORGANIZATION",
    "studio":       "ORGANIZATION",
    "institution":  "ORGANIZATION",
    # Geopolitical / location
    "city":         "GPE",
    "country":      "GPE",
    "state":        "GPE",
    "gpe":          "GPE",
    "location":     "LOCATION",
    "place":        "LOCATION",
    "landmark":     "LOCATION",
    "monument":     "LOCATION",
    "building":     "LOCATION",
    # Creative works
    # Both "work_of_art" (underscore) and "work of art" (space) are kept
    # intentionally: upstream sources differ in whitespace convention and
    # collapsing them would silently drop one form.
    "film":         "WORK_OF_ART",
    "movie":        "WORK_OF_ART",
    "book":         "WORK_OF_ART",
    "album":        "WORK_OF_ART",
    "song":         "WORK_OF_ART",
    "work_of_art":  "WORK_OF_ART",
    "work of art":  "WORK_OF_ART",
    "award":        "WORK_OF_ART",
    "prize":        "WORK_OF_ART",
    # Temporal
    "date":         "DATE",
    "year":         "DATE",
    "time":         "DATE",
    # Events / other
    "event":        "EVENT",
    "product":      "PRODUCT",
    # Out-of-universe: reachable only if a "technology" GLiNER prompt is
    # explicitly added to settings.yaml (not in the default eight-type set).
    "technology":   "TECHNOLOGY",
}

# ---------------------------------------------------------------------------
# SpaCy label map -- UPPERCASE SpaCy NER output -> canonical type.
# Preserves GPE as a distinct type so that entity IDs generated at ingestion
# time (GLiNERExtractor._spacy_extract) match entity IDs generated at query
# time (ImprovedQueryEntityExtractor._spacy_extract).
# ---------------------------------------------------------------------------
SPACY_LABEL_MAP: Final[Dict[str, str]] = {
    "PERSON":      "PERSON",
    "ORG":         "ORGANIZATION",
    "GPE":         "GPE",       # city / country -- kept distinct for graph lookups
    "LOC":         "LOCATION",
    "DATE":        "DATE",
    "EVENT":       "EVENT",
}

# ---------------------------------------------------------------------------
# SpaCy label map (flat) -- UPPERCASE SpaCy NER output -> canonical type.
# Collapses GPE -> LOCATION and adds WORK_OF_ART / FAC entries.
# Used exclusively by SpaCyEntityExtractor, where graph-store GPE/LOCATION
# discrimination is not required.
# ---------------------------------------------------------------------------
SPACY_LABEL_MAP_FLAT: Final[Dict[str, str]] = {
    "PERSON":      "PERSON",
    "ORG":         "ORGANIZATION",
    "GPE":         "LOCATION",  # flattened: no GPE/LOCATION distinction needed
    "LOC":         "LOCATION",
    "DATE":        "DATE",
    "EVENT":       "EVENT",
    "WORK_OF_ART": "WORK_OF_ART",
    "FAC":         "LOCATION",
}
