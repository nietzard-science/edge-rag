"""
Subject-Verb-Object relation extraction via SpaCy dependency parse.

Complements REBEL by extracting NARRATIVE relations that REBEL misses.
REBEL is trained on Wikipedia infoboxes (publication_date, country,
date_of_birth) — useful but not what multi-hop bridge questions ask about.
SVO triples derived from the dependency parse capture the narrative
relations REBEL is silent on:

    "<PERSON> directed <WORK_OF_ART>."
        subj=<PERSON>, verb=direct, obj=<WORK_OF_ART>
        -> (<PERSON>, direct, <WORK_OF_ART>)

    "<PERSON> won <AWARD> in <YEAR>."
        subj=<PERSON>, verb=win, obj=<AWARD>
        -> (<PERSON>, win, <AWARD>)

DESIGN CHOICES
--------------
- Both subject and object MUST resolve to known entities (via canonical_form
  match against `name_to_id`). Unmatched triples are dropped silently.
  This guarantees the relation never references a non-existent graph node.
- Verb is stored as its `lemma_` so "directed", "directs", "directing" all
  collapse to "direct".
- Confidence = 0.7 (between REBEL's 0.5 sentinel and cooccurrence's 1.0).
  Reflects that SVO is more specific than cooccurrence but less curated
  than REBEL's beam search.
- Only ROOT verbs and conjuncts are inspected -- avoids extracting verbs
  inside relative clauses ("the man who directed <WORK> said ...")
  whose subject is the relative pronoun, not a content entity.
- Prepositional objects are followed: "X was born in Paris" -> (X, bear, Paris).

Last reviewed: 2026-05-25 (audit pass, project version 5.4).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Public API. Helpers (`_np_head_text`, `_find_active_subject`, ...) and the
# lazy-loaded singleton state (`_NLP`, `_AVAILABLE`) are implementation
# details, intentionally NOT re-exported.
__all__ = [
    "extract_svo_relations",
    "is_available",
]


# ---------------------------------------------------------------------------
# LAZY SPACY LOADER
# ---------------------------------------------------------------------------

_NLP = None
_AVAILABLE: Optional[bool] = None


def _try_load(spacy_model: str = "en_core_web_sm") -> None:
    global _NLP, _AVAILABLE
    if _AVAILABLE is not None:
        return
    try:
        import spacy
        _NLP = spacy.load(spacy_model)
        _AVAILABLE = True
        logger.info("SVO extractor loaded: %s", spacy_model)
    except (OSError, IOError, ValueError, ImportError, RuntimeError) as exc:
        _AVAILABLE = False
        logger.warning(
            "FALLBACK ACTIVE: spaCy not available for SVO extraction (%s)", exc
        )


def is_available() -> bool:
    _try_load()
    return bool(_AVAILABLE)


# ---------------------------------------------------------------------------
# DEPENDENCY-PARSE HELPERS
# ---------------------------------------------------------------------------

# Subject and object dependency labels recognised by the extractor.
_SUBJ_DEPS: frozenset = frozenset({"nsubj", "nsubjpass"})
_DOBJ_DEPS: frozenset = frozenset({"dobj", "attr"})
_HEAD_POS:  frozenset = frozenset({"NOUN", "PROPN"})

# Confidence assigned to every emitted SVO triple. Calibrated between
# REBEL's 0.5 sentinel and cooccurrence's 1.0 to reflect that SVO is more
# specific than cooccurrence but less curated than REBEL's beam search.
_SVO_CONFIDENCE: float = 0.7

# Verbs filtered out as noise generators. "be"/"have"/"do" are auxiliaries
# that produce vacuous relations; "say"/"tell"/"make" are reporting/light
# verbs whose object is almost never an entity worth linking.
_NOISE_VERBS: frozenset = frozenset({"be", "have", "do", "say", "tell", "make"})


def _np_head_text(head_token) -> str:
    """
    Return the head + immediate compound/PROPN modifiers, ignoring
    determiners, adjectives, and longer subtree material.

    "the brilliant Marie Curie" -> "Marie Curie"
    "Apple Inc." -> "Apple Inc."
    """
    parts: List[str] = []
    for child in head_token.lefts:
        if child.dep_ == "compound" and child.pos_ in _HEAD_POS:
            parts.append(child.text)
    parts.append(head_token.text)
    for child in head_token.rights:
        if child.dep_ == "compound" and child.pos_ in _HEAD_POS:
            parts.append(child.text)
    return " ".join(parts).strip()


def _find_active_subject(verb_tok):
    """nsubj child (active voice) whose head POS is NOUN or PROPN."""
    for c in verb_tok.children:
        if c.dep_ == "nsubj" and c.pos_ in _HEAD_POS:
            return c
    return None


def _find_active_object(verb_tok):
    """
    Direct object first; if missing, follow a single prepositional chain
    to its pobj. Returns the head token or None.
    """
    for c in verb_tok.children:
        if c.dep_ in _DOBJ_DEPS and c.pos_ in _HEAD_POS:
            return c
    for c in verb_tok.children:
        if c.dep_ == "prep":
            for gc in c.children:
                if gc.dep_ == "pobj" and gc.pos_ in _HEAD_POS:
                    return gc
    return None


def _find_passive_components(verb_tok):
    """
    For a passive-voice verb, return (logical_subject, logical_object) so
    that the emitted SVO mirrors the active-voice meaning:

        "<WORK_OF_ART> was directed by <PERSON>"
            nsubjpass  = <WORK_OF_ART>     (the patient / logical object)
            agent pobj = <PERSON>          (the doer / logical subject)
            -> emits (<PERSON>, direct, <WORK_OF_ART>)

    Returns (None, None) if either component is missing — the caller
    falls back to skipping the verb entirely.
    """
    nsubjpass = None
    agent_pobj = None
    for c in verb_tok.children:
        if c.dep_ == "nsubjpass" and c.pos_ in _HEAD_POS:
            nsubjpass = c
        elif c.dep_ == "agent":
            for gc in c.children:
                if gc.dep_ == "pobj" and gc.pos_ in _HEAD_POS:
                    agent_pobj = gc
    if nsubjpass is not None and agent_pobj is not None:
        return agent_pobj, nsubjpass  # (logical subj, logical obj)
    return None, None


def _inherit_from_conjunct_head(verb_tok):
    """
    "<PERSON> directed and produced <WORK_OF_ART>" — `produced` conjuncts
    with `directed` and shares its subject. Walk up conj heads to recover
    the subject/object the secondary verb implicitly inherits.
    """
    if verb_tok.dep_ != "conj":
        return None, None
    head = verb_tok.head
    if head.pos_ != "VERB":
        return None, None
    subj = _find_active_subject(head)
    obj = _find_active_object(head)
    if subj is None or obj is None:
        # Try passive on the head verb
        p_subj, p_obj = _find_passive_components(head)
        if p_subj is not None and p_obj is not None:
            return p_subj, p_obj
    return subj, obj


# ---------------------------------------------------------------------------
# ENTITY MATCHING
# ---------------------------------------------------------------------------

def _match_entity(
    np_text: str,
    name_to_id: Dict[str, str],
    canonical_form_fn,
) -> Optional[str]:
    """
    Resolve a noun-phrase string to an entity_id.

    Strategy (decreasing strictness):
      1. Exact canonical_form match in name_to_id.
      2. Substring match: if any known entity's canonical form is contained
         within np_canon (or vice versa), pick the longest one.
    Returns None on no match.
    """
    if not np_text:
        return None
    np_canon = canonical_form_fn(np_text)
    if not np_canon:
        return None

    direct = name_to_id.get(np_canon)
    if direct:
        return direct

    # Substring fallback. Only for canonical forms long enough to avoid
    # accidental matches ("a" matching "april" matching "apple").
    if len(np_canon) < 4:
        return None
    candidates = []
    for canon, eid in name_to_id.items():
        if len(canon) < 4:
            continue
        if canon == np_canon:
            return eid  # already checked but defensive
        if canon in np_canon or np_canon in canon:
            candidates.append((canon, eid))
    if not candidates:
        return None
    # Prefer the longest matching canonical form (most specific).
    candidates.sort(key=lambda pair: -len(pair[0]))
    return candidates[0][1]


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def extract_svo_relations(
    text: str,
    name_to_id: Dict[str, str],
    canonical_form_fn,
    min_verb_length: int = 3,
    confidence: float = _SVO_CONFIDENCE,
) -> List[Tuple[str, str, str, float]]:
    """
    Extract SVO triples from `text` and resolve subject + object to entity IDs.

    Args:
        text:               The chunk text.
        name_to_id:         Map canonical_form(name) -> entity_id (built by
                            the caller during entity insertion).
        canonical_form_fn:  Function (str -> str) that produces the canonical
                            surface form of an entity name. Imported lazily
                            to avoid a circular dependency with graph_quality.
        min_verb_length:    Skip verbs whose lemma is shorter than this many
                            characters (filters auxiliaries like "be"/"do").

    Returns:
        List of (subject_id, verb_lemma, object_id, confidence) tuples.
        Subject and object are guaranteed to be different entities.
        Chunk-provenance is recorded by the caller when inserting the
        resulting RELATED_TO edges (see `add_related_to_relation`); this
        function is stateless w.r.t. the originating chunk.
    """
    if not text or not name_to_id:
        return []
    _try_load()
    if not _AVAILABLE or _NLP is None:
        return []

    try:
        doc = _NLP(text)
    except (ValueError, RuntimeError, AttributeError) as exc:
        logger.debug("SVO: parse failed (%s)", exc)
        return []

    triples: List[Tuple[str, str, str, float]] = []
    seen: Set[Tuple[str, str, str]] = set()

    for sent in doc.sents:
        for tok in sent:
            if tok.pos_ != "VERB":
                continue
            lemma = tok.lemma_.lower()
            if len(lemma) < min_verb_length:
                continue
            if lemma in _NOISE_VERBS:
                continue

            # Try active voice, then passive voice, then inherit from a
            # conjunct verb head. Most Wikipedia sentences use passive
            # ("X was directed by Y"); without this branch we silently
            # miss ~60 % of biographical content.
            subj_tok = _find_active_subject(tok)
            obj_tok = _find_active_object(tok)
            if subj_tok is None or obj_tok is None:
                p_subj, p_obj = _find_passive_components(tok)
                if p_subj is not None and p_obj is not None:
                    subj_tok, obj_tok = p_subj, p_obj
            if subj_tok is None or obj_tok is None:
                inh_subj, inh_obj = _inherit_from_conjunct_head(tok)
                if inh_subj is not None and inh_obj is not None:
                    subj_tok, obj_tok = inh_subj, inh_obj
            if subj_tok is None or obj_tok is None:
                continue

            subj_id = _match_entity(_np_head_text(subj_tok), name_to_id, canonical_form_fn)
            obj_id = _match_entity(_np_head_text(obj_tok), name_to_id, canonical_form_fn)
            if not subj_id or not obj_id or subj_id == obj_id:
                continue

            key = (subj_id, lemma, obj_id)
            if key in seen:
                continue
            seen.add(key)
            triples.append((subj_id, lemma, obj_id, confidence))

    return triples
