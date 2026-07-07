"""
Pronoun coreference resolution for Phase-1 chunking.

Replaces pronoun mentions in input text with their antecedent noun phrase.
Applied per-article BEFORE chunking so that GLiNER (Phase 2) and the graph
(Phase 3) capture the named entity behind each pronoun rather than dropping
the mention or classifying the pronoun as PERSON.

Worked example:
    Before:  "<PERSON> was born in <CITY>. He directed <WORK>."
             -> Phase 2 extracts: <PERSON>, <CITY>, <WORK>, "He" (PERSON)
             -> "He" is dropped by the stoplist; the link <PERSON> -> <WORK>
                is lost.
    After:   "<PERSON> was born in <CITY>. <PERSON> directed <WORK>."
             -> Phase 2 extracts: <PERSON> (twice), <CITY>, <WORK>
             -> Cooccurrence captures <PERSON>-<CITY> and <PERSON>-<WORK>
                in the same paragraph.

The magnitude of the effect on the final graph (entity count, mention count,
relation count, isolated-entity rate) depends on the corpus and the
performance of the underlying coreferee resolver and was not quantified
empirically for this paper; the design decision to enable coref by default
is qualitative (pronoun-dropped mentions are unrecoverable downstream) and
is documented as such in the methodology section.

OPTIONAL DEPENDENCY
-------------------
    pip install coreferee
    python -m coreferee install en
    python -m spacy download en_core_web_md   # or en_core_web_lg

If coreferee is not installed or no md/lg spaCy model is available, every
call returns the input text unchanged. The pipeline keeps working -- graph
density just stays at the pre-coref level. The Phase-3 ingestion log
records whether coref was applied so the ingest manifest is reproducible.

Last reviewed: 2026-05-25 (audit pass, project version 5.4).
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Public API. Underscored names below are implementation details of the
# lazy-loaded singleton and intentionally NOT re-exported.
__all__ = [
    "resolve_coreferences",
    "is_available",
]


# ---------------------------------------------------------------------------
# LAZY-LOADED SINGLETON (one spaCy + coreferee pipeline per process)
# ---------------------------------------------------------------------------

_NLP = None
_AVAILABLE: Optional[bool] = None  # tri-state: None=untried, True/False=resolved

# spaCy POS tags that indicate a pronoun-like mention worth replacing.
# We deliberately do NOT replace full noun-phrase mentions because that
# would be redundant and risks introducing duplicate text.
_PRONOUN_POS: frozenset = frozenset({"PRON", "DET", "ADV"})


def _try_load() -> None:
    """Lazy-load coreferee + a sufficiently-large spaCy model. One-shot."""
    global _NLP, _AVAILABLE
    if _AVAILABLE is not None:
        return  # already tried

    try:
        import spacy
    except ImportError:
        _AVAILABLE = False
        logger.warning(
            "FALLBACK ACTIVE: spaCy not installed; coreference resolution disabled."
        )
        return

    try:
        import coreferee  # noqa: F401  -- registers the spaCy pipe
    except ImportError:
        _AVAILABLE = False
        logger.warning(
            "FALLBACK ACTIVE: coreferee not installed; coreference resolution "
            "disabled. Install with: pip install coreferee && "
            "python -m coreferee install en"
        )
        return

    # Coreferee works with md/lg/trf -- NOT with en_core_web_sm.
    for model_name in ("en_core_web_lg", "en_core_web_md"):
        try:
            nlp = spacy.load(model_name)
            nlp.add_pipe("coreferee")
            _NLP = nlp
            _AVAILABLE = True
            logger.info(
                "Coreference resolver loaded: %s + coreferee", model_name
            )
            return
        except (OSError, IOError, ValueError, ImportError, RuntimeError):
            continue

    _AVAILABLE = False
    logger.warning(
        "FALLBACK ACTIVE: no en_core_web_md/lg available; coreference disabled. "
        "Install with: python -m spacy download en_core_web_md"
    )


def is_available() -> bool:
    """True iff coreference resolution is loaded and ready to run."""
    _try_load()
    return bool(_AVAILABLE)


# ---------------------------------------------------------------------------
# RESOLVER
# ---------------------------------------------------------------------------

def resolve_coreferences(text: str, max_chars: int = 100_000) -> str:
    """
    Replace pronoun mentions with the longest mention of their coreference chain.

    Behaviour:
      - Returns input unchanged if coreferee/spaCy is not available.
      - Returns input unchanged for empty text or text > `max_chars` (safety).
      - Returns input unchanged when no chains are found.
      - Replaces only PRON/DET/ADV mentions (pronouns and "this/that/there"),
        leaving full noun-phrase mentions alone to avoid spurious duplication.
      - Applies edits right-to-left so character offsets remain valid.

    Args:
        text:       The article text to resolve.
        max_chars:  Skip resolution for very long inputs (avoid blowing up
                    spaCy's parser on Wikipedia-scale articles).

    Returns:
        The resolved text, or the input unchanged on any fallback path.
    """
    if not text or not text.strip():
        return text
    if len(text) > max_chars:
        logger.debug("Coref: skipping oversized input (%d chars)", len(text))
        return text

    _try_load()
    if not _AVAILABLE or _NLP is None:
        return text

    try:
        doc = _NLP(text)
    except (ValueError, RuntimeError, AttributeError) as exc:
        logger.debug("Coref: spaCy parse failed (%s)", exc)
        return text

    chains = getattr(doc._, "coref_chains", None)
    if not chains:
        return text

    # Each edit is (char_start, char_end_exclusive, replacement_text).
    edits = []
    for chain in chains:
        try:
            # Use the longest mention as the canonical surface form.
            # Coreferee's Mention objects are iterable over token indices;
            # max() with `len` resolves to the longest span.
            canon = max(chain, key=lambda m: m[-1] - m[0])
            canon_text = doc[canon[0]:canon[-1] + 1].text
            for mention in chain:
                if list(mention) == list(canon):
                    continue
                tok_start, tok_end = mention[0], mention[-1]
                head = doc[tok_start]
                if head.pos_ not in _PRONOUN_POS:
                    continue
                mention_text = doc[tok_start:tok_end + 1].text
                # Skip nested matches (mention is sub-string of canon).
                if mention_text.lower() in canon_text.lower():
                    continue
                edits.append((
                    head.idx,
                    doc[tok_end].idx + len(doc[tok_end].text),
                    canon_text,
                ))
        except (IndexError, AttributeError, TypeError, ValueError) as exc:
            logger.debug("Coref: chain processing failed (%s)", exc)
            continue

    if not edits:
        return text

    edits.sort(key=lambda e: -e[0])  # right-to-left preserves indices
    result = text
    for start, end, repl in edits:
        result = result[:start] + repl + result[end:]
    return result
