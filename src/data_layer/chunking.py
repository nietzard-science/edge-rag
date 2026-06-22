"""
Chunking Module: Document Segmentation for the Edge-RAG System.

Architectural role
------------------
Sits between raw document intake and the vector/graph storage layer.
Produces LangChain Document objects consumed by `ingestion.py` and the
downstream `HybridStore` (LanceDB + KuzuDB).

Three production paths
----------------------
1. SENTENCE-BASED CHUNKING (`SpacySentenceChunker`) -- primary strategy.
   3-sentence sliding window with 1-sentence overlap (paper section 2.2).
   The overlap preserves entity bridges across chunk boundaries, which
   matters for multi-hop bridge questions where the bridging entity may
   span adjacent sentences. Sensitivity of retrieval recall to the
   (window, overlap) tuple is characterised in `chunking_ablation.py`.
   Reference: Lewis et al. (2020), "Retrieval-Augmented Generation for
   Knowledge-Intensive NLP Tasks", NeurIPS 2020.

2. SEMANTIC CHUNKING (`SemanticChunker`) -- alternative for structured
   documents. Combines TF-IDF importance scoring (Salton & Buckley,
   1988) with Shannon-entropy quality filtering (Shannon, 1948) and
   section-header-aware boundary detection. Designed for paper-style
   inputs with numbered section hierarchies. IDF uses log(N/df) with no
   add-one smoothing; a term occurring in all N chunks receives IDF=0,
   correctly down-weighting universal terms that survived the stopword
   filter.

3. UTILITY CHUNKERS (`SentenceChunker`, `FixedSizeChunker`,
   `RecursiveChunker`) -- back the "sentence", "fixed", and "recursive"
   values of the `ChunkingStrategy` enum (see `ingestion.py`). Invoked
   only when those strategies are selected for ablation studies; not
   intended for general use. The production path is the SpaCy sentence
   chunker above.

Public API
----------
Only `SpacySentenceChunker`, `SentenceChunkingConfig`, `SentenceChunk`,
and `create_sentence_chunker` are re-exported from the package
`__init__.py`. Everything else in this module is an implementation
detail consumed only by `ingestion.py`. The `__all__` declaration below
enforces this contract for `from chunking import *`.

Sentence-segmentation backends
------------------------------
Three sentence-segmentation routines exist in this file, each used in a
distinct context:

  * `SpacySentenceSegmenter._spacy_segment` -- production path used by
    `SpacySentenceChunker` when SpaCy is installed.
  * `SpacySentenceSegmenter._regex_segment` -- internal fallback used by
    `SpacySentenceChunker` when SpaCy or the requested SpaCy model
    cannot be loaded. Reference: Kiss & Strunk (2006), "Unsupervised
    multilingual sentence boundary detection."
  * `SentenceChunker` (Part 3) -- standalone regex chunker invoked when
    `ChunkingStrategy.SENTENCE` is selected in settings.yaml. Independent
    of the SpaCy chunker; preserved for ablation symmetry.

Deterministic chunk identifiers
-------------------------------
All chunk IDs are SHA-256 hashes truncated to a 20-character hex prefix
(80 bits, collision-resistant for corpora up to ~10^8 chunks under the
birthday bound):

  * Sentence chunks: SHA-256(source_doc + position + text_prefix).
  * Semantic chunks: SHA-256(source_doc + chunk_index + text_prefix).

Determinism guarantees that re-ingestion produces identical KuzuDB
node IDs, preserving cross-table foreign-key references and incremental-
update correctness.

Design constants vs configurable parameters
-------------------------------------------
The following thresholds are FIXED design constants empirically tuned on
the paper corpus and intentionally NOT exposed in `config/settings.yaml`.
Changing them would invalidate the chunking-ablation calibration:

  * `SemanticBoundaryDetector.min_boundary_distance = 200` -- minimum
    inter-boundary distance for the semantic boundary detector.
  * `AutomaticQualityFilter.TRANSCRIPT_RATIO_THRESHOLD = 0.3` -- dialog-
    label-density threshold for transcript-artifact detection.
  * `AutomaticQualityFilter.WHITESPACE_RATIO_THRESHOLD = 0.4` -- whitespace-
    fraction threshold for layout-artifact detection.
  * `SpacySentenceSegmenter.MIN_SENTENCE_CHARS = 5` -- minimum token-span
    length for a SpaCy sentence; shorter spans are sentencizer artifacts.

The stopword lists `ENGLISH_STOPWORDS` and `GERMAN_STOPWORDS` are a
project-internal curation derived from the NLTK 3.x stopword set with
project-specific additions for academic-text TF-IDF scoring. They are
embedded literally so this module has no NLTK dependency.

Settings.yaml mapping (verified against config/settings.yaml)
-------------------------------------------------------------
All tunable parameters originate from `config/settings.yaml` and are
validated against `_REQUIRED_SETTINGS` in `src/logic_layer/_settings_loader.py`
at startup. The defaults in this file are emergency fallbacks only.

  ingestion.sentences_per_chunk          -> SpacySentenceChunker.sentences_per_chunk (3)
  ingestion.sentence_overlap             -> SpacySentenceChunker.sentence_overlap (1)
  ingestion.min_chunk_size               -> SentenceChunkingConfig.min_chunk_chars (50)
  ingestion.max_chunk_chars              -> SentenceChunkingConfig.max_chunk_chars (2000)
  ingestion.word_boundary_factor         -> SemanticBoundaryDetector.word_boundary_factor (0.8)
  ingestion.spacy_model                  -> SentenceChunkingConfig.spacy_model ("en_core_web_sm")
  ingestion.entity_aware_chunking        -> SpacySentenceChunker.entity_aware (false; see note)
  ingestion.min_lexical_diversity        -> AutomaticQualityFilter.min_lexical_diversity (0.3)
  ingestion.min_information_density      -> AutomaticQualityFilter.min_information_density (2.0)
  ingestion.quality_filter.min_length    -> AutomaticQualityFilter.min_length (100)
  ingestion.quality_filter.min_words     -> AutomaticQualityFilter.min_words (15)
  chunking.chunk_size                    -> SemanticChunker.max_chunk_size (1024)
  chunking.chunk_overlap                 -> SemanticChunker.overlap (128)
  chunking.semantic.min_chunk_size       -> SemanticChunker.min_chunk_size (200)

Note on `entity_aware_chunking`: the flag is preserved in the public API
for `settings.yaml` compatibility but is currently a no-op. Setting
`entity_aware=True` emits a `logger.warning` at construction time
documenting that no entity-boundary-aware adjustment is applied. The
calibrated 1-sentence overlap already preserves bridges across
boundaries in practice; replacing the heuristic would require its own
ablation study.

Usage
-----
Sentence-based chunking (production):
    from src.data_layer.chunking import create_sentence_chunker
    chunker = create_sentence_chunker(sentences_per_chunk=3, sentence_overlap=1)
    chunks = chunker.chunk_text(text, source_doc="document.txt")

Semantic chunking (structured documents):
    from src.data_layer.chunking import create_semantic_chunker
    chunker = create_semantic_chunker(chunk_size=1024, chunk_overlap=128)
    chunks = chunker.chunk_document(document)

Last reviewed: 2026-05-25 (audit pass, project version 5.4).
"""

import hashlib
import logging
import math
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from spacy.language import Language

logger = logging.getLogger(__name__)

# Public API. Everything else in this module is an implementation detail
# consumed only by `ingestion.py` (utility chunkers behind the
# `ChunkingStrategy` enum) or internal to the chunker classes themselves.
__all__ = [
    "SpacySentenceChunker",
    "SentenceChunkingConfig",
    "SentenceChunk",
    "create_sentence_chunker",
]

# ─── LangChain ────────────────────────────────────────────────────────────────

try:
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        # Fallback to legacy path for older langchain installations
        from langchain.schema import Document  # type: ignore[no-redef,assignment]
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore[no-redef,assignment]
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        logger.warning(
            "LangChain not installed — RecursiveCharacterTextSplitter unavailable. "
            "Install with: pip install langchain-core langchain-text-splitters"
        )

        class Document:  # type: ignore[no-redef]
            """Minimal Document stub used only when langchain is absent."""
            def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
                self.page_content = page_content
                self.metadata = metadata or {}

        class RecursiveCharacterTextSplitter:  # type: ignore[no-redef]
            """Minimal text splitter stub used only when langchain is absent."""
            def __init__(
                self,
                chunk_size: int = 1024,
                chunk_overlap: int = 128,
                separators: Optional[List[str]] = None,
            ) -> None:
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.separators = separators or ["\n\n", "\n", " ", ""]

            def split_documents(self, documents: List[Any]) -> List[Any]:
                result = []
                for doc in documents:
                    text = doc.page_content
                    for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                        chunk_text = text[i:i + self.chunk_size].strip()
                        if chunk_text:
                            result.append(Document(
                                page_content=chunk_text,
                                metadata=doc.metadata.copy(),
                            ))
                return result

# ─── SpaCy ────────────────────────────────────────────────────────────────────

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning(
        "SpaCy not available — sentence chunker falls back to regex segmentation. "
        "Install with: pip install spacy && python -m spacy download en_core_web_sm"
    )


# ============================================================================
# SPACY MODEL CACHE
# ============================================================================

class SpacyModelCache:
    """
    Module-level singleton cache for SpaCy Language models.

    Prevents repeated disk loads when multiple chunkers are instantiated in
    the same process (e.g., parallel ingestion workers). Thread-safety is not
    guaranteed; this is acceptable for the target single-threaded edge pipeline.
    """

    _instances: Dict[str, Any] = {}

    @classmethod
    def get_model(
        cls, model_name: str, disable: Optional[List[str]] = None
    ) -> Optional["Language"]:
        """Return a cached SpaCy Language model, loading it from disk on first access."""
        if not SPACY_AVAILABLE:
            return None

        disable = disable or []
        cache_key = "%s__%s" % (model_name, "_".join(sorted(disable)))

        if cache_key not in cls._instances:
            try:
                nlp = spacy.load(model_name, disable=disable)
                # Add a sentencizer only if neither senter nor sentencizer is present.
                # SpaCy small models ship with a rule-based sentencizer; full
                # pipeline models include the neural senter. We add the lightweight
                # rule-based sentencizer as a safe default when neither is active.
                if "sentencizer" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
                    nlp.add_pipe("sentencizer")
                cls._instances[cache_key] = nlp
                logger.info("SpaCy model loaded and cached: %s", model_name)
            except OSError as exc:
                logger.warning(
                    "SpaCy model '%s' not found: %s. "
                    "Falling back to regex sentence segmentation.",
                    model_name, exc,
                )
                cls._instances[cache_key] = None

        return cls._instances[cache_key]

    @classmethod
    def clear_cache(cls) -> None:
        """Evict all cached models. Primarily used in tests to reset state."""
        cls._instances.clear()


# ============================================================================
# STOPWORDS FOR TF-IDF SCORING
# ============================================================================

ENGLISH_STOPWORDS = frozenset({
    'a', 'an', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
    'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'having', 'do', 'does', 'did', 'doing', 'will', 'would', 'could', 'should',
    'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'may',
    'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around', 'at',
    'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond', 'by',
    'down', 'during', 'except', 'for', 'from', 'in', 'inside', 'into', 'near', 'of',
    'off', 'on', 'onto', 'out', 'outside', 'over', 'past', 'since', 'through',
    'throughout', 'till', 'to', 'toward', 'towards', 'under', 'underneath', 'until',
    'unto', 'up', 'upon', 'with', 'within', 'without', 'and', 'but', 'or', 'nor',
    'yet', 'so', 'both', 'either', 'neither', 'not', 'only', 'own', 'same', 'than',
    'too', 'very', 'just', 'also', 'now', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'each', 'every', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'any', 'many', 'much', 'as', 'if', 'then', 'because', 'while', 'although',
    'though', 'once', 'unless', 'whether', 's', 't', 'd', 'll', 've', 're', 'm',
})

GERMAN_STOPWORDS = frozenset({
    'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einer', 'einem',
    'einen', 'eines', 'und', 'oder', 'aber', 'wenn', 'weil', 'dass', 'ist', 'sind',
    'war', 'waren', 'wird', 'werden', 'wurde', 'wurden', 'hat', 'haben', 'hatte',
    'hatten', 'sein', 'ihr', 'ihre', 'ihrer', 'ihrem', 'ihren', 'sich', 'auch',
    'als', 'so', 'wie', 'bei', 'mit', 'zu', 'zur', 'zum', 'von', 'vom', 'für',
    'auf', 'aus', 'an', 'in', 'im', 'am', 'um', 'nach', 'über', 'unter', 'vor',
    'hinter', 'neben', 'zwischen', 'durch', 'gegen', 'ohne', 'bis', 'seit',
    'während', 'trotz', 'wegen', 'es', 'er', 'sie', 'wir', 'ich', 'du', 'man',
    'nicht', 'nur', 'noch', 'schon', 'sehr', 'mehr', 'kann', 'können', 'muss',
    'müssen', 'soll', 'sollen', 'will', 'wollen', 'darf', 'dürfen',
})

ALL_STOPWORDS = ENGLISH_STOPWORDS | GERMAN_STOPWORDS


# ============================================================================
# PART 1: SEMANTIC CHUNKING
# ============================================================================

@dataclass
class ChunkMetadata:
    """Structured metadata for a semantically-chunked document segment."""

    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    heading_level: int = 0
    is_header: bool = False
    page_number: Optional[int] = None
    importance_score: float = 0.0
    lexical_diversity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chapter": self.chapter,
            "section": self.section,
            "subsection": self.subsection,
            "heading_level": self.heading_level,
            "is_header": self.is_header,
            "page_number": self.page_number,
            "importance_score": self.importance_score,
            "lexical_diversity": self.lexical_diversity,
        }


class HeaderExtractor:
    """
    Extract hierarchical document structure using language-agnostic regex patterns.

    Maintains stateful chapter/section context across successive calls to
    extract_headers() within a single document traversal. Call reset() before
    processing a new document to prevent context from leaking across document
    boundaries. SemanticChunker.chunk_document() calls reset() automatically.
    """

    CHAPTER_PATTERNS = [
        r'^(\d+)\.\s+([A-Z\u00C0-\u024F][^\n]{3,80})$',
        r'^([IVX]+)\.\s+([A-Z\u00C0-\u024F][^\n]{3,80})$',
        r'^\w+\s+(\d+)[:\s]+([^\n]{3,80})$',
    ]

    SECTION_PATTERNS = [
        r'^(\d+\.\d+)[\.\s]+([A-Z\u00C0-\u024F][^\n]{3,80})$',
    ]

    SUBSECTION_PATTERNS = [
        r'^(\d+\.\d+\.\d+)[\.\s]+([A-Z\u00C0-\u024F][^\n]{3,80})$',
    ]

    def __init__(self) -> None:
        self.current_chapter: Optional[str] = None
        self.current_section: Optional[str] = None
        self.current_subsection: Optional[str] = None

    def reset(self) -> None:
        """Reset document-level header context. Must be called between documents."""
        self.current_chapter = None
        self.current_section = None
        self.current_subsection = None

    def _scan_all_headers(self, text: str) -> None:
        """Scan all lines of a chunk to update the running chapter/section context."""
        for line in text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            for pattern in self.CHAPTER_PATTERNS:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    number, title = match.groups()
                    self.current_chapter = "%s. %s" % (number, title)
                    self.current_section = None
                    self.current_subsection = None
                    break
            for pattern in self.SECTION_PATTERNS:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    number, title = match.groups()
                    self.current_section = "%s %s" % (number, title)
                    self.current_subsection = None
                    break
            for pattern in self.SUBSECTION_PATTERNS:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    number, title = match.groups()
                    self.current_subsection = "%s %s" % (number, title)
                    break

    def extract_headers(self, text: str) -> Tuple[ChunkMetadata, str]:
        """
        Extract header information from a text chunk.

        Updates the running chapter/section context, detects whether the first
        line is a header, and returns a (ChunkMetadata, cleaned_text) pair
        where cleaned_text has the leading header line removed if applicable.
        """
        self._scan_all_headers(text)

        lines = text.strip().split('\n')
        first_line = lines[0].strip() if lines else ""

        metadata = ChunkMetadata()
        is_first_line_header = False
        header_level = 0

        for pattern in self.CHAPTER_PATTERNS:
            if re.match(pattern, first_line, re.MULTILINE):
                is_first_line_header, header_level = True, 1
                break
        if not is_first_line_header:
            for pattern in self.SECTION_PATTERNS:
                if re.match(pattern, first_line, re.MULTILINE):
                    is_first_line_header, header_level = True, 2
                    break
        if not is_first_line_header:
            for pattern in self.SUBSECTION_PATTERNS:
                if re.match(pattern, first_line, re.MULTILINE):
                    is_first_line_header, header_level = True, 3
                    break

        metadata.chapter = self.current_chapter
        metadata.section = self.current_section
        metadata.subsection = self.current_subsection
        metadata.heading_level = header_level
        metadata.is_header = is_first_line_header

        cleaned_text = '\n'.join(lines[1:]).strip() if is_first_line_header else text
        return metadata, cleaned_text


class SemanticBoundaryDetector:
    """
    Detect natural semantic boundaries in text for the SemanticChunker.

    Boundary patterns (in priority order):
      - Double newline:         paragraph break — strongest structural signal
      - Sentence-end + newline: sentence-final punctuation followed by uppercase
                                start — article section transitions
      - Punctuation + blank:    sentence group separated by blank line
      - Colon + newline:        list or enumeration introduction

    A candidate is accepted only when the distance from the previous accepted
    boundary exceeds word_boundary_factor * max_chunk_size AND exceeds
    min_boundary_distance, preventing excessively short chunks.

    word_boundary_factor maps to settings.yaml: ingestion.word_boundary_factor (0.8).
    min_boundary_distance is a FIXED design constant (default 200) intentionally
    not exposed in settings.yaml; see module docstring "Design constants vs
    configurable parameters" for the rationale. Override is supported in code
    for ablation only.
    """

    BOUNDARY_PATTERNS = [
        r'\n\n+',                           # paragraph break
        r'\.\s*\n(?=[A-Z\u00C0-\u024F])',  # sentence end before capitalised line
        r'[.!?]\s*\n\s*\n',                # sentence end + blank line
        r':\s*\n',                          # colon-terminated line (list/enum intro)
    ]

    def __init__(
        self,
        min_boundary_distance: int = 200,   # fixed design constant (see class docstring)
        word_boundary_factor: float = 0.8,  # settings.yaml: ingestion.word_boundary_factor
    ) -> None:
        self.min_boundary_distance = min_boundary_distance
        self.word_boundary_factor = word_boundary_factor

    def find_semantic_boundaries(self, text: str, max_chunk_size: int = 1024) -> List[int]:
        """
        Return sorted character positions at which the text may be split.

        A boundary is accepted when the distance from the previous boundary is
        >= word_boundary_factor * max_chunk_size AND >= min_boundary_distance.
        """
        boundaries = [0]
        potential_boundaries: List[int] = []

        for pattern in self.BOUNDARY_PATTERNS:
            for match in re.finditer(pattern, text):
                pos = match.end()
                if pos >= self.min_boundary_distance:
                    potential_boundaries.append(pos)

        potential_boundaries = sorted(set(potential_boundaries))

        current_position = 0
        acceptance_threshold = self.word_boundary_factor * max_chunk_size

        for boundary in potential_boundaries:
            distance = boundary - current_position
            if distance >= self.min_boundary_distance and distance >= acceptance_threshold:
                boundaries.append(boundary)
                current_position = boundary

        if boundaries[-1] != len(text):
            boundaries.append(len(text))

        return boundaries


class AutomaticQualityFilter:
    """
    Statistical quality gate for text chunks.

    A chunk is retained only if it passes all five filters (applied in order
    of ascending computational cost). Thresholds map to settings.yaml under
    ingestion.quality_filter.*

    Class-level ratio thresholds are FIXED design constants derived from
    empirical evaluation on the paper corpus and intentionally NOT exposed in
    settings.yaml (see module docstring "Design constants vs configurable
    parameters"):
      TRANSCRIPT_RATIO_THRESHOLD:  dialog-label density above which text is
                                   classified as a dialog transcript artifact.
      WHITESPACE_RATIO_THRESHOLD:  whitespace fraction above which text is
                                   classified as a layout/table artifact.
    Subclassing is supported for ablation; downstream production code does not
    override these.
    """

    # Empirically tuned on the paper corpus; fixed design constants.
    TRANSCRIPT_RATIO_THRESHOLD: float = 0.3
    WHITESPACE_RATIO_THRESHOLD: float = 0.4

    def __init__(
        self,
        min_length: int = 100,                 # settings.yaml: ingestion.quality_filter.min_length
        min_words: int = 15,                   # settings.yaml: ingestion.quality_filter.min_words
        min_lexical_diversity: float = 0.3,    # settings.yaml: ingestion.min_lexical_diversity
        min_information_density: float = 2.0,  # settings.yaml: ingestion.min_information_density
    ) -> None:
        self.min_length = min_length
        self.min_words = min_words
        self.min_lexical_diversity = min_lexical_diversity
        self.min_information_density = min_information_density

    def calculate_lexical_diversity(self, text: str) -> float:
        """
        Compute type-token ratio (TTR) as a measure of vocabulary richness.

        TTR = |unique_tokens| / |total_tokens|, range [0, 1]. Values below
        min_lexical_diversity indicate repetitive or boilerplate text.
        """
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    def calculate_information_density(self, text: str) -> float:
        """
        Compute Shannon entropy of the unigram word distribution.

        H = -sum_i p_i * log2(p_i) over all word types.

        Reference: Shannon, C.E. (1948). "A Mathematical Theory of
        Communication." Bell System Technical Journal, 27, 379-423.

        Low entropy (< min_information_density) indicates layout artifacts,
        highly repetitive lists, or near-empty chunks.
        """
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0

        word_counts = Counter(words)
        total_words = len(words)

        entropy = 0.0
        for count in word_counts.values():
            p = count / total_words
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def detect_transcript_pattern(self, text: str) -> bool:
        """Return True if the text resembles a dialog transcript (short speaker labels)."""
        pattern = r'(?:^|\n)\s*\w{1,3}\s*:\s*.{10,}'
        matches = re.findall(pattern, text)
        lines = text.split('\n')
        if not lines:
            return False
        return len(matches) / len(lines) > self.TRANSCRIPT_RATIO_THRESHOLD

    def detect_excessive_whitespace(self, text: str) -> bool:
        """Return True if whitespace exceeds the configured threshold (layout artifact)."""
        if not text:
            return True
        return (text.count(' ') + text.count('\t')) / len(text) > self.WHITESPACE_RATIO_THRESHOLD

    def should_keep_chunk(self, text: str) -> Tuple[bool, str, float]:
        """
        Apply all quality filters in sequence.

        Returns (keep, reason_string, lexical_diversity).  The lexical_diversity
        value is returned so that callers (SemanticChunker.chunk_document) can
        reuse it without recomputing, since it is already computed here.

        Filters are ordered from cheapest to most expensive.
        """
        if len(text) < self.min_length:
            return False, "too_short (%d chars)" % len(text), 0.0

        words = re.findall(r'\b\w+\b', text)
        if len(words) < self.min_words:
            return False, "too_few_words (%d words)" % len(words), 0.0

        if self.detect_transcript_pattern(text):
            return False, "transcript_pattern_detected", 0.0

        diversity = self.calculate_lexical_diversity(text)
        if diversity < self.min_lexical_diversity:
            return False, "low_lexical_diversity (%.2f)" % diversity, diversity

        density = self.calculate_information_density(text)
        if density < self.min_information_density:
            return False, "low_information_density (%.2f bits/word)" % density, diversity

        if self.detect_excessive_whitespace(text):
            return False, "layout_artifact", diversity

        return True, "passed", diversity


class TFIDFScorer:
    """
    TF-IDF importance scorer for text chunks.

    Computes normalised TF-IDF scores to rank chunk relevance within a
    document corpus. Stopwords are excluded from both TF and DF calculations.

    Reference: Salton, G. & Buckley, C. (1988). "Term-weighting approaches
    in automatic text retrieval." Information Processing & Management,
    24(5), 513-523.
    """

    def __init__(self, stopwords: Optional[frozenset] = None) -> None:
        self.stopwords = stopwords if stopwords is not None else ALL_STOPWORDS
        self.document_frequency: Dict[str, int] = {}
        self.total_chunks: int = 0
        self.chunk_term_frequencies: List[Counter] = []

    def reset(self) -> None:
        """Reset scorer state before analyzing a new corpus."""
        self.document_frequency = {}
        self.total_chunks = 0
        self.chunk_term_frequencies = []

    def _tokenize_and_filter(self, text: str) -> List[str]:
        """Tokenize and remove stopwords and single/two-character tokens."""
        words = re.findall(r'\b\w+\b', text.lower())
        # Minimum token length 3: single and two-character tokens are almost
        # never meaningful content terms in English or German academic text.
        return [w for w in words if w not in self.stopwords and len(w) > 2]

    def analyze_corpus(self, chunks: List[str]) -> None:
        """Build per-chunk TF tables and corpus-level DF table."""
        self.reset()
        self.total_chunks = len(chunks)

        for chunk in chunks:
            words = self._tokenize_and_filter(chunk)
            term_freq = Counter(words)
            self.chunk_term_frequencies.append(term_freq)
            for term in set(words):
                self.document_frequency[term] = self.document_frequency.get(term, 0) + 1

    def calculate_chunk_importance(self, chunk_index: int) -> float:
        """
        Return mean TF-IDF score for all content terms in the chunk.

        Returns 0.0 if the corpus has not been analyzed or the chunk is empty.
        IDF = log(N / df); a term in all N chunks receives IDF = 0 (intentional:
        universal terms carry no discriminative weight).
        """
        if self.total_chunks == 0 or chunk_index >= len(self.chunk_term_frequencies):
            return 0.0

        term_freq = self.chunk_term_frequencies[chunk_index]
        if not term_freq:
            return 0.0

        tfidf_score = 0.0
        for term, tf in term_freq.items():
            df = self.document_frequency.get(term, 1)
            idf = math.log(self.total_chunks / df) if df > 0 else 0.0
            tfidf_score += tf * idf

        total_terms = sum(term_freq.values())
        return tfidf_score / total_terms if total_terms > 0 else 0.0


class SemanticChunker:
    """
    Semantic chunking orchestrator for structured documents (Paper Section 2.3).

    Combines SemanticBoundaryDetector, HeaderExtractor, AutomaticQualityFilter,
    and TFIDFScorer. Falls back to RecursiveCharacterTextSplitter only on
    ValueError, RuntimeError, or AttributeError; other exceptions propagate.

    Parameter → settings.yaml mapping:
      max_chunk_size         → chunking.chunk_size (1024)
      min_chunk_size         → chunking.semantic.min_chunk_size (200)
      overlap                → chunking.chunk_overlap (128)
      word_boundary_factor   → ingestion.word_boundary_factor (0.8)
      min_words              → ingestion.quality_filter.min_words (15)
      min_lexical_diversity  → ingestion.min_lexical_diversity (0.3)
      min_info_density       → ingestion.min_information_density (2.0)
    """

    def __init__(
        self,
        max_chunk_size: int = 1024,              # settings.yaml: chunking.chunk_size
        min_chunk_size: int = 200,               # settings.yaml: chunking.semantic.min_chunk_size
        overlap: int = 128,                      # settings.yaml: chunking.chunk_overlap
        word_boundary_factor: float = 0.8,       # settings.yaml: ingestion.word_boundary_factor
        min_words: int = 15,                     # settings.yaml: ingestion.quality_filter.min_words
        min_lexical_diversity: float = 0.3,      # settings.yaml: ingestion.min_lexical_diversity
        min_info_density: float = 2.0,           # settings.yaml: ingestion.min_information_density
    ) -> None:
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap

        self.header_extractor = HeaderExtractor()
        self.boundary_detector = SemanticBoundaryDetector(
            min_boundary_distance=min_chunk_size,
            word_boundary_factor=word_boundary_factor,
        )
        self.quality_filter = AutomaticQualityFilter(
            min_length=min_chunk_size,
            min_words=min_words,
            min_lexical_diversity=min_lexical_diversity,
            min_information_density=min_info_density,
        )
        self.tfidf_scorer = TFIDFScorer()

        # Ordered separator priority: paragraph > line > sentence > word > char
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        logger.info(
            "SemanticChunker initialized: max_size=%d, min_size=%d, overlap=%d",
            max_chunk_size, min_chunk_size, overlap,
        )

    @staticmethod
    def _generate_chunk_id(source_doc: str, chunk_index: int, text: str) -> str:
        """
        Generate a deterministic, content-addressed chunk identifier.

        SHA-256 over (source_doc + chunk_index + text_prefix) ensures stable
        IDs on re-ingestion, preserving KuzuDB foreign key references.
        """
        content = "%s:%d:%s" % (source_doc, chunk_index, text[:50])
        return hashlib.sha256(content.encode()).hexdigest()[:20]

    def _find_overlap_start(self, text: str, boundary: int, target_overlap: int) -> int:
        """
        Find overlap start that respects word boundaries.

        Walks backward from (boundary - target_overlap) to the nearest
        whitespace, then forward past the whitespace to find the word start.
        Prevents mid-word splits at the beginning of overlapping chunks.
        """
        if boundary < target_overlap:
            return 0

        pos = boundary - target_overlap
        while pos > 0 and not text[pos].isspace():
            pos -= 1
        while pos < boundary and text[pos].isspace():
            pos += 1
        return pos

    def _extract_raw_chunks(self, text: str) -> List[str]:
        """Extract raw text chunks using detected semantic boundaries."""
        boundaries = self.boundary_detector.find_semantic_boundaries(
            text, self.max_chunk_size
        )

        chunks = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]

            # Apply word-boundary-aware overlap for all chunks except the first
            if i > 0 and start >= self.overlap:
                start = self._find_overlap_start(text, boundaries[i], self.overlap)

            chunk_text = text[start:end].strip()
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)

        return chunks

    def chunk_document(self, document: "Document") -> List["Document"]:
        """
        Segment a Document into semantically-coherent chunks with metadata.

        HeaderExtractor state is reset at the start of each call to prevent
        chapter/section context from leaking across document boundaries when
        this chunker instance is reused.

        Chunk IDs are deterministic SHA-256 hashes over (source_doc, index,
        text_prefix), ensuring stable KuzuDB node IDs across re-ingestion.
        """
        text = document.page_content
        base_metadata = document.metadata.copy()
        source_doc = base_metadata.get("source_file", "unknown")

        try:
            raw_chunks = self._extract_raw_chunks(text)
        except (ValueError, RuntimeError, AttributeError) as exc:
            logger.warning(
                "Semantic chunking failed (%s), using fallback: %s",
                type(exc).__name__, exc,
            )
            return self.fallback_splitter.split_documents([document])

        if not raw_chunks:
            # Short or structure-free document: delegate to character-level splitter.
            # This is expected for short documents; log at info level so the caller
            # knows which documents bypassed semantic segmentation.
            logger.info(
                "No semantic boundaries found in '%s'; delegating to fallback splitter",
                source_doc,
            )
            return self.fallback_splitter.split_documents([document])

        self.tfidf_scorer.analyze_corpus(raw_chunks)

        processed_chunks: List["Document"] = []
        filter_stats: Dict[str, Any] = {"kept": 0, "filtered": 0, "reasons": {}}

        self.header_extractor.reset()

        for i, chunk_text in enumerate(raw_chunks):
            metadata, cleaned_text = self.header_extractor.extract_headers(chunk_text)

            keep, reason, lexical_diversity = self.quality_filter.should_keep_chunk(cleaned_text)
            if not keep:
                filter_stats["filtered"] += 1
                filter_stats["reasons"][reason] = filter_stats["reasons"].get(reason, 0) + 1
                logger.debug("Filtered chunk %d: %s", i, reason)
                continue

            filter_stats["kept"] += 1

            importance_score = self.tfidf_scorer.calculate_chunk_importance(i)
            # lexical_diversity already computed by should_keep_chunk — reuse to avoid
            # a second O(n) pass over the text.

            chunk_index = len(processed_chunks)
            enriched_metadata = base_metadata.copy()
            enriched_metadata.update({
                "chunk_id": self._generate_chunk_id(source_doc, chunk_index, cleaned_text),
                "chunk_index": chunk_index,
                "chunk_size": len(cleaned_text),
                "chapter": metadata.chapter,
                "section": metadata.section,
                "subsection": metadata.subsection,
                "heading_level": metadata.heading_level,
                "is_header": metadata.is_header,
                "chunking_method": "semantic_automatic",
                "importance_score": round(importance_score, 4),
                "lexical_diversity": round(lexical_diversity, 4),
            })

            processed_chunks.append(
                Document(page_content=cleaned_text, metadata=enriched_metadata)
            )

        if filter_stats["filtered"] > 0 or len(raw_chunks) > 10:
            logger.info(
                "Semantic chunking '%s': %d raw -> %d kept (filtered %d: %s)",
                source_doc, len(raw_chunks), filter_stats["kept"],
                filter_stats["filtered"], filter_stats["reasons"],
            )

        return processed_chunks

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Primary interface consumed by ingestion.py (DocumentIngestionPipeline).

        Wraps chunk_document() and returns a list of dicts with keys 'text' and
        'metadata', compatible with the ingestion pipeline's internal format.
        """
        metadata = metadata or {}
        doc = Document(page_content=text, metadata=metadata)
        result_docs = self.chunk_document(doc)
        return [
            {"text": d.page_content, "metadata": d.metadata}
            for d in result_docs
        ]


def create_semantic_chunker(
    chunk_size: int = 1024,                 # settings.yaml: chunking.chunk_size
    chunk_overlap: int = 128,               # settings.yaml: chunking.chunk_overlap
    min_chunk_size: int = 200,              # settings.yaml: chunking.semantic.min_chunk_size
    word_boundary_factor: float = 0.8,      # settings.yaml: ingestion.word_boundary_factor
    min_words: int = 15,                    # settings.yaml: ingestion.quality_filter.min_words
    min_lexical_diversity: float = 0.3,     # settings.yaml: ingestion.min_lexical_diversity
    min_info_density: float = 2.0,          # settings.yaml: ingestion.min_information_density
) -> "SemanticChunker":
    """
    Factory for SemanticChunker. All parameters map to settings.yaml entries.
    Defaults are emergency fallbacks; production code should pass values from
    config/settings.yaml.
    """
    return SemanticChunker(
        max_chunk_size=chunk_size,
        min_chunk_size=min_chunk_size,
        overlap=chunk_overlap,
        word_boundary_factor=word_boundary_factor,
        min_words=min_words,
        min_lexical_diversity=min_lexical_diversity,
        min_info_density=min_info_density,
    )


# ============================================================================
# PART 2: SENTENCE-BASED CHUNKING (Primary Strategy)
# ============================================================================

@dataclass
class SentenceChunkingConfig:
    """
    Configuration for the 3-sentence sliding-window chunker.

    Default values match the settings.yaml entries listed below. All values
    are validated in __post_init__ to surface misconfiguration at instantiation
    time rather than during processing.

      sentences_per_chunk → settings.yaml: ingestion.sentences_per_chunk (3)
      sentence_overlap    → settings.yaml: ingestion.sentence_overlap (1)
      min_chunk_chars     → settings.yaml: ingestion.min_chunk_size (50)
      max_chunk_chars     → settings.yaml: ingestion.max_chunk_chars (2000)
      spacy_model         → settings.yaml: ingestion.spacy_model ("en_core_web_sm")
      entity_aware        → settings.yaml: ingestion.entity_aware_chunking (false)
    """

    sentences_per_chunk: int = 3
    sentence_overlap: int = 1
    min_chunk_chars: int = 50
    max_chunk_chars: int = 2000
    spacy_model: str = "en_core_web_sm"
    # NER and parser are disabled for sentence segmentation to minimise latency.
    # Named entity recognition is handled separately by GLiNER (entity_extraction.py).
    disable_components: List[str] = field(default_factory=lambda: ["ner", "parser"])
    entity_aware: bool = False
    include_sentence_offsets: bool = True

    def __post_init__(self) -> None:
        if self.sentences_per_chunk < 1:
            raise ValueError(
                "sentences_per_chunk must be >= 1, got %d" % self.sentences_per_chunk
            )
        if self.sentence_overlap < 0:
            raise ValueError(
                "sentence_overlap must be >= 0, got %d" % self.sentence_overlap
            )
        if self.sentence_overlap >= self.sentences_per_chunk:
            raise ValueError(
                "sentence_overlap (%d) must be < sentences_per_chunk (%d)"
                % (self.sentence_overlap, self.sentences_per_chunk)
            )


@dataclass
class SentenceInfo:
    """Metadata for a single sentence within a chunk."""

    text: str
    start_char: int
    end_char: int
    index: int


@dataclass
class SentenceChunk:
    """
    A chunk comprising multiple consecutive sentences.

    chunk_id is a deterministic SHA-256-based identifier derived from
    source_doc, position, and the first 50 characters of the chunk text.
    This ensures that re-ingestion produces identical KuzuDB node IDs,
    preserving graph integrity across ingestion runs.

    The 20-character hex prefix (80 bits) provides sufficient collision
    resistance for corpora up to ~10^8 chunks (birthday bound).
    """

    chunk_id: str
    text: str
    sentences: List[SentenceInfo]
    position: int
    source_doc: str
    char_start: int
    char_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def sentence_count(self) -> int:
        return len(self.sentences)

    @property
    def sentence_indices(self) -> List[int]:
        return [s.index for s in self.sentences]

    def to_langchain_document(self) -> "Document":
        return Document(
            page_content=self.text,
            metadata={
                "chunk_id": self.chunk_id,
                "position": self.position,
                "source_doc": self.source_doc,
                "source_file": self.source_doc,
                "sentence_count": self.sentence_count,
                "char_start": self.char_start,
                "char_end": self.char_end,
                "sentence_indices": self.sentence_indices,
                "chunk_method": "sentence_spacy_3_window",
                **self.metadata,
            }
        )


class SpacySentenceSegmenter:
    """
    Sentence boundary detection backed by SpaCy.

    Loads the model from SpacyModelCache to avoid repeated disk reads.
    Automatically falls back to a regex-based segmenter when SpaCy is
    unavailable or the requested model is not installed.
    """

    # Minimum character length for a token span to be treated as a sentence.
    # Shorter spans are typically fragment artifacts from the sentencizer.
    MIN_SENTENCE_CHARS: int = 5

    def __init__(self, config: SentenceChunkingConfig) -> None:
        self.config = config
        self.nlp: Optional["Language"] = None
        self.using_spacy: bool = False
        self._load_model()

    def _load_model(self) -> None:
        """Load SpaCy model via SpacyModelCache (no-op if already cached)."""
        if not SPACY_AVAILABLE:
            logger.warning("SpaCy not available; using regex sentence segmentation fallback")
            return

        self.nlp = SpacyModelCache.get_model(
            self.config.spacy_model,
            disable=self.config.disable_components,
        )
        if self.nlp is not None:
            self.using_spacy = True

    def segment(self, text: str) -> List[SentenceInfo]:
        """Segment text into SentenceInfo objects; returns [] for None or empty input."""
        if not text:
            return []
        if self.using_spacy and self.nlp is not None:
            return self._spacy_segment(text)
        return self._regex_segment(text)

    def _spacy_segment(self, text: str) -> List[SentenceInfo]:
        doc = self.nlp(text)
        sentences = []
        for idx, sent in enumerate(doc.sents):
            sent_text = sent.text.strip()
            if len(sent_text) < self.MIN_SENTENCE_CHARS:
                continue
            sentences.append(SentenceInfo(
                text=sent_text,
                start_char=sent.start_char,
                end_char=sent.end_char,
                index=idx,
            ))
        return sentences

    def _regex_segment(self, text: str) -> List[SentenceInfo]:
        """
        Regex-based fallback segmenter.

        Splits on sentence-final punctuation followed by whitespace and an
        uppercase character. Accuracy degrades for text with dense abbreviation
        usage; SpaCy is strongly preferred when available.
        """
        if not text.strip():
            return []

        sentences = []
        current_pos = 0
        for sent_idx, part in enumerate(re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)):
            part_stripped = part.strip()
            if len(part_stripped) < self.MIN_SENTENCE_CHARS:
                continue

            start = text.find(part_stripped, current_pos)
            if start == -1:
                # part_stripped could not be located after current_pos.
                # This should not occur with a simple regex split on the same text,
                # but guard defensively to avoid incorrect character offsets.
                logger.debug(
                    "Regex segmenter: substring not found at expected position "
                    "(sent_idx=%d, current_pos=%d); using approximate offset.",
                    sent_idx, current_pos,
                )
                start = current_pos
            end = start + len(part_stripped)
            current_pos = end

            sentences.append(SentenceInfo(
                text=part_stripped,
                start_char=start,
                end_char=end,
                index=sent_idx,
            ))

        return sentences


class SpacySentenceChunker:
    """
    3-sentence sliding-window chunker — primary ingestion strategy (Paper §2.2).

    Each chunk spans sentences_per_chunk consecutive sentences with
    sentence_overlap sentences shared between adjacent windows. The overlap
    prevents entity bridges from being severed at chunk boundaries, which is
    critical for multi-hop retrieval in HotpotQA-style bridge queries.

    Reference: Lewis, P. et al. (2020). "Retrieval-Augmented Generation for
    Knowledge-Intensive NLP Tasks." NeurIPS 2020.
    (3-sentence window size validated by ablation study, Paper §4.2.)

    Chunk identifiers are deterministic SHA-256 hashes over (source_doc,
    position, text_prefix), ensuring that KuzuDB graph node IDs remain stable
    across re-ingestion runs.  The 20-char hex prefix (80 bits) provides
    sufficient collision resistance for corpora up to ~10^8 chunks.
    """

    def __init__(
        self,
        sentences_per_chunk: int = 3,       # settings.yaml: ingestion.sentences_per_chunk
        sentence_overlap: int = 1,           # settings.yaml: ingestion.sentence_overlap
        min_chunk_chars: int = 50,           # settings.yaml: ingestion.min_chunk_size
        max_chunk_chars: int = 2000,         # settings.yaml: ingestion.max_chunk_chars
        spacy_model: str = "en_core_web_sm", # settings.yaml: ingestion.spacy_model
        entity_aware: bool = False,          # settings.yaml: ingestion.entity_aware_chunking
    ) -> None:
        self.config = SentenceChunkingConfig(
            sentences_per_chunk=sentences_per_chunk,
            sentence_overlap=sentence_overlap,
            min_chunk_chars=min_chunk_chars,
            max_chunk_chars=max_chunk_chars,
            spacy_model=spacy_model,
            entity_aware=entity_aware,
        )

        self.segmenter = SpacySentenceSegmenter(self.config)
        # entity_aware is accepted to maintain API compatibility with
        # settings.yaml (`ingestion.entity_aware_chunking`) but is currently
        # a no-op. Warn loudly so a user who flips the setting expecting an
        # effect does not get silent acceptance. The 1-sentence overlap
        # already preserves entity bridges across boundaries in practice.
        if entity_aware:
            logger.warning(
                "SpacySentenceChunker: entity_aware=True requested but is a "
                "no-op in this release; no entity-boundary-aware adjustment "
                "is applied. See chunking.py module docstring for rationale."
            )

        logger.info(
            "SpacySentenceChunker initialized: %d-sentence windows, overlap=%d",
            self.config.sentences_per_chunk, self.config.sentence_overlap,
        )

    @staticmethod
    def _generate_chunk_id(source_doc: str, position: int, text: str) -> str:
        """
        Generate a deterministic, content-addressed chunk identifier.

        SHA-256 over (source_doc + position + text_prefix) ensures identical
        IDs on re-ingestion, preserving KuzuDB foreign key references.
        """
        content = "%s:%d:%s" % (source_doc, position, text[:50])
        return hashlib.sha256(content.encode()).hexdigest()[:20]

    def chunk_text(
        self,
        text: str,
        source_doc: str = "unknown",
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SentenceChunk]:
        """Segment text into overlapping SentenceChunk objects."""
        if text is None:
            logger.warning("chunk_text received None for source_doc=%r", source_doc)
            return []

        base_metadata = base_metadata or {}
        sentences = self.segmenter.segment(text)

        if not sentences:
            logger.warning("No sentences found in document: %r", source_doc)
            return []

        return self._sliding_window_chunk(sentences, source_doc, base_metadata)

    def _sliding_window_chunk(
        self,
        sentences: List[SentenceInfo],
        source_doc: str,
        base_metadata: Dict[str, Any],
    ) -> List[SentenceChunk]:
        """
        Build overlapping sentence windows.

        Window size = sentences_per_chunk; step = window_size - overlap.
        A trailing-sentences handler ensures no sentences are dropped when
        len(sentences) is not evenly divisible by step_size.
        """
        window_size = self.config.sentences_per_chunk
        step_size = max(1, window_size - self.config.sentence_overlap)

        chunks: List[SentenceChunk] = []
        position = 0
        i = 0

        while i < len(sentences):
            window_end = min(i + window_size, len(sentences))
            window_sentences = sentences[i:window_end]
            chunk_text = " ".join(s.text for s in window_sentences)

            # Extend window if chunk falls below minimum size
            if len(chunk_text) < self.config.min_chunk_chars and window_end < len(sentences):
                window_end = min(window_end + 1, len(sentences))
                window_sentences = sentences[i:window_end]
                chunk_text = " ".join(s.text for s in window_sentences)

            # Truncate window if chunk exceeds maximum size
            while len(chunk_text) > self.config.max_chunk_chars and len(window_sentences) > 1:
                window_sentences = window_sentences[:-1]
                chunk_text = " ".join(s.text for s in window_sentences)

            if len(chunk_text) < self.config.min_chunk_chars:
                i += step_size
                continue

            chunks.append(SentenceChunk(
                chunk_id=self._generate_chunk_id(source_doc, position, chunk_text),
                text=chunk_text,
                sentences=window_sentences,
                position=position,
                source_doc=source_doc,
                char_start=window_sentences[0].start_char,
                char_end=window_sentences[-1].end_char,
                metadata={
                    **base_metadata,
                    "chunk_method": "sentence_spacy_3_window",
                    "sentences_per_chunk": self.config.sentences_per_chunk,
                    "sentence_overlap": self.config.sentence_overlap,
                },
            ))

            position += 1
            i += step_size

            if i >= len(sentences) - 1 and window_end >= len(sentences):
                break

        # Trailing sentences: emit a final chunk for sentences not covered by the
        # main loop (occurs when len(sentences) % step_size != 0).
        if sentences and chunks:
            last_covered_idx = chunks[-1].sentences[-1].index
            remaining = [s for s in sentences if s.index > last_covered_idx]
            if remaining:
                chunk_text = " ".join(s.text for s in remaining)
                if len(chunk_text) >= self.config.min_chunk_chars:
                    chunks.append(SentenceChunk(
                        chunk_id=self._generate_chunk_id(source_doc, position, chunk_text),
                        text=chunk_text,
                        sentences=remaining,
                        position=position,
                        source_doc=source_doc,
                        char_start=remaining[0].start_char,
                        char_end=remaining[-1].end_char,
                        metadata={
                            **base_metadata,
                            "chunk_method": "sentence_spacy_3_window",
                            "is_final_chunk": True,
                        },
                    ))

        return chunks

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Primary interface consumed by ingestion.py (DocumentIngestionPipeline).

        Returns a list of dicts with keys 'text' and 'metadata', compatible
        with the ingestion pipeline's internal format.

        'chunk_index' is set to chunk.position so that storage.py can create
        ordered NEXT_CHUNK edges in KuzuDB (chunk_index=0 for all chunks would
        otherwise make sequential ordering impossible).
        """
        metadata = metadata or {}
        source_doc = metadata.get("source_file", metadata.get("source", "unknown"))
        chunks = self.chunk_text(text, source_doc=source_doc, base_metadata=metadata)

        return [
            {
                "text": chunk.text,
                "metadata": {
                    **chunk.metadata,
                    "chunk_id": chunk.chunk_id,
                    "position": chunk.position,
                    "chunk_index": chunk.position,  # required by storage.py NEXT_CHUNK ordering
                    "source_file": source_doc,
                    "sentence_start": chunk.sentences[0].index if chunk.sentences else 0,
                    "sentence_end": chunk.sentences[-1].index + 1 if chunk.sentences else 0,
                    "sentence_count": chunk.sentence_count,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                },
            }
            for chunk in chunks
        ]

    def chunk_to_documents(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List["Document"]:
        """Convert chunked text directly to LangChain Document objects."""
        metadata = metadata or {}
        source_doc = metadata.get("source_file", metadata.get("source", "unknown"))
        chunks = self.chunk_text(text, source_doc=source_doc, base_metadata=metadata)
        return [chunk.to_langchain_document() for chunk in chunks]

    def chunk_documents(self, documents: List["Document"]) -> List["Document"]:
        """Chunk a list of LangChain Documents into sentence-window Documents."""
        all_chunks: List["Document"] = []
        for doc in documents:
            source = doc.metadata.get("source_file", doc.metadata.get("source", "unknown"))
            chunks = self.chunk_text(
                doc.page_content, source_doc=source, base_metadata=doc.metadata
            )
            all_chunks.extend(chunk.to_langchain_document() for chunk in chunks)
        return all_chunks


def create_sentence_chunker(
    sentences_per_chunk: int = 3,       # settings.yaml: ingestion.sentences_per_chunk
    sentence_overlap: int = 1,           # settings.yaml: ingestion.sentence_overlap
    spacy_model: str = "en_core_web_sm", # settings.yaml: ingestion.spacy_model
    entity_aware: bool = False,          # settings.yaml: ingestion.entity_aware_chunking
    **kwargs,
) -> SpacySentenceChunker:
    """
    Factory for SpacySentenceChunker. All parameters map to settings.yaml entries.
    Defaults are emergency fallbacks; production code should pass values from
    config/settings.yaml.
    """
    return SpacySentenceChunker(
        sentences_per_chunk=sentences_per_chunk,
        sentence_overlap=sentence_overlap,
        spacy_model=spacy_model,
        entity_aware=entity_aware,
        **kwargs,
    )


# ============================================================================
# PART 3: UTILITY CHUNKERS
# ============================================================================
# These implementations are used by ingestion.py for the "sentence", "fixed",
# and "recursive" strategies. They live here so that all chunking logic is
# consolidated in one module.
# ============================================================================

class SentenceChunker:
    """
    Regex-based sentence chunker — fast fallback when SpaCy is unavailable.

    Groups N sentences per chunk using a simple punctuation-boundary regex.
    This strategy is used when ChunkingStrategy.SENTENCE is selected, or as
    a fallback when SpaCy cannot be loaded.

    Reference: Kiss, T. & Strunk, J. (2006). "Unsupervised multilingual sentence
    boundary detection." Computational Linguistics, 32(4), 485-525.
    """

    SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

    def __init__(self, sentences_per_chunk: int = 3, min_chunk_size: int = 50) -> None:
        self.sentences_per_chunk = sentences_per_chunk
        self.min_chunk_size = min_chunk_size

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using a punctuation-boundary regex."""
        if not text or not text.strip():
            return []
        return [s.strip() for s in self.SENTENCE_PATTERN.split(text) if s.strip()]

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Chunk text by grouping N sentences.

        Returns a list of dicts with keys 'text' and 'metadata', compatible
        with the ingestion pipeline's internal format.
        """
        metadata = metadata or {}
        if not text or not text.strip():
            return []

        sentences = self.split_into_sentences(text)
        if not sentences:
            return []

        chunks = []
        for i in range(0, len(sentences), self.sentences_per_chunk):
            chunk_sents = sentences[i:i + self.sentences_per_chunk]
            chunk_text = " ".join(chunk_sents)
            if len(chunk_text.strip()) < self.min_chunk_size:
                continue
            chunk_meta = metadata.copy()
            chunk_meta.update({
                "sentence_start": i,
                "sentence_end": i + len(chunk_sents),
                "sentence_count": len(chunk_sents),
                "chunk_index": len(chunks),
                "chunk_method": "sentence_regex",
            })
            chunks.append({"text": chunk_text, "metadata": chunk_meta})
        return chunks


class FixedSizeChunker:
    """
    Fixed-size chunking with configurable overlap and word-boundary snapping.

    Breaks text into segments of at most chunk_size characters, trying to
    end each segment at a word boundary rather than mid-token.

    Reference: Lewis, P. et al. (2020). "Retrieval-Augmented Generation for
    Knowledge-Intensive NLP Tasks." NeurIPS 2020.
    """

    def __init__(
        self,
        chunk_size: int = 1024,          # settings.yaml: ingestion.chunk_size
        chunk_overlap: int = 128,        # settings.yaml: ingestion.chunk_overlap
        min_chunk_size: int = 50,        # settings.yaml: ingestion.min_chunk_size
        word_boundary_factor: float = 0.8,  # settings.yaml: ingestion.word_boundary_factor
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.word_boundary_factor = word_boundary_factor

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Chunk text into fixed-size pieces with overlap.

        Returns a list of dicts with keys 'text' and 'metadata'.
        """
        metadata = metadata or {}
        if not text or not text.strip():
            return []

        chunks = []
        start = 0

        while start < len(text):
            prev_start = start
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Snap to word boundary if not at end of text
            if end < len(text):
                last_space = chunk_text.rfind(' ')
                if last_space > self.chunk_size * self.word_boundary_factor:
                    chunk_text = chunk_text[:last_space]
                    end = start + last_space

            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunk_meta = metadata.copy()
                chunk_meta.update({
                    "char_start": start,
                    "char_end": end,
                    "chunk_index": len(chunks),
                    "chunk_method": "fixed",
                })
                chunks.append({"text": chunk_text.strip(), "metadata": chunk_meta})

            # Advance with overlap; guard against infinite loop
            start = end - self.chunk_overlap
            start = max(start, prev_start + 1)
            if start >= len(text) - self.min_chunk_size:
                break

        return chunks


class RecursiveChunker:
    """
    Wrapper for RecursiveCharacterTextSplitter from LangChain.

    Recursively splits on paragraph, line, sentence, and word boundaries
    until chunks are below chunk_size. Falls back to fixed-size chunking when
    LangChain is not installed, emitting a FALLBACK warning.
    """

    def __init__(
        self,
        chunk_size: int = 1024,          # settings.yaml: ingestion.chunk_size
        chunk_overlap: int = 128,        # settings.yaml: ingestion.chunk_overlap
        min_chunk_size: int = 50,        # settings.yaml: ingestion.min_chunk_size
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self._splitter = None
        self._using_fallback = False

    def _get_splitter(self):
        """Lazy initialise splitter; log warning when LangChain is absent."""
        if self._splitter is None:
            if LANGCHAIN_AVAILABLE:
                self._splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
            else:
                logger.warning(
                    "FALLBACK ACTIVE: LangChain not available -- "
                    "RecursiveChunker using simple fixed-size fallback. "
                    "Install with: pip install langchain-text-splitters"
                )
                self._splitter = FixedSizeChunker(
                    self.chunk_size, self.chunk_overlap, self.min_chunk_size
                )
                self._using_fallback = True
        return self._splitter

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Chunk using RecursiveCharacterTextSplitter or fixed-size fallback.

        Returns a list of dicts with keys 'text' and 'metadata'.
        """
        metadata = metadata or {}
        if not text or not text.strip():
            return []

        splitter = self._get_splitter()

        if self._using_fallback:
            return splitter.chunk(text, metadata)

        texts = splitter.split_text(text)
        chunks = []
        for chunk_text in texts:
            if len(chunk_text.strip()) < self.min_chunk_size:
                continue
            chunk_meta = metadata.copy()
            chunk_meta.update({
                "chunk_index": len(chunks),
                "chunk_method": "recursive",
            })
            chunks.append({"text": chunk_text.strip(), "metadata": chunk_meta})
        return chunks


# ============================================================================
# SELF-VERIFICATION  (python -m src.data_layer.chunking)
# ============================================================================

def _main() -> None:
    """Smoke demo and test runner for direct module invocation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    _sample = (
        "Albert Einstein was born on March 14, 1879, in Ulm, Germany. "
        "He developed the theory of relativity. Einstein received the Nobel "
        "Prize in Physics in 1921. He emigrated to the United States in 1933 "
        "and worked at Princeton University until his death in 1955."
    )

    # ── Sentence Chunker ──────────────────────────────────────────────────────
    sc = create_sentence_chunker(sentences_per_chunk=3, sentence_overlap=1)
    sentence_chunks = sc.chunk_text(_sample, source_doc="demo.txt")
    assert sentence_chunks, "Expected at least one sentence chunk"
    for chunk in sentence_chunks:
        logger.info(
            "SentenceChunk [%d] id=%s sents=%s  %.80s",
            chunk.position, chunk.chunk_id, chunk.sentence_indices, chunk.text,
        )
    logger.info("Sentence chunker smoke demo: %d chunks produced", len(sentence_chunks))

    # ── Semantic Chunker ──────────────────────────────────────────────────────
    try:
        _doc = Document(page_content=_sample, metadata={"source_file": "demo.txt"})
        sem = create_semantic_chunker(chunk_size=300, chunk_overlap=50, min_chunk_size=80)
        sem_chunks = sem.chunk_document(_doc)
        logger.info("Semantic chunker smoke demo: %d chunks produced", len(sem_chunks))
        for c in sem_chunks:
            logger.info(
                "SemanticChunk score=%.3f  %.70s",
                c.metadata.get("importance_score", 0), c.page_content,
            )
    except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as exc:
        logger.warning("Semantic chunker smoke demo skipped: %s", exc)

    logger.info("Smoke demo passed.")

    # ── pytest ────────────────────────────────────────────────────────────────
    test_file = (
        __import__("pathlib").Path(__file__).parent.parent.parent
        / "test_system" / "test_chunking.py"
    )
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "pytest", str(test_file), "-v"],
        check=False,
    )
    sys.exit(proc.returncode)


if __name__ == "__main__":
    _main()
