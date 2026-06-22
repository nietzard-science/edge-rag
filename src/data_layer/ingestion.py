"""
Document Ingestion Pipeline -- Configurable Chunking Strategies

Author: Jan Nietzard

================================================================================
OVERVIEW
================================================================================

Central ingestion pipeline for all chunking operations.
Supports five strategies that can be selected via config/settings.yaml.

================================================================================
CHUNKING STRATEGIES
================================================================================

    1. sentence       - Regex-based sentence chunking (SentenceChunker)
    2. sentence_spacy - SpaCy 3-sentence window (SpacySentenceChunker, paper §2.2)
    3. semantic       - Semantic boundaries using TF-IDF (SemanticChunker)
    4. fixed          - Fixed character count with word-boundary snap (FixedSizeChunker)
    5. recursive      - RecursiveCharacterTextSplitter from LangChain

All chunker implementations live in chunking.py. This module is responsible
only for strategy selection, configuration, and the DocumentIngestionPipeline
orchestration layer.

================================================================================
CONFIGURATION
================================================================================

All parameters originate from config/settings.yaml (ingestion: block).
Use create_ingestion_config(cfg) to build DocumentIngestionConfig from a loaded
YAML dict rather than constructing it manually.

================================================================================
USAGE
================================================================================

    from src.data_layer.ingestion import DocumentIngestionPipeline, DocumentIngestionConfig
    from src.data_layer.ingestion import create_ingestion_config

    # From settings.yaml dict
    config = create_ingestion_config(yaml_cfg)
    pipeline = DocumentIngestionPipeline(config)
    chunks = pipeline.process_text("Your text here...")

    # Quick start with explicit parameters
    from src.data_layer.ingestion import create_data_layer_pipeline
    pipeline = create_data_layer_pipeline(strategy="sentence_spacy")

================================================================================

Last reviewed: 2026-05-25 (audit pass, project version 5.4).
"""

import logging
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Public API. All five symbols are re-exported via src/data_layer/__init__.py
# and consumed by tests + src/pipeline/ingestion_pipeline.py.
__all__ = [
    "DocumentIngestionPipeline",
    "DocumentIngestionConfig",
    "ChunkingStrategy",
    "create_ingestion_config",
    "create_data_layer_pipeline",
]


# ============================================================================
# OPTIONAL DEPENDENCIES
# ============================================================================

# LangChain Document — used by process_documents() API
try:
    from langchain_core.documents import Document
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.schema import Document  # type: ignore[no-redef,assignment]
        _LANGCHAIN_AVAILABLE = True
    except ImportError:
        _LANGCHAIN_AVAILABLE = False

        @dataclass
        class Document:  # type: ignore[no-redef]
            """Minimal Document stub used only when langchain is absent."""
            page_content: str
            metadata: Dict[str, Any] = None

            def __post_init__(self):
                if self.metadata is None:
                    self.metadata = {}

        logger.debug("LangChain not available; using minimal Document stub")

# Chunking implementations — all strategies live in chunking.py
try:
    from .chunking import (
        SpacySentenceChunker,
        SemanticChunker,
        SentenceChunker,
        FixedSizeChunker,
        RecursiveChunker,
        create_sentence_chunker,
        create_semantic_chunker,
    )
    _CHUNKING_AVAILABLE = True
except ImportError as _chunking_err:
    logger.warning(
        "FALLBACK ACTIVE: chunking module not available (%s). "
        "Only the built-in SentenceChunker (regex) will work. "
        "Install SpaCy: pip install spacy && python -m spacy download en_core_web_sm",
        _chunking_err,
    )
    _CHUNKING_AVAILABLE = False
    SpacySentenceChunker = None  # type: ignore[assignment,misc]
    SemanticChunker = None       # type: ignore[assignment,misc]

    # Minimal inline fallback — keeps the pipeline functional without chunking.py
    class SentenceChunker:  # type: ignore[no-redef]
        """Inline regex fallback used only when chunking.py is absent."""
        import re as _re
        _PATTERN = _re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

        def __init__(self, sentences_per_chunk: int = 3, min_chunk_size: int = 50) -> None:
            self.sentences_per_chunk = sentences_per_chunk
            self.min_chunk_size = min_chunk_size

        def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
            metadata = metadata or {}
            sentences = [s.strip() for s in self._PATTERN.split(text) if s.strip()]
            chunks = []
            for i in range(0, len(sentences), self.sentences_per_chunk):
                t = " ".join(sentences[i:i + self.sentences_per_chunk])
                if len(t) >= self.min_chunk_size:
                    m = metadata.copy()
                    m["chunk_index"] = len(chunks)
                    m["chunk_method"] = "sentence_regex_fallback"
                    chunks.append({"text": t, "metadata": m})
            return chunks

    FixedSizeChunker = None   # type: ignore[assignment,misc]
    RecursiveChunker = None   # type: ignore[assignment,misc]
    create_sentence_chunker = None  # type: ignore[assignment]
    create_semantic_chunker = None  # type: ignore[assignment]


# ============================================================================
# CONFIGURATION
# ============================================================================

class ChunkingStrategy(Enum):
    """Available chunking strategies (maps to settings.yaml ingestion.chunking_strategy)."""
    SENTENCE = "sentence"             # Regex-based N-sentences per chunk
    SENTENCE_SPACY = "sentence_spacy" # SpaCy 3-sentence window (paper §2.2)
    SEMANTIC = "semantic"             # Semantic TF-IDF boundaries
    FIXED = "fixed"                   # Fixed character count with word-boundary snap
    RECURSIVE = "recursive"           # LangChain RecursiveCharacterTextSplitter


@dataclass
class DocumentIngestionConfig:
    """
    Configuration for the data-layer ``DocumentIngestionPipeline`` (chunking).

    NOTE — distinct from ``src.pipeline.ingestion_pipeline.IngestionConfig``:
    that one configures the FULL ingestion run (chunking + GLiNER/REBEL entity
    extraction + embeddings + storage). THIS class configures only chunking
    strategy selection for the data-layer ``DocumentIngestionPipeline``.

    All defaults match config/settings.yaml (ingestion: block).
    Construct from settings.yaml via create_ingestion_config(cfg) rather
    than setting individual fields to avoid drift from the YAML source of truth.

    Attributes
    ----------
    chunking_strategy : str
        One of "sentence", "sentence_spacy", "semantic", "fixed", "recursive".
    chunk_size : int
        Max chunk size in characters (fixed / recursive / semantic strategies).
    chunk_overlap : int
        Overlap in characters between consecutive chunks.
    min_chunk_size : int
        Minimum chunk length; shorter chunks are silently dropped.
    sentences_per_chunk : int
        Window size for sentence-based strategies.
    sentence_overlap : int
        Number of sentences shared between consecutive windows.
    max_chunk_chars : int
        Hard cap on sentence-chunk length; long chunks are split at this boundary.
    spacy_model : str
        SpaCy model name for sentence segmentation.
    entity_aware_chunking : bool
        Reserved for future entity-boundary snapping; currently unused.
    min_lexical_diversity : float
        Minimum type-token ratio for semantic quality filter.
    min_information_density : float
        Minimum Shannon entropy (bits) for semantic quality filter.
    word_boundary_factor : float
        Fixed-chunk word-boundary snapping fraction (0–1).
    extract_entities : bool
        Reserved for compatibility; entity extraction is handled by
        EntityExtractionPipeline (entity_extraction.py).
    add_source_metadata : bool
        Attach source_id to chunk metadata.
    """
    # Core
    chunking_strategy: str = "sentence_spacy"
    chunk_size: int = 1024
    chunk_overlap: int = 128
    min_chunk_size: int = 50

    # Sentence strategy
    sentences_per_chunk: int = 3
    sentence_overlap: int = 1
    max_chunk_chars: int = 2000
    spacy_model: str = "en_core_web_sm"
    entity_aware_chunking: bool = False

    # Semantic strategy
    min_lexical_diversity: float = 0.3
    min_information_density: float = 2.0

    # Fixed-chunking word-boundary heuristic
    word_boundary_factor: float = 0.8

    # Metadata enrichment
    extract_entities: bool = False
    add_source_metadata: bool = True

    def __post_init__(self) -> None:
        valid = [s.value for s in ChunkingStrategy]
        if self.chunking_strategy not in valid:
            raise ValueError(
                "Invalid chunking_strategy: %r. Must be one of: %s"
                % (self.chunking_strategy, valid)
            )
        if self.chunk_size < 50:
            raise ValueError("chunk_size must be >= 50, got %d" % self.chunk_size)
        if self.sentences_per_chunk < 1:
            raise ValueError("sentences_per_chunk must be >= 1, got %d" % self.sentences_per_chunk)
        if self.sentence_overlap >= self.sentences_per_chunk:
            raise ValueError(
                "sentence_overlap (%d) must be < sentences_per_chunk (%d)"
                % (self.sentence_overlap, self.sentences_per_chunk)
            )


def create_ingestion_config(cfg: Dict[str, Any]) -> DocumentIngestionConfig:
    """
    Build DocumentIngestionConfig from a settings.yaml dict ('ingestion:' block).

    This is the canonical way to construct DocumentIngestionConfig in production.
    All parameters are sourced from settings.yaml via the caller, which keeps
    this module decoupled from YAML-loading logic.

    Parameters
    ----------
    cfg : dict
        The full parsed settings.yaml as a dict, or just the 'ingestion:' sub-dict.

    Returns
    -------
    DocumentIngestionConfig
        Fully populated config; falls back to class defaults for any key absent
        from cfg.

    Example
    -------
        from src.logic_layer._settings_loader import _load_settings
        settings = _load_settings()
        config = create_ingestion_config(settings.get("ingestion", {}))
    """
    ingestion = cfg.get("ingestion", cfg)   # accept either full settings or sub-dict
    qf = ingestion.get("quality_filter", {})

    return DocumentIngestionConfig(
        chunking_strategy=ingestion.get("chunking_strategy", "sentence_spacy"),
        chunk_size=ingestion.get("chunk_size", 1024),
        chunk_overlap=ingestion.get("chunk_overlap", 128),
        min_chunk_size=ingestion.get("min_chunk_size", 50),
        sentences_per_chunk=ingestion.get("sentences_per_chunk", 3),
        sentence_overlap=ingestion.get("sentence_overlap", 1),
        max_chunk_chars=ingestion.get("max_chunk_chars", 2000),
        spacy_model=ingestion.get("spacy_model", "en_core_web_sm"),
        entity_aware_chunking=ingestion.get("entity_aware_chunking", False),
        min_lexical_diversity=ingestion.get(
            "min_lexical_diversity",
            qf.get("min_lexical_diversity", 0.3),
        ),
        min_information_density=ingestion.get(
            "min_information_density",
            qf.get("min_information_density", 2.0),
        ),
        word_boundary_factor=ingestion.get("word_boundary_factor", 0.8),
        extract_entities=ingestion.get("extract_entities", False),
        add_source_metadata=ingestion.get("add_source_metadata", True),
    )


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class DocumentIngestionPipeline:
    """
    Central Document Ingestion Pipeline.

    Selects and initialises the chunker specified by config.chunking_strategy,
    then exposes process_text / process_texts / process_documents methods that
    chunk raw text into a normalised list-of-dicts format.

    Entity extraction is handled separately by EntityExtractionPipeline
    (entity_extraction.py); this pipeline does not perform NER or RE.
    """

    def __init__(self, config: Optional[DocumentIngestionConfig] = None) -> None:
        """
        Initialise pipeline with configuration.

        Parameters
        ----------
        config : DocumentIngestionConfig, optional
            Uses DocumentIngestionConfig() defaults if None. Prefer
            create_ingestion_config() to ensure values match settings.yaml.
        """
        self.config = config or DocumentIngestionConfig()
        self.chunker = self._create_chunker()
        logger.info(
            "DocumentIngestionPipeline initialised: strategy=%s",
            self.config.chunking_strategy,
        )

    def _create_chunker(self):
        """Instantiate the appropriate chunker for config.chunking_strategy."""
        strategy = self.config.chunking_strategy

        if strategy == ChunkingStrategy.SENTENCE.value:
            return SentenceChunker(
                sentences_per_chunk=self.config.sentences_per_chunk,
                min_chunk_size=self.config.min_chunk_size,
            )

        if strategy == ChunkingStrategy.SENTENCE_SPACY.value:
            if _CHUNKING_AVAILABLE and SpacySentenceChunker is not None:
                try:
                    return create_sentence_chunker(
                        sentences_per_chunk=self.config.sentences_per_chunk,
                        sentence_overlap=self.config.sentence_overlap,
                        spacy_model=self.config.spacy_model,
                        entity_aware=self.config.entity_aware_chunking,
                        min_chunk_chars=self.config.min_chunk_size,
                        max_chunk_chars=self.config.max_chunk_chars,
                    )
                except (ImportError, OSError, ValueError, RuntimeError) as exc:
                    logger.warning(
                        "FALLBACK ACTIVE: SpaCy sentence chunker init failed (%s). "
                        "Falling back to regex SentenceChunker. "
                        "Ensure SpaCy is installed: pip install spacy && "
                        "python -m spacy download %s",
                        exc, self.config.spacy_model,
                    )
            else:
                logger.warning(
                    "FALLBACK ACTIVE: sentence_spacy requested but SpaCy unavailable. "
                    "Falling back to regex SentenceChunker.",
                )
            return SentenceChunker(
                sentences_per_chunk=self.config.sentences_per_chunk,
                min_chunk_size=self.config.min_chunk_size,
            )

        if strategy == ChunkingStrategy.SEMANTIC.value:
            if _CHUNKING_AVAILABLE and SemanticChunker is not None:
                try:
                    return create_semantic_chunker(
                        chunk_size=self.config.chunk_size,
                        chunk_overlap=self.config.chunk_overlap,
                        min_chunk_size=self.config.min_chunk_size,
                        word_boundary_factor=self.config.word_boundary_factor,
                        min_lexical_diversity=self.config.min_lexical_diversity,
                        min_info_density=self.config.min_information_density,
                    )
                except (ImportError, OSError, ValueError, RuntimeError) as exc:
                    logger.warning(
                        "FALLBACK ACTIVE: semantic chunker init failed (%s). "
                        "Falling back to RecursiveChunker.",
                        exc,
                    )
            else:
                logger.warning(
                    "FALLBACK ACTIVE: semantic strategy requested but chunking "
                    "module unavailable. Falling back to RecursiveChunker.",
                )
            return RecursiveChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                min_chunk_size=self.config.min_chunk_size,
            )

        if strategy == ChunkingStrategy.FIXED.value:
            if _CHUNKING_AVAILABLE and FixedSizeChunker is not None:
                return FixedSizeChunker(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                    min_chunk_size=self.config.min_chunk_size,
                    word_boundary_factor=self.config.word_boundary_factor,
                )
            logger.warning(
                "FALLBACK ACTIVE: fixed strategy requested but chunking module "
                "unavailable. Falling back to SentenceChunker.",
            )
            return SentenceChunker(
                sentences_per_chunk=self.config.sentences_per_chunk,
                min_chunk_size=self.config.min_chunk_size,
            )

        if strategy == ChunkingStrategy.RECURSIVE.value:
            if _CHUNKING_AVAILABLE and RecursiveChunker is not None:
                return RecursiveChunker(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                    min_chunk_size=self.config.min_chunk_size,
                )
            logger.warning(
                "FALLBACK ACTIVE: recursive strategy requested but chunking "
                "module unavailable. Falling back to SentenceChunker.",
            )
            return SentenceChunker(
                sentences_per_chunk=self.config.sentences_per_chunk,
                min_chunk_size=self.config.min_chunk_size,
            )

        raise ValueError("Unknown chunking strategy: %s" % strategy)

    def process_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        source_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Process a single raw text into chunks.

        Parameters
        ----------
        text : str
            Raw text content to chunk.
        metadata : dict, optional
            Base metadata merged into each chunk's metadata dict.
        source_id : str, optional
            Source identifier added to metadata when add_source_metadata is True.

        Returns
        -------
        List[Dict]
            Each dict has keys 'text' (str) and 'metadata' (dict).
        """
        metadata = metadata or {}
        if not text or not text.strip():
            return []

        if source_id and self.config.add_source_metadata:
            metadata = {**metadata, "source_id": source_id}

        return self.chunker.chunk(text, metadata)

    def process_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        source_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Process multiple texts into a flat list of chunks.

        Also assigns a cross-document global_chunk_id to each chunk.

        Parameters
        ----------
        texts : List[str]
        metadatas : List[dict], optional
        source_ids : List[str], optional

        Returns
        -------
        List[Dict]
        """
        if not texts:
            return []

        metadatas = metadatas or [{} for _ in texts]
        source_ids = source_ids or [None] * len(texts)

        all_chunks: List[Dict] = []
        for text, meta, src in zip(texts, metadatas, source_ids):
            for chunk in self.process_text(text, meta, src):
                chunk["metadata"]["global_chunk_id"] = len(all_chunks)
                all_chunks.append(chunk)

        logger.info(
            "Processed %d texts -> %d chunks (strategy: %s)",
            len(texts), len(all_chunks), self.config.chunking_strategy,
        )
        return all_chunks

    def process_documents(self, documents: List["Document"]) -> List["Document"]:
        """
        Process a list of LangChain Document objects into chunked Documents.

        Parameters
        ----------
        documents : List[Document]

        Returns
        -------
        List[Document]
        """
        if not documents:
            return []

        chunks = self.process_texts(
            [doc.page_content for doc in documents],
            [doc.metadata for doc in documents],
        )
        return [
            Document(page_content=c["text"], metadata=c["metadata"])
            for c in chunks
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Return a snapshot of pipeline configuration and availability flags."""
        return {
            "chunking_strategy": self.config.chunking_strategy,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "sentences_per_chunk": self.config.sentences_per_chunk,
            "chunking_module_available": _CHUNKING_AVAILABLE,
        }


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_data_layer_pipeline(
    strategy: str = "sentence_spacy",
    chunk_size: int = 1024,
    sentences_per_chunk: int = 3,
    **kwargs,
) -> DocumentIngestionPipeline:
    """
    Factory for quick pipeline creation with explicit parameters.

    Prefer create_ingestion_config() when settings.yaml is available.

    Parameters
    ----------
    strategy : str
        Chunking strategy: "sentence", "sentence_spacy", "semantic",
        "fixed", or "recursive".
    chunk_size : int
    sentences_per_chunk : int
    **kwargs
        Additional DocumentIngestionConfig fields.

    Returns
    -------
    DocumentIngestionPipeline
    """
    config = DocumentIngestionConfig(
        chunking_strategy=strategy,
        chunk_size=chunk_size,
        sentences_per_chunk=sentences_per_chunk,
        **kwargs,
    )
    return DocumentIngestionPipeline(config)


# ============================================================================
# SMOKE DEMO AND TEST RUNNER
# ============================================================================

def _main() -> None:
    """Smoke demo and test runner for direct module invocation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    sample = (
        "Albert Einstein was a German-born theoretical physicist. "
        "He developed the theory of relativity. "
        "Einstein received the Nobel Prize in Physics in 1921. "
        "He was awarded for his explanation of the photoelectric effect. "
        "Einstein published more than 300 scientific papers during his career."
    )

    for strat in ("sentence_spacy", "fixed", "recursive", "sentence"):
        pipeline = create_data_layer_pipeline(strategy=strat, min_chunk_size=20)
        chunks = pipeline.process_text(sample, {"source": "smoke_test"})
        logger.info("strategy=%-15s -> %d chunks", strat, len(chunks))
        assert chunks, "Expected at least one chunk for strategy=%s" % strat

    logger.info("Smoke demo passed.")

    test_file = Path(__file__).parent.parent.parent / "test_system" / "test_chunking.py"
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "pytest", str(test_file), "-v", "-k", "ingestion"],
        check=False,
    )
    sys.exit(proc.returncode)


if __name__ == "__main__":
    _main()
