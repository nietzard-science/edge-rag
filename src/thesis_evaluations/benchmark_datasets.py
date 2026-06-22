"""
Multi-dataset benchmark harness for the three-agent edge-RAG pipeline.

This is the dependency root for the entire thesis_evaluations package. It
provides: dataset loaders (HotpotQA, 2WikiMultiHopQA, StrategyQA), a
per-dataset store manager (separate vector + graph store per dataset to
prevent cross-dataset leakage), the pipeline factory, the evaluation
runner with pipeline-vs-LLM failure separation, and the retrieval-lane /
component ablation drivers.

The evaluator distinguishes four mutually exclusive failure modes that
the paper argument depends on:
    pipeline_failed         : gold supporting paragraphs not retrieved
    pipeline_ok_llm_failed  : retrieval ok, LLM returned an error sentinel
    pipeline_ok_llm_wrong   : retrieval ok, LLM produced wrong answer
    pipeline_ok_llm_ok      : retrieval ok, LLM produced correct answer

The headline correctness verdict is Soft-EM (token-F1 >= ANSWER_F1_THRESHOLD,
default 0.6); strict EM is reported alongside for transparency.

Exports
-------
- AVAILABLE_DATASETS, LOADERS                   -- dataset registry
- TestQuestion, Article, EvalResult,
  ConfigResult, AblationResults                 -- record dataclasses
- ABLATION_CONFIGS, COMPONENT_CONFIGS           -- typed ablation matrices
- LaneConfig, ComponentConfig                   -- NamedTuples for above
- ANSWER_F1_THRESHOLD                           -- Soft-EM cut-off
- HotpotQALoader / WikiMultiHopLoader /
  StrategyQALoader                              -- loaders behind DatasetLoader
- StoreManager                                  -- per-dataset paths
- create_langchain_documents(...)               -- chunking entry point
- run_ingestion(...)                            -- vector + graph build
- normalize_answer / compute_exact_match /
  compute_f1                                    -- official-style metrics
- create_pipeline(...)                          -- pipeline factory
- evaluate_dataset(...)                         -- evaluation runner
- cmd_ingest / cmd_evaluate / cmd_ablation /
  cmd_status / cmd_test                         -- CLI sub-commands
- load_config_file(...)                         -- settings.yaml loader
- main()                                        -- CLI entry point

Dependencies / Requirements
---------------------------
- src.pipeline.agent_pipeline.AgentPipeline     -- pipeline
- src.data_layer.{storage, embeddings,
  hybrid_retriever, chunking}                   -- retrieval stack
- src.logic_layer._settings_loader              -- 35-key config validator
- ollama server at config.llm.base_url
- LanceDB (vector store), KuzuDB (graph store)
- HuggingFace `datasets`                        -- dataset loaders
- tqdm                                          -- progress bars (no-op fallback)

CLI sub-commands (see main()): ingest | evaluate | ablation | status | test.

Usage (run as a module from the project root; -X utf8 required on Windows):
    python -X utf8 -m src.thesis_evaluations.benchmark_datasets ingest --dataset hotpotqa --samples 500
    python -X utf8 -m src.thesis_evaluations.benchmark_datasets evaluate --dataset hotpotqa --samples 20 --seed 42
    python -X utf8 -m src.thesis_evaluations.benchmark_datasets ablation --samples 100
    python -X utf8 -m src.thesis_evaluations.benchmark_datasets status

References
----------
Cormack et al. (2009) "Reciprocal Rank Fusion outperforms Condorcet and
    individual rank learning methods." SIGIR.

Last reviewed: 2026-06-01 (audit pass, project version 5.5)
"""

import argparse
import gc
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

# Project-root bootstrap: this script lives at
# `<root>/src/thesis_evaluations/benchmark_datasets.py`. Walk two levels up
# from __file__ to find the project root, then prepend it to sys.path so the
# `from src.pipeline...` / `from src.data_layer...` imports resolve no matter
# how the script is launched (python path/to/file.py, python -m, etc.).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.logic_layer._settings_loader import _load_settings

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Minimal no-op shim so the harness runs without tqdm installed. Covers
    # only the tqdm surface this module uses (iteration, context manager,
    # update, set_postfix); it is NOT a full tqdm replacement. Any new tqdm
    # method used elsewhere must be added here or guarded by TQDM_AVAILABLE.
    class tqdm:
        def __init__(self, iterable=None, **kwargs):
            self._iterable = iterable
        def __iter__(self):
            return iter(self._iterable) if self._iterable is not None else iter([])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            pass
        def set_postfix(self, **kwargs):
            pass

# ============================================================================
# IMPORTS WITH FALLBACK LOGIC
# ============================================================================

# LangChain (Core)
try:
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Mock Document class
    @dataclass
    class Document:
        page_content: str
        metadata: Dict[str, Any] = field(default_factory=dict)

# Chunking (SpacySentenceChunker is primary)
CHUNKING_AVAILABLE = False
SpacySentenceChunker = None

try:
    from src.data_layer.chunking import SpacySentenceChunker, create_sentence_chunker
    CHUNKING_AVAILABLE = True
except ImportError:
    pass

# Ingestion Pipeline (optional fallback — used when primary path unavailable)
INGESTION_PIPELINE_AVAILABLE = False
DocumentIngestionPipeline = None
DocumentIngestionConfig = None

try:
    from src.data_layer.ingestion import DocumentIngestionPipeline, DocumentIngestionConfig
    INGESTION_PIPELINE_AVAILABLE = True
except ImportError:
    pass

# Storage
STORAGE_AVAILABLE = False
try:
    from src.data_layer.storage import HybridStore, StorageConfig
    from src.data_layer.embeddings import BatchedOllamaEmbeddings
    from src.data_layer.hybrid_retriever import (
        HybridRetriever, RetrievalConfig, RetrievalMode, create_hybrid_retriever,
    )
    STORAGE_AVAILABLE = True
except ImportError:
    pass

# Pipeline (new structure)
PIPELINE_AVAILABLE = False
AgentPipeline = None

try:
    from src.pipeline import AgentPipeline, create_full_pipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    pass

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(level: str = "INFO", quiet_modules: bool = True) -> logging.Logger:
    """Setup logging with optional quiet mode for sub-modules."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    if quiet_modules:
        logging.getLogger("src.data_layer").setLevel(logging.WARNING)
        logging.getLogger("src.logic_layer").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


def _log_module_availability() -> None:
    """Log which optional sub-modules imported successfully.

    Called from main() rather than at import time so that importing this
    module (e.g. from another evaluator) has no logging side effects.
    """
    logger.info("Module availability:")
    logger.info("  LangChain:          %s", LANGCHAIN_AVAILABLE)
    logger.info("  Chunking:           %s", CHUNKING_AVAILABLE)
    logger.info("  IngestionPipeline:  %s", INGESTION_PIPELINE_AVAILABLE)
    logger.info("  Storage:            %s", STORAGE_AVAILABLE)
    logger.info("  AgentPipeline:      %s", PIPELINE_AVAILABLE)


# Module-level logger only — no handler/config side effects at import time.
# setup_logging() (root config) and _log_module_availability() run in main().
logger = logging.getLogger(__name__)

# ============================================================================
# MODULE CONSTANTS
# ============================================================================

AVAILABLE_DATASETS = ["hotpotqa", "2wikimultihop", "musique", "strategyqa"]

# Why: the production model lives in config/settings.yaml. This constant is the
# emergency fallback when no settings entry and no --model flag is given so the
# default is documented in code rather than a magic string scattered through
# the CLI handlers.
_DEFAULT_MODEL_FALLBACK = "qwen2:1.5b"

# Why: project-anchored data / cache / results roots so the CLI works
# regardless of the current working directory. _PROJECT_ROOT is defined above.
#
# Container/CI override: the pre-built per-dataset stores (LanceDB/KuzuDB/BM25 +
# questions.json) are mounted at runtime. INDEX_DIR (preferred) or DATASET_DIR
# point _DATA_ROOT at that mount (e.g. /app/data/indices) so no code edit is
# needed to relocate the stores. Both resolve to the same per-dataset root
# because StoreManager.get_paths() derives vector/graph/questions from one base.
def _resolve_data_root() -> Path:
    env_root = os.environ.get("INDEX_DIR") or os.environ.get("DATASET_DIR")
    if env_root:
        return Path(env_root)
    return _PROJECT_ROOT / "data"


_DATA_ROOT = _resolve_data_root()
_CACHE_ROOT = _PROJECT_ROOT / "cache"
_EVAL_RESULTS_ROOT = _PROJECT_ROOT / "evaluation_results"
# Lane-ablation outputs (per-run JSONL + ablation_<ts>.json) used to land
# loose at the eval-root top level. They now go under runs/lane_ablation/ so
# the top level stays scannable and the aggregator's run-dir discovery is not
# polluted by stray files.
_LANE_ABLATION_DIR = _EVAL_RESULTS_ROOT / "runs" / "lane_ablation"

# Why: per-source RRF weights default to 1.0 (vanilla equal-weight RRF) so a
# no-flag run uses the production retrieval contract. settings.yaml overrides
# when present.
_DEFAULT_VECTOR_WEIGHT = 1.0
_DEFAULT_GRAPH_WEIGHT = 1.0
_DEFAULT_BM25_WEIGHT = 1.0

# Why: nomic-embed-text emits 768-dimensional vectors. Hard-coded by the
# embedding model choice; surfaced as a named constant so a model swap touches
# one line, not three call sites.
_NOMIC_EMBED_DIM = 768

# Why: ingestion writes documents in batches to bound peak memory; 100 keeps
# one batch's embeddings (100 * 768 * 4B ~= 0.3 MiB) well under RAM pressure
# on the edge target.
_INGESTION_BATCH_SIZE = 100
_EMBEDDING_BATCH_SIZE = 32  # Ollama's per-request batch (network side)

# Why: chunking defaults match the production ingestion contract.
_DEFAULT_CHUNK_SENTENCES = 3
_DEFAULT_SENTENCE_OVERLAP = 1
_MIN_CHUNK_CHARS = 50

# Why: Verifier max-docs fallback used when the pipeline does not expose its
# value (offline tests, mocked verifier). 5 mirrors the production setting.
_VERIFIER_MAX_DOCS_FALLBACK = 5

# Why: vector-store similarity-threshold fallback when neither settings.yaml
# nor a passed config dict supplies one. 0.3 matches the documented production
# default (vector_store.similarity_threshold in config/settings.yaml).
_VECTOR_SIMILARITY_THRESHOLD_FALLBACK = 0.3

# Why: text->title cache uses a fixed-length key so chunk truncation in the
# retrieval pipeline does not cause cache misses. 200 chars covers the longest
# article-title sentence we have observed without producing degenerate keys.
_TEXT_KEY_LEN = 200

# Why: 5-sample window for the yes/no rule in compute_exact_match. Captures
# "Yes, because ..." / "No -- the answer is ..." patterns but rejects yes/no
# tokens buried in long explanatory answers.
_YESNO_PREFIX_TOKEN_COUNT = 5

# Why: error-message truncation when logging a per-question failure -- long
# tracebacks would flood the eval log on a systematic failure.
_LOG_ERROR_MSG_TRUNC = 80

# Why: keep articles_info.json bounded for spot-checking; the full title list
# is recoverable from the ingested store.
_ARTICLES_INFO_TITLES_CAP = 100

# Why: when --seed is not given, auto-generate a seed in this range. Logged so
# the run is reproducible. 5-digit cap keeps the log line readable.
_AUTO_SEED_RANGE = 100_000

# Why: hoist regexes used by normalize_answer to module scope -- they are on
# the metric hot path (called per question per metric per row).
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_RE = re.compile(r"[^\w\s]")


# ----------------------------------------------------------------------------
# Ablation matrices
# ----------------------------------------------------------------------------

class LaneConfig(NamedTuple):
    """One retrieval-lane ablation cell.

    BM25 is a separate sparse-lexical lane (rank_bm25), not a vector approach,
    so it gets its own ablation dimension. Architectural constraint:
    HybridRetriever fires BM25 only in VECTOR / HYBRID mode (mode derived from
    vector/graph weights -- see create_pipeline), so a clean "BM25 alone" row
    is not separable. The rows below expose each lane's MARGINAL contribution:
        BM25  marginal = vector_bm25 - vector_only
        graph marginal = hybrid_all  - vector_bm25
        dense vs graph = vector_only vs graph_only
    """
    name: str
    vector_weight: float
    graph_weight: float
    bm25_weight: float
    enable_bm25: bool


ABLATION_CONFIGS: Tuple[LaneConfig, ...] = (
    LaneConfig("vector_only", 1.0, 0.0, 0.0, False),  # dense ANN alone
    LaneConfig("vector_bm25", 1.0, 0.0, 1.0, True),   # dense + sparse
    LaneConfig("graph_only",  0.0, 1.0, 0.0, False),  # entity-path alone
    LaneConfig("hybrid_all",  1.0, 1.0, 1.0, True),   # all three lanes
)


class ComponentConfig(NamedTuple):
    """One component-ablation cell (planner / verifier / iterations).

    Used with --component-ablation; baseline retrieval is equal-weight hybrid.
    """
    name: str
    enable_planner: bool
    enable_verifier: bool
    max_iterations: int


COMPONENT_CONFIGS: Tuple[ComponentConfig, ...] = (
    ComponentConfig("full",        True,  True,  1),
    ComponentConfig("no_planner",  False, True,  1),
    ComponentConfig("no_verifier", True,  False, 1),
    ComponentConfig("iter_2",      True,  True,  2),
    ComponentConfig("iter_3",      True,  True,  3),
)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TestQuestion:
    """Universal test question format."""
    id: str
    question: str
    answer: str
    dataset: str
    question_type: str = "unknown"
    level: str = "unknown"
    supporting_facts: List = field(default_factory=list)

@dataclass
class Article:
    """Universal article/document format."""
    id: str
    title: str
    text: str
    sentences: List[str]
    dataset: str

@dataclass
class EvalResult:
    """Single question evaluation result.

    Separates pipeline correctness (retrieval) from model correctness (LLM
    answer) so the paper can argue: "of N questions where retrieval found
    all gold paragraphs, X% were also answered correctly — the remaining
    gap is model capacity, not pipeline architecture."
    """
    question_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    exact_match: bool
    f1_score: float
    retrieval_count: int
    time_ms: float
    dataset: str
    question_type: str

    # Soft correctness verdict: token-F1 >= ANSWER_F1_THRESHOLD. The fair
    # headline metric — strict exact_match under-credits formal-vs-common
    # name variants (see ANSWER_F1_THRESHOLD docstring).
    answer_correct: bool = False

    # Retrieval quality (independent of LLM)
    gold_titles: List[str] = field(default_factory=list)
    retrieved_titles: List[str] = field(default_factory=list)
    retrieval_recall: float = 0.0
    retrieval_precision: float = 0.0
    sf_f1: float = 0.0
    all_gold_retrieved: bool = False
    # Delivery check: are ALL gold paragraphs still present in the chunks the
    # Verifier actually forwards to the LLM (top max_docs of filtered_context)?
    # all_gold_retrieved counts the Navigator output (<=max_context_chunks);
    # this counts the post-cap LLM-visible window. The gap between the two is
    # "delivery loss" -- gold retrieved by the pipeline but cut by the
    # max_docs cap before the model ever sees it.
    gold_in_final_context: bool = False

    # Failure-mode separation
    llm_error: bool = False
    llm_error_type: str = ""
    pipeline_succeeded_llm_failed: bool = False

    # Planner diagnostics. matched_pattern identifies which Planner pattern
    # produced the plan; classifier_preempt records whether a classifier
    # short-circuit fired. Both surface in the per-question JSONL so per-
    # pattern hit-rates and SF-F1 deltas can be computed without parsing
    # debug logs. None when the field is unset (e.g. mocked tests).
    planner_query_type: str = ""
    hop_count: int = 0
    n_entities: int = 0
    matched_pattern: Optional[str] = None
    classifier_preempt: Optional[str] = None

    # Verifier diagnostics. pre_validation_status surfaces the
    # pre_validation.status enum value (e.g. PASSED / INSUFFICIENT_EVIDENCE);
    # pre_validation_chunks_in/out track how many context chunks the three
    # pre-generation filters dropped, so the diagnostic harness can
    # distinguish "pre-val fired but inert" from "pre-val dropped a gold
    # chunk". bridge_retry_fired records whether the bridge-entity
    # exclusion retry fired and changed the final answer.
    verifier_iterations: int = 0
    all_verified: bool = False
    confidence: str = ""
    pre_validation_status: Optional[str] = None
    pre_validation_chunks_in: int = 0
    pre_validation_chunks_out: int = 0
    bridge_retry_fired: bool = False

@dataclass
class ConfigResult:
    """Results for one configuration on one dataset."""
    dataset: str
    config_name: str
    vector_weight: float
    graph_weight: float
    n_questions: int
    exact_match: float
    f1_score: float
    avg_time_ms: float
    coverage: float
    # Soft-EM: fraction with token-F1 >= ANSWER_F1_THRESHOLD (fair headline
    # answer-correctness metric; strict exact_match kept above for transparency).
    soft_em: float = 0.0
    by_type: Dict[str, Dict] = field(default_factory=dict)

    # Retrieval-level aggregates (pipeline correctness)
    avg_sf_f1: float = 0.0
    sf_recall_rate: float = 0.0          # fraction of Qs where all gold retrieved
    # fraction of Qs where all gold survives into the LLM-visible window
    final_context_recall_rate: float = 0.0
    # fraction of Qs where gold WAS retrieved but did NOT reach the LLM (cut by
    # the max_docs cap) — isolates "delivery loss" from "retrieval loss".
    delivery_loss_rate: float = 0.0
    retrieval_only_em: float = 0.0       # EM among Qs where all gold retrieved
    llm_error_rate: float = 0.0
    pipeline_failed_rate: float = 0.0    # gold not fully retrieved
    pipeline_ok_llm_failed_rate: float = 0.0
    pipeline_ok_llm_wrong_rate: float = 0.0
    pipeline_ok_llm_ok_rate: float = 0.0

@dataclass
class AblationResults:
    """Complete ablation study results."""
    timestamp: str
    datasets: List[str]
    configs: List[str]
    results: Dict[str, List[ConfigResult]]
    # SHA-256 (short) of the config that produced this run — frozen-config
    # provenance (P0-T1). Optional so older callers/tests still construct
    # AblationResults without it.
    config_sha256: Optional[str] = None

    def to_dict(self) -> Dict:
        output = {
            "timestamp": self.timestamp,
            "datasets": self.datasets,
            "configs": self.configs,
            "config_sha256": self.config_sha256,
            "results": {}
        }
        for ds, results in self.results.items():
            output["results"][ds] = [asdict(r) for r in results]
        return output

# ============================================================================
# DATASET LOADERS
# ============================================================================

class DatasetLoader(ABC):
    """Abstract base for dataset loaders."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def load(self, n_samples: int = None) -> Tuple[List[Article], List[TestQuestion]]:
        pass

class HotpotQALoader(DatasetLoader):
    """HotpotQA: Multi-hop reasoning over Wikipedia."""
    
    @property
    def name(self) -> str:
        return "hotpotqa"
    
    def load(self, n_samples: int = None) -> Tuple[List[Article], List[TestQuestion]]:
        from datasets import load_dataset
        
        logger.info("Loading HotpotQA from HuggingFace...")
        ds = load_dataset("hotpot_qa", "distractor", split="validation")
        
        if n_samples:
            ds = ds.select(range(min(n_samples, len(ds))))
        
        articles_dict = {}
        questions = []
        
        for idx, item in enumerate(ds):
            if idx % 100 == 0 and idx > 0:
                logger.info("  Processing %d/%d...", idx, len(ds))

            q = TestQuestion(
                id=item["id"],
                question=item["question"],
                answer=item["answer"],
                dataset="hotpotqa",
                question_type=item["type"],
                level=item["level"],
                supporting_facts=list(zip(
                    item["supporting_facts"]["title"],
                    item["supporting_facts"]["sent_id"]
                )),
            )
            questions.append(q)
            
            for title, sentences in zip(
                item["context"]["title"],
                item["context"]["sentences"]
            ):
                if title not in articles_dict:
                    articles_dict[title] = Article(
                        id=f"hotpotqa_{len(articles_dict)}",
                        title=title,
                        text=" ".join(sentences),
                        sentences=list(sentences),
                        dataset="hotpotqa",
                    )
        
        articles = list(articles_dict.values())
        logger.info("  HotpotQA: %d articles, %d questions",
                    len(articles), len(questions))

        return articles, questions

class WikiMultiHopLoader(DatasetLoader):
    """2WikiMultiHopQA: Requires 2 Wikipedia articles."""
    
    @property
    def name(self) -> str:
        return "2wikimultihop"
    
    def load(self, n_samples: int = None) -> Tuple[List[Article], List[TestQuestion]]:
        from datasets import load_dataset
        
        logger.info("Loading 2WikiMultiHopQA from HuggingFace...")
        
        try:
            ds = load_dataset("framolfese/2WikiMultihopQA", split="validation")
        except Exception as e:
            logger.error("2WikiMultiHopQA not available: %s", e)
            return [], []
        
        if n_samples:
            ds = ds.select(range(min(n_samples, len(ds))))
        
        articles_dict = {}
        questions = []
        
        for idx, item in enumerate(ds):
            if idx % 100 == 0 and idx > 0:
                logger.info("  Processing %d/%d...", idx, len(ds))

            q = TestQuestion(
                id=item.get("id", f"2wiki_{idx}"),
                question=item["question"],
                answer=item["answer"],
                dataset="2wikimultihop",
                question_type=item.get("type", "unknown"),
                supporting_facts=list(zip(
                    item.get("supporting_facts", {}).get("title", []),
                    item.get("supporting_facts", {}).get("sent_id", [])
                )) if item.get("supporting_facts") else [],
            )
            questions.append(q)
            
            context = item.get("context", {})
            
            if isinstance(context, dict):
                titles = context.get("title", [])
                sentences_list = context.get("sentences", [])
                
                for title, sentences in zip(titles, sentences_list):
                    if title and title not in articles_dict:
                        if isinstance(sentences, list):
                            text = " ".join(str(s) for s in sentences)
                            sent_list = [str(s) for s in sentences]
                        else:
                            text = str(sentences)
                            sent_list = [text]
                        
                        articles_dict[title] = Article(
                            id=f"2wiki_{len(articles_dict)}",
                            title=title,
                            text=text,
                            sentences=sent_list,
                            dataset="2wikimultihop",
                        )
        
        articles = list(articles_dict.values())
        logger.info("  2WikiMultiHop: %d articles, %d questions",
                    len(articles), len(questions))

        return articles, questions

class MuSiQueLoader(DatasetLoader):
    """MuSiQue: 2-4 hop multi-hop QA over a distractor paragraph pool.

    Trivedi et al. (2022), *MuSiQue: Multihop Questions via Single-hop Question
    Composition* (TACL). Harder and more strictly retrieval-bound than HotpotQA
    / 2Wiki — chosen as the third retrieval dataset to turn the two-point
    cross-dataset inversion (§8) into a three-point reasoning-type trend.

    Field mapping (HF schema → universal format):
      question                       → question
      answer (+ answer_aliases)      → answer (aliases kept for lenient scoring)
      paragraphs[].{title, paragraph_text}  → corpus pool (the distractor setting,
                                              analogous to 2Wiki context)
      supporting-fact titles         → titles of paragraphs referenced by
                                       question_decomposition[].paragraph_support_idx.
                                       MuSiQue marks support per decomposition
                                       step (an index into paragraphs), not via a
                                       per-paragraph is_supporting flag, so we
                                       resolve the indices to titles here.
    """

    @property
    def name(self) -> str:
        return "musique"

    def load(self, n_samples: int = None) -> Tuple[List[Article], List[TestQuestion]]:
        from datasets import load_dataset

        logger.info("Loading MuSiQue (musique_ans) from HuggingFace...")
        ds = None
        for repo in ("dgslibisey/MuSiQue", "voidful/musique"):
            try:
                ds = load_dataset(repo, split="validation")
                logger.info("  loaded from %s", repo)
                break
            except Exception as e:  # noqa: BLE001 — try the next mirror
                logger.warning("  %s unavailable: %s", repo, e)
        if ds is None:
            logger.error("MuSiQue not available from any known mirror.")
            return [], []

        if n_samples:
            ds = ds.select(range(min(n_samples, len(ds))))

        articles_dict: Dict[str, Article] = {}
        questions: List[TestQuestion] = []

        for idx, item in enumerate(ds):
            if idx % 100 == 0 and idx > 0:
                logger.info("  Processing %d/%d...", idx, len(ds))

            paragraphs = item.get("paragraphs", []) or []

            # Supporting-fact titles: resolve paragraph_support_idx (per
            # decomposition step) → paragraph title. Some steps have a null
            # support idx (the composed final step); skip those.
            support_titles: List[str] = []
            for step in item.get("question_decomposition", []) or []:
                sidx = step.get("paragraph_support_idx")
                if sidx is None:
                    continue
                if 0 <= sidx < len(paragraphs):
                    title = paragraphs[sidx].get("title")
                    if title:
                        support_titles.append(title)
            # Fall back to any paragraph flagged is_supporting, if the mirror
            # provides that field instead of decomposition indices.
            if not support_titles:
                for p in paragraphs:
                    if p.get("is_supporting") and p.get("title"):
                        support_titles.append(p["title"])

            # sent_id is 0 for every supporting title (MuSiQue support is
            # paragraph-level, not sentence-level) — keeps the (title, sent_id)
            # shape the SF-recall scorer expects.
            q = TestQuestion(
                id=str(item.get("id", f"musique_{idx}")),
                question=item["question"],
                answer=item["answer"],
                dataset="musique",
                question_type=item.get("question_type", "multi_hop"),
                supporting_facts=[(t, 0) for t in support_titles],
            )
            # Attach aliases for the lenient scorer if the universal question
            # format does not carry them (stored as an attribute, read opportu-
            # nistically by the scorer; harmless if unused).
            aliases = item.get("answer_aliases") or []
            if aliases:
                setattr(q, "answer_aliases", list(aliases))
            questions.append(q)

            for p in paragraphs:
                title = p.get("title")
                text = p.get("paragraph_text", "") or ""
                if title and title not in articles_dict and text.strip():
                    articles_dict[title] = Article(
                        id=f"musique_{len(articles_dict)}",
                        title=title,
                        text=text,
                        sentences=[text],
                        dataset="musique",
                    )

        articles = list(articles_dict.values())
        logger.info("  MuSiQue: %d articles, %d questions",
                    len(articles), len(questions))
        return articles, questions


class StrategyQALoader(DatasetLoader):
    """StrategyQA: Yes/No questions with implicit reasoning."""
    
    @property
    def name(self) -> str:
        return "strategyqa"
    
    def load(self, n_samples: int = None) -> Tuple[List[Article], List[TestQuestion]]:
        from datasets import load_dataset
        
        logger.info("Loading StrategyQA from HuggingFace...")
        
        ds = None
        try:
            ds = load_dataset("ChilleD/StrategyQA", split="train")
        except Exception:
            try:
                ds = load_dataset("wics/strategy-qa", "strategyQA", split="test")
            except Exception as e:
                logger.error("StrategyQA not available: %s", e)
                return [], []
        
        if n_samples:
            ds = ds.select(range(min(n_samples, len(ds))))
        
        articles = []
        questions = []
        
        for idx, item in enumerate(ds):
            raw_answer = item.get("answer", item.get("label", False))
            if isinstance(raw_answer, bool):
                answer = "yes" if raw_answer else "no"
            elif isinstance(raw_answer, int):
                answer = "yes" if raw_answer == 1 else "no"
            else:
                answer = "yes" if raw_answer else "no"
            
            q = TestQuestion(
                id=f"strategyqa_{idx}",
                question=item["question"],
                answer=answer,
                dataset="strategyqa",
                question_type="boolean",
            )
            questions.append(q)
            
            facts = None
            for field_name in ["facts", "evidence", "paragraphs", "decomposition"]:
                if field_name in item and item[field_name]:
                    facts = item[field_name]
                    break
            
            if facts and isinstance(facts, list):
                for i, fact in enumerate(facts):
                    if isinstance(fact, str) and len(fact.strip()) > 10:
                        articles.append(Article(
                            id=f"strategyqa_fact_{idx}_{i}",
                            title=f"Fact_{idx}_{i}",
                            text=fact.strip(),
                            sentences=[fact.strip()],
                            dataset="strategyqa",
                        ))
        
        logger.info("  StrategyQA: %d facts, %d questions",
                    len(articles), len(questions))

        if len(articles) == 0:
            logger.warning("  No facts found - StrategyQA requires external knowledge!")

        return articles, questions

LOADERS: Dict[str, DatasetLoader] = {
    "hotpotqa": HotpotQALoader(),
    "2wikimultihop": WikiMultiHopLoader(),
    "musique": MuSiQueLoader(),
    "strategyqa": StrategyQALoader(),
}

# ============================================================================
# STORE MANAGER
# ============================================================================

class StoreManager:
    """Manages separate vector stores and knowledge graphs per dataset."""

    def __init__(self, base_path: Optional[Path] = None):
        # Why: anchor on _DATA_ROOT so the CLI works regardless of cwd.
        self.base_path = base_path if base_path is not None else _DATA_ROOT
    
    def get_paths(self, dataset: str) -> Dict[str, Path]:
        """Return all storage paths for a dataset."""
        ds_path = self.base_path / dataset
        return {
            "root": ds_path,
            "vector": ds_path / "vector",
            "graph": ds_path / "graph",
            "questions": ds_path / "questions.json",
            "articles_info": ds_path / "articles_info.json",
        }
    
    def ensure_dirs(self, dataset: str) -> None:
        """Create directory structure for a dataset."""
        paths = self.get_paths(dataset)
        paths["root"].mkdir(parents=True, exist_ok=True)
    
    def clear_dataset(self, dataset: str, chunks_only: bool = False) -> None:
        """
        Clear data for a dataset.

        Two modes:
          chunks_only=True:   Delete ONLY chunks_export.json (Phase 1 output).
                              Leaves vector/graph/extraction_results.json
                              intact. Use when re-running Phase 1 while the
                              KuzuDB graph may be locked by another process
                              (Windows holds Kuzu file locks until the OS
                              releases them).

          chunks_only=False:  Full reset. Rescues every .json file in the
                              tree (chunks_export, questions, articles_info,
                              extraction_results — all expensive to
                              regenerate, especially the Colab output),
                              then deletes the rest. PermissionError on
                              individual files (typically Kuzu .lock files
                              still held by the OS) is logged as a warning
                              and skipped — partial cleanup is acceptable
                              because re-ingestion will MERGE-overwrite.
        """
        paths = self.get_paths(dataset)
        root = paths["root"]
        if not root.exists():
            return

        # Mode A: chunks-only -> remove just chunks_export.json
        if chunks_only:
            chunks_file = root / "chunks_export.json"
            if chunks_file.exists():
                try:
                    chunks_file.unlink()
                    logger.info("Cleared: %s", chunks_file)
                except PermissionError as exc:
                    logger.warning("Could not remove %s: %s", chunks_file, exc)
            return

        # Mode B: full reset with .json rescue + lock-tolerant rmtree.
        rescued: Dict[Path, bytes] = {}
        for json_file in root.rglob("*.json"):
            try:
                rescued[json_file.relative_to(root)] = json_file.read_bytes()
                logger.info("  Protected before --clear: %s", json_file)
            except OSError as exc:
                logger.warning("  Could not read %s: %s", json_file, exc)

        # Best-effort rmtree. PermissionError on individual files (typical
        # cause: a still-held Kuzu .lock file) is logged but does not abort.
        def _on_error(func, path, exc_info):
            logger.warning("  Could not remove %s: %s", path, exc_info[1])

        try:
            shutil.rmtree(root, onerror=_on_error)
            logger.info("Cleared (best-effort): %s", root)
        except Exception as exc:
            logger.warning("Partial clear of %s: %s", root, exc)

        # Restore the rescued JSON files
        for relpath, data in rescued.items():
            target = root / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            logger.info("  Restored: %s", target)

    def save_questions(self, questions: List[TestQuestion], dataset: str) -> None:
        """Save test questions."""
        self.ensure_dirs(dataset)
        paths = self.get_paths(dataset)

        with open(paths["questions"], "w", encoding="utf-8") as f:
            json.dump([asdict(q) for q in questions], f, indent=2, ensure_ascii=False)

        logger.info("Saved %d questions to %s", len(questions), paths["questions"])

    def load_questions(self, dataset: str) -> List[TestQuestion]:
        """Load test questions."""
        paths = self.get_paths(dataset)

        if not paths["questions"].exists():
            logger.error("Questions not found: %s", paths["questions"])
            return []
        
        with open(paths["questions"], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return [TestQuestion(**q) for q in data]
    
    def save_articles_info(self, articles: List[Article], dataset: str) -> None:
        """Save article metadata (count + bounded title sample for spot checks)."""
        self.ensure_dirs(dataset)
        paths = self.get_paths(dataset)

        info = {
            "count": len(articles),
            "dataset": dataset,
            "titles": [a.title for a in articles[:_ARTICLES_INFO_TITLES_CAP]],
        }
        
        with open(paths["articles_info"], 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)
    
    def dataset_exists(self, dataset: str) -> bool:
        """Check if dataset is ingested."""
        paths = self.get_paths(dataset)
        return paths["vector"].exists() and paths["questions"].exists()
    
    def get_status(self) -> Dict[str, bool]:
        """Get ingestion status for all datasets."""
        return {ds: self.dataset_exists(ds) for ds in AVAILABLE_DATASETS}

# ============================================================================
# DOCUMENT CREATION & INGESTION
# ============================================================================

def create_langchain_documents(
    articles: List[Article],
    chunk_sentences: int = 3,
    sentence_overlap: int = 1,
    apply_coreference: bool = True,
) -> List[Document]:
    """
    Convert articles to LangChain Documents using SpacySentenceChunker.

    Pipeline:
      1. (optional) Coreference resolution per-article — replaces pronouns
         with their antecedent noun phrases so GLiNER can later capture the
         underlying named entity (Phase 2) and the graph captures the right
         cooccurrence pairs (Phase 3).
      2. SpacySentenceChunker (3-sentence sliding window) — primary path.
      3. Simple sentence-grouping fallback if SpaCy is unavailable.

    Coreference is silently skipped when `coreferee` or `en_core_web_md/lg`
    is not installed — the pipeline keeps working with reduced graph density.
    Pass apply_coreference=False to disable explicitly (e.g. for ablation).
    """

    # Lazy import — keeps benchmark_datasets.py functional when the data
    # layer is unavailable for non-ingestion subcommands.
    coref_resolver = None
    if apply_coreference:
        try:
            from src.data_layer.coreference import resolve_coreferences, is_available
            if is_available():
                coref_resolver = resolve_coreferences
                logger.info("Coreference resolution: ENABLED")
            else:
                logger.info("Coreference resolution: SKIPPED (coreferee or md/lg model missing)")
        except ImportError:
            logger.info("Coreference resolution: SKIPPED (module not importable)")

    # Primary path: SpacySentenceChunker.
    if CHUNKING_AVAILABLE and SpacySentenceChunker is not None:
        logger.info("Using SpacySentenceChunker (3-sentence window)")

        try:
            chunker = create_sentence_chunker(
                sentences_per_chunk=chunk_sentences,
                sentence_overlap=sentence_overlap,
                min_chunk_chars=50,
            )

            all_documents = []
            chunk_id = 0

            for article in articles:
                # Apply coreference resolution if available — the chunker
                # then sees pronoun-resolved text and produces chunks where
                # GLiNER can re-identify the named entity behind every "He".
                article_text = article.text
                if coref_resolver is not None:
                    article_text = coref_resolver(article_text)
                chunk_results = chunker.chunk_text(
                    article_text,
                    source_doc=article.title
                )
                
                # Convert to LangChain Documents
                for chunk in chunk_results:
                    doc = Document(
                        page_content=chunk.text,
                        metadata={
                            "chunk_id": chunk_id,
                            "source_file": f"{article.dataset}_{article.title}",
                            "article_title": article.title,
                            "dataset": article.dataset,
                            "sentence_count": chunk.sentence_count,
                            "position": chunk.position,
                        }
                    )
                    all_documents.append(doc)
                    chunk_id += 1
            
            logger.info("Created %d chunks using SpaCy chunker", len(all_documents))
            return all_documents

        except Exception as e:
            logger.warning("SpaCy chunker failed: %s - using fallback", e)

    # Fallback path (always available).
    logger.info("Using fallback sentence grouping")
    return _create_documents_fallback(articles, chunk_sentences)

def _create_documents_fallback(articles: List[Article], chunk_sentences: int = 3) -> List[Document]:
    """Fallback chunking without SpaCy."""
    documents = []
    chunk_id = 0
    
    for article in articles:
        sentences = article.sentences
        
        for i in range(0, len(sentences), chunk_sentences):
            chunk_sents = sentences[i:i + chunk_sentences]
            chunk_text = " ".join(chunk_sents)
            
            if len(chunk_text.strip()) < 50:
                continue
            
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "chunk_id": chunk_id,
                    "source_file": f"{article.dataset}_{article.title}",
                    "article_title": article.title,
                    "dataset": article.dataset,
                    "sentence_start": i,
                    "sentence_end": i + len(chunk_sents),
                }
            )
            documents.append(doc)
            chunk_id += 1
    
    return documents

def run_ingestion(
    documents: List[Document],
    vector_path: Path,
    graph_path: Path,
    config: Dict,
    dataset_name: str,
) -> None:
    """Ingest documents into vector store and knowledge graph."""
    
    if not STORAGE_AVAILABLE:
        logger.error("Storage module not available!")
        logger.error("Install: pip install lancedb kuzu")
        return
    
    logger.info("Ingesting %d documents for %s...", len(documents), dataset_name)

    vector_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize embeddings
    embedding_config = config.get("embeddings", {})
    perf_config = config.get("performance", {})

    cache_path = _CACHE_ROOT / f"{dataset_name}_embeddings.db"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = BatchedOllamaEmbeddings(
        model_name=embedding_config.get("model_name", "nomic-embed-text"),
        base_url=embedding_config.get("base_url", "http://localhost:11434"),
        batch_size=perf_config.get("batch_size", _EMBEDDING_BATCH_SIZE),
        cache_path=cache_path,
        device=perf_config.get("device", "cpu"),
    )

    # Initialize storage
    vector_config = config.get("vector_store", {})

    storage_config = StorageConfig(
        vector_db_path=vector_path,
        graph_db_path=graph_path,
        embedding_dim=embedding_config.get("embedding_dim", _NOMIC_EMBED_DIM),
        similarity_threshold=vector_config.get("similarity_threshold", _VECTOR_SIMILARITY_THRESHOLD_FALLBACK),
        normalize_embeddings=vector_config.get("normalize_embeddings", True),
        distance_metric=vector_config.get("distance_metric", "cosine"),
        # Fallbacks mirror the StorageConfig dataclass defaults.
        overfetch_factor=vector_config.get("overfetch_factor", 3),
        graph_text_max_chars=vector_config.get("graph_text_max_chars", 500),
        # GLiNER + REBEL -> Entity nodes in KuzuDB.
        enable_entity_extraction=True,
    )

    hybrid_store = HybridStore(config=storage_config, embeddings=embeddings)

    # Ingest in batches
    start_time = time.time()
    batch_size = perf_config.get("ingestion_batch_size", _INGESTION_BATCH_SIZE)
    n_batches = (len(documents) + batch_size - 1) // batch_size

    with tqdm(total=len(documents), desc=f"Ingesting {dataset_name}", unit="doc",
              disable=not TQDM_AVAILABLE) as pbar:
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_start = time.time()
            hybrid_store.add_documents(batch)
            batch_elapsed = time.time() - batch_start
            pbar.update(len(batch))
            done = i + len(batch)
            pct = 100 * done / len(documents)
            elapsed_total = time.time() - start_time
            remaining = (elapsed_total / done) * (len(documents) - done) if done > 0 else 0
            logger.info(
                "  [%5.1f%%] Batch %d/%d | %.0fs/batch | remaining ~%.1fh",
                pct, i // batch_size + 1, n_batches, batch_elapsed,
                remaining / 3600,
            )
            pbar.set_postfix(batch=f"{i // batch_size + 1}/{n_batches}")
    
    hybrid_store.save()
    
    elapsed = time.time() - start_time
    logger.info("  Ingestion complete: %.1fs", elapsed)

# ============================================================================
# EVALUATION METRICS
# ============================================================================

# Token-F1 threshold above which an answer counts as "correct" for the
# headline answer-quality verdict (Soft-EM). Strict Exact Match is too brittle
# for frequent formal-vs-common name variants — a gold answer giving a full
# legal name vs a predicted common short form is EM=0 but F1=0.8 and is
# unambiguously correct. F1 (the official HotpotQA answer metric) is reported
# alongside strict EM; a chunk that overlaps the gold by >= this fraction of
# tokens is treated as a correct answer. 0.6 admits partial/middle-name and
# nickname variants while excluding answers that share only one token.
ANSWER_F1_THRESHOLD: float = 0.6


def normalize_answer(text: str) -> str:
    """Lower-case, strip leading articles and punctuation, collapse whitespace.

    Mirrors the official HotpotQA normalisation. Regexes are module constants
    (`_ARTICLES_RE`, `_PUNCT_RE`) because this function is on the metric hot
    path (called twice per question per ablation row).
    """
    text = text.lower().strip()
    text = _ARTICLES_RE.sub(" ", text)
    text = _PUNCT_RE.sub("", text)
    return " ".join(text.split())

@lru_cache(maxsize=4096)
def _gold_phrase_pattern(gold_norm: str) -> "re.Pattern[str]":
    """Compile and cache the word-boundary-anchored multi-word-gold pattern."""
    return re.compile(r"\b" + re.escape(gold_norm) + r"\b")


def compute_exact_match(prediction: str, gold: str) -> bool:
    """Compute exact match (EM) following the official HotpotQA metric.

    Rules (in order):
    1. Exact string match after normalisation.
    2. For multi-word gold answers (>=2 tokens): gold must appear as a
       contiguous word-boundary-anchored substring of the prediction
       (handles "the <LANDMARK>" inside "<LANDMARK>, <CITY>").
    3. For yes/no: the gold token must appear as a standalone word in
       the first `_YESNO_PREFIX_TOKEN_COUNT` tokens of the prediction --
       NOT as a substring of a longer word (prevents "no" matching
       inside "I don't know").

    Deliberately NOT used: bare `gold in pred` substring check, which
    causes "no" to match "I don't know." as a false positive.
    """
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)

    if not gold_norm:
        return False

    # Rule 1: exact match
    if pred_norm == gold_norm:
        return True

    # Rule 2: multi-word gold as contiguous phrase (word-boundary anchored).
    # Pattern is cached so repeated questions with the same gold do not
    # recompile.
    gold_tokens = gold_norm.split()
    if len(gold_tokens) >= 2:
        if _gold_phrase_pattern(gold_norm).search(pred_norm):
            return True

    # Rule 3: yes/no -- standalone word match in the prefix window.
    if gold_norm in ("yes", "no"):
        pred_words = pred_norm.split()[:_YESNO_PREFIX_TOKEN_COUNT]
        if gold_norm in pred_words:
            return True

    return False

def compute_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    
    num_common = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)

# ============================================================================
# PIPELINE INITIALIZATION
# ============================================================================

def create_pipeline(
    dataset: str,
    config: Dict,
    store_manager: StoreManager,
    vector_weight: float = 1.0,
    graph_weight: float = 1.0,
    bm25_weight: float = 1.0,
    model_name: str = None,
    enable_planner: bool = True,
    enable_verifier: bool = True,
    max_iterations: int = 1,
    enable_pre_validation: bool = True,
    enable_bm25: Optional[bool] = None,
    cross_source_boost: Optional[float] = None,
):
    """Build an AgentPipeline bound to one dataset's stores.

    Parameters thread the per-source RRF weights (vector / graph / bm25),
    the BM25 toggle, the cross-source RRF boost β (cross_source_boost; None =
    use settings.yaml), and the component-ablation flags (planner / verifier /
    pre-validation / max_iterations) through to the pipeline config. The
    `enable_pre_validation=False` path forces the three pre-generation
    filters off so the verifier ablation produces a distinct configuration
    from the planner ablation; otherwise both rows would emit identical
    settings (zero marginal delta would be structural, not measured).
    """
    
    if not PIPELINE_AVAILABLE:
        logger.error("Pipeline module not available!")
        logger.error("Install: Check src/pipeline/agent_pipeline.py")
        raise ImportError("AgentPipeline not found")
    
    # Get paths for this dataset
    paths = store_manager.get_paths(dataset)
    
    # Update config
    pipeline_config = config.copy()
    pipeline_config["paths"] = pipeline_config.get("paths", {}).copy()
    pipeline_config["paths"]["vector"] = str(paths["vector"])
    pipeline_config["paths"]["graph_db"] = str(paths["graph"])

    # Model override (--model flag) — all model names live in settings.yaml
    if model_name is not None:
        pipeline_config["llm"] = pipeline_config.get("llm", {}).copy()
        pipeline_config["llm"]["model_name"] = model_name

    # Component ablation flags
    pipeline_config["agent"] = pipeline_config.get("agent", {}).copy()
    pipeline_config["agent"]["enable_planner"] = enable_planner
    pipeline_config["agent"]["enable_verifier"] = enable_verifier
    pipeline_config["agent"]["max_verification_iterations"] = max_iterations

    # Why: when False, the three pre-validation checks (entity-path,
    # contradiction, credibility) are forced off so the +Verifier ablation
    # row produces a distinct configuration from the +Planner row.
    # Without this override, both rows have byte-identical settings and any
    # delta is structural noise, not measurement.
    pipeline_config["verifier"] = pipeline_config.get("verifier", {}).copy()
    if not enable_pre_validation:
        pipeline_config["verifier"]["enable_entity_path_validation"] = False
        pipeline_config["verifier"]["enable_contradiction_detection"] = False
        pipeline_config["verifier"]["enable_credibility_scoring"] = False
        logger.info(
            "create_pipeline: pre-validation DISABLED (entity-path / "
            "contradiction / credibility all off)"
        )

    # Set per-lane RRF weights (Cormack et al. 2009). The pipeline has three
    # independent retrieval lanes fused via RRF:
    #   - vector : dense ANN over the embedding model (LanceDB)
    #   - graph  : entity-path traversal (KuzuDB)
    #   - bm25   : sparse lexical term-frequency ranking (rank_bm25)
    # BM25 is a genuinely separate lane (lexical, not embedding-based), so it
    # gets its own ablation weight + on/off toggle just like vector and graph.
    pipeline_config["rag"] = pipeline_config.get("rag", {}).copy()
    pipeline_config["rag"]["vector_weight"] = vector_weight
    pipeline_config["rag"]["graph_weight"] = graph_weight
    pipeline_config["rag"]["bm25_weight"] = bm25_weight

    # enable_bm25: when explicitly set, override settings.yaml. Needed for a
    # clean single-lane ablation -- BM25 fires in VECTOR and HYBRID modes
    # (HybridRetriever gates it by `enable_bm25 and mode in {VECTOR, HYBRID}`),
    # so a "pure vector" row must set enable_bm25=False to silence the sparse
    # lane. None = leave the settings.yaml value untouched (default behaviour).
    if enable_bm25 is not None:
        pipeline_config["rag"]["enable_bm25"] = bool(enable_bm25)

    # cross_source_boost (RRF β): extra RRF credit for a chunk surfaced by more
    # than one lane (Cormack et al. 2009 give the base RRF; β>1 up-weights
    # cross-lane consensus). Exposed as an override so the β ablation (B-3) can
    # sweep it WITHOUT editing config/settings.yaml. None = use the settings.yaml
    # value (default 1.2), preserving current behaviour.
    if cross_source_boost is not None:
        pipeline_config["rag"]["cross_source_boost"] = float(cross_source_boost)

    # RetrievalMode selects the vector/graph lanes. NOTE: BM25 is orthogonal —
    # it is controlled by enable_bm25/bm25_weight above, not by this mode. In
    # GRAPH mode the BM25 lane does not fire (HybridRetriever restricts it to
    # VECTOR/HYBRID). So "graph only" = graph_weight>0, vector_weight=0; "vector
    # only (no sparse)" = vector_weight>0, graph_weight=0, enable_bm25=False.
    if graph_weight == 0:
        pipeline_config["rag"]["retrieval_mode"] = "vector"
    elif vector_weight == 0:
        pipeline_config["rag"]["retrieval_mode"] = "graph"
    else:
        pipeline_config["rag"]["retrieval_mode"] = "hybrid"
    
    # Initialize storage components
    if not STORAGE_AVAILABLE:
        raise ImportError("Storage module required for pipeline")
    
    embedding_config = pipeline_config.get("embeddings", {})
    perf_config = pipeline_config.get("performance", {})
    cache_path = _CACHE_ROOT / f"{dataset}_embeddings.db"

    embeddings = BatchedOllamaEmbeddings(
        model_name=embedding_config.get("model_name", "nomic-embed-text"),
        base_url=embedding_config.get("base_url", "http://localhost:11434"),
        batch_size=perf_config.get("batch_size", _EMBEDDING_BATCH_SIZE),
        cache_path=cache_path,
    )

    vector_config = pipeline_config.get("vector_store", {})
    storage_config = StorageConfig(
        vector_db_path=paths["vector"],
        graph_db_path=paths["graph"],
        embedding_dim=embedding_config.get("embedding_dim", _NOMIC_EMBED_DIM),
        similarity_threshold=vector_config.get("similarity_threshold", _VECTOR_SIMILARITY_THRESHOLD_FALLBACK),
        normalize_embeddings=vector_config.get("normalize_embeddings", True),
        distance_metric=vector_config.get("distance_metric", "cosine"),
        # Fallbacks mirror the StorageConfig dataclass defaults.
        overfetch_factor=vector_config.get("overfetch_factor", 3),
        graph_text_max_chars=vector_config.get("graph_text_max_chars", 500),
    )

    hybrid_store = HybridStore(config=storage_config, embeddings=embeddings)

    # Determine retrieval mode
    if graph_weight == 0:
        retrieval_mode = RetrievalMode.VECTOR
    elif vector_weight == 0:
        retrieval_mode = RetrievalMode.GRAPH
    else:
        retrieval_mode = RetrievalMode.HYBRID

    # Wrap HybridStore in HybridRetriever (Navigator needs .retrieve()).
    # Build the RetrievalConfig from settings.yaml via the canonical factory
    # so every retrieval knob (vector_store.top_k_vectors, rag.bm25_top_k,
    # rag.rrf_k, rag.cross_source_boost, graph.max_hops, top_k_entities,
    # enable_hop3, hub_mention_cap, rag.vector/graph/bm25_weight) is honoured
    # from a single source of truth. The per-config ablation `mode` (derived
    # from the vector/graph weights above) overrides rag.retrieval_mode
    # afterwards.
    retriever = create_hybrid_retriever(hybrid_store, embeddings, pipeline_config)
    retriever.config.mode = retrieval_mode

    pipeline = create_full_pipeline(
        hybrid_retriever=retriever,
        graph_store=hybrid_store.graph_store,
        config=pipeline_config,
    )

    # Why: attach the BatchedOllamaEmbeddings instance so cmd_evaluate can
    # print cache-hit-rate, batch-count and mean per-text latency at the end
    # of the run. The metrics are owned by the embeddings object and
    # accumulate across both ingestion (indirectly via the cache file) and
    # evaluation (directly via embed_query calls). The underscore prefix
    # signals "diagnostic accessor only" -- production code must not depend
    # on this attribute.
    pipeline._embeddings = embeddings

    logger.info(
        "Pipeline created for %s (v=%s, g=%s)",
        dataset, vector_weight, graph_weight,
    )

    return pipeline


def _close_pipeline(pipeline) -> None:
    """Release a pipeline's KuzuDB file lock promptly.

    Called from the ablation loop's `finally` so the lock is freed even when a
    config raises mid-run — otherwise the next config cannot open the same
    graph (KuzuDB holds an exclusive C++ file lock). Closing is best-effort:
    the store is reached via the retriever (`pipeline.hybrid_retriever.store`),
    which exposes HybridStore.close() → KuzuGraphStore.close().
    """
    if pipeline is None:
        return
    try:
        retriever = getattr(pipeline, "hybrid_retriever", None)
        store = getattr(retriever, "store", None) if retriever is not None else None
        if store is not None and hasattr(store, "close"):
            store.close()
    except Exception as exc:
        logger.warning("    Pipeline cleanup (store.close) failed: %s", exc)
    finally:
        gc.collect()

# ============================================================================
# EVALUATION RUNNER -- pipeline vs. LLM failure separation
# ============================================================================
#
# The benchmark must answer two independent questions:
#   1. Did the pipeline (S_P + S_N) retrieve the correct supporting facts?
#      Measured by supporting-fact F1 against HotpotQA gold paragraphs.
#   2. Did the LLM (S_V) produce the correct answer GIVEN those facts?
#      Measured by EM/F1 restricted to questions where retrieval succeeded.
#
# These can fail independently. A timeout on the SLM is a model failure, not
# a pipeline failure. A missing gold paragraph is a pipeline failure, no
# matter how good the LLM is. The paper argument depends on distinguishing
# the two.

# Module-level text→title index. Populated by the retriever monkey-patch
# below; consulted by _retrieved_titles_for_chunks() to look up the source
# title of a Navigator-filtered chunk. Keyed by the first 200 chars of chunk
# text (case-folded) so we tolerate downstream truncation.
_TEXT_TO_TITLE: Dict[str, str] = {}


def _text_key(text: str) -> str:
    """Stable lookup key for the text->title cache. See `_TEXT_KEY_LEN`."""
    return (text or "").strip()[:_TEXT_KEY_LEN].lower()


def _norm_title(title: str) -> str:
    """Lowercase, strip a leading 'hotpotqa_'/'2wiki_'/etc. dataset prefix,
    collapse whitespace. Mirrors the diagnose_verbose.py logic so SF
    comparison uses the same key space."""
    t = (title or "").strip().lower()
    if "_" in t:
        prefix, _, rest = t.partition("_")
        if prefix and " " not in prefix and rest:
            t = rest
    return " ".join(t.split())


def _gold_titles_from_supporting_facts(supporting_facts: List) -> List[str]:
    """HotpotQA/2WikiMHQA supporting_facts → set of normalised gold titles.

    supporting_facts is a list of (title, sent_id) tuples (see
    load_hotpotqa). We only need the unique titles."""
    seen, out = set(), []
    for entry in supporting_facts or []:
        title = entry[0] if isinstance(entry, (list, tuple)) and entry else str(entry)
        norm = _norm_title(title)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _install_retriever_title_capture(pipeline) -> Optional[callable]:
    """Monkey-patch HybridRetriever.retrieve() on the live pipeline so every
    raw RetrievalResult registers its (text→source_doc) mapping in
    _TEXT_TO_TITLE.

    Returns the original retrieve() so the caller can restore it after the
    benchmark run. None if no retriever is reachable from the pipeline.

    This is the inline alternative to threading source_doc through the
    Navigator/Verifier — the navigator strips chunks to text-only, so we
    rebuild the title lookup at the retriever boundary."""
    # Locate the HybridRetriever on the live pipeline. AgentPipeline exposes
    # it as `hybrid_retriever` (the canonical attribute used in
    # src/pipeline/agent_pipeline.py); older paths used `retriever`. The
    # Navigator stores it as `retriever` after set_retriever() is called.
    # Without this fallback chain matching ALL possible attribute names,
    # the patch silently no-ops and SF-F1 = 0.0 across every question —
    # which is exactly what was observed before this fix.
    retriever = None
    for attr in ("hybrid_retriever", "retriever", "_retriever"):
        candidate = getattr(pipeline, attr, None)
        if candidate is not None and hasattr(candidate, "retrieve"):
            retriever = candidate
            break
    if retriever is None:
        nav = getattr(pipeline, "navigator", None) or getattr(pipeline, "_navigator", None)
        if nav is not None:
            retriever = getattr(nav, "retriever", None) or getattr(nav, "_retriever", None)
    if retriever is None or not hasattr(retriever, "retrieve"):
        logger.warning(
            "Could not locate HybridRetriever on pipeline (tried "
            "hybrid_retriever / retriever / _retriever / navigator.retriever) "
            "— SF metrics will be 0. Pipeline attrs available: %s",
            [a for a in dir(pipeline) if not a.startswith('_')][:20],
        )
        return None
    logger.info("SF-title capture installed on %s.%s",
                type(pipeline).__name__,
                "hybrid_retriever" if retriever is getattr(pipeline, "hybrid_retriever", None)
                else "retriever")

    original = retriever.retrieve

    def _wrapped(*args, **kwargs):
        ret = original(*args, **kwargs)
        # HybridRetriever.retrieve() returns Tuple[List[RetrievalResult],
        # RetrievalMetrics]. Previous version of this patch iterated the
        # tuple directly and produced (RetrievalResult, RetrievalMetrics)
        # pairs — the metrics object has no .text / .source_doc, so every
        # capture silently fell through and SF-F1 was 0.0 across all
        # questions. The unpacking below is the actual fix.
        if isinstance(ret, tuple) and len(ret) >= 1:
            results_iter = ret[0]
        else:
            results_iter = ret
        try:
            for r in (results_iter or []):
                txt = (r.text if hasattr(r, "text")
                       else r.get("text", "") if isinstance(r, dict)
                       else "")
                src = (r.source_doc if hasattr(r, "source_doc")
                       else r.get("source_doc", "") if isinstance(r, dict)
                       else "")
                if txt and src:
                    _TEXT_TO_TITLE.setdefault(_text_key(txt), src)
        except Exception as exc:
            logger.debug("title-capture failed (non-fatal): %s", exc)
        return ret

    retriever.retrieve = _wrapped
    return original


def _retrieved_titles_for_chunks(chunks: List[str]) -> List[str]:
    """Map filtered-context chunks back to their source-doc titles using
    the text→title index built by the retriever monkey-patch."""
    out: List[str] = []
    seen: set = set()
    for c in chunks or []:
        title = _TEXT_TO_TITLE.get(_text_key(c)) or _TEXT_TO_TITLE.get(_text_key(c[:200]))
        if not title:
            continue
        norm = _norm_title(title)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _compute_sf_metrics(retrieved: List[str], gold: List[str]) -> Tuple[float, float, float, bool]:
    """Supporting-fact precision, recall, F1, and 'all_gold_retrieved' flag.

    HotpotQA-style: titles only, set-based (sent_id ignored — at chunk
    granularity we cannot resolve sentence-level supporting facts)."""
    gold_set = set(gold or [])
    retrieved_set = set(retrieved or [])
    if not gold_set:
        return 0.0, 0.0, 0.0, False
    if not retrieved_set:
        return 0.0, 0.0, 0.0, False
    tp = len(gold_set & retrieved_set)
    precision = tp / len(retrieved_set) if retrieved_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    all_gold = gold_set.issubset(retrieved_set)
    return precision, recall, f1, all_gold


def _classify_llm_error(answer: str) -> Tuple[bool, str]:
    """Detect Verifier error sentinels.

    Verifier emits answers prefixed with '[Error:' on LLM-side failures
    (timeout, API error, no valid answer). Anything else is a real model
    response, even if substantively wrong."""
    if not answer or not answer.startswith("[Error:"):
        return False, ""
    low = answer.lower()
    if "timeout" in low:
        return True, "timeout"
    if "connect" in low or "ollama" in low:
        return True, "connection"
    if "api returned" in low:
        return True, "api"
    if "no valid answer" in low:
        return True, "no_answer"
    return True, "other"


def _extract_planner_diagnostics(
    planner_result: Dict[str, Any],
) -> Tuple[str, int, int, Optional[str], Optional[str]]:
    """Extract (query_type, hop_count, n_entities, matched_pattern,
    classifier_preempt) from PlannerResult.to_dict().

    matched_pattern and classifier_preempt are surfaced so per-question JSONL
    can be filtered and aggregated by pattern without parsing debug logs.
    """
    if not isinstance(planner_result, dict):
        return "", 0, 0, None, None
    qt = planner_result.get("query_type", "") or ""
    hops = planner_result.get("hop_sequence", []) or []
    entities = planner_result.get("entities", []) or []
    matched_pattern = planner_result.get("matched_pattern")
    classifier_preempt = planner_result.get("classifier_preempt")
    return qt, len(hops), len(entities), matched_pattern, classifier_preempt


def _extract_verifier_diagnostics(verifier_result: Dict[str, Any]) -> Tuple[int, bool, str]:
    """(iterations, all_verified, confidence) from Verifier output."""
    if not isinstance(verifier_result, dict):
        return 0, False, ""
    iters = int(verifier_result.get("iterations", 0) or 0)
    allv = bool(verifier_result.get("all_verified", False))
    conf = str(verifier_result.get("confidence", "") or "")
    return iters, allv, conf


def _extract_pre_validation_diagnostics(
    verifier_result: Dict[str, Any],
) -> Tuple[Optional[str], int, int]:
    """Extract (status, chunks_in, chunks_out) from the pre_validation nested
    dict produced by Verifier.generate_and_verify().

    Returns (None, 0, 0) when verifier_result is missing or pre_validation
    was not run (enable_verifier=False).
    """
    if not isinstance(verifier_result, dict):
        return None, 0, 0
    pre = verifier_result.get("pre_validation") or {}
    if not isinstance(pre, dict):
        return None, 0, 0
    status = pre.get("status")
    if status is not None and not isinstance(status, str):
        # ValidationStatus enum -> str via asdict serialization
        status = getattr(status, "value", str(status))
    filtered = pre.get("filtered_context", []) or []
    details = pre.get("details", {}) or {}
    chunks_in = int(details.get("input_chunk_count", 0) or 0) if isinstance(details, dict) else 0
    chunks_out = len(filtered) if isinstance(filtered, list) else 0
    return (str(status) if status is not None else None, chunks_in, chunks_out)


# ============================================================================
# EVALUATION RUNNER
# ============================================================================

def evaluate_dataset(
    dataset: str,
    questions: List[TestQuestion],
    pipeline,
    config_name: str,
    vector_weight: float,
    graph_weight: float,
    jsonl_out: Optional[Path] = None,
    retrieval_only: bool = False,
    answer_f1_threshold: Optional[float] = None,
) -> ConfigResult:
    """Evaluate dataset with given configuration.

    Args:
        jsonl_out: If given, append one JSON line per question to this file
            with the full EvalResult -- for paper analysis.
        retrieval_only: If True, skip the LLM (verifier) entirely and only
            measure pipeline retrieval quality. EM/F1 are forced to 0 in
            this mode; SF-F1/recall are the meaningful metrics.
        answer_f1_threshold: Soft-EM threshold (token-F1 >= threshold counts
            as correct). When None, falls back to the module default
            ANSWER_F1_THRESHOLD so callers that do not need per-run override
            remain backward compatible.
    """
    f1_threshold = (
        ANSWER_F1_THRESHOLD if answer_f1_threshold is None
        else float(answer_f1_threshold)
    )

    results: List[EvalResult] = []

    # Reset text→title cache for this run and install retriever hook.
    _TEXT_TO_TITLE.clear()
    original_retrieve = _install_retriever_title_capture(pipeline)

    # Verifier's per-query chunk cap (max_docs) — the LLM-visible window size.
    # Used to compute gold_in_final_context (delivery check). Read from the live
    # pipeline's verifier config; falls back to _VERIFIER_MAX_DOCS_FALLBACK
    # (=5, production setting) if the verifier or its config is unreachable
    # (offline tests, mocked verifier).
    _verifier_max_docs = _VERIFIER_MAX_DOCS_FALLBACK
    try:
        _v = getattr(pipeline, "verifier", None)
        _vc = getattr(_v, "config", None) if _v is not None else None
        if _vc is not None and getattr(_vc, "max_docs", None):
            _verifier_max_docs = int(_vc.max_docs)
    except (AttributeError, TypeError, ValueError) as exc:
        # Defensive: any of the three getattr / int-cast steps can fail under
        # mocked pipelines or unusual config types. Stay on the fallback and
        # log at debug so a reviewer can still see the cause if needed.
        logger.debug("verifier max_docs lookup failed (fallback=%d): %s",
                     _VERIFIER_MAX_DOCS_FALLBACK, exc)

    # Optional: disable verifier for retrieval-only evaluation.
    saved_enable_verifier = None
    if retrieval_only and hasattr(pipeline, "enable_verifier"):
        saved_enable_verifier = pipeline.enable_verifier
        pipeline.enable_verifier = False

    desc = f"Evaluating {dataset} [{config_name}]" + (" [retrieval-only]" if retrieval_only else "")

    try:
        for q in tqdm(questions, desc=desc, unit="q"):
            try:
                start = time.time()
                result = pipeline.process(q.question)
                elapsed = (time.time() - start) * 1000

                # ── Answer-level metrics (LLM correctness) ───────────────
                if retrieval_only:
                    em, f1, ans_correct = False, 0.0, False
                    predicted = ""
                else:
                    em = compute_exact_match(result.answer, q.answer)
                    f1 = compute_f1(result.answer, q.answer)
                    # Soft correctness: strict EM OR token-F1 >= threshold.
                    ans_correct = bool(em) or f1 >= f1_threshold
                    predicted = result.answer

                # ── Retrieval-level metrics (pipeline correctness) ───────
                filtered_chunks: List[str] = []
                if hasattr(result, 'navigator_result'):
                    nav = result.navigator_result
                    if isinstance(nav, dict):
                        filtered_chunks = nav.get('filtered_context', []) or []
                retrieval_count = len(filtered_chunks)

                gold_titles = _gold_titles_from_supporting_facts(q.supporting_facts)
                retrieved_titles = _retrieved_titles_for_chunks(filtered_chunks)
                sf_p, sf_r, sf_f1, all_gold = _compute_sf_metrics(retrieved_titles, gold_titles)

                # Delivery check: the Verifier forwards only the top `max_docs`
                # of filtered_context to the LLM (membership owned by RRF order;
                # the reorder runs within that window). Recompute gold presence
                # against that post-cap window to separate delivery loss from
                # retrieval loss.
                final_chunks = filtered_chunks[:_verifier_max_docs]
                final_titles = _retrieved_titles_for_chunks(final_chunks)
                _, _, _, gold_in_final = _compute_sf_metrics(final_titles, gold_titles)

                # ── Failure-mode separation ──────────────────────────────
                llm_err, llm_err_type = (False, "") if retrieval_only else _classify_llm_error(predicted)
                # Pipeline OK + LLM failed = retrieval was complete but model errored.
                pipeline_ok_llm_failed = bool(all_gold and llm_err)

                # -- Planner / Verifier diagnostics -----------------------
                # Pull matched_pattern + classifier_preempt for per-pattern
                # aggregation downstream; pre_validation status + chunk-flow
                # for the pre-generation-filter diagnostic.
                p_qtype, hop_count, n_ents, p_pattern, p_preempt = (
                    _extract_planner_diagnostics(
                        getattr(result, "planner_result", {}) or {}
                    )
                )
                v_iters, v_verified, v_conf = _extract_verifier_diagnostics(
                    getattr(result, "verifier_result", {}) or {}
                )
                pv_status, pv_in, pv_out = _extract_pre_validation_diagnostics(
                    getattr(result, "verifier_result", {}) or {}
                )

                eval_result = EvalResult(
                    question_id=q.id,
                    question=q.question,
                    gold_answer=q.answer,
                    predicted_answer=predicted,
                    exact_match=em,
                    f1_score=f1,
                    answer_correct=ans_correct,
                    retrieval_count=retrieval_count,
                    time_ms=elapsed,
                    dataset=q.dataset,
                    question_type=q.question_type,
                    gold_titles=gold_titles,
                    retrieved_titles=retrieved_titles,
                    retrieval_recall=sf_r,
                    retrieval_precision=sf_p,
                    sf_f1=sf_f1,
                    all_gold_retrieved=all_gold,
                    gold_in_final_context=gold_in_final,
                    llm_error=llm_err,
                    llm_error_type=llm_err_type,
                    pipeline_succeeded_llm_failed=pipeline_ok_llm_failed,
                    planner_query_type=p_qtype,
                    hop_count=hop_count,
                    n_entities=n_ents,
                    matched_pattern=p_pattern,
                    classifier_preempt=p_preempt,
                    verifier_iterations=v_iters,
                    all_verified=v_verified,
                    confidence=v_conf,
                    pre_validation_status=pv_status,
                    pre_validation_chunks_in=pv_in,
                    pre_validation_chunks_out=pv_out,
                )
                results.append(eval_result)

                # Per-question JSONL (one line per question) for paper analysis.
                if jsonl_out is not None:
                    try:
                        jsonl_out.parent.mkdir(parents=True, exist_ok=True)
                        with open(jsonl_out, "a", encoding="utf-8") as fh:
                            fh.write(json.dumps(asdict(eval_result), ensure_ascii=False) + "\n")
                    except Exception as exc:
                        logger.warning("JSONL write failed: %s", exc)

            except Exception as e:
                logger.warning("    Error on Q%s: %s", q.id, str(e)[:_LOG_ERROR_MSG_TRUNC])
    finally:
        # Restore patched retriever and verifier flag so the pipeline isn't
        # permanently mutated (matters for ablation: same pipeline runs
        # multiple configurations).
        if original_retrieve is not None:
            for attr in ("hybrid_retriever", "retriever", "_retriever"):
                cand = getattr(pipeline, attr, None)
                if cand is not None and hasattr(cand, "retrieve"):
                    cand.retrieve = original_retrieve
                    break
            else:
                nav = getattr(pipeline, "navigator", None) or getattr(pipeline, "_navigator", None)
                if nav is not None:
                    inner = getattr(nav, "retriever", None) or getattr(nav, "_retriever", None)
                    if inner is not None:
                        inner.retrieve = original_retrieve
        if saved_enable_verifier is not None:
            pipeline.enable_verifier = saved_enable_verifier

    if not results:
        return None

    # ── Aggregate metrics ────────────────────────────────────────────────
    n = len(results)
    em_rate = sum(1 for r in results if r.exact_match) / n
    soft_em_rate = sum(1 for r in results if r.answer_correct) / n
    avg_f1 = sum(r.f1_score for r in results) / n
    avg_time = sum(r.time_ms for r in results) / n
    coverage = sum(1 for r in results if r.retrieval_count > 0) / n

    # Retrieval-level aggregates. "retrieval_only_em" uses the fair (soft)
    # correctness verdict so model accuracy isn't understated by strict EM.
    avg_sf_f1 = sum(r.sf_f1 for r in results) / n
    n_all_gold = sum(1 for r in results if r.all_gold_retrieved)
    sf_recall_rate = n_all_gold / n
    # Delivery: gold survives into the LLM-visible window (post max_docs cap).
    final_context_recall_rate = sum(1 for r in results if r.gold_in_final_context) / n
    # Delivery loss: retrieved by the pipeline but cut before the LLM saw it.
    delivery_loss_rate = sum(
        1 for r in results if r.all_gold_retrieved and not r.gold_in_final_context
    ) / n
    retrieval_only_em = (
        sum(1 for r in results if r.all_gold_retrieved and r.answer_correct) / n_all_gold
        if n_all_gold > 0 else 0.0
    )
    llm_error_rate = sum(1 for r in results if r.llm_error) / n

    # Failure decomposition (mutually exclusive buckets, sum to 1.0):
    #   pipeline_failed       : not all_gold_retrieved
    #   pipeline_ok_llm_failed : all_gold AND llm error (timeout/etc.)
    #   pipeline_ok_llm_wrong  : all_gold AND no llm error AND not correct
    #   pipeline_ok_llm_ok     : all_gold AND correct (soft verdict, F1>=thr)
    pipeline_failed = sum(1 for r in results if not r.all_gold_retrieved) / n
    ok_llm_failed = sum(1 for r in results if r.all_gold_retrieved and r.llm_error) / n
    ok_llm_wrong = sum(
        1 for r in results
        if r.all_gold_retrieved and not r.llm_error and not r.answer_correct
    ) / n
    ok_llm_ok = sum(
        1 for r in results if r.all_gold_retrieved and r.answer_correct
    ) / n

    # By question type
    by_type: Dict[str, Dict] = {}
    for qtype in set(r.question_type for r in results):
        type_results = [r for r in results if r.question_type == qtype]
        tn = len(type_results)
        by_type[qtype] = {
            "count": tn,
            "exact_match": sum(1 for r in type_results if r.exact_match) / tn,
            "soft_em": sum(1 for r in type_results if r.answer_correct) / tn,
            "f1": sum(r.f1_score for r in type_results) / tn,
            "sf_f1": sum(r.sf_f1 for r in type_results) / tn,
            "sf_recall_rate": sum(1 for r in type_results if r.all_gold_retrieved) / tn,
            "llm_error_rate": sum(1 for r in type_results if r.llm_error) / tn,
        }

    return ConfigResult(
        dataset=dataset,
        config_name=config_name,
        vector_weight=vector_weight,
        graph_weight=graph_weight,
        n_questions=n,
        exact_match=em_rate,
        f1_score=avg_f1,
        soft_em=soft_em_rate,
        avg_time_ms=avg_time,
        coverage=coverage,
        by_type=by_type,
        avg_sf_f1=avg_sf_f1,
        sf_recall_rate=sf_recall_rate,
        final_context_recall_rate=final_context_recall_rate,
        delivery_loss_rate=delivery_loss_rate,
        retrieval_only_em=retrieval_only_em,
        llm_error_rate=llm_error_rate,
        pipeline_failed_rate=pipeline_failed,
        pipeline_ok_llm_failed_rate=ok_llm_failed,
        pipeline_ok_llm_wrong_rate=ok_llm_wrong,
        pipeline_ok_llm_ok_rate=ok_llm_ok,
    )

# ============================================================================
# COMMANDS
# ============================================================================

def cmd_ingest(args, config: Dict, store_manager: StoreManager):
    """Ingest command."""
    
    if args.dataset == "all":
        datasets = AVAILABLE_DATASETS
    else:
        datasets = [d.strip() for d in args.dataset.split(",")]
    
    logger.info("=" * 70)
    logger.info("BENCHMARK INGESTION")
    logger.info("=" * 70)
    logger.info("Datasets: %s", datasets)
    logger.info("Samples per dataset: %s", args.samples)
    logger.info("Chunking: %s sentences, overlap=%s",
                args.chunk_sentences, args.chunk_overlap)
    logger.info("=" * 70)

    for dataset in datasets:
        logger.info("\n%s", "-" * 70)
        logger.info("DATASET: %s", dataset.upper())
        logger.info("%s", "-" * 70)

        if dataset not in LOADERS:
            logger.error("Unknown dataset: %s", dataset)
            continue

        if args.clear:
            # When --chunks-only is set we only want to wipe Phase-1 output;
            # the existing graph (often locked by KuzuDB on Windows) and the
            # Colab-produced extraction_results.json must NOT be touched.
            store_manager.clear_dataset(
                dataset,
                chunks_only=getattr(args, "chunks_only", False),
            )

        if store_manager.dataset_exists(dataset) and not args.clear:
            logger.info("  Already exists. Use --clear to re-ingest.")
            continue

        # Load dataset
        loader = LOADERS[dataset]
        articles, questions = loader.load(n_samples=args.samples)

        if not articles and not questions:
            logger.warning("  No data loaded for %s", dataset)
            continue
        
        # Save questions
        store_manager.save_questions(questions, dataset)
        store_manager.save_articles_info(articles, dataset)
        
        # Create documents and ingest
        if articles:
            documents = create_langchain_documents(
                articles,
                chunk_sentences=args.chunk_sentences,
                sentence_overlap=args.chunk_overlap,
                apply_coreference=not getattr(args, "no_coreference", False),
            )
            logger.info("  Created %d document chunks", len(documents))

            # --chunks-only: export chunks as JSON and stop (decoupled
            # ingestion part 1; the JSON is then processed in Colab).
            if getattr(args, "chunks_only", False):
                out_dir = _DATA_ROOT / dataset
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "chunks_export.json"
                chunks_data = [
                    {"text": doc.page_content, "metadata": {k: str(v) for k, v in doc.metadata.items()}}
                    for doc in documents
                ]
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(chunks_data, f, ensure_ascii=False, indent=2)
                logger.info("  [chunks-only] Exported: %s  (%.1f MB)",
                            out_path, out_path.stat().st_size / 1024 / 1024)
                logger.info("  [chunks-only] Next step: upload file to Google Colab.")
                continue

            paths = store_manager.get_paths(dataset)
            run_ingestion(
                documents,
                paths["vector"],
                paths["graph"],
                config,
                dataset,
            )
    
    # Print status
    logger.info("\n%s", "=" * 70)
    logger.info("INGESTION STATUS")
    logger.info("=" * 70)

    status = store_manager.get_status()
    for ds, exists in status.items():
        mark = "[OK]" if exists else "[--]"
        logger.info("  %s %s", mark, ds)
    
    logger.info("="*70)

def cmd_evaluate(args, config: Dict, store_manager: StoreManager):
    """Evaluate command."""

    # Soft-EM threshold resolution: settings.yaml -> CLI -> module default.
    # No module-level mutation: the resolved value is threaded into
    # evaluate_dataset for this call only, so subsequent in-process callers
    # (e.g. cmd_ablation in a notebook) are not affected.
    cfg_thr = (config.get("benchmark", {}) or {}).get("answer_f1_threshold")
    cli_thr = getattr(args, "answer_f1_threshold", None)
    f1_threshold = (
        float(cli_thr) if cli_thr is not None
        else float(cfg_thr) if cfg_thr is not None
        else ANSWER_F1_THRESHOLD
    )
    logger.info("Soft-EM answer-correctness threshold: F1 >= %.2f", f1_threshold)

    dataset = args.dataset

    if not store_manager.dataset_exists(dataset):
        logger.error("Dataset not ingested: %s", dataset)
        logger.error("Run: python -m src.thesis_evaluations.benchmark_datasets "
                     "ingest --dataset %s", dataset)
        return

    logger.info("=" * 70)
    logger.info("EVALUATION: %s", dataset.upper())
    logger.info("=" * 70)

    questions = store_manager.load_questions(dataset)
    if not questions:
        return

    # Question selection: either a fixed slice (--range) or a random sample
    # (--samples, default 20). --seed lets you reproduce a previous random
    # selection. The two are mutually exclusive: --range wins if both given.
    range_spec = getattr(args, "range", None)
    if range_spec:
        sep = next((s for s in ("-", "_", ":") if s in range_spec), None)
        if sep is None:
            logger.error("--range must contain a separator (-, _, or :); got %r", range_spec)
            return
        try:
            start_s, end_s = range_spec.split(sep, 1)
            start, end = int(start_s), int(end_s)
        except ValueError:
            logger.error("--range expects two integers; got %r", range_spec)
            return
        if start < 0 or end > len(questions) or start >= end:
            logger.error(
                "--range %d-%d out of bounds for %d questions",
                start, end, len(questions),
            )
            return
        questions = questions[start:end]
        logger.info("Question range: [%d..%d) -> %d questions", start, end, len(questions))
    elif args.samples:
        seed = getattr(args, "seed", None)
        if seed is None:
            seed = random.randint(0, _AUTO_SEED_RANGE - 1)
        logger.info(
            "Question sample seed: %d  (re-run with --seed %d to reproduce)",
            seed, seed,
        )
        rng = random.Random(seed)
        questions = rng.sample(questions, min(args.samples, len(questions)))


    model_name = (
        getattr(args, "model", None)
        or config.get("llm", {}).get("model_name", _DEFAULT_MODEL_FALLBACK)
    )
    enable_planner = not getattr(args, "no_planner", False)
    enable_verifier = not getattr(args, "no_verifier", False)
    max_iterations = getattr(args, "iterations", 1) or 1
    logger.info("Questions: %d", len(questions))
    logger.info("Config: vector=%s, graph=%s, model=%s",
                args.vector_weight, args.graph_weight, model_name)
    logger.info("Components: planner=%s, verifier=%s, iter=%s",
                enable_planner, enable_verifier, max_iterations)

    # Create pipeline
    pipeline = create_pipeline(
        dataset, config, store_manager,
        vector_weight=args.vector_weight,
        graph_weight=args.graph_weight,
        model_name=model_name,
        enable_planner=enable_planner,
        enable_verifier=enable_verifier,
        max_iterations=max_iterations,
    )

    try:
        config_name = f"v{args.vector_weight}_g{args.graph_weight}_{model_name}"

        retrieval_only = getattr(args, "retrieval_only", False)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        jsonl_dir = _LANE_ABLATION_DIR
        jsonl_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = jsonl_dir / f"{dataset}_{model_name.replace(':','-')}_{ts}.jsonl"
        # Truncate any prior partial file with the same name.
        if jsonl_path.exists():
            jsonl_path.unlink()

        result = evaluate_dataset(
            dataset, questions, pipeline,
            config_name, args.vector_weight, args.graph_weight,
            jsonl_out=jsonl_path,
            retrieval_only=retrieval_only,
            answer_f1_threshold=f1_threshold,
        )

        # Print results
        bar = "-" * 70
        logger.info("\n%s", bar)
        logger.info("RESULTS%s", " [retrieval-only]" if retrieval_only else "")
        logger.info("%s", bar)
        cfg_hash = getattr(args, "_config_hash", None)
        if cfg_hash:
            logger.info("  Config sha256:         %s   (frozen-config provenance)", cfg_hash)
        logger.info("  Answer F1:             %.3f   (primary answer-quality metric)", result.f1_score)
        logger.info("  Soft-EM (F1>=%g):     %.2f%%   (correct incl. name variants)",
                    f1_threshold, result.soft_em * 100)
        logger.info("  Exact Match (strict):  %.2f%%", result.exact_match * 100)
        logger.info("  Coverage:              %.2f%%", result.coverage * 100)
        logger.info("  Avg Time:              %.0fms", result.avg_time_ms)
        logger.info("")
        logger.info("  Pipeline (S_P + S_N) -- retrieval quality:")
        logger.info("    Supporting-fact F1:  %.3f", result.avg_sf_f1)
        logger.info("    All gold retrieved (Navigator output): %.2f%%", result.sf_recall_rate * 100)
        logger.info("    All gold in LLM window (post max_docs): %.2f%%", result.final_context_recall_rate * 100)
        logger.info("    Delivery loss (retrieved but cut):      %.2f%%", result.delivery_loss_rate * 100)
        if not retrieval_only:
            logger.info("")
            logger.info("  Failure decomposition (sum to 100%%):")
            logger.info("    Pipeline failed:           %.2f%%  (gold not fully retrieved)", result.pipeline_failed_rate * 100)
            logger.info("    Pipeline ok, LLM failed:   %.2f%%  (timeout/api error)", result.pipeline_ok_llm_failed_rate * 100)
            logger.info("    Pipeline ok, LLM wrong:    %.2f%%", result.pipeline_ok_llm_wrong_rate * 100)
            logger.info("    Pipeline ok, LLM ok:       %.2f%%  (F1>=%g)", result.pipeline_ok_llm_ok_rate * 100, f1_threshold)
            logger.info("")
            logger.info("  LLM error rate:        %.2f%%", result.llm_error_rate * 100)
            logger.info("  Correct | all-gold-retrieved: %.2f%%  "
                        "(model accuracy when retrieval succeeds)",
                        result.retrieval_only_em * 100)

        if result.by_type:
            logger.info("\n  By Question Type:")
            for qtype, stats in result.by_type.items():
                logger.info(
                    "    %s: SoftEM=%.2f%% F1=%.3f EM=%.2f%% SF-F1=%.3f "
                    "SF-Recall=%.2f%% LLM-err=%.2f%%",
                    qtype, stats.get("soft_em", 0.0) * 100, stats["f1"],
                    stats["exact_match"] * 100, stats.get("sf_f1", 0.0),
                    stats.get("sf_recall_rate", 0.0) * 100,
                    stats.get("llm_error_rate", 0.0) * 100,
                )

        # Print embedding-cache metrics so the paper report can defend
        # "cache hit rate: X%" with a concrete number. See create_pipeline
        # (pipeline._embeddings) for how the metrics accumulator is wired
        # through.
        emb = getattr(pipeline, "_embeddings", None)
        if emb is not None and hasattr(emb, "get_metrics"):
            try:
                m = emb.get_metrics()
                logger.info("")
                logger.info("  Embedding cache (BatchedOllamaEmbeddings):")
                logger.info("    Total texts processed:   %8,d", m.get("total_texts", 0))
                logger.info("    Cache hit rate:          %7.1f%%", m.get("cache_hit_rate", 0.0))
                logger.info("    Cache hits / misses:     %6,d / %6,d",
                            m.get("cache_hits", 0), m.get("cache_misses", 0))
                logger.info("    Batch requests issued:   %8,d", m.get("batch_count", 0))
                logger.info("    Avg time per text:       %7.2f ms",
                            m.get("avg_time_per_text_ms", 0.0))
            except Exception as _exc:
                logger.debug("Embedding metrics unavailable: %s", _exc)

        # Run-provenance sidecar: pin the config hash + run parameters next to
        # the per-question JSONL so a results file is self-describing (which
        # config produced it, on what slice). Best-effort — a sidecar failure
        # must not lose the JSONL that was just written.
        try:
            meta_path = jsonl_path.with_suffix(".meta.json")
            meta = {
                "dataset": dataset,
                "model": model_name,
                "config_path": str(getattr(args, "config", None)
                                   or (_PROJECT_ROOT / "config" / "settings.yaml")),
                "config_sha256": getattr(args, "_config_hash", None),
                "n_questions": len(questions),
                "range": getattr(args, "range", None),
                "seed": getattr(args, "seed", None),
                "retrieval_only": retrieval_only,
                "vector_weight": args.vector_weight,
                "graph_weight": args.graph_weight,
                "answer_f1_threshold": f1_threshold,
                "timestamp": ts,
                "jsonl": jsonl_path.name,
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            logger.info("  Run metadata:         %s", meta_path)
        except OSError as exc:  # noqa: BLE001 — provenance is advisory
            logger.warning("Run-metadata sidecar failed: %s", exc)

        logger.info("")
        logger.info("  Per-question results: %s", jsonl_path)
        logger.info("=" * 70)

    finally:
        del pipeline
        gc.collect()

def _log_config_result(result: ConfigResult) -> None:
    """Log one ablation cell's headline metrics (lazy %-format)."""
    logger.info(
        "    EM: %.2f%%, F1: %.3f, SF-F1: %.3f, SF-Recall: %.2f%%, "
        "LLM-err: %.2f%%, Latency: %.0fms",
        result.exact_match * 100, result.f1_score, result.avg_sf_f1,
        result.sf_recall_rate * 100, result.llm_error_rate * 100,
        result.avg_time_ms,
    )


def cmd_ablation(args, config: Dict, store_manager: StoreManager):
    """Ablation study command."""
    
    logger.info("="*70)
    logger.info("ABLATION STUDY")
    logger.info("="*70)
    
    # Parse datasets
    if args.dataset == "all":
        datasets = AVAILABLE_DATASETS
    else:
        datasets = [d.strip() for d in args.dataset.split(",")]
    
    # Verify all datasets are ingested
    for dataset in datasets:
        if not store_manager.dataset_exists(dataset):
            logger.error("Dataset not ingested: %s", dataset)
            logger.error("Run: python -m src.thesis_evaluations.benchmark_datasets "
                         "ingest --dataset %s", dataset)
            return

    model_name = (
        getattr(args, "model", None)
        or config.get("llm", {}).get("model_name", _DEFAULT_MODEL_FALLBACK)
    )
    do_component = getattr(args, "component_ablation", False)
    logger.info("Datasets: %s", datasets)
    logger.info("Samples per dataset: %s", args.samples)
    logger.info("Model: %s", model_name)
    logger.info("Retrieval configs: %d", len(ABLATION_CONFIGS))
    logger.info("Component ablation: %s", do_component)
    logger.info("=" * 70)

    # Run ablation
    all_results: Dict[str, List] = {}
    used_run_names: List[str] = []

    for dataset in datasets:
        logger.info("\n%s", "=" * 70)
        logger.info("DATASET: %s", dataset.upper())
        logger.info("%s", "=" * 70)

        questions = store_manager.load_questions(dataset)
        if args.samples:
            questions = questions[:args.samples]

        logger.info("Questions: %d", len(questions))

        dataset_results = []

        retrieval_only = getattr(args, "retrieval_only", False)
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        jsonl_dir = _LANE_ABLATION_DIR
        jsonl_dir.mkdir(parents=True, exist_ok=True)

        # -- Retrieval-weight ablation ------------------------------------
        for lane in ABLATION_CONFIGS:
            run_name = f"{lane.name}_{model_name}"
            logger.info(
                "\n  [Retrieval] %s (v=%s, g=%s, bm25=%s, enable_bm25=%s)",
                run_name, lane.vector_weight, lane.graph_weight,
                lane.bm25_weight, lane.enable_bm25,
            )

            pipeline = None
            try:
                pipeline = create_pipeline(
                    dataset, config, store_manager,
                    vector_weight=lane.vector_weight,
                    graph_weight=lane.graph_weight,
                    bm25_weight=lane.bm25_weight,
                    enable_bm25=lane.enable_bm25,
                    model_name=model_name,
                )

                jsonl_path = jsonl_dir / (
                    f"{dataset}_{model_name.replace(':','-')}_"
                    f"{run_name.replace(':','-')}_{run_ts}.jsonl"
                )
                if jsonl_path.exists():
                    jsonl_path.unlink()

                result = evaluate_dataset(
                    dataset, questions, pipeline,
                    run_name, lane.vector_weight, lane.graph_weight,
                    jsonl_out=jsonl_path,
                    retrieval_only=retrieval_only,
                )

                if result:
                    dataset_results.append(result)
                    if run_name not in used_run_names:
                        used_run_names.append(run_name)
                    _log_config_result(result)

            except Exception as e:
                logger.error("    Failed: %s", e)
            finally:
                _close_pipeline(pipeline)

        # -- Component ablation (optional) --------------------------------
        if do_component:
            logger.info("\n  %s", "-" * 60)
            logger.info("  COMPONENT ABLATION (equal-weight hybrid)")
            logger.info("  %s", "-" * 60)
            for comp in COMPONENT_CONFIGS:
                run_name = f"comp_{comp.name}_{model_name}"
                logger.info(
                    "\n  [Component] %s (planner=%s, verifier=%s, iter=%d)",
                    run_name, comp.enable_planner, comp.enable_verifier,
                    comp.max_iterations,
                )

                pipeline = None
                try:
                    pipeline = create_pipeline(
                        dataset, config, store_manager,
                        vector_weight=_DEFAULT_VECTOR_WEIGHT,
                        graph_weight=_DEFAULT_GRAPH_WEIGHT,
                        model_name=model_name,
                        enable_planner=comp.enable_planner,
                        enable_verifier=comp.enable_verifier,
                        max_iterations=comp.max_iterations,
                    )

                    jsonl_path = jsonl_dir / (
                        f"{dataset}_{model_name.replace(':','-')}_"
                        f"{run_name.replace(':','-')}_{run_ts}.jsonl"
                    )
                    if jsonl_path.exists():
                        jsonl_path.unlink()

                    result = evaluate_dataset(
                        dataset, questions, pipeline,
                        run_name,
                        _DEFAULT_VECTOR_WEIGHT, _DEFAULT_GRAPH_WEIGHT,
                        jsonl_out=jsonl_path,
                        retrieval_only=retrieval_only,
                    )

                    if result:
                        dataset_results.append(result)
                        if run_name not in used_run_names:
                            used_run_names.append(run_name)
                        _log_config_result(result)

                except Exception as e:
                    logger.error("    Failed: %s", e)
                finally:
                    _close_pipeline(pipeline)

        all_results[dataset] = dataset_results

    # Save results
    ablation_results = AblationResults(
        timestamp=datetime.now().isoformat(),
        datasets=datasets,
        configs=used_run_names,
        results=all_results,
        config_sha256=getattr(args, "_config_hash", None),
    )
    
    output_dir = _LANE_ABLATION_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"ablation_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info("\nResults saved to: %s", output_path)

    print_ablation_table(ablation_results)

def _compute_mur(ds_results: List) -> Dict[str, Optional[float]]:
    """Marginal Utility Ratio: dF1 / dLatency(s) vs. the baseline config.

    Baseline = first entry (vector_only for the lane ablation, full for the
    component ablation). Sign convention, by case:
      * slower + better (dLat>0, dF1>0): positive ratio (F1 gained per
        extra second) -- the normal "is the cost worth it?" reading.
      * slower + worse  (dLat>0, dF1<0): negative ratio -- paying latency
        AND losing accuracy.
      * faster + better (dLat<0, dF1>0): +inf -- strictly dominates the
        baseline (rendered "inf" in the table).
      * faster + worse  (dLat<0, dF1<0): positive ratio (neg/neg) -- read
        as "accuracy lost per second SAVED"; NOT comparable in sign to the
        slower+better case, so interpret the cell alongside the raw F1 and
        latency rows, not in isolation.
    Returns None for the baseline row and for any config whose latency
    differs from the baseline by < 1 ms (not reliably measurable).
    """
    if not ds_results:
        return {}
    baseline = ds_results[0]
    mur: Dict[str, Optional[float]] = {baseline.config_name: None}
    for r in ds_results[1:]:
        delta_f1 = r.f1_score - baseline.f1_score
        delta_s = (r.avg_time_ms - baseline.avg_time_ms) / 1000.0
        if abs(delta_s) < 0.001:           # < 1ms difference -> not measurable
            mur[r.config_name] = None
        elif delta_s > 0:
            mur[r.config_name] = delta_f1 / delta_s
        else:                              # Faster than baseline
            mur[r.config_name] = float("inf") if delta_f1 > 0 else delta_f1 / delta_s
    return mur


def print_ablation_table(results: AblationResults):
    """Print formatted ablation results with MUR metric."""

    col_w = 14
    n_cols = len(results.configs)
    total_w = 16 + col_w * n_cols

    print("\n" + "="*total_w)
    print("ABLATION STUDY RESULTS")
    print("="*total_w)

    # Header
    header = f"{'Dataset':<16}"
    for cfg in results.configs:
        short = cfg[:col_w-1]
        header += f"{short:>{col_w}}"
    print(header)
    print("-" * total_w)

    for dataset in results.datasets:
        ds_results = results.results.get(dataset, [])

        if not ds_results:
            print(f"{dataset:<16} (no results)")
            continue

        mur_scores = _compute_mur(ds_results)

        # EM row
        row = f"{dataset + ' (EM)':<16}"
        for cfg_name in results.configs:
            r = next((r for r in ds_results if r.config_name == cfg_name), None)
            row += f"{r.exact_match:>{col_w}.1%}" if r else f"{'N/A':>{col_w}}"
        print(row)

        # F1 row
        row = f"{'  (F1)':<16}"
        for cfg_name in results.configs:
            r = next((r for r in ds_results if r.config_name == cfg_name), None)
            row += f"{r.f1_score:>{col_w}.3f}" if r else f"{'':>{col_w}}"
        print(row)

        # SF-F1 row (pipeline retrieval quality)
        row = f"{'  (SF-F1)':<16}"
        for cfg_name in results.configs:
            r = next((r for r in ds_results if r.config_name == cfg_name), None)
            row += f"{r.avg_sf_f1:>{col_w}.3f}" if r else f"{'':>{col_w}}"
        print(row)

        # SF-Recall (all-gold-retrieved rate)
        row = f"{'  (SF-Recall)':<16}"
        for cfg_name in results.configs:
            r = next((r for r in ds_results if r.config_name == cfg_name), None)
            row += f"{r.sf_recall_rate:>{col_w}.1%}" if r else f"{'':>{col_w}}"
        print(row)

        # EM | all-gold-retrieved  (model accuracy when retrieval is correct)
        row = f"{'  (EM|retr.ok)':<16}"
        for cfg_name in results.configs:
            r = next((r for r in ds_results if r.config_name == cfg_name), None)
            row += f"{r.retrieval_only_em:>{col_w}.1%}" if r else f"{'':>{col_w}}"
        print(row)

        # LLM error rate
        row = f"{'  (LLM-err)':<16}"
        for cfg_name in results.configs:
            r = next((r for r in ds_results if r.config_name == cfg_name), None)
            row += f"{r.llm_error_rate:>{col_w}.1%}" if r else f"{'':>{col_w}}"
        print(row)

        # Latency row
        row = f"{'  (ms)':<16}"
        for cfg_name in results.configs:
            r = next((r for r in ds_results if r.config_name == cfg_name), None)
            row += f"{r.avg_time_ms:>{col_w}.0f}" if r else f"{'':>{col_w}}"
        print(row)

        # MUR row  (dF1 / dLatency_s)
        row = f"{'  (MUR)':<16}"
        for cfg_name in results.configs:
            mur_val = mur_scores.get(cfg_name)
            if mur_val is None:
                row += f"{'-':>{col_w}}"
            elif mur_val == float("inf"):
                row += f"{'inf':>{col_w}}"
            else:
                row += f"{mur_val:>{col_w}.3f}"
        print(row)
        print()

    print("="*total_w)

    print("\nBEST CONFIGURATION PER DATASET (by F1):")
    for dataset in results.datasets:
        ds_results = results.results.get(dataset, [])
        if ds_results:
            best = max(ds_results, key=lambda r: r.f1_score)
            mur_val = _compute_mur(ds_results).get(best.config_name)
            mur_str = f", MUR={mur_val:.3f}" if mur_val is not None and mur_val != float("inf") else ""
            print(f"  {dataset:<15}: {best.config_name}  (F1={best.f1_score:.3f}{mur_str})")

    print("="*total_w + "\n")
    print("Legend:")
    print("  EM/F1        = final answer correctness vs. gold (LLM output).")
    print("  SF-F1        = supporting-fact F1: did the pipeline retrieve the right paragraphs?")
    print("  SF-Recall    = % of questions where ALL gold supporting paragraphs were retrieved.")
    print("  EM|retr.ok   = EM among questions where retrieval succeeded (model accuracy")
    print("                 conditioned on correct retrieval - isolates LLM capability).")
    print("  LLM-err      = % of questions where the LLM returned a [Error:...] sentinel")
    print("                 (timeout/connection/api). Distinguishes pipeline failure from model failure.")
    print("  MUR          = dF1 / dLatency(s) vs. baseline (first config). Higher = better trade-off.")
    print()

def cmd_demo(args, config: Dict, store_manager: StoreManager):
    """One-command, single-query demo: plan -> chunks -> answer -> envelope.

    Runs ONE question end-to-end through the full S_P -> S_N -> S_V pipeline and
    prints the stage breakdown a reviewer wants to see live: the Planner's
    query-type + sub-queries, the chunks the Navigator forwarded, the Verifier's
    answer + confidence, the wall-clock latency, and the peak resident-set size.
    The latency + RSS together are the live evidence for the <60 s / ~2 GB edge
    envelope claimed in the paper.

    Binds to one dataset's pre-built stores (default hotpotqa) exactly like the
    `evaluate` path, so it exercises the real retrieval stack, not a stub.
    """
    if not PIPELINE_AVAILABLE:
        print("[ERROR] AgentPipeline not available — check src/pipeline/.")
        return

    question = args.question
    dataset = args.dataset
    model_name = getattr(args, "model", None)

    # Peak-RSS sampler. psutil is in requirements but optional at runtime; the
    # demo degrades to "n/a" rather than failing if it is missing.
    try:
        import psutil  # type: ignore
        _proc = psutil.Process()
    except Exception:  # noqa: BLE001 — optional diagnostic only
        _proc = None

    def _rss_mb() -> Optional[float]:
        if _proc is None:
            return None
        try:
            return _proc.memory_info().rss / (1024 * 1024)
        except Exception:  # noqa: BLE001
            return None

    print("\n" + "=" * 70)
    print("EDGE-RAG DEMO  —  single query, full S_P -> S_N -> S_V trace")
    print("=" * 70)
    print(f"Dataset : {dataset}")
    print(f"Model   : {model_name or config.get('llm', {}).get('model_name', _DEFAULT_MODEL_FALLBACK)}")
    print(f"Question: {question}\n")

    rss_start = _rss_mb()
    pipeline = create_pipeline(
        dataset=dataset,
        config=config,
        store_manager=store_manager,
        model_name=model_name,
    )
    rss_after_load = _rss_mb()

    start = time.time()
    result = pipeline.process(question)
    elapsed_s = time.time() - start
    rss_peak = _rss_mb()

    # ── S_P: Planner ────────────────────────────────────────────────────────
    planner = getattr(result, "planner_result", {}) or {}
    qtype, hop_count, n_ents, pattern, preempt = _extract_planner_diagnostics(planner)
    sub_queries = []
    if isinstance(planner, dict):
        sub_queries = planner.get("sub_queries", []) or []
    print("-" * 70)
    print(f"[S_P] PLANNER   query_type={qtype or 'n/a'}  hops={hop_count}  "
          f"entities={n_ents}" + (f"  pattern={pattern}" if pattern else ""))
    for i, sq in enumerate(sub_queries):
        print(f"        sub-query {i}: {sq}")

    # ── S_N: Navigator ──────────────────────────────────────────────────────
    nav = getattr(result, "navigator_result", {}) or {}
    filtered = nav.get("filtered_context", []) if isinstance(nav, dict) else []
    titles = _retrieved_titles_for_chunks(filtered)
    print("-" * 70)
    print(f"[S_N] NAVIGATOR forwarded {len(filtered)} chunk(s) to the Verifier")
    for i, chunk in enumerate(filtered[:5]):
        title = titles[i] if i < len(titles) else "?"
        snippet = " ".join(str(chunk).split())[:160]
        print(f"        [{i}] ({title}) {snippet}{'…' if len(str(chunk)) > 160 else ''}")
    if len(filtered) > 5:
        print(f"        … +{len(filtered) - 5} more")

    # ── S_V: Verifier ───────────────────────────────────────────────────────
    verifier = getattr(result, "verifier_result", {}) or {}
    v_iters, v_verified, v_conf = _extract_verifier_diagnostics(verifier)
    print("-" * 70)
    print(f"[S_V] VERIFIER  confidence={v_conf or 'n/a'}  "
          f"iterations={v_iters}  all_verified={v_verified}")
    print(f"\nANSWER: {result.answer}")

    # ── Edge-envelope evidence ──────────────────────────────────────────────
    print("-" * 70)
    print(f"Latency : {elapsed_s:.2f} s  (target < 60 s)")
    if rss_peak is not None:
        line = f"Peak RSS: {rss_peak:.0f} MB  (target ~2 GB envelope)"
        if rss_start is not None and rss_after_load is not None:
            line += f"   [start {rss_start:.0f} -> after-load {rss_after_load:.0f} -> peak {rss_peak:.0f}]"
        print(line)
    else:
        print("Peak RSS: n/a  (install psutil to report memory)")
    print("=" * 70 + "\n")


def cmd_status(args, config: Dict, store_manager: StoreManager):
    """Show status command."""
    
    print("\n" + "="*50)
    print("DATASET STATUS")
    print("="*50)
    
    status = store_manager.get_status()

    for dataset in AVAILABLE_DATASETS:
        exists = status.get(dataset, False)
        mark = "[OK]" if exists else "[--]"

        if exists:
            questions = store_manager.load_questions(dataset)
            n_questions = len(questions) if questions else 0
            print(f"  {mark} {dataset:<15} ({n_questions} questions)")
        else:
            print(f"  {mark} {dataset:<15} (not ingested)")

    print("=" * 50)
    print("\nTo ingest: python -m src.thesis_evaluations.benchmark_datasets "
          "ingest --dataset <name>")
    print("=" * 50 + "\n")

def cmd_test(args, config: Dict, store_manager: StoreManager):
    """Self-test command."""

    print("\n" + "=" * 70)
    print("BENCHMARK_DATASETS.PY - SELF-TEST")
    print("=" * 70)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Module imports
    print("\nTest 1: Module Availability")
    print(f"  LangChain:     {'[OK]' if LANGCHAIN_AVAILABLE else '[--]'}")
    print(f"  Chunking:      {'[OK]' if CHUNKING_AVAILABLE else '[--]'}")
    print(f"  Storage:       {'[OK]' if STORAGE_AVAILABLE else '[--]'}")
    print(f"  AgentPipeline: {'[OK]' if PIPELINE_AVAILABLE else '[--]'}")

    # Test 2: Data classes
    print("\nTest 2: Data Classes")
    try:
        TestQuestion("t1", "Q?", "A", "test")
        Article("a1", "Title", "Text", ["S1"], "test")
        print("  [OK] TestQuestion, Article")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")
        tests_failed += 1

    # Test 3: Metrics
    print("\nTest 3: Evaluation Metrics")
    try:
        em = compute_exact_match("Paris", "paris")
        f1 = compute_f1("Paris France", "Paris")
        assert em is True
        assert 0 < f1 < 1
        print("  [OK] compute_exact_match, compute_f1")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")
        tests_failed += 1

    # Test 4: Loaders
    print("\nTest 4: Dataset Loaders")
    try:
        assert "hotpotqa" in LOADERS
        assert "2wikimultihop" in LOADERS
        assert "strategyqa" in LOADERS
        print("  [OK] All loaders registered")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")
        tests_failed += 1

    # Summary
    print("\n" + "=" * 70)
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 70 + "\n")

    return 0 if tests_failed == 0 else 1

# ============================================================================
# MAIN
# ============================================================================

# ---------------------------------------------------------------------------
# Reproducibility: frozen-config hashing (P0-T1)
# ---------------------------------------------------------------------------
# The headline evaluation is pinned to config/frozen_paper.yaml — a byte-copy
# of config/settings.yaml taken at the moment the reported numbers were
# produced. Every results block and summary.json stamps the SHA-256 of the
# config that was actually loaded, so a reviewer can confirm a given run used
# the frozen contract and not a drifted settings.yaml. The check is advisory
# (log + warn), never a hard abort: ablation lanes deliberately vary config.
FROZEN_CONFIG_PATH = _PROJECT_ROOT / "config" / "frozen_paper.yaml"


def _config_hash(config_path: Path) -> str:
    """Return the short SHA-256 (first 12 hex chars) of a config file.

    Hashes the raw file bytes — not the parsed dict — so the digest is stable,
    cheap, and independent of YAML-loader key ordering. Returns "<unhashable>"
    if the file cannot be read, so logging never raises in an eval run.
    """
    try:
        digest = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
        return digest[:12]
    except OSError as exc:  # noqa: BLE001 — diagnostics only, never fatal
        logger.debug("Config hash failed for %s: %s", config_path, exc)
        return "<unhashable>"


def log_frozen_config(config_path: Path) -> str:
    """Log the loaded config path + its hash, warn on drift from the frozen one.

    Returns the short hash string so callers can stamp it into a summary dict.
    Emits a prominent WARNING when ``config_path`` resolves to something other
    than ``frozen_paper.yaml`` and the two files' hashes differ — the signal a
    reviewer (or future-you) needs that a headline run was not on the frozen
    contract. Non-fatal by design: a mismatch is reported, never raised.
    """
    config_path = Path(config_path)
    h = _config_hash(config_path)
    logger.info("Config loaded: %s (sha256:%s)", config_path.name, h)
    if FROZEN_CONFIG_PATH.exists():
        frozen_h = _config_hash(FROZEN_CONFIG_PATH)
        if config_path.resolve() != FROZEN_CONFIG_PATH.resolve() and h != frozen_h:
            logger.warning(
                "Config drift: loaded %s (sha256:%s) != frozen_paper.yaml "
                "(sha256:%s). Headline numbers must be reproduced with "
                "--config config/frozen_paper.yaml.",
                config_path.name, h, frozen_h,
            )
    else:
        logger.warning(
            "frozen_paper.yaml not found at %s — reproducibility hash cannot "
            "be cross-checked. Create it: copy config/settings.yaml.",
            FROZEN_CONFIG_PATH,
        )
    return h


def load_config_file(config_path: Optional[Path] = None) -> Dict:
    """Load configuration from YAML.

    Routes through the unified ``_load_settings`` so the 35-key reproducibility
    validator runs here too. The hardcoded fallback dict below is kept as a
    last-resort emergency default if the path does not exist at all.
    """
    if config_path is None:
        config_path = _PROJECT_ROOT / "config" / "settings.yaml"
    if config_path.exists():
        return _load_settings(settings_path=config_path) or {}

    logger.warning("Config not found: %s, using defaults", config_path)
    return {
        "embeddings": {
            "model_name": "nomic-embed-text",
            "base_url": "http://localhost:11434",
            "embedding_dim": _NOMIC_EMBED_DIM,
        },
        "vector_store": {
            "similarity_threshold": _VECTOR_SIMILARITY_THRESHOLD_FALLBACK,
            "distance_metric": "cosine",
            "normalize_embeddings": True,
        },
        "rag": {
            "retrieval_mode": "hybrid",
            "vector_weight": _DEFAULT_VECTOR_WEIGHT,
            "graph_weight": _DEFAULT_GRAPH_WEIGHT,
        },
        "performance": {
            "batch_size": _EMBEDDING_BATCH_SIZE,
            "device": "cpu",
        },
    }

def main():
    setup_logging()
    _log_module_availability()

    parser = argparse.ArgumentParser(
        description="Multi-Dataset Benchmark System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # INGEST
    ingest_p = subparsers.add_parser("ingest", help="Ingest dataset(s)")
    ingest_p.add_argument("--dataset", "-d", type=str, default="hotpotqa")
    ingest_p.add_argument("--samples", "-n", type=int, default=500)
    ingest_p.add_argument("--chunk-sentences", type=int, default=_DEFAULT_CHUNK_SENTENCES)
    ingest_p.add_argument("--chunk-overlap", type=int, default=_DEFAULT_SENTENCE_OVERLAP)
    ingest_p.add_argument("--clear", action="store_true")
    ingest_p.add_argument("--chunks-only", action="store_true",
                          help="Only build chunks and export as JSON, no ingestion")
    ingest_p.add_argument("--no-coreference", action="store_true",
                          help="Disable per-article coreference resolution. "
                               "Default: ON (requires coreferee + en_core_web_md/lg, "
                               "silently skipped if not installed)")
    
    # EVALUATE
    eval_p = subparsers.add_parser("evaluate", help="Evaluate single dataset")
    eval_p.add_argument("--dataset", "-d", type=str, required=True)
    eval_p.add_argument(
        "--config", type=str, default=None, metavar="PATH",
        help="Config YAML to load (default: config/settings.yaml). Pass "
             "config/frozen_paper.yaml to reproduce the headline numbers; the "
             "loaded file's SHA-256 is logged into the results block.",
    )
    eval_p.add_argument(
        "--samples", "-n", type=int, default=20,
        help="Number of questions to sample randomly (default: 20). "
             "Ignored if --range is set.",
    )
    eval_p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for question sampling. Auto-generated and logged "
             "for reproducibility if not set.",
    )
    eval_p.add_argument(
        "--range", type=str, default=None, metavar="START-END",
        help="Deterministic question slice questions[START:END] (separators "
             "'-', '_' or ':'). Use '0-20' for the FIRST 20 questions (old "
             "default behaviour), or '10-30' for a defined band. Overrides "
             "--samples random sampling when set.",
    )
    eval_p.add_argument(
        "--answer-f1-threshold", type=float, default=None, metavar="F1",
        help="Token-F1 threshold for the Soft-EM correctness verdict "
             "(default from settings.yaml benchmark.answer_f1_threshold, 0.6). "
             "Set 1.0 to fall back to strict exact-match.",
    )
    # Default 1.0/1.0 = vanilla equal-weight RRF, matching config/settings.yaml.
    # A no-flag eval run therefore uses the same weights as the production path.
    eval_p.add_argument("--vector-weight", type=float, default=1.0)
    eval_p.add_argument("--graph-weight", type=float, default=1.0)
    eval_p.add_argument("--model", "-m", type=str, default=None,
                        help="Model name (e.g. phi3, llama3.2:3b). Default: from settings.yaml")
    eval_p.add_argument("--no-planner", action="store_true",
                        help="Skip S_P (ablation: no planner)")
    eval_p.add_argument("--no-verifier", action="store_true",
                        help="Skip S_V (ablation: no verifier)")
    eval_p.add_argument("--iterations", type=int, default=1,
                        help="Number of verifier iterations (1/2/3). Default: 1")
    eval_p.add_argument("--retrieval-only", action="store_true",
                        help="Skip the LLM entirely; only measure pipeline retrieval (SF-F1). "
                             "EM/F1 are forced to 0 in this mode. Useful for isolating "
                             "Planner/Navigator quality without LLM latency cost.")

    # ABLATION
    ablation_p = subparsers.add_parser("ablation", help="Run ablation study")
    ablation_p.add_argument("--dataset", "-d", type=str, default="all")
    ablation_p.add_argument(
        "--config", type=str, default=None, metavar="PATH",
        help="Config YAML to load (default: config/settings.yaml). The loaded "
             "file's SHA-256 is logged and written into summary.json.",
    )
    ablation_p.add_argument("--samples", "-n", type=int, default=100)
    ablation_p.add_argument("--model", "-m", type=str, default=None,
                            help="Model name (e.g. phi3, llama3.2:3b). Default: from settings.yaml")
    ablation_p.add_argument("--component-ablation", action="store_true",
                            help="Also run planner/verifier/iterations component ablation")
    ablation_p.add_argument("--retrieval-only", action="store_true",
                            help="Skip the LLM entirely across all ablation configs. Only "
                                 "supporting-fact metrics (SF-F1, SF-Recall) are meaningful. "
                                 "Much faster — useful for tuning retrieval-side config without "
                                 "paying LLM latency on every config.")
    
    # STATUS
    subparsers.add_parser("status", help="Show ingestion status")
    
    # TEST
    test_p = subparsers.add_parser("test", help="Run self-tests")
    test_p.add_argument("--verbose", "-v", action="store_true")

    # DEMO — single query, full S_P->S_N->S_V trace + envelope evidence
    demo_p = subparsers.add_parser(
        "demo",
        help="Run ONE query end-to-end and show the stage breakdown + envelope",
    )
    demo_p.add_argument("--question", "-q", type=str, required=True,
                        help="The question to answer end-to-end.")
    demo_p.add_argument("--dataset", "-d", type=str, default="hotpotqa",
                        help="Dataset whose pre-built stores to query (default: hotpotqa).")
    demo_p.add_argument("--config", type=str, default=None, metavar="PATH",
                        help="Config YAML to load (default: config/settings.yaml, "
                             "or $CONFIG_PATH). Pass config/frozen_paper.yaml to "
                             "match the reported contract.")
    demo_p.add_argument("--model", "-m", type=str, default=None,
                        help="Override the LLM model (default: from config).")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return

    # Reproducibility: honour an explicit --config (e.g. frozen_paper.yaml) on
    # the evaluate/ablation paths; default to settings.yaml otherwise. The
    # loaded file's SHA-256 is logged here and stamped into every results
    # summary downstream (see log_frozen_config). config_path is None for
    # sub-commands without the flag, which load the default settings.yaml.
    config_path = getattr(args, "config", None)
    if config_path:
        config = load_config_file(Path(config_path))
        args._config_hash = log_frozen_config(Path(config_path))
    else:
        config = load_config_file()
        # Still log + hash the default so every run records its provenance.
        if args.command in ("evaluate", "ablation"):
            args._config_hash = log_frozen_config(_PROJECT_ROOT / "config" / "settings.yaml")
    store_manager = StoreManager()
    
    if args.command == "ingest":
        cmd_ingest(args, config, store_manager)
    elif args.command == "evaluate":
        cmd_evaluate(args, config, store_manager)
    elif args.command == "ablation":
        cmd_ablation(args, config, store_manager)
    elif args.command == "status":
        cmd_status(args, config, store_manager)
    elif args.command == "test":
        cmd_test(args, config, store_manager)
    elif args.command == "demo":
        cmd_demo(args, config, store_manager)

if __name__ == "__main__":
    main()