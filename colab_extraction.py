"""
Colab Phase-2 extractor: chunks_export.json -> extraction_results.json.

GPU-accelerated GLiNER NER + REBEL relation extraction over the chunks
produced by Phase 1, emitting the Phase-3-compatible JSON schema.

Pipeline (top-to-bottom cells)
------------------------------
  1   Setup + Drive mount + device detection
  1.5 Resumable checkpoint (atomic save, hash-bound, 5%-cadence)
  2   Dataclasses (ExtractedEntity / ExtractedRelation /
                   ChunkExtractionResult / ExtractionConfig)
  3   GLiNERExtractor (junk-filter, OOM-resilient batching)
  4   REBELExtractor (batched generate, log-prob calibrated confidence,
                      dynamic max-length, whole-word relevance match)
  5   Load chunks (local upload OR Drive copy)
  6   Run pipeline (with resume from checkpoint when compatible)
  7   Export Phase-3 schema (local + Drive copy)
  8   Spot-check (5 chunks with relations)
  9   Junk-filter sanity check
  10  Optional browser download

Critical configuration (fixed across the paper-evaluation datasets)
-------------------------------------------------------------------
  entity_types     OntoNotes-5 core, 9 GLiNER prompts -> 8 canonical types
                   (person, organization, location, city, country, date,
                   event, work of art, product). DO NOT add domain-
                   specific types (state, landmark, album, ...): they
                   transfer poorly and break reproducibility.
  _LABEL_MAP       Production map; every prompt above must resolve to an
                   OntoNotes-5 canonical type.
  _GLINER_JUNK     Stop-list filter applied at extraction time (pronouns,
                   nationality adjectives, ambiguous abbreviations).
  num_beams=_NUM_BEAMS, ner_threshold=_NER_THRESHOLD_DEFAULT,
  min_entities_for_re=_MIN_ENTITIES_FOR_RE_DEFAULT -- production.

Resumability contract
---------------------
The checkpoint is keyed by (sha1(chunks_export.json), sha1(config
signature)). Any change to either invalidates the prior checkpoint and
forces a fresh run, so a stale checkpoint cannot contaminate the output.

Exports (classes / functions referenced across cells)
-----------------------------------------------------
- ExtractedEntity, ExtractedRelation, ChunkExtractionResult
- ExtractionConfig.signature()       -- config-hash for checkpoint binding
- CheckpointSaver                    -- atomic, resumable 5%-cadence saver
- GLiNERExtractor.extract_batch(...) -- GPU NER with OOM-resilient batching
- REBELExtractor.extract_batch(...)  -- batched RE with log-prob confidence
- _is_junk(name) / _sha1_file / _sha1_text / _atomic_write_json
- _load_checkpoint / _checkpoint_compatible

Dependencies / Requirements
---------------------------
- Google Colab runtime with GPU (T4 recommended, ~12-18 min on ~9 400 chunks)
- gliner, transformers, torch, tqdm (pinned in cell 1 -- verify before
  final paper run)
- Google Drive mount for resumable checkpoint + Drive-resident output
- chunks_export.json at /content/chunks_export.json (uploaded) OR
                       /content/drive/MyDrive/chunks_export.json

References
----------
Murray & Chiang (2018). "Correcting Length Bias in Neural Machine
    Translation." WMT 2018. (length-normalised beam-score used for
    triplet log-prob confidence.)
Weischedel et al. (2013). OntoNotes Release 5.0. LDC2013T19. (entity-
    type schema for the prompt set.)

Usage
-----
  1. Colab -> Runtime -> Change runtime type -> GPU (T4).
  2. Upload chunks_export.json via the file sidebar OR place it at
     /content/drive/MyDrive/chunks_export.json before running cell 5.
  3. Run all cells top-to-bottom. On crash/disconnect, re-run all cells;
     the checkpoint resumes within ~5% of where it stopped.
  4. extraction_results.json is auto-saved to Drive; cell 10 download
     is optional.

Last reviewed: 2026-05-30 (audit pass, project version 5.4)
"""

# === CELL 1: Setup ========================================================

# Why: pin the dependency versions used for the paper-run extraction so a
# Colab runtime change does not silently re-rank the entity outputs. Verify
# these match `requirements_frozen.txt` before the final paper run.
!pip install -q gliner==0.2.13 transformers==4.46.3 torch==2.5.1 tqdm==4.67.1

import hashlib
import json
import logging
import math
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from google.colab import drive

# Mount Drive so checkpoint + final output survive Colab restarts.
drive.mount('/content/drive')

DRIVE_ROOT = "/content/drive/MyDrive"
CHECKPOINT_PATH = f"{DRIVE_ROOT}/extraction_checkpoint.json"
OUT_LOCAL = "extraction_results.json"
OUT_DRIVE = f"{DRIVE_ROOT}/extraction_results.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else " - SLOW (no GPU)"))
print(f"Checkpoint: {CHECKPOINT_PATH}")


# ---------------------------------------------------------------------------
# Module constants (algorithm-side hyperparameters)
# ---------------------------------------------------------------------------

# Why: recall-optimised NER threshold; the junk filter + Phase-3 cleanup
# absorb the false positives this admits.
_NER_THRESHOLD_DEFAULT = 0.15
# Why: GLiNER batch size that fits the T4 GPU envelope with a comfortable
# margin for OOM-retry halving (16 -> 8 -> 4 -> 2 -> 1).
_NER_BATCH_SIZE_DEFAULT = 16

# Why: log-prob fallback used when generate() does not return scores
# (older transformers versions). 0.5 mirrors the legacy sentinel so the
# fallback path produces the same ranking as the pre-calibration version.
_RE_FALLBACK_CONFIDENCE = 0.5
# Why: minimum log-prob-derived confidence below which a triplet is
# rejected. Set to the legacy sentinel so the new calibrated path keeps
# producing at least the candidates the old constant-confidence path did.
_RE_CONFIDENCE_THRESHOLD_DEFAULT = 0.5
# Why: REBEL batch size matched to T4 memory + dynamic padding cost.
_RE_BATCH_SIZE_DEFAULT = 8
# Why: skip REBEL when a chunk has <2 entities (no possible binary relation).
_MIN_ENTITIES_FOR_RE_DEFAULT = 2

# Why: beam-search width used in `_generate_with_logprob`. 5 matches the
# REBEL paper's reported beam width and the Phase-3 evaluation contract.
_NUM_BEAMS = 5
# Why: HF length_penalty=1.0 -> `sequences_scores` is the per-token average
# log-prob (Murray & Chiang 2018), the cleanest base for `math.exp(s)` to
# yield a calibrated probability.
_LENGTH_PENALTY = 1.0

# Why: REBEL input/output token budgets. Input tokenizer cap of 256 covers
# the 3-sentence chunk size; output max is dynamic (1.2x input, clipped to
# [64, 384]) so long chunks with many candidate triplets are not truncated.
_REBEL_INPUT_MAX_TOKENS = 256
_REBEL_OUTPUT_MIN_TOKENS = 64
_REBEL_OUTPUT_MAX_TOKENS = 384
_REBEL_OUTPUT_LEN_FACTOR = 1.2

# Why: clamp the exponentiated log-prob to [1e-4, 1.0]. The lower bound
# prevents catastrophic underflow (a triplet with prob 0 is still rankable
# above another triplet with prob 0).
_PROB_CLAMP_MIN = 1e-4
_PROB_CLAMP_MAX = 1.0

# Why: regex-fallback confidence when GLiNER fails to load (CPU diagnostic
# mode). Matches the legacy sentinel.
_REGEX_FALLBACK_CONFIDENCE = 0.5
# Why: entity-id is a 12-char MD5 truncation -- enough collision space for
# a single dataset's entity vocabulary (~50k entities). Phase 3 recomputes
# a 24-char SHA-256 id, so this length is purely for the Colab artefact.
_ENTITY_ID_LEN = 12
# Why: entities shorter than 3 chars are usually noise (single letters,
# abbreviations stripped of punctuation).
_MIN_ENTITY_NAME_LEN = 3

# Why: 5% checkpoint cadence keeps wall-clock overhead negligible (the JSON
# write is microseconds against minutes of GPU work).
_SAVE_EVERY_PCT_DEFAULT = 5.0
# Why: 1 MiB I/O buffer for the chunks-file sha1.
_FILE_SHA1_BLOCK_SIZE = 1 << 20

# Why: schema version of the extraction_results.json. Bump when the
# relation.confidence semantics change (e.g. constant sentinel -> real
# log-prob). Phase 3 reads this to decide whether to trust confidence.
_EXTRACTION_SCHEMA_VERSION = 2

# Why: spot-check chunk-text truncation in cell 8.
_SPOT_CHECK_TEXT_TRUNC = 120
# Why: rule width for the printed report sections.
_BAR_WIDTH = 70

# Why: seed for any stochastic component (currently none, but hardens
# reproducibility against transformers / tokenizers patch versions that
# may introduce internal randomness).
_RANDOM_SEED = 42

# Why: multi-word title-cased English noun-phrase pattern. Used by the
# GLiNER regex fallback when the model fails to load. Hoisted to module
# scope so it compiles once, not once per chunk.
_MULTIWORD_TITLECASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")


# === CELL 1.5: Checkpoint helpers (atomic save / load / hashing) =========
# Saves after every 5 % of progress. The checkpoint key is the
# (chunks_hash, config_hash) pair: any change to either invalidates the
# existing checkpoint so a fresh run cannot inherit stale partial state.
#
# File layout:
#   {
#     "version": 1,
#     "chunks_hash": "<sha1 of chunks_export.json>",
#     "config_hash": "<sha1 of frozen ExtractionConfig+entity_types>",
#     "phase": "ner" | "re" | "done",
#     "ner_done_chunk_ids": [...],
#     "ner_results": {chunk_id: [entity_dict, ...]},
#     "re_done_chunk_ids": [...],
#     "re_results":  {chunk_id: [relation_dict, ...]},
#     "ner_time_seconds_partial": float,
#     "re_time_seconds_partial":  float,
#     "saved_at": iso_timestamp
#   }
# Resumable design notes:
#   - NER and REBEL are saved separately because NER finishes first; if a
#     crash happens in REBEL we don't want to redo NER.
#   - We persist the FULL per-chunk entity list (not just the count) so
#     the REBEL phase can resume without re-running NER.

def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(_FILE_SHA1_BLOCK_SIZE), b""):
            h.update(blk)
    return h.hexdigest()

def _sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    """
    Write JSON atomically: write to a tempfile in the same directory, fsync,
    then os.replace() over the target. Crash-during-write cannot leave the
    target half-written.
    """
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".ckpt_", suffix=".json", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

def _load_checkpoint(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001 -- best-effort: corrupt checkpoint -> fresh run
        logger.warning("Failed to load checkpoint (%s); starting fresh.", exc)
        return None

def _checkpoint_compatible(ckpt: Dict[str, Any], chunks_hash: str, config_hash: str) -> bool:
    return (
        ckpt.get("version") == 1
        and ckpt.get("chunks_hash") == chunks_hash
        and ckpt.get("config_hash") == config_hash
    )

class CheckpointSaver:
    """
    Saves checkpoint after every 5 % of progress (configurable).

    Usage:
        cs = CheckpointSaver(path, chunks_hash, config_hash, total=N)
        cs.update_phase("ner")
        for i, chunk in enumerate(chunks):
            ... do work ...
            cs.record_ner(chunk_id, entity_dicts)
            cs.maybe_save(i + 1)
        cs.save_now()              # final save
    """
    def __init__(
        self,
        path: str,
        chunks_hash: str,
        config_hash: str,
        total: int,
        save_every_pct: float = _SAVE_EVERY_PCT_DEFAULT,
    ):
        self.path = path
        self.payload: Dict[str, Any] = {
            "version": 1,
            "chunks_hash": chunks_hash,
            "config_hash": config_hash,
            "phase": "ner",
            "ner_done_chunk_ids": [],
            "ner_results": {},
            "re_done_chunk_ids": [],
            "re_results": {},
            "ner_time_seconds_partial": 0.0,
            "re_time_seconds_partial": 0.0,
            "saved_at": "",
        }
        self.total = max(1, total)
        self.save_every = max(1, int(self.total * save_every_pct / 100.0))
        self._last_saved_at = 0

    def restore(self, existing: Dict[str, Any]) -> None:
        # Merge restored state in place; preserves the SAME chunks/config hashes.
        for k in (
            "phase",
            "ner_done_chunk_ids",
            "ner_results",
            "re_done_chunk_ids",
            "re_results",
            "ner_time_seconds_partial",
            "re_time_seconds_partial",
        ):
            if k in existing:
                self.payload[k] = existing[k]

    def update_phase(self, phase: str) -> None:
        self.payload["phase"] = phase

    def add_partial_seconds(self, key: str, delta_s: float) -> None:
        self.payload[key] = self.payload.get(key, 0.0) + float(delta_s)

    def record_ner(self, chunk_id: str, entity_dicts: List[Dict[str, Any]]) -> None:
        self.payload["ner_results"][chunk_id] = entity_dicts
        self.payload["ner_done_chunk_ids"].append(chunk_id)

    def record_re(self, chunk_id: str, relation_dicts: List[Dict[str, Any]]) -> None:
        self.payload["re_results"][chunk_id] = relation_dicts
        self.payload["re_done_chunk_ids"].append(chunk_id)

    def maybe_save(self, processed_count: int) -> None:
        if processed_count - self._last_saved_at >= self.save_every:
            self.save_now()
            self._last_saved_at = processed_count

    def save_now(self) -> None:
        self.payload["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            _atomic_write_json(self.path, self.payload)
            logger.info(
                "  Checkpoint saved at %s (phase=%s, ner=%d, re=%d)",
                self.payload["saved_at"],
                self.payload["phase"],
                len(self.payload["ner_done_chunk_ids"]),
                len(self.payload["re_done_chunk_ids"]),
            )
        except Exception as exc:  # noqa: BLE001 -- best-effort: I/O hiccup must not kill the run
            logger.warning("Checkpoint save failed: %s", exc)


# === CELL 2: Data classes (mirror src/data_layer/entity_extraction.py) ===

@dataclass
class ExtractedEntity:
    """Container for a single named entity."""
    entity_id: str
    name: str
    entity_type: str
    confidence: float
    mention_span: Tuple[int, int]
    source_chunk_id: str

    def to_dict(self) -> Dict[str, Any]:
        # Phase 3 reads both "type" and "entity_type" defensively, so the
        # original key name is kept here for backward compatibility with any
        # archived extraction_results.json files.
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "type": self.entity_type,
            "confidence": self.confidence,
            "mention_span": list(self.mention_span),
            "source_chunk_id": self.source_chunk_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExtractedEntity":
        return cls(
            entity_id=d["entity_id"],
            name=d["name"],
            entity_type=d.get("type", d.get("entity_type", "CONCEPT")),
            confidence=float(d.get("confidence", 0.0)),
            mention_span=tuple(d.get("mention_span", [0, 0])),
            source_chunk_id=d.get("source_chunk_id", ""),
        )


@dataclass
class ExtractedRelation:
    """Container for a subject-relation-object triple."""
    subject_entity: str
    relation_type: str
    object_entity: str
    confidence: float
    source_chunk_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        # Same backward-compatibility note as ExtractedEntity.to_dict.
        return {
            "subject": self.subject_entity,
            "relation": self.relation_type,
            "object": self.object_entity,
            "confidence": self.confidence,
            "source_chunks": self.source_chunk_ids,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExtractedRelation":
        return cls(
            subject_entity=d.get("subject", ""),
            relation_type=d.get("relation", ""),
            object_entity=d.get("object", ""),
            confidence=float(d.get("confidence", 0.0)),
            source_chunk_ids=list(d.get("source_chunks", [])),
        )


@dataclass
class ChunkExtractionResult:
    chunk_id: str
    text: str
    entities: List[ExtractedEntity]
    relations: List[ExtractedRelation]
    extraction_time_ms: float


@dataclass
class ExtractionConfig:
    """Mirrors src/data_layer/entity_extraction.py ExtractionConfig."""
    gliner_model: str = "urchade/gliner_small-v2.1"

    # ──────────────────────────────────────────────────────────────────────
    # FIXED: OntoNotes-5 core entity-type set
    # Reference: Weischedel et al. (2013). OntoNotes Release 5.0, LDC2013T19.
    #
    # 9 GLiNER prompts -> 8 canonical types via _LABEL_MAP:
    #   person                       -> PERSON
    #   organization                 -> ORGANIZATION
    #   location                     -> LOCATION
    #   city, country                -> GPE
    #   date                         -> DATE
    #   event                        -> EVENT
    #   work of art                  -> WORK_OF_ART
    #   product                      -> PRODUCT
    #
    # The multi-prompt expansion (city + country alongside location) gives
    # GLiNER higher recall than a single abstract label. All prompts collapse
    # to OntoNotes-5 canonical types in the graph.
    #
    # ⚠ DO NOT add domain-specific types (state, landmark, album, film, ...).
    # They transfer poorly across HotpotQA + 2WikiMultiHopQA + StrategyQA
    # and break paper reproducibility.
    # ──────────────────────────────────────────────────────────────────────
    entity_types: List[str] = field(default_factory=lambda: [
        "person",
        "organization",
        "location",
        "city",
        "country",
        "date",
        "event",
        "work of art",
        "product",
    ])
    ner_confidence_threshold: float = _NER_THRESHOLD_DEFAULT
    ner_batch_size: int = _NER_BATCH_SIZE_DEFAULT

    rebel_model: str = "Babelscape/rebel-large"
    # Why: relations whose log-prob-derived confidence falls below this
    # threshold are dropped in REBELExtractor.extract_batch (applied AFTER
    # the entity-relevance filter so the threshold acts on the calibrated
    # confidence, not the raw triplet count).
    re_confidence_threshold: float = _RE_CONFIDENCE_THRESHOLD_DEFAULT
    re_batch_size: int = _RE_BATCH_SIZE_DEFAULT
    min_entities_for_re: int = _MIN_ENTITIES_FOR_RE_DEFAULT
    selective_re: bool = True

    # Config-hash fields that invalidate the resume checkpoint when changed.
    def signature(self) -> str:
        return json.dumps({
            "gliner_model": self.gliner_model,
            "entity_types": self.entity_types,
            "ner_threshold": self.ner_confidence_threshold,
            "rebel_model": self.rebel_model,
            "re_threshold": self.re_confidence_threshold,
            "min_entities_for_re": self.min_entities_for_re,
            # Bumped to _EXTRACTION_SCHEMA_VERSION=2 when relation.confidence
            # switched from a constant 0.5 sentinel to a real log-prob.
            "schema_version": _EXTRACTION_SCHEMA_VERSION,
        }, sort_keys=True)


# === CELL 3: GLiNERExtractor (mirrors entity_extraction.py + junk filter) =

from gliner import GLiNER


# -------------------------------------------------------------------------
# JUNK FILTER
#
# Surface forms GLiNER misclassifies as PERSON or GPE at threshold 0.15.
# Mirror of DEFAULT_STOPLIST in src/data_layer/graph_quality.py -- applying
# the filter here at extraction time keeps extraction_results.json clean
# and saves the Phase-3 cleanup pass from re-doing the work.
# -------------------------------------------------------------------------
_GLINER_JUNK: frozenset = frozenset({
    # Pronouns mis-tagged as PERSON
    "he", "she", "they", "it", "we", "i", "you",
    "him", "her", "his", "hers", "us", "them", "their", "theirs",
    "this", "that", "these", "those",
    # Nationality adjectives mis-tagged as GPE
    "american", "british", "english", "german", "french", "italian",
    "spanish", "japanese", "chinese", "russian", "australian",
    "canadian", "european", "asian", "african", "indian", "korean",
    "mexican", "brazilian", "polish", "irish", "scottish", "welsh",
    "dutch", "swedish", "norwegian", "danish", "greek", "turkish",
    "arab", "arabic", "swiss", "austrian", "belgian",
    # Ambiguous abbreviations — use canonical "United States" / "United Kingdom"
    "u.s.", "us", "uk", "u.k.",
})


def _is_junk(name: str) -> bool:
    """True iff `name` (after trim+lowercase+trailing-period strip) is in the stop-list."""
    n = name.strip().lower().rstrip(".")
    return n in _GLINER_JUNK


class GLiNERExtractor:
    """
    GPU-resident GLiNER extractor. Mirrors entity_extraction.py with three
    deltas: device placement, the _is_junk() filter at the end of the
    extraction loop, and OOM-resilient batch fallback.
    """

    # ─────────────────────────────────────────────────────────────────────
    # FIXED: Full production label map.
    # Identical to GLINER_LABEL_MAP in src/data_layer/entity_types.py.
    # Every prompt in ExtractionConfig.entity_types MUST appear here so
    # _normalize_label returns an OntoNotes-5 canonical type, never a raw
    # uppercased fallback (which would create graph-side type drift).
    # ─────────────────────────────────────────────────────────────────────
    _LABEL_MAP: dict = {
        # People
        "person": "PERSON",
        "director": "PERSON", "actor": "PERSON", "politician": "PERSON",
        "scientist": "PERSON", "athlete": "PERSON",
        # Organisations
        "organization": "ORGANIZATION",
        "company": "ORGANIZATION", "studio": "ORGANIZATION",
        "institution": "ORGANIZATION",
        # Geopolitical / location
        "city": "GPE", "country": "GPE", "state": "GPE", "gpe": "GPE",
        "location": "LOCATION", "place": "LOCATION",
        "landmark": "LOCATION", "monument": "LOCATION", "building": "LOCATION",
        # Creative works
        "film": "WORK_OF_ART", "movie": "WORK_OF_ART", "book": "WORK_OF_ART",
        "album": "WORK_OF_ART", "song": "WORK_OF_ART",
        "work_of_art": "WORK_OF_ART", "work of art": "WORK_OF_ART",
        "award": "WORK_OF_ART", "prize": "WORK_OF_ART",
        # Temporal
        "date": "DATE", "year": "DATE", "time": "DATE",
        # Events / other
        "event": "EVENT",
        "product": "PRODUCT", "technology": "TECHNOLOGY",
    }

    @classmethod
    def _normalize_label(cls, label: str) -> str:
        return cls._LABEL_MAP.get(label.lower(), label.upper())

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info("Loading GLiNER model: %s", self.config.gliner_model)
            self.model = GLiNER.from_pretrained(self.config.gliner_model)
            self.model = self.model.to(DEVICE)
            logger.info("GLiNER model loaded on %s", DEVICE)
        except Exception as e:  # noqa: BLE001 -- best-effort: regex fallback if model fails
            logger.error("Failed to load GLiNER: %s", e)
            self.model = None

    def extract(self, text: str, chunk_id: str) -> List[ExtractedEntity]:
        """Single-text extraction with junk filter."""
        if self.model is None:
            return self._regex_extract(text, chunk_id)

        try:
            entities = self.model.predict_entities(
                text,
                self.config.entity_types,
                threshold=self.config.ner_confidence_threshold,
            )
            results = []
            for ent in entities:
                if _is_junk(ent["text"]):  # drop pronouns / nationality adjectives / abbreviations
                    continue
                canonical_type = self._normalize_label(ent["label"])
                results.append(ExtractedEntity(
                    entity_id=self._generate_entity_id(ent["text"], canonical_type),
                    name=ent["text"],
                    entity_type=canonical_type,
                    confidence=ent["score"],
                    mention_span=(ent["start"], ent["end"]),
                    source_chunk_id=chunk_id,
                ))
            return results
        except Exception as e:  # noqa: BLE001 -- best-effort: regex fallback if predict fails
            logger.error("GLiNER extraction failed: %s", e)
            return self._regex_extract(text, chunk_id)

    def _predict_batch_with_retry(
        self,
        batch_texts: List[str],
        batch_ids: List[str],
        initial_batch_size: int,
    ) -> List[List[Dict[str, Any]]]:
        """
        Run model.batch_predict_entities with adaptive halving on CUDA OOM.

        Strategy: try the full batch; on OOM, halve and recurse; on singleton
        OOM, fall back to per-chunk predict (which is slower but uses much
        less VRAM because GLiNER allocates per-input).
        """
        if not batch_texts:
            return []
        try:
            return self.model.batch_predict_entities(
                batch_texts,
                self.config.entity_types,
                threshold=self.config.ner_confidence_threshold,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch_texts) == 1:
                logger.warning("GLiNER OOM on singleton chunk %s - regex fallback", batch_ids[0])
                return [[]]
            mid = len(batch_texts) // 2
            return (
                self._predict_batch_with_retry(batch_texts[:mid], batch_ids[:mid], initial_batch_size)
                + self._predict_batch_with_retry(batch_texts[mid:], batch_ids[mid:], initial_batch_size)
            )
        except Exception as e:  # noqa: BLE001 -- best-effort: per-chunk fallback on non-OOM batch failure
            logger.error("GLiNER batch failed (non-OOM): %s - per-chunk fallback", e)
            results = []
            for text in batch_texts:
                try:
                    results.append(self.model.predict_entities(
                        text, self.config.entity_types,
                        threshold=self.config.ner_confidence_threshold,
                    ))
                except Exception:
                    results.append([])
            return results

    def extract_batch(
        self,
        texts: List[str],
        chunk_ids: List[str],
        checkpoint: Optional[CheckpointSaver] = None,
    ) -> List[List[ExtractedEntity]]:
        """
        Batch extraction with junk filter, OOM resilience, and per-5 %
        checkpointing. If `checkpoint` is supplied and already has results
        for some chunk_ids, those are restored from the checkpoint and not
        re-extracted.
        """
        if self.model is None:
            return [self._regex_extract(t, c) for t, c in zip(texts, chunk_ids)]

        # Restore any previously-done NER results.
        done_results: Dict[str, List[ExtractedEntity]] = {}
        if checkpoint is not None and checkpoint.payload.get("ner_results"):
            for cid, ent_dicts in checkpoint.payload["ner_results"].items():
                done_results[cid] = [ExtractedEntity.from_dict(d) for d in ent_dicts]
            logger.info("  Resuming NER: %d chunks already done.", len(done_results))

        all_results: List[List[ExtractedEntity]] = [None] * len(texts)  # type: ignore
        for i, cid in enumerate(chunk_ids):
            if cid in done_results:
                all_results[i] = done_results[cid]

        # Build the to-do list (preserving original order so progress aligns).
        pending_idx = [i for i in range(len(texts)) if all_results[i] is None]
        if not pending_idx:
            logger.info("  NER: nothing to do (all chunks restored from checkpoint).")
            return [r if r is not None else [] for r in all_results]

        batch_size = self.config.ner_batch_size
        t_start = time.time()
        processed_in_session = 0

        with tqdm(total=len(pending_idx), desc="GLiNER NER", unit="chunk") as pbar:
            for offset in range(0, len(pending_idx), batch_size):
                idx_batch = pending_idx[offset : offset + batch_size]
                batch_texts = [texts[i] for i in idx_batch]
                batch_ids = [chunk_ids[i] for i in idx_batch]

                batch_entities = self._predict_batch_with_retry(batch_texts, batch_ids, batch_size)

                for text_entities, chunk_id, i in zip(batch_entities, batch_ids, idx_batch):
                    results: List[ExtractedEntity] = []
                    for ent in text_entities:
                        if _is_junk(ent["text"]):
                            continue
                        canonical_type = self._normalize_label(ent["label"])
                        results.append(ExtractedEntity(
                            entity_id=self._generate_entity_id(ent["text"], canonical_type),
                            name=ent["text"],
                            entity_type=canonical_type,
                            confidence=ent["score"],
                            mention_span=(ent["start"], ent["end"]),
                            source_chunk_id=chunk_id,
                        ))
                    all_results[i] = results

                    if checkpoint is not None:
                        checkpoint.record_ner(chunk_id, [r.to_dict() for r in results])

                    processed_in_session += 1
                    pbar.update(1)
                    if checkpoint is not None:
                        checkpoint.maybe_save(len(done_results) + processed_in_session)

        if checkpoint is not None:
            checkpoint.add_partial_seconds("ner_time_seconds_partial", time.time() - t_start)
            checkpoint.save_now()

        return [r if r is not None else [] for r in all_results]

    def _regex_extract(self, text: str, chunk_id: str) -> List[ExtractedEntity]:
        """Last-resort regex fallback (multi-word capitalised phrases only)."""
        results = []
        for match in _MULTIWORD_TITLECASE_RE.finditer(text):
            name = match.group(1)
            if _is_junk(name):
                continue
            results.append(ExtractedEntity(
                entity_id=self._generate_entity_id(name, "CONCEPT"),
                name=name,
                entity_type="CONCEPT",
                confidence=_REGEX_FALLBACK_CONFIDENCE,
                mention_span=(match.start(), match.end()),
                source_chunk_id=chunk_id,
            ))
        return results

    @staticmethod
    def _generate_entity_id(name: str, entity_type: str) -> str:
        """
        `_ENTITY_ID_LEN`-char MD5 entity_id. Phase 3 of the local pipeline
        IGNORES this id and recomputes a 24-char SHA-256 id from
        canonical_form(name) + type, so this length is purely for the Colab
        artefact's internal consistency.
        """
        combined = f"{name.lower().strip()}:{entity_type}"
        return hashlib.md5(combined.encode()).hexdigest()[:_ENTITY_ID_LEN]


# === CELL 4: REBELExtractor (log-prob confidence + batched generate) =====

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class REBELExtractor:
    """GPU-resident REBEL relation extractor.

    Design:
      (a) Triplet confidence is the per-token sequence log-prob
          exponentiated to a probability in [_PROB_CLAMP_MIN,
          _PROB_CLAMP_MAX] -- NOT a constant sentinel. Falls back to
          _RE_FALLBACK_CONFIDENCE only when generate() does not return
          scores (older transformers versions).
      (b) generate() is batched (re_batch_size at a time) with dynamic
          padding so short chunks do not pay long-chunk cost.
      (c) Output max_length is dynamic: _REBEL_OUTPUT_LEN_FACTOR * input
          tokens, clipped to [_REBEL_OUTPUT_MIN_TOKENS,
          _REBEL_OUTPUT_MAX_TOKENS] (avoids triplet truncation on long
          chunks with many candidates).
      (d) Relevance filter uses whole-word regex matching: a short entity
          surface form cannot match as a substring inside an unrelated
          longer surface form.
    """

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = DEVICE
        self._load_model()

    def _load_model(self):
        try:
            logger.info("Loading REBEL model: %s", self.config.rebel_model)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.rebel_model)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.rebel_model)
            self.model.to(self.device)
            self.model.eval()
            logger.info("REBEL model loaded on %s", self.device)
        except Exception as e:  # noqa: BLE001 -- best-effort: skip RE entirely if model fails
            logger.error("Failed to load REBEL: %s", e)
            self.model = None

    @staticmethod
    def _whole_word_in(needle: str, haystack: str) -> bool:
        """True iff `needle` appears in `haystack` as a whole word (case-insensitive)."""
        if not needle or not haystack:
            return False
        return bool(re.search(rf"\b{re.escape(needle)}\b", haystack, re.IGNORECASE))

    def _parse_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """Parse REBEL's <triplet>/<subj>/<obj> output format."""
        triplets: List[Tuple[str, str, str]] = []
        try:
            parts = text.split("<triplet>")
            for part in parts:
                if "<subj>" in part and "<obj>" in part:
                    subj_parts = part.split("<subj>")
                    subject = subj_parts[0]
                    obj_rel_parts = subj_parts[1].split("<obj>")
                    object_ = obj_rel_parts[0]
                    relation = obj_rel_parts[1]
                    if subject and relation and object_:
                        triplets.append((subject.strip(), relation.strip(), object_.strip()))
        except Exception as e:  # noqa: BLE001 -- best-effort: parser must not crash the run
            logger.debug("Triplet parsing failed: %s", e)
        return triplets

    def _dynamic_max_len(self, batch_inputs) -> int:
        """`_REBEL_OUTPUT_LEN_FACTOR` x input length, clipped to
        [_REBEL_OUTPUT_MIN_TOKENS, _REBEL_OUTPUT_MAX_TOKENS]."""
        in_len = int(batch_inputs["input_ids"].shape[1])
        return min(
            _REBEL_OUTPUT_MAX_TOKENS,
            max(_REBEL_OUTPUT_MIN_TOKENS, int(in_len * _REBEL_OUTPUT_LEN_FACTOR)),
        )

    @torch.no_grad()
    def _generate_with_logprob(
        self,
        batch_texts: List[str],
    ) -> List[Tuple[str, float]]:
        """
        Run model.generate on a batch with `output_scores=True` and return
        a list of (decoded_text, sequence_probability) pairs in the same
        order as batch_texts. Sequence probability is exp(length-normalised
        sum of per-token log-probabilities), in [0, 1].

        Implementation:
          - Uses beam-search (num_beams=5) and returns the top-1 beam per
            input. We use `sequences_scores` which HuggingFace docs define
            as `log P(y | x) / |y|^length_penalty` (length-normalised log-
            prob; Murray & Chiang 2018). Default length_penalty=1.0 so
            sequences_scores == per-token average log-prob.
          - Exponentiating to a probability gives a calibrated confidence
            that compresses to ~[0.05, 0.95] in practice — exactly the
            ranking signal Phase 3 has been missing.
        """
        inputs = self.tokenizer(
            batch_texts,
            max_length=_REBEL_INPUT_MAX_TOKENS,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        gen = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=self._dynamic_max_len(inputs),
            num_beams=_NUM_BEAMS,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
            length_penalty=_LENGTH_PENALTY,
        )

        decoded = self.tokenizer.batch_decode(gen.sequences, skip_special_tokens=False)
        cleaned = [d.replace("<s>", "").replace("</s>", "").replace("<pad>", "") for d in decoded]

        # sequences_scores: shape [batch_size] when num_return_sequences=1 and
        # num_beams>1; it is the per-token average log-prob of the returned
        # beam. Map to a probability by exponentiating, clamped to
        # [_PROB_CLAMP_MIN, _PROB_CLAMP_MAX].
        probs: List[float] = []
        if hasattr(gen, "sequences_scores") and gen.sequences_scores is not None:
            for s in gen.sequences_scores.detach().cpu().tolist():
                p = math.exp(min(0.0, float(s)))
                probs.append(max(_PROB_CLAMP_MIN, min(_PROB_CLAMP_MAX, p)))
        else:
            probs = [_RE_FALLBACK_CONFIDENCE] * len(cleaned)

        return list(zip(cleaned, probs))

    def extract_batch(
        self,
        texts: List[str],
        entities_per_text: List[List[ExtractedEntity]],
        chunk_ids: List[str],
        checkpoint: Optional[CheckpointSaver] = None,
    ) -> List[List[ExtractedRelation]]:
        """
        Batched REBEL extraction with log-prob confidence + 5 % checkpoint
        cadence. Restores already-done chunks from the checkpoint if present.

        Returns: list of relations per input chunk, in the same order as
        `texts`. Chunks with fewer than `min_entities_for_re` entities
        contribute an empty list (no REBEL call).
        """
        if self.model is None:
            return [[] for _ in texts]

        # Restore any previously-done RE results.
        done_re: Dict[str, List[ExtractedRelation]] = {}
        if checkpoint is not None and checkpoint.payload.get("re_results"):
            for cid, rel_dicts in checkpoint.payload["re_results"].items():
                done_re[cid] = [ExtractedRelation.from_dict(d) for d in rel_dicts]
            logger.info("  Resuming RE: %d chunks already done.", len(done_re))

        all_results: List[List[ExtractedRelation]] = [None] * len(texts)  # type: ignore
        for i, cid in enumerate(chunk_ids):
            if cid in done_re:
                all_results[i] = done_re[cid]

        # Build the to-do list: only chunks with >= min_entities_for_re AND
        # not yet in done_re. Chunks with fewer entities get an empty list.
        pending_idx: List[int] = []
        for i, (cid, ents) in enumerate(zip(chunk_ids, entities_per_text)):
            if all_results[i] is not None:
                continue
            if len(ents) < self.config.min_entities_for_re:
                all_results[i] = []
                if checkpoint is not None:
                    # Record empty so a resume doesn't recount them.
                    checkpoint.record_re(cid, [])
                continue
            pending_idx.append(i)

        if not pending_idx:
            logger.info("  RE: nothing to do (all eligible chunks restored from checkpoint).")
            if checkpoint is not None:
                checkpoint.save_now()
            return [r if r is not None else [] for r in all_results]

        re_batch = max(1, self.config.re_batch_size)
        t_start = time.time()
        processed_in_session = 0

        with tqdm(total=len(pending_idx), desc="REBEL RE", unit="chunk") as pbar:
            for offset in range(0, len(pending_idx), re_batch):
                idx_batch = pending_idx[offset : offset + re_batch]
                batch_texts = [texts[i] for i in idx_batch]
                batch_ids = [chunk_ids[i] for i in idx_batch]
                batch_ents = [entities_per_text[i] for i in idx_batch]

                # ── Generate (batched). Fall back to per-chunk on OOM. ──
                try:
                    decoded_with_probs = self._generate_with_logprob(batch_texts)
                except torch.cuda.OutOfMemoryError:
                    logger.warning("REBEL OOM on batch size %d - per-chunk fallback", len(batch_texts))
                    torch.cuda.empty_cache()
                    decoded_with_probs = []
                    for one_text in batch_texts:
                        try:
                            decoded_with_probs.extend(self._generate_with_logprob([one_text]))
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            decoded_with_probs.append(("", 0.0))

                # -- Parse, filter, persist. --
                for (raw, seq_prob), chunk_id, ents, i in zip(
                    decoded_with_probs, batch_ids, batch_ents, idx_batch
                ):
                    triplets = self._parse_triplets(raw)
                    # Build entity_names ONCE per chunk.
                    entity_names = {
                        e.name for e in ents
                        if len(e.name) >= _MIN_ENTITY_NAME_LEN
                    }
                    rels: List[ExtractedRelation] = []
                    for subj, rel, obj in triplets:
                        # Why: drop low-confidence triplets BEFORE adding
                        # them to the output. Applied after relevance so
                        # the threshold acts on the calibrated log-prob,
                        # not the raw triplet count.
                        if float(seq_prob) < self.config.re_confidence_threshold:
                            continue
                        is_relevant = any(
                            self._whole_word_in(name, subj) or self._whole_word_in(name, obj)
                            for name in entity_names
                        )
                        if not is_relevant:
                            continue
                        rels.append(ExtractedRelation(
                            subject_entity=subj.strip(),
                            relation_type=rel.strip(),
                            object_entity=obj.strip(),
                            confidence=float(seq_prob),
                            source_chunk_ids=[chunk_id],
                        ))
                    all_results[i] = rels

                    if checkpoint is not None:
                        checkpoint.record_re(chunk_id, [r.to_dict() for r in rels])

                    processed_in_session += 1
                    pbar.update(1)
                    if checkpoint is not None:
                        checkpoint.maybe_save(len(done_re) + processed_in_session)

        if checkpoint is not None:
            checkpoint.add_partial_seconds("re_time_seconds_partial", time.time() - t_start)
            checkpoint.save_now()

        return [r if r is not None else [] for r in all_results]


# === CELL 5: Load chunks =================================================
# Accept either an upload at /content/chunks_export.json OR a Drive copy at
# /content/drive/MyDrive/chunks_export.json — whichever is found first.

CHUNKS_LOCAL = "chunks_export.json"
CHUNKS_DRIVE = f"{DRIVE_ROOT}/chunks_export.json"
CHUNKS_PATH = CHUNKS_LOCAL if os.path.exists(CHUNKS_LOCAL) else CHUNKS_DRIVE
if not os.path.exists(CHUNKS_PATH):
    raise FileNotFoundError(
        f"chunks_export.json not found at {CHUNKS_LOCAL} or {CHUNKS_DRIVE}. "
        "Upload it via the Colab file sidebar or copy it to your Drive."
    )

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks_data = json.load(f)

texts = [c["text"] for c in chunks_data]
chunk_ids = [c["metadata"]["chunk_id"] for c in chunks_data]
print(f"Loaded {len(chunks_data)} chunks from {CHUNKS_PATH}")


# === CELL 6: Run the pipeline (with resume) ==============================

# Why: pin every stochastic seed before any model load. Beam search is
# deterministic given identical model state, but tokenizer / transformers
# patch versions occasionally introduce internal randomness; seeding here
# hardens reproducibility against that drift.
torch.manual_seed(_RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_RANDOM_SEED)

config = ExtractionConfig()  # uses the FIXED OntoNotes-5 9-prompt list

print("Configuration:")
print(f"  GLiNER model:       {config.gliner_model}")
print(f"  Entity types ({len(config.entity_types)}):")
for t in config.entity_types:
    print(f"    - {t}")
print(f"  NER threshold:      {config.ner_confidence_threshold}")
print(f"  REBEL model:        {config.rebel_model}")
print(f"  REBEL batch size:   {config.re_batch_size}")
print(f"  Junk filter:        ON ({len(_GLINER_JUNK)} stop-list entries)")
print()

# Compute the keys that bind a checkpoint to (data + config).
chunks_hash = _sha1_file(CHUNKS_PATH)
config_hash = _sha1_text(config.signature())
print(f"  chunks_hash: {chunks_hash[:12]}…")
print(f"  config_hash: {config_hash[:12]}…")

# Load any existing checkpoint and decide whether to resume.
existing_ckpt = _load_checkpoint(CHECKPOINT_PATH)
if existing_ckpt is None:
    print("  No prior checkpoint -- fresh run.")
elif not _checkpoint_compatible(existing_ckpt, chunks_hash, config_hash):
    print("  Checkpoint exists but is INCOMPATIBLE (chunks or config changed).")
    print("  -> Discarding old checkpoint and starting fresh.")
    existing_ckpt = None
else:
    print(
        f"  Resuming from {existing_ckpt.get('saved_at', '?')}: "
        f"phase={existing_ckpt.get('phase')}, "
        f"NER {len(existing_ckpt.get('ner_done_chunk_ids', []))}/{len(texts)} done, "
        f"RE {len(existing_ckpt.get('re_done_chunk_ids', []))} chunks recorded."
    )

cs = CheckpointSaver(
    path=CHECKPOINT_PATH,
    chunks_hash=chunks_hash,
    config_hash=config_hash,
    total=len(texts),
    save_every_pct=5.0,
)
if existing_ckpt is not None:
    cs.restore(existing_ckpt)

ner = GLiNERExtractor(config)
re_ext = REBELExtractor(config)

# ── NER ──────────────────────────────────────────────────────────────────
cs.update_phase("ner")
print(f"\nGLiNER NER on {len(texts)} chunks (save every 5 %)...")
ner_start = time.time()
all_entities = ner.extract_batch(texts, chunk_ids, checkpoint=cs)
ner_time = (time.time() - ner_start) + cs.payload.get("ner_time_seconds_partial", 0.0)
total_ents = sum(len(e) for e in all_entities)
print(f"  -> {total_ents} entities in {ner_time:.1f}s (incl. resumed time)")

# ── REBEL RE ─────────────────────────────────────────────────────────────
cs.update_phase("re")
print(f"\nREBEL RE (selective: >= {config.min_entities_for_re} entities, "
      f"log-prob confidence, batched generate)...")
re_start = time.time()
all_relations = re_ext.extract_batch(texts, all_entities, chunk_ids, checkpoint=cs)
re_time = (time.time() - re_start) + cs.payload.get("re_time_seconds_partial", 0.0)
total_rels = sum(len(r) for r in all_relations)
re_calls = sum(1 for ents in all_entities if len(ents) >= config.min_entities_for_re)
print(f"  -> {total_rels} relations in {re_time:.1f}s ({re_calls} REBEL calls)")

cs.update_phase("done")
cs.save_now()


# === CELL 7: Export (Phase 3-compatible schema) ==========================

extraction_results = []
for text, chunk_id, entities, relations in zip(texts, chunk_ids, all_entities, all_relations):
    for rel in relations:
        rel.source_chunk_ids = [chunk_id]
    extraction_results.append(ChunkExtractionResult(
        chunk_id=chunk_id,
        text=text,
        entities=entities,
        relations=relations,
        extraction_time_ms=0.0,
    ))

unique_entity_ids = {e.entity_id for ents in all_entities for e in ents}

# Per-canonical-type histogram (so the metadata records what GLiNER actually
# produced after the junk filter).
type_counts: Dict[str, int] = {}
for ents in all_entities:
    for e in ents:
        type_counts[e.entity_type] = type_counts.get(e.entity_type, 0) + 1

# Confidence distribution summary for sanity-checking the log-prob fix.
rel_confidences = [rel.confidence for rels in all_relations for rel in rels]
if rel_confidences:
    rel_conf_min = min(rel_confidences)
    rel_conf_max = max(rel_confidences)
    rel_conf_mean = sum(rel_confidences) / len(rel_confidences)
    rel_confidence_is_constant = (rel_conf_min == rel_conf_max)
else:
    rel_conf_min = rel_conf_max = rel_conf_mean = 0.0
    rel_confidence_is_constant = True

stats = {
    "total_chunks": len(chunks_data),
    "total_entities": total_ents,
    "unique_entities": len(unique_entity_ids),
    "total_relations": total_rels,
    "chunks_with_entities": sum(1 for e in all_entities if e),
    "chunks_with_relations": sum(1 for r in all_relations if r),
    "ner_calls": len(texts),
    "re_calls": re_calls,
    "ner_time_seconds": round(ner_time, 1),
    "re_time_seconds": round(re_time, 1),
    "device": DEVICE,
    "gliner_model": config.gliner_model,
    "rebel_model": config.rebel_model,
    "ner_threshold": config.ner_confidence_threshold,
    "re_threshold": config.re_confidence_threshold,
    "num_beams": _NUM_BEAMS,
    "min_entities_for_re": config.min_entities_for_re,
    "entity_types": config.entity_types,         # FIXED OntoNotes-5 list
    "junk_filter_active": True,
    "junk_filter_size": len(_GLINER_JUNK),
    "type_distribution": type_counts,
    "relation_confidence": {
        "min":  round(rel_conf_min, 4),
        "max":  round(rel_conf_max, 4),
        "mean": round(rel_conf_mean, 4),
        "is_constant": rel_confidence_is_constant,
    },
    "rebel_confidence_is_constant": rel_confidence_is_constant,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "checkpoint_path": CHECKPOINT_PATH,
    "chunks_hash": chunks_hash,
    "config_hash": config_hash,
}

output = {
    "metadata": stats,
    "results": [
        {
            "chunk_id": r.chunk_id,
            "entities": [e.to_dict() for e in r.entities],
            "relations": [rel.to_dict() for rel in r.relations],
            "extraction_time_ms": r.extraction_time_ms,
        }
        for r in extraction_results
    ],
}

# Write locally AND to Drive so a runtime disconnect after the run does not
# lose the result.
for path in (OUT_LOCAL, OUT_DRIVE):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001 -- best-effort: a single write target may fail
        logger.warning("Could not write %s: %s", path, exc)

mb = os.path.getsize(OUT_LOCAL) / 1024 / 1024 if os.path.exists(OUT_LOCAL) else 0.0

print(f"\n{'=' * _BAR_WIDTH}")
print(f"EXPORT: {OUT_LOCAL} ({mb:.1f} MB)  +  copied to {OUT_DRIVE}")
print(f"{'=' * _BAR_WIDTH}")
print(f"  Chunks:                {stats['total_chunks']}")
print(f"  Entities:              {stats['total_entities']} ({stats['unique_entities']} unique)")
print(f"  Relations:             {stats['total_relations']}")
print(f"  Chunks w/ entities:    {stats['chunks_with_entities']}")
print(f"  Chunks w/ relations:   {stats['chunks_with_relations']}")
print(f"  NER:                   {ner_time:.1f}s ({stats['ner_calls']} calls)")
print(f"  RE:                    {re_time:.1f}s ({stats['re_calls']} calls)")
print(f"  Total:                 {ner_time + re_time:.1f}s")
print()
print("  Type distribution (after junk filter):")
for t in sorted(type_counts.keys()):
    print(f"    {t:<20} {type_counts[t]:>7,}")
print()
print(f"  REBEL confidence:      min={rel_conf_min:.3f}  max={rel_conf_max:.3f}  "
      f"mean={rel_conf_mean:.3f}  constant={rel_confidence_is_constant}")
if rel_confidence_is_constant:
    print("    WARNING: confidence collapsed to a constant -- log-prob path may have")
    print("             fallen back to the sentinel. Inspect _generate_with_logprob.")
print(f"{'=' * _BAR_WIDTH}")


# === CELL 8: Spot-check (first 5 chunks with relations) ==================

print("\n" + "-" * _BAR_WIDTH)
print("SPOT-CHECK (first 5 chunks with relations)")
print("-" * _BAR_WIDTH)

shown = 0
for r in extraction_results:
    if r.relations:
        print(f"\nChunk {r.chunk_id}: {r.text[:_SPOT_CHECK_TEXT_TRUNC]}...")
        print(f"  Entities: {[(e.name, e.entity_type) for e in r.entities]}")
        for rel in r.relations:
            print(f"  -> {rel.subject_entity} --[{rel.relation_type}|p={rel.confidence:.3f}]"
                  f"--> {rel.object_entity}")
        shown += 1
        if shown >= 5:
            break

if shown == 0:
    print("  No relations found - inspect the entity extraction!")


# === CELL 9: Sanity check -- junk should be absent =======================

print("\n" + "-" * _BAR_WIDTH)
print("JUNK-FILTER SANITY CHECK")
print("-" * _BAR_WIDTH)

# Confirm none of the stop-list strings appear in the output.
all_names = [e.name for ents in all_entities for e in ents]
leaked = [n for n in all_names if _is_junk(n)]
if leaked:
    print(f"  WARNING: {len(leaked)} junk entities slipped through:")
    for n in set(leaked):
        print(f"    - {n!r}")
else:
    print(f"  OK - no pronouns or nationality adjectives in {len(all_names)} extracted entities.")


# === CELL 10: Download (optional -- file is already on Drive) ============

from google.colab import files
try:
    files.download(OUT_LOCAL)
except Exception as exc:  # noqa: BLE001 -- best-effort: browser download is optional
    print(f"  (Browser download skipped: {exc}.  File is on Drive at {OUT_DRIVE}.)")
