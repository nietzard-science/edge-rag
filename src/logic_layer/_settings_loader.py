"""
Project-wide settings loader for ``config/settings.yaml``.

This module is the single canonical entry point for opening, parsing, and
validating the project settings file. All other modules — logic_layer,
data_layer, pipeline, evaluation scripts — go through ``_load_settings()``
instead of calling ``yaml.safe_load`` directly. Centralising the load:

  1. Resolves the YAML path relative to the project root, not the current
     working directory, so a script launched from any subdirectory behaves
     identically.
  2. Runs ``_validate_settings()`` automatically — a missing required key
     warns once and the caller falls back to its dataclass default. This
     is the reproducibility guard that catches the silent-default bug
     class (TECHNICAL_ARCHITECTURE.md §11.16.5).
  3. Returns ``{}`` on any I/O or parse error so callers never crash on
     a malformed YAML — they exercise their dataclass defaults instead.

Internal module — not part of the public API. Imported by planner.py,
navigator.py, controller.py, verifier.py, the ingestion pipeline, and the
evaluation scripts that need the full settings dict.

Exports
-------
    _load_settings(settings_path=None) -> Dict[str, Any]
        Load the canonical (or a caller-supplied) settings YAML, run the
        required-key validator, and return the parsed dict.
    _validate_settings(cfg) -> None
        Emit a ``WARNING`` for every missing required key in ``cfg``.
        Backed by the 37-key ``_REQUIRED_SETTINGS`` tuple covering every
        parameter that meaningfully affects EM/SF metrics (LLM context
        budget, embeddings, vector store, graph, RAG fusion + BM25,
        Navigator filter chain, Verifier validation thresholds, agent
        pipeline flags, entity extraction, ingestion, benchmark Soft-EM
        threshold).

Dependencies
------------
    PyYAML        (parse YAML)
    stdlib only otherwise (pathlib, logging, warnings, typing).

Last reviewed: 2026-05-26 (audit pass, project version 5.4).
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Project root = three levels up: this file -> logic_layer -> src -> root.
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


def _default_settings_path() -> Path:
    """Resolve the default settings YAML, honouring the ``CONFIG_PATH`` env var.

    Containerised / CI runs mount the config at an arbitrary path and point
    ``CONFIG_PATH`` at it (e.g. ``/app/config/frozen_paper.yaml``). Local runs
    set nothing and fall back to ``config/settings.yaml`` next to the source
    tree. Resolved per-call (not at import) so a test or subprocess can set the
    env var after this module is first imported.
    """
    env_path = os.environ.get("CONFIG_PATH")
    if env_path:
        return Path(env_path)
    return _PROJECT_ROOT / "config" / "settings.yaml"


# Back-compat module attribute. Some call sites import this symbol directly;
# it reflects the env at import time. Prefer ``_default_settings_path()`` for a
# live read.
_DEFAULT_SETTINGS_PATH: Path = _default_settings_path()


def _apply_env_overrides(cfg: Dict[str, Any]) -> None:
    """Apply environment-variable overrides to a loaded settings dict in place.

    ``OLLAMA_HOST`` (e.g. ``http://ollama:11434`` inside docker-compose)
    overrides BOTH the LLM and the embedding base URLs, because both clients
    read their endpoint from this dict (``llm.base_url`` in verifier.py /
    controller config, ``embeddings.base_url`` in embeddings.py). Defaulting is
    left to the dataclasses (``http://localhost:11434``) so a local run with no
    env var is unchanged.
    """
    ollama_host = os.environ.get("OLLAMA_HOST")
    if ollama_host:
        ollama_host = ollama_host.rstrip("/")
        cfg.setdefault("llm", {})["base_url"] = ollama_host
        cfg.setdefault("embeddings", {})["base_url"] = ollama_host
        logger.info("OLLAMA_HOST override applied: base_url=%s", ollama_host)


def _load_settings(settings_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load and validate ``config/settings.yaml``.

    Path is resolved relative to this file's location, so the function works
    regardless of the current working directory. Callers that load a
    non-default settings file (e.g., a held-out evaluation config) may pass
    ``settings_path`` explicitly. With no explicit path, the ``CONFIG_PATH``
    environment variable is honoured (container/CI mounts), falling back to
    ``config/settings.yaml``. After parsing, ``_apply_env_overrides`` rewrites
    the Ollama base URLs from ``OLLAMA_HOST`` (for the docker-compose sidecar).
    Returns ``{}`` (still env-overridden) if the file is missing or unparseable
    so callers can fall back to their dataclass defaults.

    Parameters
    ----------
    settings_path : Optional[Path]
        Override path to a settings YAML. ``None`` (default) → load the
        project's canonical ``config/settings.yaml``.

    Returns
    -------
    Dict[str, Any]
        Parsed settings dict, or ``{}`` on any I/O or parse error.
    """
    import yaml  # PyYAML is a required dependency (see requirements.txt)

    path = Path(settings_path) if settings_path is not None else _default_settings_path()
    if not path.exists():
        logger.warning(
            "_load_settings: settings.yaml not found at %s — "
            "config dataclass defaults will be used as emergency fallbacks.",
            path,
        )
        # An OLLAMA_HOST override must still reach the callers even when the YAML
        # is missing, so they can construct clients against the sidecar.
        empty: Dict[str, Any] = {}
        _apply_env_overrides(empty)
        return empty
    try:
        with open(path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        _apply_env_overrides(cfg)
        _validate_settings(cfg)
        return cfg
    except yaml.YAMLError as exc:
        logger.error(
            "_load_settings: failed to parse %s (%s) — "
            "config dataclass defaults will be used as emergency fallbacks.",
            path,
            exc,
        )
        empty = {}
        _apply_env_overrides(empty)
        return empty


# Keys that must be present for reproducible the paper's evaluation.
# Format: tuple of nested keys, e.g. ("llm", "temperature") -> cfg["llm"]["temperature"].
#
# Every parameter listed here MEANINGFULLY affects EM/SF metrics. If any one is
# missing from settings.yaml the system silently falls back to a hardcoded
# dataclass default, which can change results without notice (precise case: the
# 2026-05-24 audit found vector_store.top_k_vectors was silently 10 — the
# dataclass default — instead of the documented settings value 20, halving the
# vector retrieval funnel during evaluation). Growing this list is the
# reproducibility guard that catches that class of bug.
_REQUIRED_SETTINGS: Tuple[Tuple[str, ...], ...] = (
    # ── LLM / Verifier prompt context budget ──────────────────────────
    ("llm", "model_name"),
    ("llm", "base_url"),
    ("llm", "temperature"),
    ("llm", "max_tokens"),
    ("llm", "timeout"),
    ("llm", "max_docs"),
    ("llm", "max_context_chars"),
    ("llm", "max_chars_per_doc"),
    # ── Embeddings ────────────────────────────────────────────────────
    ("embeddings", "model_name"),
    # ── Vector store ──────────────────────────────────────────────────
    ("vector_store", "top_k_vectors"),
    ("vector_store", "similarity_threshold"),
    ("vector_store", "distance_metric"),
    # ── Graph ─────────────────────────────────────────────────────────
    ("graph", "max_hops"),
    ("graph", "top_k_entities"),
    ("graph", "hub_mention_cap"),
    ("graph", "hub_fanout_cap"),
    ("graph", "search_budget_seconds"),
    ("graph", "enable_hop3"),
    # ── RAG / retrieval fusion ────────────────────────────────────────
    ("rag", "retrieval_mode"),
    ("rag", "rrf_k"),
    ("rag", "cross_source_boost"),
    ("rag", "enable_bm25"),
    ("rag", "bm25_top_k"),
    # ── Navigator filter chain ────────────────────────────────────────
    ("navigator", "relevance_threshold_factor"),
    ("navigator", "redundancy_threshold"),
    ("navigator", "max_context_chunks"),
    ("navigator", "top_k_per_subquery"),
    ("navigator", "rrf_k"),
    ("navigator", "enable_reranker"),
    ("navigator", "contradiction_min_value"),
    # ── Verifier validation ───────────────────────────────────────────
    ("verifier", "entity_coverage_threshold"),
    ("verifier", "confidence_high_threshold"),
    # ── Agent pipeline ────────────────────────────────────────────────
    ("agent", "max_verification_iterations"),
    ("agent", "enable_verifier"),
    # ── Entity extraction (ingestion + query-time) ────────────────────
    ("entity_extraction", "gliner", "confidence_threshold"),
    # ── Ingestion ─────────────────────────────────────────────────────
    ("ingestion", "sentences_per_chunk"),
    # ── Benchmark (Soft-EM verdict threshold) ─────────────────────────
    ("benchmark", "answer_f1_threshold"),
)


def _validate_settings(cfg: Dict[str, Any]) -> None:
    """
    Warn when required settings.yaml keys are absent.

    A missing key means the system silently falls back to a hardcoded
    dataclass default — a reproducibility risk for the paper's evaluation. This
    function emits a ``WARNING`` (not an error) so missing keys never
    prevent the system from starting, but are always visible in the logs.

    Channel asymmetry
    -----------------
    Missing keys are reported on *both* the Python ``warnings`` channel
    (per-key, so a reviewer running `python -W error::UserWarning` can
    promote any missing key to a hard error during a reproducibility
    audit) AND the ``logging`` channel (also per-key). Parse errors and
    missing-file events in ``_load_settings`` go to ``logging`` only
    because they are unique, terminal events for that load — promoting
    them to ``warnings`` would not give a reviewer per-key resolution.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Parsed settings dict returned by ``yaml.safe_load()``.
    """
    import warnings as _warnings

    for key_path in _REQUIRED_SETTINGS:
        node: Any = cfg
        found = True
        for key in key_path:
            if not isinstance(node, dict) or key not in node:
                found = False
                break
            node = node[key]
        if not found:
            dotted = ".".join(key_path)
            # stacklevel=4 surfaces the warning at the user's call site
            # rather than inside this loader. Frame budget:
            #   1: warnings.warn itself
            #   2: this function (_validate_settings)
            #   3: _load_settings (the wrapper)
            #   4: caller of _load_settings  ← surfaced here
            _warnings.warn(
                f"settings.yaml: key {dotted!r} is missing — hardcoded "
                f"dataclass default will be used instead. Evaluation "
                f"results may not be reproducible. Check config/settings.yaml.",
                stacklevel=4,
            )
            logger.warning(
                "_validate_settings: required key %r absent from settings.yaml",
                dotted,
            )
