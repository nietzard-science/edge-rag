"""
Full-pipeline trace diagnostic.

Runs the complete S_P -> S_N -> S_V pipeline for a single question and
prints the inputs / outputs / latencies of every stage. No production
code is modified; all observability is installed via runtime monkey-
patching on the live `AgentPipeline` and its sub-agents.

Two modes of gold-evidence tracking are supported:
  (A) supporting-facts mode (preferred): the gold paragraph TITLES (from
      the dataset's supporting_facts schema) are the authoritative
      "is the gold evidence still present?" signal. A chunk is gold iff
      its source-doc title matches; relatives' biographies that merely
      mention the answer string never produce false positives.
  (B) fallback mode: no supporting_facts -> heuristic match of the
      answer string's content words against chunk text. Marked with a
      one-time warning so the operator knows it is lossy.

Used to localise where a wrong answer lost the gold evidence: each
filter stage prints a GOLD OK / GOLD PARTIAL / GOLD LOST marker so the
operator can read down the trace and stop at the first regression.

Exports
-------
- main()                      -- CLI entry point
- build_pipeline(cfg, dataset)-- construct AgentPipeline + sub-agents
- patch_planner(planner)      -- attach S_P observability hook
- patch_retriever(retriever)  -- attach HybridRetriever hook
- patch_navigator(navigator)  -- attach S_N RRF / filter / reranker hooks
- patch_verifier(verifier)    -- attach S_V prompt / answer hook
- patch_controller_bridge()   -- attach bridge-entity-extraction hook
- print_call_trace()          -- post-run sys.settrace dump
- load_question(idx, dataset) -- read questions.json for the given dataset

Dependencies / Requirements
---------------------------
- src.pipeline.AgentPipeline + sub-agents (Planner / Navigator /
  Verifier / HybridRetriever)
- src.data_layer.{embeddings, storage, hybrid_retriever}
- src.logic_layer (create_planner, create_verifier, AgenticController)
- ollama server at config.embeddings.base_url (unless --skip-llm)
- LanceDB + KuzuDB at data/<dataset>/{vector,graph}
- pyyaml

Usage (single line; -X utf8 required on Windows / PowerShell):
    python -X utf8 diagnose_verbose.py --idx 0
    python -X utf8 diagnose_verbose.py --idx 5 --skip-llm
    python -X utf8 diagnose_verbose.py --question "<text>" --gold "<answer>"
    python -X utf8 diagnose_verbose.py --idx 0 --trace-calls
    python -X utf8 diagnose_verbose.py --idx 0 --no-color > trace.txt

Flags
-----
    --idx N           Question N from data/<dataset>/questions.json (default: 0)
    --dataset NAME    Dataset name (default: hotpotqa)
    --question TEXT   Custom free-text question instead of questions.json
    --gold TEXT       Gold answer for stage tracking
    --gold-docs TEXT  Comma-separated gold paragraph titles for --question mode
    --skip-llm        Skip the Verifier (retrieval debugging, no LLM wait)
    --trace-calls     sys.settrace: print every function call in src/
    --no-color        Plain output (for pipe / file capture)

Last reviewed: 2026-05-30 (audit pass, project version 5.4)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from textwrap import wrap

# Why: project-root bootstrap so the `from src.*` imports in build_pipeline
# resolve from any cwd. This file lives in tools/, so the project root is the
# parent of the parent.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# Why: emergency fallbacks if settings.yaml is missing. Production runs
# read these from config; the diagnostic still functions on a bare repo.
_OLLAMA_DEFAULT_MODEL = "nomic-embed-text"
_OLLAMA_DEFAULT_URL = "http://localhost:11434"
_DEFAULT_DATASET = "hotpotqa"

# Why: Soft-EM (token-F1) threshold fallback when settings.yaml does not
# declare `benchmark.answer_f1_threshold`. Matches the benchmark default.
_SOFT_EM_F1_THRESHOLD_FALLBACK = 0.6

# Why: stopwords excluded from the gold-answer content-word check
# (mode B / fallback gold tracking). Closed-class English function words
# that never carry the answer.
_GOLD_STOPWORDS = frozenset({
    "that", "this", "with", "from", "have", "been", "were", "will", "would",
})
# Why: minimum content-word length for the gold-answer match. Anything
# shorter is too noisy (single-letter abbreviations, common 3-letter words).
_GOLD_MIN_WORD_LEN = 4

# Why: rule widths for the printed report -- 72 was the original section
# bar, 68 the subsection, 60 the inline "..." rule.
_BAR_WIDTH = 72
_SECTION_RULE_WIDTH = 68
_INLINE_RULE_WIDTH = 60

# Why: chunk text-preview truncation caps. 200 is the cache key length
# (must tolerate later truncation); the others govern report verbosity.
_TEXT_KEY_LEN = 200
_CHUNK_PREVIEW_TRUNC = 300
_REMOVED_PREVIEW_TRUNC = 150
_FUSED_PREVIEW_TRUNC = 120

# Why: text-wrap widths for the report body and the bordered prompt/answer
# blocks. 90 covers an 80-column terminal with 6 indent + 4 prefix; 86 is
# the bordered-block inner width.
_WRAP_WIDTH_BODY = 90
_WRAP_WIDTH_BORDER = 86

# Why: top-K reported on the fused-RRF output. 5 / 8 are display caps,
# not algorithmic constants.
_FUSED_DISPLAY_TOP_K = 5
_RRF_DISPLAY_TOP_K = 8

USE_COLOR = True


# ---------------------------------------------------------------------------
# Gold-tracking state
# ---------------------------------------------------------------------------
# Set in main() and read by all patch closures via module-level access.
_GOLD_ANSWER: str = ""
# Gold supporting-paragraph titles (supporting_facts) -- the authoritative
# signal for "is the gold evidence still present?". Empty list -> fall
# back to the answer-string heuristic (and print a warning the first time).
_GOLD_DOC_TITLES: list = []
# Single-flag mutable state; was previously wrapped in a list-cell for
# closure mutation, now a plain module global.
_GOLD_FALLBACK_WARNED: bool = False
# List-cell (not a bare int) so the _nav_wrapper closure can mutate it via
# _HOP_COUNTER[0] += 1 without a `global` declaration — see usage below.
_HOP_COUNTER: list = [0]

def _c(code: str, text: str) -> str:
    return f"{code}{text}\033[0m" if USE_COLOR else text

def cyan(t):    return _c("\033[96m", t)
def green(t):   return _c("\033[92m", t)
def yellow(t):  return _c("\033[93m", t)
def red(t):     return _c("\033[91m", t)
def bold(t):    return _c("\033[1m", t)
def dim(t):     return _c("\033[2m", t)
def magenta(t): return _c("\033[95m", t)
def blue(t):    return _c("\033[94m", t)


# ─── Gold tracking ───────────────────────────────────────────────────────────
#
# Two modes:
#   (A) supporting-facts mode (preferred): we know the gold paragraph titles
#       from HotpotQA `supporting_facts`. A chunk "is gold" iff its source-doc
#       title equals one of those titles. This is exact and never fires on a
#       relative's biography that merely mentions the answer string.
#   (B) fallback mode: no supporting_facts ⇒ heuristically match the answer
#       string's content words against chunk text (the old behaviour). Marked
#       with a one-time warning so the reader knows it's lossy.

def _gold_words(gold: str) -> list:
    """Content words from the gold answer used by fallback mode B.

    Includes tokens of length >= `_GOLD_MIN_WORD_LEN` that are not in the
    closed-class `_GOLD_STOPWORDS` set.
    """
    return [
        w for w in gold.lower().split()
        if len(w) >= _GOLD_MIN_WORD_LEN and w not in _GOLD_STOPWORDS
    ]


def _norm_title(t: str) -> str:
    """Normalise a doc title for comparison.

    Why:    chunks_export's source_file field prepends "<dataset>_" to the
            article title, but the dataset's supporting_facts schema lists
            the raw article title. Strip the prefix so both representations
            collapse to the same key for set membership.
    What:   lower-case, strip a leading dataset prefix (whitespace-free
            first underscore-delimited token), collapse internal whitespace.
    Misses: a legitimate title whose first whitespace-free token happens to
            match a dataset name would lose that token. Acceptable: the
            tracked supporting_facts titles never contain underscores in
            their first word.
    """
    t = (t or "").strip().lower()
    if "_" in t:
        prefix, _, rest = t.partition("_")
        if prefix and " " not in prefix and rest:
            t = rest
    return " ".join(t.split())


_GOLD_TITLES_NORM: set = set()   # populated in main() from supporting_facts

# The Navigator's _rrf_fusion drops the source_doc from chunk dicts (it groups by
# text), so downstream filter stages only have {"text", "rrf_score", ...}. We
# therefore build a text→title index from the HybridRetriever's raw results
# (which DO carry .source_doc) and look the title up by chunk text everywhere
# else. Keyed by the first ~200 chars of text to tolerate later truncation.
_TEXT_TO_TITLE: dict = {}


def _text_key(text: str) -> str:
    """Stable lookup key for the text -> title cache.

    Keyed on the first `_TEXT_KEY_LEN` chars so later truncation by the
    Navigator (which strips chunk metadata at the RRF boundary) still
    resolves to the same title.
    """
    return (text or "").strip()[:_TEXT_KEY_LEN].lower()


def _register_titles(results) -> None:
    """Record text→source_doc for a batch of RetrievalResult-like objects/dicts."""
    for r in results:
        title = _chunk_doc_title(r)
        txt = (r.get("text", "") if isinstance(r, dict) else getattr(r, "text", "")) or ""
        if title and txt:
            _TEXT_TO_TITLE.setdefault(_text_key(txt), title)


def _chunk_doc_title(item) -> str:
    """Best-effort extraction of the source-document title from a chunk —
    a RetrievalResult-like object, a fused dict, or (via the text→title index)
    a plain string / a stripped-down dict that only has "text"."""
    # direct attributes / keys
    if isinstance(item, dict):
        direct = (item.get("source_doc")
                  or item.get("source_file")
                  or (item.get("metadata", {}) or {}).get("source_file")
                  or (item.get("metadata", {}) or {}).get("article_title"))
        if direct:
            return direct
        txt = item.get("text", "")
    elif isinstance(item, str):
        txt = item
    else:
        for attr in ("source_doc", "source_file", "source"):
            v = getattr(item, attr, None)
            if v:
                return v
        meta = getattr(item, "metadata", None)
        if isinstance(meta, dict):
            v = meta.get("source_file") or meta.get("article_title")
            if v:
                return v
        txt = getattr(item, "text", "") or ""
    # fall back to the text→title index
    return _TEXT_TO_TITLE.get(_text_key(txt), "")


def _is_gold_chunk(item) -> bool:
    """True if this chunk is one of the gold supporting paragraphs (mode A),
    or — in fallback mode B — if it contains all content words of the answer."""
    if _GOLD_TITLES_NORM:
        return _norm_title(_chunk_doc_title(item)) in _GOLD_TITLES_NORM
    # fallback mode
    gold = _GOLD_ANSWER
    if not gold or gold in ("?", "(unknown)"):
        return False
    words = _gold_words(gold)
    if not words:
        return False
    t = (item if isinstance(item, str) else
         (item.get("text", "") if isinstance(item, dict) else getattr(item, "text", ""))).lower()
    return all(w in t for w in words)


def _gold_marker_inline(item) -> str:
    """Short inline marker for per-chunk listings, or '' if not gold."""
    if not _is_gold_chunk(item):
        return ""
    if _GOLD_TITLES_NORM:
        return f"  {green('← GOLD ¶')}"
    return f"  {green('← GOLD (answer-string)')}"


def _gold_check_texts(texts, stage: str, gold: str = "") -> None:
    """Report which gold supporting paragraphs are still present after `stage`.

    In supporting-facts mode this lists hit/missing titles; in fallback
    mode it behaves like the answer-string check (with a one-time warning).
    `texts`: list of str OR dicts with a "text" key OR result objects.
    """
    global _GOLD_FALLBACK_WARNED
    bar = "  " + "." * _INLINE_RULE_WIDTH

    if _GOLD_TITLES_NORM:
        present_titles = {_norm_title(_chunk_doc_title(it)) for it in texts}
        present_titles.discard("")
        hit  = sorted(t for t in _GOLD_TITLES_NORM if t in present_titles)
        miss = sorted(t for t in _GOLD_TITLES_NORM if t not in present_titles)
        print(f"{bar}")
        if not miss:
            print(f"  {green(bold('OK GOLD'))}  all {len(hit)} gold paragraph(s) present  "
                  f"[{stage}]  {dim('para: ' + ', '.join(hit))}")
        elif hit:
            print(f"  {yellow(bold('~ GOLD PARTIAL'))}  {len(hit)}/{len(_GOLD_TITLES_NORM)} present  "
                  f"[{stage}]")
            print(f"      {green('present:')} {', '.join(hit)}")
            print(f"      {red('MISSING:')} {', '.join(miss)}")
        else:
            print(f"  {red(bold('XX GOLD LOST'))}  none of the {len(_GOLD_TITLES_NORM)} gold "
                  f"paragraph(s) present  [{stage}]")
            print(f"      {red('MISSING:')} {', '.join(miss)}")
        print(f"{bar}")
        return

    # -- fallback mode (no supporting_facts) ------------------------------
    if not _GOLD_FALLBACK_WARNED:
        _msg = ("* gold-tracking fallback: no supporting_facts -- matching "
                "answer-string content words against chunk text (lossy; may "
                "produce false positives on related-entity passages).")
        print(f"  {yellow(_msg)}")
        _GOLD_FALLBACK_WARNED = True
    g = gold or _GOLD_ANSWER
    if not g or g in ("?", "(unknown)"):
        return
    words = _gold_words(g)
    if not words:
        return
    hits = []
    for i, item in enumerate(texts):
        t = (item if isinstance(item, str) else
             (item.get("text", "") if isinstance(item, dict) else getattr(item, "text", ""))).lower()
        if all(w in t for w in words):
            hits.append(i + 1)
    print(f"{bar}")
    if hits:
        extra = f" (+ {len(hits)-1} more)" if len(hits) > 1 else ""
        print(f"  {green(bold('OK GOLD'))}  '{g}'  ~ chunk #{hits[0]}{extra}  [{stage}]")
    else:
        print(f"  {red(bold('XX GOLD LOST'))}  '{g}'  no chunk contains all answer words  [{stage}]")
    print(f"{bar}")


# ─── Output helpers ──────────────────────────────────────────────────────────

def section(title: str) -> None:
    bar = "=" * _BAR_WIDTH
    print(f"\n{bold(bar)}")
    print(f"  {bold(cyan(title))}")
    print(f"{bold(bar)}")

def subsection(title: str) -> None:
    print(f"\n  {bold(yellow('>> ' + title))}")
    print(f"  {'-' * _SECTION_RULE_WIDTH}")

def field(label: str, value, indent_lvl: int = 4) -> None:
    prefix = " " * indent_lvl
    label_str = bold(label + ":")
    value_str = str(value)
    if "\n" in value_str or len(value_str) > 100:
        print(f"{prefix}{label_str}")
        for line in value_str.splitlines():
            for wrapped in wrap(line, width=_WRAP_WIDTH_BODY) or [""]:
                print(f"{prefix}  {dim(wrapped)}")
    else:
        print(f"{prefix}{label_str} {value_str}")

def chunk_block(idx: int, text: str, score=None, extra: str = "",
                max_chars: int = _CHUNK_PREVIEW_TRUNC) -> None:
    score_str = f"  {dim(f'score={score:.4f}')}" if score is not None else ""
    extra_str = f"  {dim(extra)}" if extra else ""
    print(f"  {bold(green(f'Chunk #{idx+1}'))}{score_str}{extra_str}")
    preview = text[:max_chars].replace("\n", " ")
    if len(text) > max_chars:
        preview += dim("...")
    for line in wrap(preview, width=_WRAP_WIDTH_BODY):
        print(f"    {dim(line)}")

def removed_block(idx: int, text: str, reason: str) -> None:
    print(f"  {red(f'XX Chunk #{idx+1} REMOVED')}  {dim(f'[{reason}]')}")
    preview = text[:_REMOVED_PREVIEW_TRUNC].replace("\n", " ")
    for line in wrap(preview, width=_WRAP_WIDTH_BORDER + 2):
        print(f"    {dim(line)}")

def prompt_block(prompt: str) -> None:
    print(f"  {bold(magenta('PROMPT ->'))}")
    border = "  " + "." * _SECTION_RULE_WIDTH
    print(border)
    for line in prompt.splitlines():
        for wrapped in wrap(line, width=_WRAP_WIDTH_BORDER) or [""]:
            print(f"  {dim(wrapped)}")
    print(border)

def answer_block(answer: str, latency_ms: float = None) -> None:
    lat = f"  {dim(f'({latency_ms:.0f} ms)')}" if latency_ms else ""
    color = red if answer.startswith("[Error:") else green
    print(f"  {bold(color('ANSWER ->'))}{lat}")
    border = "  " + "." * _SECTION_RULE_WIDTH
    print(border)
    for line in answer.splitlines():
        for wrapped in wrap(line, width=_WRAP_WIDTH_BORDER) or [""]:
            print(f"  {color(wrapped)}")
    print(border)


# ─── Monkey-Patch Utilities ──────────────────────────────────────────────────

def _patch(obj, method_name: str, wrapper_factory):
    original = getattr(obj, method_name)
    setattr(obj, method_name, wrapper_factory(original))


# =============================================================================
# PART 1: sys.settrace — track every function call in src/
# =============================================================================

_call_log: list = []          # (file, function)
_seen_files: set = set()      # unique src/ files

def _make_tracer(src_root: Path):
    """
    Return a trace function that records every function call inside src/.
    Enabled via --trace-calls.
    """
    src_str = str(src_root / "src")

    def _tracer(frame, event, arg):
        if event != "call":
            return _tracer
        filename = frame.f_code.co_filename
        if src_str in filename:
            rel = Path(filename).relative_to(src_root)
            func = frame.f_code.co_name
            _call_log.append((str(rel), func))
            _seen_files.add(str(rel))
        return _tracer

    return _tracer

def print_call_trace() -> None:
    section("FUNCTION CALL TRACE (all src/ calls)")
    subsection(f"Unique files ({len(_seen_files)})")
    for f in sorted(_seen_files):
        print(f"    {blue('📄')} {f}")

    subsection(f"Call sequence ({len(_call_log)} calls)")
    prev_file = None
    prev_func = None
    repeat_count = 0

    def _flush_repeat():
        if repeat_count > 1:
            print(f"    {dim('→')} {prev_func}()  {dim(f'×{repeat_count}')}")
        elif repeat_count == 1:
            print(f"    {dim('→')} {prev_func}()")

    for rel, func in _call_log:
        if rel != prev_file:
            _flush_repeat()
            repeat_count = 0
            print(f"\n  {bold(blue(rel))}")
            prev_file = rel
            prev_func = None
        if func == prev_func:
            repeat_count += 1
        else:
            _flush_repeat()
            prev_func = func
            repeat_count = 1
    _flush_repeat()


# =============================================================================
# PLANNER HOOK
# =============================================================================

def patch_planner(planner) -> None:
    def _wrap_plan(original):
        def _plan(query: str):
            section("S_P — PLANNER")
            field("Query", query)

            t0 = time.time()
            result = original(query)
            ms = (time.time() - t0) * 1000

            subsection("RetrievalPlan")
            field("query_type",  result.query_type.value)
            field("strategy",    result.strategy.value)
            field("confidence",  f"{result.confidence:.3f}")

            field("sub_queries", "")
            for i, sq in enumerate(result.sub_queries, 1):
                print(f"      {bold(str(i) + '.')} {sq}")

            field("entities", "")
            if result.entities:
                for e in result.entities:
                    # EntityInfo has: text, label, confidence (no entity_type!)
                    print(f"      {bold(e.text)}"
                          f"  {dim(e.label)}"
                          f"  {dim(f'conf={e.confidence:.2f}')}"
                          f"  {dim('bridge') if e.is_bridge else ''}")
            else:
                print(f"      {red('(no entities detected — entity-mention filter will be disabled!)')}")

            if result.hop_sequence:
                field("hop_sequence", "")
                for hop in result.hop_sequence:
                    print(f"      {dim(str(hop))}")

            field("Duration", f"{ms:.0f} ms")
            return result
        return _plan

    _patch(planner, "plan", _wrap_plan)


# =============================================================================
# HYBRID RETRIEVER HOOK — shows GLiNER entities + raw vector/graph results
# =============================================================================

def patch_retriever(retriever) -> None:
    """
    Patch HybridRetriever.retrieve() to expose GLiNER query entities,
    raw vector/graph results, and the fused top-K.
    This is the most important hook: it determines whether the target
    chunk is even pulled from the database.

    GLiNER entities are read from metrics.query_entities (no double call
    of the extractor).
    """
    orig_retrieve = retriever.retrieve

    def _retrieve(query: str, top_k=None, entity_hints=None):
        subsection(f"HybridRetriever.retrieve()  query={query!r}")

        # ── execute retrieve() ───────────────────────────────────────────────
        results, metrics = orig_retrieve(query, top_k, entity_hints=entity_hints)
        # Record text→source_doc so downstream stages (which drop source_doc)
        # can still resolve the gold paragraph by chunk text.
        _register_titles(results)

        # ── read GLiNER entities from metrics (no double call) ───────────────
        print(f"    {bold('GLiNER query entities:')}")
        if metrics.query_entities:
            for e in metrics.query_entities:
                print(f"      {bold(e)}  {dim('(used for graph search)')}")
        else:
            print(f"      {red('(no entities detected -> graph search will be SKIPPED!)')}")

        # ── show metrics ─────────────────────────────────────────────────────
        print(f"\n    {bold('Retrieval metrics:')}")
        print(f"      Vector: {metrics.vector_results} hits  "
              f"{dim(f'({metrics.vector_time_ms:.0f} ms)')}")
        print(f"      Graph:  {metrics.graph_results} hits  "
              f"{dim(f'({metrics.graph_time_ms:.0f} ms)')}")
        print(f"      Fused:  {metrics.final_results} results  "
              f"{dim(f'({metrics.fusion_time_ms:.0f} ms)')}")

        # ── top-5 results with retrieval method and matched entities ─────────
        print(f"\n    {bold('Top-5 fused results:')}")
        for i, r in enumerate(results[:5]):
            src     = getattr(r, "source_doc", getattr(r, "source", "?"))
            score   = getattr(r, "rrf_score", getattr(r, "score", 0))
            method  = getattr(r, "retrieval_method", "?")
            matched = getattr(r, "matched_entities", [])
            v_score = getattr(r, "vector_score", None)
            g_score = getattr(r, "graph_score", None)
            txt     = getattr(r, "text", str(r))[:120].replace("\n", " ")

            method_color = blue if method == "graph" else (cyan if method == "hybrid" else dim)
            print(f"      {bold(f'#{i+1}')}"
                  f"  {green(f'rrf={score:.4f}')}"
                  f"  [{method_color(method)}]"
                  f"  {dim(f'src={src}')}")
            if v_score is not None:
                print(f"        {dim(f'vector_score={v_score:.4f}')}", end="")
            if g_score is not None:
                print(f"  {dim(f'graph_score={g_score:.4f}')}", end="")
            if v_score is not None or g_score is not None:
                print()
            if matched:
                print(f"        {dim(f'matched_entities: {matched}')}")
            for line in wrap(txt, 84):
                print(f"        {dim(line)}")

        return results, metrics

    retriever.retrieve = _retrieve


# =============================================================================
# NAVIGATOR HOOKS
# =============================================================================

def patch_navigator(navigator) -> None:

    # ── _rrf_fusion ──────────────────────────────────────────────────────────
    def _wrap_rrf(original):
        def _rrf(results, k=None):
            fused = original(results, k)
            k_val = navigator.config.rrf_k if k is None else k
            subsection(f"RRF Fusion  k={k_val}  "
                       f"({len(results)} raw entries → {len(fused)} unique chunks)")
            for i, r in enumerate(fused[:8]):
                score = r.get("rrf_score", 0)
                src   = r.get("source_count", "?")
                qc    = r.get("query_count", "?")
                txt   = r["text"][:120].replace("\n", " ")
                gold_marker = _gold_marker_inline(r)
                src_doc = r.get("source_doc") or _chunk_doc_title(r) or "?"
                print(f"    {bold(f'#{i+1}')}"
                      f"  {green(f'rrf={score:.4f}')}"
                      f"  {dim(f'src_count={src} query_count={qc}')}"
                      f"  {dim(f'src={src_doc}')}"
                      f"{gold_marker}")
                for line in wrap(txt, 86):
                    print(f"      {dim(line)}")
            if len(fused) > 8:
                print(f"    {dim(f'… {len(fused)-8} more chunks')}")
            _gold_check_texts(fused, "after RRF fusion")
            return fused
        return _rrf

    # ── _relevance_filter ────────────────────────────────────────────────────
    def _wrap_relevance(original):
        def _filt(results):
            before = len(results)
            threshold = 0.0
            if results:
                max_score = max(r["rrf_score"] for r in results)
                threshold = navigator.config.relevance_threshold_factor * max_score
            filtered = original(results)
            after = len(filtered)
            removed = before - after
            subsection(f"Filter 1 - Relevance  ({before} -> {after}"
                       + (f", {red(str(removed) + ' removed')})" if removed else ")"))
            if results:
                print(f"    {bold('Threshold:')} {threshold:.4f}  "
                      f"{dim(f'= {navigator.config.relevance_threshold_factor} x max({max_score:.4f})')}")
            if removed:
                kept_texts = {r["text"] for r in filtered}
                for i, r in enumerate(results):
                    if r["text"] not in kept_texts:
                        removed_block(i, r["text"],
                                      f"rrf={r['rrf_score']:.4f} < {threshold:.4f}")
            else:
                print(f"    {dim('(all chunks above threshold — no filtering)')}")
            _gold_check_texts(filtered, "after relevance filter")
            return filtered
        return _filt

    # ── _redundancy_filter ───────────────────────────────────────────────────
    def _wrap_redundancy(original):
        def _filt(results):
            before = len(results)
            filtered = original(results)
            after = len(filtered)
            removed = before - after
            subsection(f"Filter 2 - Redundancy  Jaccard threshold={navigator.config.redundancy_threshold}"
                       f"  ({before} -> {after}"
                       + (f", {red(str(removed) + ' removed')})" if removed else ")"))
            if removed:
                kept_texts = {r["text"] for r in filtered}
                for i, r in enumerate(results):
                    if r["text"] not in kept_texts:
                        removed_block(i, r["text"], "Jaccard duplicate")
            if not removed:
                print(f"    {dim('(no duplicates)')}")
            _gold_check_texts(filtered, "after redundancy filter")
            return filtered
        return _filt

    # ── _contradiction_filter ────────────────────────────────────────────────
    def _wrap_contradiction(original):
        def _filt(results):
            before = len(results)
            filtered = original(results)
            after = len(filtered)
            removed = before - after
            subsection(f"Filter 3 - Contradiction  overlap>={navigator.config.contradiction_overlap_threshold}"
                       f"  ratio>={navigator.config.contradiction_ratio_threshold}"
                       f"  min_value>={navigator.config.contradiction_min_value}"
                       f"  ({before} -> {after}"
                       + (f", {red(str(removed) + ' removed')})" if removed else ")"))
            if removed:
                kept_texts = {r["text"] for r in filtered}
                for i, r in enumerate(results):
                    if r["text"] not in kept_texts:
                        removed_block(i, r["text"], "Numeric contradiction")
            if not removed:
                print(f"    {dim('(no contradictions)')}")
            _gold_check_texts(filtered, "after contradiction filter")
            return filtered
        return _filt

    # ── _entity_overlap_pruning ──────────────────────────────────────────────
    def _wrap_entity_overlap(original):
        def _filt(results):
            before = len(results)
            filtered = original(results)
            after = len(filtered)
            removed = before - after
            subsection(f"Filter 4 - Entity overlap  ({before} -> {after}"
                       + (f", {red(str(removed) + ' removed')})" if removed else ")"))
            if removed:
                kept_texts = {r["text"] for r in filtered}
                for i, r in enumerate(results):
                    if r["text"] not in kept_texts:
                        removed_block(i, r["text"], "Entity set is a subset")
            if not removed:
                print(f"    {dim('(no subsets)')}")
            _gold_check_texts(filtered, "after entity-overlap filter")
            return filtered
        return _filt

    # ── _entity_mention_filter ───────────────────────────────────────────────
    def _wrap_entity_mention(original):
        def _filt(results, entity_names, *args, **kwargs):
            before = len(results)
            filtered = original(results, entity_names, *args, **kwargs)
            after = len(filtered)
            removed = before - after

            # Detect safety-fallback: all chunks would have been filtered but
            # the filter returned the full list unchanged as a last-resort.
            # Heuristic: removed==0, entity_names non-empty, and no chunk
            # actually contains any entity → fallback must have fired.
            safety_fallback = False
            if removed == 0 and entity_names and results:
                import re as _re
                def _mentions(text, names):
                    t = text.lower()
                    for name in names:
                        tokens = name.lower().split()
                        if len(tokens) > 1:
                            if name.lower() in t:
                                return True
                        else:
                            if len(name) >= 5 and _re.search(r'\b' + _re.escape(name.lower()) + r'\b', t):
                                return True
                    return False
                safety_fallback = not any(_mentions(r["text"], entity_names) for r in results)

            label = f"{before} -> {after}"
            if removed:
                label += f", {red(str(removed) + ' removed')}"
            elif safety_fallback:
                label += f", {yellow('safety fallback!')}"
            subsection(f"Filter 5 - Entity mention  ({label})")

            print(f"    {bold('Searched entities:')} "
                  + (", ".join(bold(e) for e in entity_names) if entity_names
                     else red("(none!) -> filter disabled, all chunks kept")))

            if not entity_names:
                print(f"    {yellow('* CAUSE: planner entities empty')}")
                print(f"    {yellow('* CONSEQUENCE: irrelevant chunks pass through — Verifier sees poor context')}")
            elif safety_fallback:
                print(f"    {yellow('* SAFETY FALLBACK: no chunk contains any of the searched entities.')}")
                print(f"    {yellow('* CAUSE: article likely not in the database (missing from ingestion)')}")
                print(f"    {yellow('* CONSEQUENCE: all 10 irrelevant chunks stay — Verifier has useless context')}")
            elif removed:
                kept_texts = {r["text"] for r in filtered}
                for i, r in enumerate(results):
                    if r["text"] not in kept_texts:
                        removed_block(i, r["text"], "No entity mention in the text")

            if after < before or not entity_names or safety_fallback:
                print(f"    {bold('Remaining chunks:')}")
                for i, r in enumerate(filtered):
                    chunk_block(i, r["text"], r.get("rrf_score"))
            _gold_check_texts(filtered, "after entity-mention filter")
            return filtered
        return _filt

    # ── _context_shrinkage ───────────────────────────────────────────────────
    def _wrap_shrinkage(original):
        def _filt(results, max_chars_per_chunk=None):
            shrunk = original(results, max_chars_per_chunk)
            limit = max_chars_per_chunk or navigator.config.max_chars_per_doc
            total_before = sum(len(r["text"]) for r in results)
            total_after  = sum(len(r["text"]) for r in shrunk)
            reduction = 100 * (1 - total_after / max(total_before, 1))
            subsection(f"Filter 6 - Context shrinkage  "
                       f"limit={limit} chars/chunk  "
                       f"({total_before} -> {total_after} chars, {reduction:.0f}% reduction)")
            for i, r in enumerate(shrunk):
                orig_len = len(results[i]["text"]) if i < len(results) else "?"
                new_len  = len(r["text"])
                trunc = f"truncated {orig_len}->{new_len}" if orig_len != new_len else "unchanged"
                chunk_block(i, r["text"], r.get("rrf_score"), extra=trunc)
            _gold_check_texts(shrunk, "after context shrinkage")
            return shrunk
        return _filt

    _patch(navigator, "_rrf_fusion",             _wrap_rrf)
    _patch(navigator, "_relevance_filter",       _wrap_relevance)
    _patch(navigator, "_redundancy_filter",      _wrap_redundancy)
    _patch(navigator, "_contradiction_filter",   _wrap_contradiction)
    _patch(navigator, "_entity_overlap_pruning", _wrap_entity_overlap)
    _patch(navigator, "_entity_mention_filter",  _wrap_entity_mention)
    _patch(navigator, "_context_shrinkage",      _wrap_shrinkage)

    # ── navigate() ───────────────────────────────────────────────────────────
    orig_navigate = navigator.navigate

    def _nav_wrapper(retrieval_plan, sub_queries, entity_names=None):
        _HOP_COUNTER[0] += 1
        hop_label = f"Hop {_HOP_COUNTER[0]}" if _HOP_COUNTER[0] > 1 else "Single-Pass"
        section(f"S_N — NAVIGATOR  [{hop_label}]")
        field("Sub-Queries", len(sub_queries))
        for i, sq in enumerate(sub_queries, 1):
            print(f"    {bold(str(i) + '.')} {sq}")
        field("Entity names (for entity-mention filter)", entity_names or [])
        print(f"    {bold('max_context_chunks:')} {navigator.config.max_context_chunks}")
        print(f"    {bold('top_k_per_subquery:')} {navigator.config.top_k_per_subquery}")

        result = orig_navigate(retrieval_plan, sub_queries, entity_names)

        subsection("NAVIGATOR RESULT")
        field("Raw chunks (before all filters)",    len(result.raw_context))
        field("Filtered chunks (after all filters)", len(result.filtered_context))
        print(f"\n    {bold('Filter counters:')}")
        filter_keys = [
            "pre_filter_count",
            "after_relevance_filter",
            "after_redundancy_filter",
            "after_contradiction_filter",
            "after_entity_overlap_pruning",
            "after_entity_mention_filter",
        ]
        prev = result.metadata.get("pre_filter_count", "?")
        for k in filter_keys:
            v = result.metadata.get(k, "?")
            diff = ""
            if k != "pre_filter_count" and isinstance(v, int) and isinstance(prev, int):
                delta = v - prev
                if delta < 0:
                    diff = f"  {red(f'-{abs(delta)} removed')}"
                elif delta == 0:
                    diff = f"  {dim('unchanged')}"
            print(f"      {bold(k)}: {v}{diff}")
            prev = v

        if result.metadata.get("retrieval_errors"):
            print(f"    {red('Retrieval errors:')} {result.metadata['retrieval_errors']}")

        # Gold check on final Navigator result
        _gold_check_texts(result.filtered_context, f"Navigator result [{hop_label}]")
        return result

    navigator.navigate = _nav_wrapper


# =============================================================================
# VERIFIER HOOKS
# =============================================================================

def patch_verifier(verifier) -> None:

    # ── _reorder_by_question_relevance ────────────────────────────────────────
    orig_reorder = verifier._reorder_by_question_relevance

    def _reorder_wrapper(query, context, *args, **kwargs):
        reordered = orig_reorder(query, context, *args, **kwargs)
        subsection(f"_reorder_by_question_relevance()  {len(context)} chunks")
        if reordered != context:
            print(f"    {bold('Order changed')} — LLM sees chunks in this order:")
        else:
            print(f"    {dim('(order unchanged)')}")
        for i, c in enumerate(reordered):
            orig_pos = context.index(c) + 1 if c in context else "?"
            gold_marker = _gold_marker_inline(c)
            pos_change = f"  {dim(f'(was #{orig_pos})')}" if orig_pos != i + 1 else ""
            print(f"    {bold(f'#{i+1}')}{pos_change}{gold_marker}")
            preview = c[:120].replace("\n", " ")
            for line in wrap(preview, 86):
                print(f"      {dim(line)}")
        _gold_check_texts(reordered, "after reorder (LLM input)")
        return reordered

    verifier._reorder_by_question_relevance = _reorder_wrapper

    # ── _format_context ───────────────────────────────────────────────────────
    orig_format = verifier._format_context

    def _fmt_wrapper(context, *args, **kwargs):
        formatted = orig_format(context, *args, **kwargs)
        subsection(f"_format_context()  {len(context)} chunks -> {len(formatted)} chars")
        max_docs = verifier.config.max_docs
        max_chars = verifier.config.max_chars_per_doc
        print(f"    {bold('Limits:')} max_docs={max_docs}, max_chars_per_doc={max_chars}, "
              f"max_context_chars={verifier.config.max_context_chars}")
        if len(context) > max_docs:
            print(f"    {yellow(f'* {len(context)} chunks -> only the first {max_docs} will be used!')}")
        return formatted

    verifier._format_context = _fmt_wrapper

    # ── _extract_claims ───────────────────────────────────────────────────────
    orig_extract = verifier._extract_claims

    def _claims_wrapper(answer):
        claims = orig_extract(answer)
        print(f"\n    {bold('Extracted claims')} ({len(claims)}):")
        for c in claims:
            print(f"      {dim('·')} {c}")
        return claims

    verifier._extract_claims = _claims_wrapper

    # ── _call_llm ─────────────────────────────────────────────────────────────
    orig_call_llm = verifier._call_llm

    def _wrap_call_llm(prompt: str, *args, **kwargs):
        # Forward *args/**kwargs so the wrapper tracks the real _call_llm
        # signature (e.g. the structured-CoT path passes max_tokens=...).
        prompt_block(prompt)
        answer, latency_ms = orig_call_llm(prompt, *args, **kwargs)
        answer_block(answer, latency_ms)
        return answer, latency_ms

    verifier._call_llm = _wrap_call_llm

    # ── generate_and_verify ───────────────────────────────────────────────────
    orig_gen = verifier.generate_and_verify

    def _wrap_generate(query, context, entities=None, hop_sequence=None,
                       query_type=None, bridge_entities=None,
                       chunk_is_graph_based=None, **_kw):
        # B2 (verifier audit, 2026-05-15) added chunk_is_graph_based; **_kw
        # absorbs any future kwargs so diagnose_verbose.py is forward-
        # compatible with verifier-signature additions.
        section("S_V — VERIFIER")
        field("Query",       query)
        field("Chunks in",   len(context))
        field("Entities",    entities or [])
        if bridge_entities:
            field("Bridge entities", bridge_entities)

        # Run pre-validation and display its result
        subsection("Pre-Generation Validation")
        try:
            pre_val = verifier.pre_validator.validate(
                context=context, query=query,
                entities=entities, hop_sequence=hop_sequence,
                chunk_is_graph_based=chunk_is_graph_based,
            )
            print(f"    {bold('Status:')}          {pre_val.status.value}")
            print(f"    {bold('entity_path_valid:')} {pre_val.entity_path_valid}")
            print(f"    {bold('Contradictions:')}")
            if pre_val.contradictions:
                for c in pre_val.contradictions[:3]:
                    print(f"      {red('XX')} {dim(str(c)[:120])}")
            else:
                print(f"      {dim('(none)')}")
            print(f"    {bold('Filtered context:')} "
                  f"{len(pre_val.filtered_context)}/{len(context)} chunks kept")
            if len(pre_val.filtered_context) < len(context):
                kept = set(id(c) for c in pre_val.filtered_context)
                for i, c in enumerate(context):
                    if id(c) not in kept:
                        print(f"      {red(f'XX chunk #{i+1} removed by pre-validator:')}")
                        print(f"        {dim(c[:100])}")
            print(f"    {bold('Credibility scores:')}")
            if hasattr(pre_val, "credibility_scores") and pre_val.credibility_scores:
                for i, sc in enumerate(pre_val.credibility_scores[:5]):
                    score_val = sc.score if hasattr(sc, "score") else sc
                    print(f"      Chunk #{i+1}: {score_val:.3f}")
            else:
                print(f"      {dim('(none)')}")
        except Exception as ex:
            print(f"    {red(f'Pre-validation error: {ex}')}")

        print(f"\n  {bold(yellow('Context chunks sent to Verifier:'))}")
        for i, c in enumerate(context):
            chunk_block(i, c)

        result = orig_gen(
            query, context, entities, hop_sequence,
            query_type=query_type,
            bridge_entities=bridge_entities,
            chunk_is_graph_based=chunk_is_graph_based,
            **_kw,
        )

        subsection("VERIFIER RESULT")
        field("Answer",       result.answer)
        conf = result.confidence
        conf_str = conf.value if hasattr(conf, "value") else str(conf)
        field("Confidence",   conf_str)
        field("Iterations",   result.iterations)
        field("All verified", result.all_verified)
        if result.verified_claims:
            print(f"    {bold('Verified claims')} ({len(result.verified_claims)}):")
            for c in result.verified_claims:
                print(f"      {green('✓')} {dim(c)}")
        if result.violated_claims:
            print(f"    {bold('Violated claims')} ({len(result.violated_claims)}):")
            for c in result.violated_claims:
                print(f"      {red('✗')} {dim(c)}")
        return result

    verifier.generate_and_verify = _wrap_generate


# =============================================================================
# PIPELINE AUFBAUEN
# =============================================================================

def load_question(idx: int, dataset: str = _DEFAULT_DATASET) -> dict:
    """Read questions[idx] from data/<dataset>/questions.json."""
    questions_path = PROJECT_ROOT / "data" / dataset / "questions.json"
    if not questions_path.exists():
        print(red(f"Error: {questions_path} not found"))
        sys.exit(1)
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)
    if idx >= len(questions):
        print(red(f"Index {idx} too large (max: {len(questions)-1})"))
        sys.exit(1)
    return questions[idx]


# =============================================================================
# CONTROLLER HOOK — make bridge-entity extraction visible
# =============================================================================

def patch_controller_bridge(_pipeline_unused=None) -> None:
    """Wrap AgenticController._extract_bridge_entities for observability.

    AgenticController is a static-helper container (no instances), so the
    patch swaps the staticmethod at class level. Every caller (currently
    only AgentPipeline._iterative_navigate) then sees the wrapped version.

    The argument is accepted for backward compatibility with older call
    sites that passed an instance; it is no longer used.
    """
    from src.logic_layer.controller import AgenticController as _AC
    orig_extract = _AC._extract_bridge_entities  # unbound staticmethod

    def _wrapped_extract(chunks, exclude, *args, **kwargs):
        # *args/**kwargs forward any new parameters added to the underlying
        # _extract_bridge_entities signature (e.g. the `query=...` kwarg
        # added for relevance-ranked extraction). Without this, every
        # signature change would crash diagnose_verbose.py.
        result = orig_extract(chunks, exclude, *args, **kwargs)
        bar = "  " + "." * _INLINE_RULE_WIDTH
        print(f"\n{bar}")
        if result:
            print(f"  {green(bold('** BRIDGE ENTITIES DETECTED:'))}  "
                  + "  ".join(bold(e) for e in result))
            print(f"  {dim(f'(from {len(chunks)} chunk(s), exclude={exclude})')}")
        else:
            print(f"  {yellow(bold('* NO BRIDGE ENTITIES FOUND'))}  "
                  f"{dim(f'(exclude={exclude}, chunks: {len(chunks)})')}")
            if chunks:
                preview = chunks[0][:120].replace("\n", " ")
                print(f"  {dim(f'Chunk preview: {preview}')}")
        print(f"{bar}\n")
        return result

    # Class-level patch (replaces the staticmethod). AgenticController is
    # a static-helper container with no instances, so the attribute is
    # overridden on the class itself rather than on a bound object.
    _AC._extract_bridge_entities = staticmethod(_wrapped_extract)


def build_pipeline(cfg: dict, dataset: str = _DEFAULT_DATASET):
    """Construct the production AgentPipeline + sub-agents for diagnostics.

    AgenticController is a static-helper container; the production entry
    point is `src.pipeline.AgentPipeline`. This builder mirrors
    `create_full_pipeline` but returns the individual agents alongside the
    pipeline so the diagnostic hooks (patch_planner / patch_navigator /
    patch_verifier) can attach to each one.
    """
    from src.data_layer.embeddings import BatchedOllamaEmbeddings
    from src.data_layer.hybrid_retriever import create_hybrid_retriever
    from src.data_layer.storage import HybridStore, StorageConfig
    from src.logic_layer import create_planner, create_verifier
    from src.pipeline.agent_pipeline import AgentPipeline

    vector_path = PROJECT_ROOT / "data" / dataset / "vector"
    graph_path  = PROJECT_ROOT / "data" / dataset / "graph"

    emb_cfg = cfg.get("embeddings", {})
    embeddings = BatchedOllamaEmbeddings(
        model_name=emb_cfg.get("model_name", _OLLAMA_DEFAULT_MODEL),
        base_url=emb_cfg.get("base_url", _OLLAMA_DEFAULT_URL),
    )
    # read_only=True: the demo/diagnostic never writes the stores, and KuzuDB's
    # write mode takes an exclusive file lock (one process per store). Opening
    # read-only lets several demo terminals share the same dataset.
    storage_cfg = StorageConfig(
        vector_db_path=vector_path, graph_db_path=graph_path, read_only=True,
    )
    store = HybridStore(config=storage_cfg, embeddings=embeddings)

    # Build the retriever through the SAME factory the evaluation suite uses
    # (create_hybrid_retriever), so a demo/diagnostic run applies the full
    # frozen contract: vector_store.top_k_vectors (20, not the dataclass 10),
    # rag.bm25_top_k, per-source weights, graph.hub_mention_cap /
    # hub_fanout_cap / search_budget_seconds, similarity_threshold, and the
    # rag.retrieval_mode toggle. A hand-rolled RetrievalConfig here previously
    # read a non-existent rag.top_k_vectors key and silently halved the
    # retrieval funnel — exactly the silent-default bug class the settings
    # validator exists to catch.
    retriever = create_hybrid_retriever(store, embeddings, cfg)

    planner  = create_planner(cfg)
    verifier = create_verifier(cfg=cfg)
    verifier.set_graph_store(store.graph_store)

    # AgentPipeline lazy-initialises its Navigator using `cfg`. Construct
    # explicitly here so the diagnostic hooks can attach before .process().
    pipeline = AgentPipeline(
        planner=planner,
        verifier=verifier,
        hybrid_retriever=retriever,
        graph_store=store.graph_store,
        config=cfg,
    )
    # Force agent construction now (otherwise `pipeline.navigator` is None
    # until the first .process() call).
    pipeline._lazy_init_agents()

    return pipeline, planner, verifier, store, retriever


# =============================================================================
# MAIN
# =============================================================================

def main():
    global USE_COLOR, _GOLD_ANSWER, _GOLD_TITLES_NORM

    parser = argparse.ArgumentParser(
        description="Full pipeline trace with output from every function"
    )
    parser.add_argument("--idx",         type=int,  default=0)
    parser.add_argument("--dataset", "-d", type=str, default=_DEFAULT_DATASET,
                        help="Dataset name; resolves to data/<dataset>/{vector,graph,questions.json}")
    parser.add_argument("--question",    type=str,  default=None)
    parser.add_argument("--gold",        type=str,  default=None,
                        help="Gold answer for stage tracking (otherwise read from questions.json)")
    parser.add_argument("--gold-docs",   type=str,  default=None,
                        help="Comma-separated gold paragraph titles for --question mode "
                             "(otherwise read from questions.json supporting_facts)")
    parser.add_argument("--skip-llm",    action="store_true")
    parser.add_argument("--trace-calls", action="store_true",
                        help="sys.settrace: show every src/ function call")
    parser.add_argument("--no-color",    action="store_true")
    args = parser.parse_args()

    if args.no_color:
        USE_COLOR = False

    # ── Load the question ────────────────────────────────────────────────────
    gold_doc_titles: list = []
    if args.question:
        q_text = args.question
        gold   = args.gold or "(unknown)"
        q_type = "custom"
        if args.gold_docs:
            gold_doc_titles = [t.strip() for t in args.gold_docs.split(",") if t.strip()]
    else:
        q = load_question(args.idx, dataset=args.dataset)
        q_text = q["question"]
        gold   = args.gold or q.get("answer", "?")
        q_type = q.get("question_type", "?")
        # HotpotQA supporting_facts: [[title, sent_idx], ...] — collect unique titles
        for sf in q.get("supporting_facts", []):
            if isinstance(sf, (list, tuple)) and sf:
                gold_doc_titles.append(str(sf[0]))
            elif isinstance(sf, str):
                gold_doc_titles.append(sf)
        if args.gold_docs:  # explicit override always wins
            gold_doc_titles = [t.strip() for t in args.gold_docs.split(",") if t.strip()]

    # Set the gold answer + gold paragraph titles globally (patch hooks read these)
    _GOLD_ANSWER = gold
    # de-dup while preserving order
    seen = set()
    _GOLD_DOC_TITLES[:] = [t for t in gold_doc_titles if not (t in seen or seen.add(t))]
    _GOLD_TITLES_NORM = {_norm_title(t) for t in _GOLD_DOC_TITLES}
    _GOLD_TITLES_NORM.discard("")

    # ── Load config ──────────────────────────────────────────────────────────
    import yaml
    with open(PROJECT_ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # -- Header -----------------------------------------------------------
    print("\n" + bold("=" * _BAR_WIDTH))
    print(f"  {bold(cyan('PIPELINE VERBOSE TRACE'))}")
    print(bold("=" * _BAR_WIDTH))
    field("Question",     q_text)
    field("Gold answer",  gold)
    field("Type",         q_type)
    if _GOLD_DOC_TITLES:
        field("Gold paragraphs", ", ".join(_GOLD_DOC_TITLES) + "  (tracked by source-doc title)")
    else:
        print(f"  {yellow('* no supporting_facts — gold-tracking will use the lossy answer-string heuristic')}")
    if args.skip_llm:
        print(f"\n  {yellow('* --skip-llm: Verifier will NOT be executed')}")
    if args.trace_calls:
        print(f"\n  {blue('* --trace-calls: every src/ function call will be logged')}")

    # ── Enable sys.settrace ───────────────────────────────────────────────────
    if args.trace_calls:
        sys.settrace(_make_tracer(PROJECT_ROOT))

    # ── Build pipeline ────────────────────────────────────────────────────────
    print(f"\n{dim('Loading pipeline...')} ", end="", flush=True)
    t0 = time.time()
    pipeline, planner, verifier, store, retriever = build_pipeline(cfg, dataset=args.dataset)
    print(f"{green('OK')}  {dim(f'({(time.time()-t0)*1000:.0f} ms)')}")

    # ── Install hooks ─────────────────────────────────────────────────────────
    patch_planner(planner)
    patch_retriever(retriever)
    patch_navigator(pipeline.navigator)
    patch_controller_bridge()  # class-level patch (no instances exist)

    if not args.skip_llm:
        patch_verifier(verifier)
    else:
        def _skip_verifier(query, context, entities=None, hop_sequence=None,
                            query_type=None, bridge_entities=None,
                            chunk_is_graph_based=None, **_kw):
            from src.logic_layer import VerificationResult
            section("S_V — VERIFIER  (skipped via --skip-llm)")
            field("Context chunks", len(context))
            if _GOLD_TITLES_NORM:
                # Navigator drops source metadata before S_V (context is
                # plain strings). At this stage only answer-string matching
                # is possible; the authoritative paragraph-level tracking
                # already happened at the Navigator output above.
                print(f"    {dim('(verifier context is text-only — paragraph-level gold tracking '
                              'happened at the Navigator output above; answer-string check below)')}")
            if context:
                for i, c in enumerate(context):
                    txt = c if isinstance(c, str) else getattr(c, "text", str(c))
                    words = _gold_words(_GOLD_ANSWER) if _GOLD_ANSWER else []
                    has_ans = bool(words) and all(w in txt.lower() for w in words)
                    chunk_block(i, txt)
                    if has_ans:
                        print("    " + green(f"  <- answer string '{_GOLD_ANSWER}' appears in this chunk"))
                # answer-string fallback report (titles unavailable here by design)
                bar = "  " + "." * _INLINE_RULE_WIDTH
                any_ans = any(
                    _GOLD_ANSWER and _gold_words(_GOLD_ANSWER)
                    and all(w in (c if isinstance(c, str) else getattr(c, "text", "")).lower()
                            for w in _gold_words(_GOLD_ANSWER))
                    for c in context
                )
                print(f"{bar}")
                if any_ans:
                    print(f"  {green(bold('OK'))}  answer string present in S_V context  [Verifier input (skip-llm)]")
                else:
                    print(f"  {red(bold('XX'))}  answer string NOT in S_V context  [Verifier input (skip-llm)]")
                print(f"{bar}")
            else:
                print(f"    {red('* NO CONTEXT — Navigator returned 0 chunks!')}")
                print(f"    {yellow('  -> Possible causes: retrieval error, entity-mention filter too aggressive')}")
            result = VerificationResult(
                answer="[skipped]",
                iterations=0,
                verified_claims=[],
                violated_claims=[],
                all_verified=False,
                pre_validation=None,
                iteration_history=[],
                timing_ms=0.0,
            )
            subsection("VERIFIER RESULT  (simulated)")
            field("Answer",     result.answer)
            field("Confidence", result.confidence.value)
            field("Iterations", result.iterations)
            return result
        verifier.generate_and_verify = _skip_verifier

    # -- Run pipeline -----------------------------------------------------
    # Orchestration lives on AgentPipeline.process(); the result is a
    # PipelineResult dataclass exposing per-stage timing fields.
    t_start = time.time()
    pipeline_result = pipeline.process(q_text)
    total_ms = (time.time() - t_start) * 1000

    # ── Disable sys.settrace ─────────────────────────────────────────────────
    if args.trace_calls:
        sys.settrace(None)
        print_call_trace()

    # ── Summary ──────────────────────────────────────────────────────────────
    section("SUMMARY")
    pred = pipeline_result.answer or ""
    field("Question",   q_text)
    field("Gold",       gold)
    field("Prediction", pred)
    vcfg = getattr(verifier, "config", None)
    field("Model",      vcfg.model_name if vcfg else "(unknown)")
    field("Total time", f"{total_ms:.0f} ms")

    # PipelineResult exposes per-stage timing as top-level fields.
    field("S_P", f"{pipeline_result.planner_time_ms:.0f} ms")
    field("S_N", f"{pipeline_result.navigator_time_ms:.0f} ms")
    field("S_V", f"{pipeline_result.verifier_time_ms:.0f} ms")

    # EM-Check
    import re
    def _norm(t: str) -> str:
        t = t.lower()
        t = re.sub(r'\b(a|an|the)\b', ' ', t)
        t = re.sub(r'[^\w\s]', '', t)
        return ' '.join(t.split())

    pred_n, gold_n = _norm(pred), _norm(gold)
    em = pred_n == gold_n or (
        gold_n and bool(re.search(r'\b' + re.escape(gold_n) + r'\b', pred_n))
    )

    # Soft-EM (token-F1 >= threshold) -- the headline correctness verdict
    # the benchmark uses. Mirrored here so per-trace diagnostics agree
    # with the aggregate eval; otherwise the diagnostic shows "WRONG" on
    # answers the benchmark counts as correct (e.g. a prediction that is
    # the gold answer minus a trailing qualifier word). Threshold is read
    # from settings.yaml (benchmark.answer_f1_threshold; module fallback
    # `_SOFT_EM_F1_THRESHOLD_FALLBACK` mirrors the benchmark default).
    def _token_f1(p_norm: str, g_norm: str) -> float:
        p_toks, g_toks = p_norm.split(), g_norm.split()
        if not p_toks or not g_toks:
            return 0.0
        from collections import Counter
        common = Counter(p_toks) & Counter(g_toks)
        n_common = sum(common.values())
        if n_common == 0:
            return 0.0
        prec = n_common / len(p_toks)
        rec  = n_common / len(g_toks)
        return 2 * prec * rec / (prec + rec)

    try:
        from src.logic_layer._settings_loader import _load_settings
        _f1_threshold = float(
            _load_settings().get("benchmark", {}).get(
                "answer_f1_threshold", _SOFT_EM_F1_THRESHOLD_FALLBACK,
            )
        )
    except Exception:  # noqa: BLE001 -- best-effort: settings missing -> fall back
        _f1_threshold = _SOFT_EM_F1_THRESHOLD_FALLBACK
    f1 = _token_f1(pred_n, gold_n)
    soft_em = bool(em) or f1 >= _f1_threshold

    field("Token F1",   f"{f1:.3f}")
    field("Strict EM",  "yes" if em else "no")
    field("Soft-EM",    f"yes (F1>={_f1_threshold:g})" if soft_em else f"no (F1<{_f1_threshold:g})")
    print(f"\n  Result: {bold(green('OK CORRECT') if soft_em else red('XX WRONG'))}")

    print(f"\n{bold('=' * _BAR_WIDTH)}\n")


if __name__ == "__main__":
    main()
