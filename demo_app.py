"""
demo_app.py — single-query demo of the edge-RAG pipeline (Phase-1 demo paper).

Watch one question flow S_P (Planner) -> S_N (Navigator) -> S_V (Verifier) and
see the edge envelope measured live: per-stage timing, total wall time against
the 60 s budget, and peak resident memory against the ~2 GB / 16 GB envelope.

This is the live artifact for the demo paper / screencast. It reuses the
production pipeline builder and the per-stage trace hooks from
``diagnose_verbose.py`` (so the demo and the diagnostic show identical internals)
and adds two things the diagnostic lacks:

  1. **Live peak-RSS sampling** around ``pipeline.process()`` — a background
     thread polls this process's RSS and the ollama runner's RSS so the demo can
     state "peak X MB, within the 16 GB envelope" on screen.
  2. **An EDGE ENVELOPE panel** — total time vs the 60 s budget and peak RSS vs
     the budget, the headline property of the system.

Unlike ``diagnose_verbose.py`` (which is benchmark/gold-tracking oriented and
keyed by question index), this entry point is built for arbitrary free-text
questions and loads the **frozen** paper config by default, so a demo run uses
exactly the reported contract.

THE TWO PAPER WALKTHROUGHS (copy-paste, PowerShell / Windows)
-------------------------------------------------------------
E1 — end-to-end bridge question (paper Fig. "demo" + "trace"; answer
"Henry J. Kaiser", HIGH confidence, envelope card at the end).
Plain form (verbose diagnostic trace — every filter-chain step):

    python -X utf8 demo_app.py --question "Kaiser Ventures corporation was founded by an American industrialist who became known as the father of modern American shipbuilding?"

Recommended for a LIVE audience — same question, compact attendee view
(planner hops, top-3 per lane, fused list, one-line filter summary,
final context, answer + envelope card — no trace noise):

    python -X utf8 demo_app.py --presenter --question "Kaiser Ventures corporation was founded by an American industrialist who became known as the father of modern American shipbuilding?"

E2 — graph-rescue toggle in ONE command (paper Fig. "rescue"; graph OFF
misses the gold "Bratislava" article, graph ON rescues it at ~#7):

    python -X utf8 demo_app.py --compare --question "Július Satinský was born in a city that has a current population of what?"

    # manual two-command variant of the same toggle:
    python -X utf8 demo_app.py --skip-llm --no-graph --question "..."   # graph OFF
    python -X utf8 demo_app.py --skip-llm            --question "..."   # graph ON

Other flags (any question, not just E1/E2):
    --dataset hotpotqa                 dataset stores to query (default: hotpotqa)
    --config config/settings.yaml      non-frozen config (default: frozen_paper.yaml)
    --skip-llm                         retrieval only, no Verifier/LLM call
    --interactive / -i                 REPL: pipeline stays warm, 2nd+ questions
                                        show steady-state latency (no cold-start hit)
    --presenter / -p                   compact attendee view (see E1 above)
    --compare / -c                     graph OFF vs ON side by side (see E2 above)

Depends-on: P0-T1 (frozen_paper.yaml). Requires the dataset's LanceDB + KuzuDB
stores to be ingested and ollama running (qwen2:1.5b + nomic-embed-text).
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# The diagnostic trace helpers now live in tools/; add it to the path so the
# `from diagnose_verbose import ...` below resolves without a package prefix.
_TOOLS_DIR = PROJECT_ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

# Reuse the production builder, the per-stage trace hooks, and the pretty-print
# helpers from the diagnostic so the demo shows the identical pipeline internals
# without duplicating ~600 lines of trace formatting.
from diagnose_verbose import (  # noqa: E402
    build_pipeline,
    patch_planner,
    patch_retriever,
    patch_navigator,
    patch_verifier,
    patch_controller_bridge,
    section,
    field,
    bold,
    green,
    yellow,
    red,
    cyan,
    dim,
)

# Enable ANSI colour on Windows consoles. PowerShell / cmd do not interpret
# raw \033[..] escapes by default, which is why the demo looked colourless;
# colorama translates them into Win32 console calls (and is a harmless no-op
# on Linux/macOS). Without this the screencast/screenshot shows plain text.
try:
    import colorama  # type: ignore
    colorama.just_fix_windows_console()
except Exception:  # noqa: BLE001 — colour is cosmetic; never fail the demo for it
    pass

# Edge-envelope thresholds the demo reports against. 60 s is the latency budget
# used throughout the paper; the memory envelope is the 16 GB device target,
# with ~2 GB highlighted as the measured operating point (latency_memory_profile
# reported ~2 GB peak RSS).
_LATENCY_BUDGET_S = 60.0
_MEMORY_BUDGET_MB = 16 * 1024
_MEMORY_OPERATING_POINT_MB = 2 * 1024

# Frozen paper config is the default so a demo run uses the reported contract.
_FROZEN_CONFIG = PROJECT_ROOT / "config" / "frozen_paper.yaml"
_DEFAULT_CONFIG = PROJECT_ROOT / "config" / "settings.yaml"

_BYTES_PER_MB = 1024 * 1024

try:
    import psutil  # type: ignore
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False


# ---------------------------------------------------------------------------
# Live memory sampling
# ---------------------------------------------------------------------------

def _process_tree_rss_mb() -> float:
    """RSS (MB) of this Python process plus every ollama process.

    The retrieval stack (LanceDB / KuzuDB / embedding buffers) lives in THIS
    process; the LLM weights live in the ollama runner. Their sum is the system
    footprint the edge claim is about. Returns 0.0 if psutil is unavailable
    (the panel then prints "n/a" rather than a misleading 0).
    """
    if not _PSUTIL_OK:
        return 0.0
    total = 0.0
    try:
        total += psutil.Process(os.getpid()).memory_info().rss / _BYTES_PER_MB
    except Exception:  # noqa: BLE001 — sampling must never raise into the demo
        pass
    try:
        for proc in psutil.process_iter(["name"]):
            try:
                if "ollama" in (proc.info.get("name") or "").lower():
                    total += proc.memory_info().rss / _BYTES_PER_MB
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:  # noqa: BLE001
        pass
    return total


class _PeakRSSSampler:
    """Background thread that records the peak process-tree RSS during a call.

    ``pipeline.process()`` is a single blocking call, so a sampler thread polls
    RSS every ``interval`` seconds between start() and stop() and keeps the max.
    Best-effort: if psutil is missing the peak stays 0.0 and the panel says so.
    """

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, _process_tree_rss_mb())
            self._stop.wait(self.interval)

    def __enter__(self) -> "_PeakRSSSampler":
        # Seed with an immediate reading so a very fast call still has a value.
        self.peak_mb = _process_tree_rss_mb()
        if _PSUTIL_OK:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak_mb = max(self.peak_mb, _process_tree_rss_mb())


# ---------------------------------------------------------------------------
# Envelope panel
# ---------------------------------------------------------------------------

def _print_envelope_panel(total_ms: float, planner_ms: float, navigator_ms: float,
                          verifier_ms: float, peak_rss_mb: float,
                          cold_start: bool = False) -> None:
    """Print the headline edge-envelope readout for the demo.

    ``cold_start`` marks the FIRST question of a fresh process, whose latency
    includes the one-time warm-up (GLiNER + cross-encoder weights, lazy BM25
    index build). The paper's 60 s budget is a steady-state contract, so an
    over-budget cold first question is reported as COLD START (yellow), not as
    an envelope failure (red) — and never the other way around: a warm
    over-budget question still fails red.
    """
    section("EDGE ENVELOPE  (the headline property)")

    # Latency vs the 60 s budget.
    total_s = total_ms / 1000.0
    lat_ok = total_s <= _LATENCY_BUDGET_S
    lat_mark = (green("WITHIN") if lat_ok
                else yellow("OVER — COLD START") if cold_start
                else red("OVER"))
    field("Total latency",
          f"{total_s:6.2f} s  /  {_LATENCY_BUDGET_S:.0f} s budget   [{lat_mark}]")
    if not lat_ok and cold_start:
        field("", yellow("(first question of this session — includes one-time "
                         "warm-up; ask another question for the steady-state "
                         "timing the paper reports, e.g. via --interactive)"))
    elif not lat_ok:
        field("", yellow("(warm run over budget — this one counts against the "
                         "envelope)"))
    field("  S_P (Planner)",   f"{planner_ms:8.0f} ms")
    field("  S_N (Navigator)", f"{navigator_ms:8.0f} ms")
    field("  S_V (Verifier)",  f"{verifier_ms:8.0f} ms  "
          + dim("(LLM generation — usually dominates)"))

    # Memory vs the 16 GB envelope (with the ~2 GB operating point called out).
    if peak_rss_mb > 0:
        mem_ok = peak_rss_mb <= _MEMORY_BUDGET_MB
        mem_mark = green("WITHIN") if mem_ok else red("OVER")
        op_note = (green("at/below the ~2 GB operating point")
                   if peak_rss_mb <= _MEMORY_OPERATING_POINT_MB
                   else yellow("above the ~2 GB operating point but within 16 GB")
                   if mem_ok else red("exceeds the 16 GB envelope"))
        field("Peak RSS",
              f"{peak_rss_mb:7.0f} MB  /  {_MEMORY_BUDGET_MB} MB envelope   [{mem_mark}]")
        field("  Footprint", op_note)
    else:
        field("Peak RSS", dim("n/a (psutil not installed — `pip install psutil`)"))

    print()
    mem_ok = (peak_rss_mb == 0) or (peak_rss_mb <= _MEMORY_BUDGET_MB)
    if total_s <= _LATENCY_BUDGET_S and mem_ok:
        verdict = green(bold("OK  fits the edge envelope"))
    elif cold_start and mem_ok:
        verdict = yellow(bold("~~  cold start over budget (one-time warm-up "
                              "included) — steady-state is the reported metric"))
    else:
        verdict = red(bold("XX  exceeds the edge envelope"))
    print(f"  Verdict: {verdict}")


# ---------------------------------------------------------------------------
# One-shot summary card (the screenshot target for the demo paper / Figure 1)
# ---------------------------------------------------------------------------

def _print_summary_card(question: str, answer: str, confidence: str,
                        total_ms: float, planner_ms: float, navigator_ms: float,
                        verifier_ms: float, peak_rss_mb: float,
                        dataset: str, cold_start: bool = False) -> None:
    """Print a single, high-contrast, screenshot-able summary of the whole run.

    Designed to be captured as one image (paper Figure 1 / screencast end
    card): a boxed panel with the question, the answer, confidence, and the
    two edge-envelope verdicts (latency + memory) in contrasting colours.
    A cold FIRST question that is over budget (one-time warm-up included)
    frames yellow with an explicit COLD START verdict — the 60 s budget is a
    steady-state contract — while a warm over-budget question still fails red.
    """
    total_s = total_ms / 1000.0
    lat_ok = total_s <= _LATENCY_BUDGET_S
    mem_ok = (peak_rss_mb == 0) or (peak_rss_mb <= _MEMORY_BUDGET_MB)
    all_ok = lat_ok and mem_ok
    cold_over = (not lat_ok) and cold_start and mem_ok

    width = 72
    bar = "═" * width
    # Colour the frame by the overall verdict so the screenshot reads at a glance.
    frame = (green if all_ok else yellow if cold_over else red)

    def row(text: str) -> str:
        return "  " + frame("║ ") + text

    print("\n  " + frame("╔" + bar + "╗"))
    title = bold(cyan("  EDGE-RAG  -  one-question summary"))
    print(row(title))
    print("  " + frame("╠" + bar + "╣"))

    # Question (wrapped to the card width).
    import textwrap
    q_lines = textwrap.wrap(question, width=width - 12) or ["(none)"]
    print(row(bold("Question : ") + q_lines[0]))
    for cont in q_lines[1:]:
        print(row("           " + cont))

    # Answer — the single most important line; make it loud.
    ans = (answer or "(no answer)").strip()
    print(row(bold("Answer   : ") + green(bold(ans))))

    # Confidence with a colour by level.
    conf = str(confidence or "n/a").upper()
    conf_col = (green if "HIGH" in conf else
                yellow if "MED" in conf else
                red if "LOW" in conf else cyan)
    print(row(bold("Confidence: ") + conf_col(bold(conf)) + dim(f"   ·  dataset={dataset}")))

    print("  " + frame("╟" + "─" * width + "╢"))

    # Latency verdict.
    lat_mark = (green("WITHIN 60 s") if lat_ok
                else yellow("OVER 60 s — COLD START") if cold_over
                else red("OVER 60 s"))
    print(row(bold("Latency  : ")
              + f"{total_s:5.1f}s  ["
              + lat_mark + "]  "
              + dim(f"P={planner_ms:.0f} N={navigator_ms:.0f} V={verifier_ms:.0f} ms")))

    # Memory verdict.
    if peak_rss_mb > 0:
        mem_mark = green("WITHIN 16 GB") if mem_ok else red("OVER 16 GB")
        op = (green("~2 GB op-point")
              if peak_rss_mb <= _MEMORY_OPERATING_POINT_MB
              else yellow("> 2 GB, < 16 GB"))
        print(row(bold("Memory   : ")
                  + f"{peak_rss_mb:5.0f} MB  [" + mem_mark + "]  " + op))
    else:
        print(row(bold("Memory   : ") + dim("n/a (pip install psutil)")))

    print("  " + frame("╠" + bar + "╣"))
    if all_ok:
        verdict = green(bold("  ✔  FITS THE EDGE ENVELOPE  (CPU-only · no GPU · no cloud)"))
    elif cold_over:
        verdict = yellow(bold("  ~  COLD START (one-time warm-up) — ask another "
                              "question for steady state"))
    else:
        verdict = red(bold("  ✘  EXCEEDS THE EDGE ENVELOPE"))
    print(row(verdict))
    print("  " + frame("╚" + bar + "╝") + "\n")


# ---------------------------------------------------------------------------
# Presenter / compare support: lightweight per-lane + fused capture
# ---------------------------------------------------------------------------

def _title_from_source(source_doc: str, dataset: str) -> str:
    """'hotpotqa_Nicki Minaj' -> 'Nicki Minaj' (strip the dataset prefix)."""
    s = str(source_doc or "?")
    prefix = f"{dataset}_"
    return s[len(prefix):] if s.startswith(prefix) else s


class _LaneCapture:
    """Record per-lane and fused retrieval results for the presenter/compare views.

    Wraps the three lane primitives (``store.vector_search``,
    ``store.graph_search``, ``retriever._bm25_search``) and the fused
    ``retriever.retrieve`` on the LIVE instances, so the demo can show
    "which lane surfaced which article" without the full diagnostic trace.
    Capture only — inputs and outputs pass through unchanged, so the
    pipeline behaves identically with or without the wrapper.
    """

    def __init__(self, store, retriever, dataset: str) -> None:
        self.dataset = dataset
        self.reset()
        self._install(store, retriever)

    def reset(self) -> None:
        self.vector: list = []    # titles, capture order (= rank order per call)
        self.bm25: list = []      # titles
        self.graph: list = []     # (title, hops, bridge_entity)
        self.fused: list = []     # (title, retrieval_method)
        self._text2title: dict = {}

    def _dedup_add(self, lst: list, item) -> None:
        if item not in lst:
            lst.append(item)

    def _install(self, store, retriever) -> None:
        _orig_vec, _orig_graph = store.vector_search, store.graph_search
        _orig_bm25, _orig_retrieve = retriever._bm25_search, retriever.retrieve

        def vec_wrap(*a, **kw):
            out = _orig_vec(*a, **kw)
            for r in out:
                src = (r.get("metadata") or {}).get("source_file", "?")
                self._dedup_add(self.vector, _title_from_source(src, self.dataset))
            return out

        def graph_wrap(*a, **kw):
            out = _orig_graph(*a, **kw)
            for r in out:
                t = _title_from_source(r.get("source_file", "?"), self.dataset)
                entry = (t, r.get("hops", 0), r.get("bridge_entity"))
                if all(e[0] != t for e in self.graph):
                    self.graph.append(entry)
            return out

        def bm25_wrap(*a, **kw):
            out = _orig_bm25(*a, **kw)
            for r in out:
                src = (r.get("metadata") or {}).get("source_file", "?")
                self._dedup_add(self.bm25, _title_from_source(src, self.dataset))
            return out

        def retrieve_wrap(*a, **kw):
            results, metrics = _orig_retrieve(*a, **kw)
            for r in results:
                t = _title_from_source(getattr(r, "source_doc", "?"), self.dataset)
                self._dedup_add(self.fused, (t, getattr(r, "retrieval_method", "?")))
                # Map chunk text -> title so the final filtered context (plain
                # texts) can be labelled. Keyed by an 80-char prefix; context
                # shrinkage rarely alters chunks below its 800-char budget.
                self._text2title[getattr(r, "text", "")[:80]] = t
            return results, metrics

        store.vector_search = vec_wrap
        store.graph_search = graph_wrap
        retriever._bm25_search = bm25_wrap
        retriever.retrieve = retrieve_wrap

    def title_of_chunk(self, chunk_text: str) -> str:
        t = self._text2title.get((chunk_text or "")[:80])
        if t is not None:
            return t
        # Fallback: prefix containment against the recorded map.
        frag = (chunk_text or "")[:60]
        for k, v in self._text2title.items():
            if k.startswith(frag) or frag.startswith(k[:60]):
                return v
        return "?"


def _print_presenter_summary(result, capture: "_LaneCapture") -> None:
    """Compact attendee view: planner hops, top-3 per lane, fused list,
    one-line filter summary, final context titles. Mirrors exactly what the
    paper says the attendee 'sees and learns' — without the diagnostic trace."""
    plan = result.planner_result if isinstance(result.planner_result, dict) else {}
    nav = result.navigator_result if isinstance(result.navigator_result, dict) else {}
    meta = nav.get("metadata") or {}

    # ── S_P: what the rule-based planner emitted (no LLM call) ──────────────
    section("S_P PLANNER  (rule-based, no LLM)")
    ents = [e.get("text", "") for e in plan.get("entities", []) if e.get("text")]
    field("Query type", str(plan.get("query_type", "n/a"))
          + (f"   pattern={plan.get('matched_pattern')}" if plan.get("matched_pattern") else ""))
    if ents:
        field("Entities", ", ".join(ents[:6]))
    hops = plan.get("hop_sequence") or []
    if hops:
        for h in hops:
            tag = " [bridge]" if h.get("is_bridge") else ""
            # Printed directly (not via field()) so long hop text still
            # renders bright: field() dims + wraps any value over 100 chars,
            # which would make hop 1 (often the full original question) grey.
            label = bold(f"  hop {h.get('step_id', '?')}:")
            value = f"{str(h.get('sub_query', ''))[:160]}{tag}"
            print(f"    {label} {value}")
    else:
        field("Hops", "single-pass (no decomposition)")

    # ── S_N: the three lanes side by side ────────────────────────────────────
    section("S_N LANES  (top-3 retrieved articles per lane)")
    field("dense ", " | ".join(capture.vector[:3]) or dim("(no hits)"))
    field("bm25  ", " | ".join(capture.bm25[:3]) or dim("(no hits)"))
    if capture.graph:
        parts = []
        for t, hops_n, bridge in capture.graph[:3]:
            parts.append(f"{t} " + (dim(f"(bridge: {bridge})") if bridge
                                    else dim("(direct mention)")))
        field("graph ", " | ".join(parts))
    else:
        field("graph ", dim("(no hits — lane off or no entity anchor)"))

    fused = [f"{t}{dim('[' + m + ']')}" for t, m in capture.fused[:6]]
    if fused:
        field("RRF fused", " | ".join(fused))

    # ── Filter chain: one line ────────────────────────────────────────────────
    filtered = nav.get("filtered_context") or []
    counts = [meta.get("pre_filter_count"),
              meta.get("after_relevance_filter"),
              meta.get("after_redundancy_filter"),
              meta.get("after_contradiction_filter"),
              meta.get("after_entity_overlap_pruning"),
              meta.get("after_entity_mention_filter"),
              len(filtered)]
    if counts[0] is not None:
        chain = " -> ".join(str(c) for c in counts if c is not None)
        field("Filters", chain + " chunks "
              + dim("(fused -> relevance -> redundancy -> contradiction -> "
                    "entity-overlap -> entity-mention -> cap/shrink)"))

    # ── Final context the Verifier reads ──────────────────────────────────────
    titles = [capture.title_of_chunk(c) for c in filtered]
    section(f"FINAL CONTEXT  ({len(filtered)} chunks -> Verifier)")
    for i, t in enumerate(titles):
        field(f"  [{i}]", t)


def _print_compare_panel(question: str, off_titles: list, on_titles: list,
                         off_s: float, on_s: float) -> None:
    """Side-by-side retrieved-title lists: graph lane OFF vs ON, with
    graph-rescued documents highlighted. The paper's central retrieval claim
    as one ten-second visual."""
    col = 36

    def cell(t: str) -> str:
        return t[:col - 2].ljust(col)

    rescued = [t for t in on_titles if t not in off_titles]
    dropped = [t for t in off_titles if t not in on_titles]

    section("GRAPH-RESCUE COMPARE  (same question, one flag difference)")
    field("Question", question[:70])
    print()
    print("  " + bold(cell("graph OFF  (dense+BM25)")) + bold("graph ON  (hybrid, default)"))
    print("  " + "-" * (col * 2 + 4))
    for i in range(max(len(off_titles), len(on_titles))):
        left = off_titles[i] if i < len(off_titles) else ""
        right = on_titles[i] if i < len(on_titles) else ""
        l_txt = cell(f"{i+1}. {left}" if left else "")
        if right and right in rescued:
            r_txt = green(bold(f"{i+1}. {right}")) + "  " + green("<< GRAPH RESCUE")
        else:
            r_txt = f"{i+1}. {right}" if right else ""
        print("  " + (dim(l_txt) if left and left in dropped else l_txt) + r_txt)
    print("  " + "-" * (col * 2 + 4))
    if rescued:
        field("Rescued by graph lane", green(bold(" | ".join(rescued))))
        if dropped:
            field("Displaced distractor", dim(" | ".join(dropped)))
        print("  " + green("The two runs differ ONLY by the graph lane — the "
                           "rescued document is attributable to it alone."))
    else:
        field("Rescued by graph lane", dim("(none — lanes agree on this "
                                           "question; try a bridge question "
                                           "whose gold document shares no "
                                           "words with the question)"))
    field("Latency", f"OFF {off_s:.1f} s  |  ON {on_s:.1f} s   "
          + dim("(retrieval-only, no LLM)"))
    print()


def _run_compare(pipeline, retriever, capture: "_LaneCapture",
                 question: str) -> None:
    """Run `question` with the graph lane OFF then ON and print the panel.

    Requires the pipeline to have been built with enable_verifier=False and
    enable_caching=False (main() enforces this for --compare): caching keys on
    the query string alone, so a cache hit would silently reuse the OFF result
    for the ON run.
    """
    from src.data_layer.hybrid_retriever import RetrievalMode

    def one_pass(mode) -> tuple:
        retriever.config.mode = mode
        capture.reset()
        t0 = time.time()
        result = pipeline.process(question)
        elapsed = time.time() - t0
        nav = result.navigator_result if isinstance(result.navigator_result, dict) else {}
        chunks = nav.get("filtered_context") or []
        return [capture.title_of_chunk(c) for c in chunks], elapsed

    # Warm the Ollama embedding endpoint before either timed pass. Ollama is a
    # separate server process from this pipeline: even a "warm" pipeline can
    # hit a cold embedding-model load on its first HTTP call if the model had
    # idled out of Ollama's own residency. Untimed here, that one-time cost
    # would otherwise land entirely on the OFF pass (it runs first) and make
    # the OFF/ON gap look like a graph-lane effect when it is not.
    try:
        retriever._embed_query(question)
    except (OSError, RuntimeError, ValueError, ConnectionError, AttributeError):
        pass

    off_titles, off_s = one_pass(RetrievalMode.VECTOR)
    on_titles, on_s = one_pass(RetrievalMode.HYBRID)
    _print_compare_panel(question, off_titles, on_titles, off_s, on_s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Keep the console clean for the audience/screencast: transformers'
    # per-layer weight-materialisation report and HF progress bars otherwise
    # dump hundreds of lines on the first (cold) reranker/GLiNER load.
    # setdefault: an explicitly configured environment still wins.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    parser = argparse.ArgumentParser(
        description="Single-query edge-RAG demo (S_P -> S_N -> S_V) with live "
                    "latency + peak-RSS against the edge envelope.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--question", "-q", type=str, default=None,
                        help="The question to run end-to-end (free text). "
                             "Required unless --interactive is given.")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="After the first question, keep the pipeline warm "
                             "and prompt for further questions (empty line or "
                             "'exit' quits). The 2nd+ questions demonstrate the "
                             "steady-state latency the paper reports — the "
                             "one-time warm-up is paid only by the first.")
    parser.add_argument("--dataset", "-d", type=str, default="hotpotqa",
                        help="Dataset stores to query "
                             "(data/<dataset>/{vector,graph}). Default: hotpotqa.")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML. Default: config/frozen_paper.yaml "
                             "(the reported contract); falls back to "
                             "config/settings.yaml if the frozen file is absent.")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip the Verifier LLM call (retrieval-only demo).")
    parser.add_argument("--retrieval-mode", choices=["hybrid", "vector", "graph"],
                        default=None,
                        help="Override the retrieval lanes: 'hybrid' = dense+BM25"
                             "+graph (default); 'vector' = dense+BM25 only (graph "
                             "OFF); 'graph' = graph only. Use this to run the "
                             "graph-rescue toggle: compare --retrieval-mode vector "
                             "vs. hybrid on the same bridge question.")
    parser.add_argument("--no-graph", action="store_true",
                        help="Shortcut for --retrieval-mode vector: disable the "
                             "knowledge-graph lane (dense+BM25 only).")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI colour (for plain-text capture).")
    parser.add_argument("--presenter", "-p", action="store_true",
                        help="Compact attendee view: planner hops, top-3 "
                             "articles per retrieval lane, RRF-fused list, "
                             "one-line filter summary, final context, answer "
                             "+ envelope card. Hides the full diagnostic "
                             "trace (the default output).")
    parser.add_argument("--compare", "-c", action="store_true",
                        help="Graph-rescue toggle in ONE command: run the "
                             "question with the graph lane OFF (dense+BM25) "
                             "and ON (hybrid) and print the two retrieved "
                             "title lists side by side, highlighting graph-"
                             "rescued documents. Retrieval-only (no LLM); "
                             "overrides --retrieval-mode/--no-graph/--skip-llm.")
    args = parser.parse_args()

    # A question is required up front unless --interactive prompts for one.
    # Reject whitespace-only questions here with a usage message instead of
    # letting AgentPipeline.process() raise a raw ValueError traceback.
    if args.question is not None and not args.question.strip():
        parser.error("--question must be a non-empty string")
    if args.question is None and not args.interactive:
        parser.error("--question is required (or pass --interactive)")

    if args.no_color:
        # diagnose_verbose reads its module-level USE_COLOR; flip it there too.
        import diagnose_verbose as dv
        dv.USE_COLOR = False

    # ── Dataset validation (before any model/store loading) ─────────────────
    # A typo'd dataset name would otherwise silently create empty
    # data/<typo>/{vector,graph} stores and "succeed" with an unanswerable
    # empty context — fail fast with the list of datasets actually present.
    dataset_dir = PROJECT_ROOT / "data" / args.dataset
    if not (dataset_dir / "vector").exists():
        available = sorted(
            p.name for p in (PROJECT_ROOT / "data").iterdir()
            if p.is_dir() and (p / "vector").exists()
        ) if (PROJECT_ROOT / "data").exists() else []
        avail_str = ", ".join(available) if available else (
            "(none — run scripts/fetch_data.ps1 or see REPRODUCE.md)")
        print(f"\n  {red(bold('Unknown dataset'))} {args.dataset!r}: "
              f"no stores at {dataset_dir / 'vector'}")
        print(f"  Available datasets: {avail_str}")
        sys.exit(1)

    # Resolve config: explicit --config wins; else frozen; else settings.
    if args.config:
        config_path = Path(args.config)
    elif _FROZEN_CONFIG.exists():
        config_path = _FROZEN_CONFIG
    else:
        config_path = _DEFAULT_CONFIG

    # Load through the canonical settings loader (not raw yaml.safe_load) so
    # the demo gets the same treatment as every production entry point: the
    # 37-key _REQUIRED_SETTINGS validation (missing keys warn instead of
    # silently falling back to dataclass defaults) and the OLLAMA_HOST env
    # override for containerised / relocated Ollama endpoints.
    from src.logic_layer._settings_loader import _load_settings
    cfg = _load_settings(config_path)
    if not cfg:
        print(f"  {red('Could not load config:')} {config_path}")
        sys.exit(1)

    # ── --compare: single-command graph-rescue toggle ────────────────────────
    # Compare mode owns the retrieval-mode switching itself (OFF pass, then ON
    # pass) and is retrieval-only by construction. It also REQUIRES caching
    # off: the pipeline cache keys on the query string alone, so a cache hit
    # would silently replay the OFF result for the ON pass.
    if args.compare:
        args.skip_llm = True
        args.retrieval_mode = None
        args.no_graph = False
        cfg.setdefault("agent", {})["enable_caching"] = False

    # ── Retrieval-mode override (the graph-rescue toggle) ───────────────────
    # --no-graph is a shortcut for --retrieval-mode vector. Mutating cfg["rag"]
    # here is the single reproducible toggle the paper describes: re-run the
    # same question with mode=vector (graph OFF) vs mode=hybrid (graph ON).
    mode_override = "vector" if args.no_graph else args.retrieval_mode
    if mode_override:
        cfg.setdefault("rag", {})["retrieval_mode"] = mode_override

    # ── --skip-llm: retrieval-only mode ──────────────────────────────────────
    # Disabling the Verifier in the pipeline config is what actually skips the
    # LLM call: AgentPipeline.process() gates S_V on agent.enable_verifier.
    # (Merely not attaching the verifier trace hook below would still run the
    # full generation — the retrieval-only demo must not touch Ollama's LLM.)
    if args.skip_llm:
        cfg.setdefault("agent", {})["enable_verifier"] = False

    # ── Header ──────────────────────────────────────────────────────────────
    print("\n" + bold("=" * 78))
    print(f"  {bold(cyan('EDGE-RAG DEMO'))}  —  S_P -> S_N -> S_V on a single question")
    print(bold("=" * 78))
    field("Question", args.question or dim("(interactive — prompted below)"))
    field("Dataset",  args.dataset)
    field("Config",   f"{config_path.name}")
    field("Retrieval", cfg.get("rag", {}).get("retrieval_mode", "hybrid")
          + ("  (graph lane OFF)" if mode_override == "vector" else ""))
    if args.compare:
        field("Mode", cyan(bold("COMPARE — graph lane OFF vs ON, retrieval-only")))
    elif args.skip_llm:
        field("Mode", yellow("retrieval-only (--skip-llm: Verifier/LLM disabled)"))
    if args.presenter:
        field("View", dim("presenter (compact attendee view; full trace hidden)"))
    if not _PSUTIL_OK:
        print(f"  {yellow('* psutil not installed — peak-RSS readout disabled '
                          '(pip install psutil)')}")

    # ── Build pipeline (reuses the production builder) ──────────────────────
    # The first build after Ollama idles out its models can take 10-35 s
    # (embedding-model cold load) — say so instead of looking hung, and turn
    # an infrastructure failure into one actionable line, not a stack trace.
    print(f"\n{dim('Loading pipeline... (this process has no pipeline yet, so this '
                    'always builds it from scratch — Ollama model + NER/reranker '
                    'weights — can take 1-2 min; a fresh `python demo_app.py` call '
                    'pays this again even if you ran it a moment ago, use '
                    '--interactive to ask several questions in one warm process)')} ",
          end="", flush=True)
    t0 = time.time()
    try:
        pipeline, planner, verifier, store, retriever = build_pipeline(
            cfg, dataset=args.dataset
        )
    except ConnectionError as exc:
        print(red("FAILED"))
        print(f"\n  {red(bold('Cannot reach Ollama:'))} {exc}")
        print(f"  {yellow('Fix: start it with `ollama serve`, pre-warm with '
                          '`ollama run nomic-embed-text`, then re-run.')}\n")
        sys.exit(1)
    except RuntimeError as exc:
        print(red("FAILED"))
        print(f"\n  {red(bold('Pipeline startup failed:'))} {exc}\n")
        sys.exit(1)
    print(f"{green('OK')}  {dim(f'({(time.time() - t0) * 1000:.0f} ms)')}")

    # ── Attach the per-stage trace hooks (same as the diagnostic) ───────────
    # Presenter and compare modes replace the verbose trace with the compact
    # _LaneCapture-backed views, so the diagnostic hooks stay off there.
    capture: Optional[_LaneCapture] = None
    if args.presenter or args.compare:
        capture = _LaneCapture(store, retriever, args.dataset)
    else:
        patch_planner(planner)
        patch_retriever(retriever)
        patch_navigator(pipeline.navigator)
        patch_controller_bridge()
        if not args.skip_llm:
            patch_verifier(verifier)

    def _run_one_question(question: str, cold_start: bool) -> None:
        """Run one question end-to-end and print answer, envelope, and card.

        ``cold_start`` marks the first question of this process — its latency
        includes the one-time warm-up, which the envelope panel and summary
        card report as COLD START (yellow) rather than an envelope failure.
        """
        if capture is not None:
            capture.reset()

        # ── Run, sampling peak RSS for the duration ──────────────────────────
        with _PeakRSSSampler() as sampler:
            t_start = time.time()
            result = pipeline.process(question)
            total_ms = (time.time() - t_start) * 1000

        # ── Presenter view (compact; replaces the verbose trace) ─────────────
        if args.presenter and capture is not None:
            _print_presenter_summary(result, capture)

        # ── Answer + confidence ──────────────────────────────────────────────
        section("ANSWER")
        field("Answer", result.answer or "(no answer)")
        conf = getattr(result, "confidence", None)
        if conf is not None:
            # PipelineResult.confidence may be an enum or a float depending on path.
            field("Confidence", getattr(conf, "value", conf))
        if getattr(result, "cached_result", False):
            field("", dim("(served from the FIFO result cache — identical "
                          "repeat question)"))

        # ── Reranker status (paper-contract visibility) ──────────────────────
        # The cross-encoder loads lazily on the first navigate() call and
        # silently degrades to raw RRF order if it cannot load (e.g. offline
        # with a cold HF cache) — surface that state instead of hiding it.
        _rr = getattr(pipeline.navigator, "_reranker", None)
        _rr_enabled = getattr(pipeline.navigator.config, "enable_reranker", False)
        if not _rr_enabled:
            field("Reranker", dim("disabled by config"))
        elif _rr is False:
            field("Reranker", red("OFF — model failed to load; ranking is raw "
                                  "RRF (differs from the paper contract)"))
        else:
            field("Reranker", green("active")
                  + dim(f"  ({pipeline.navigator.config.reranker_model})"))

        # ── Edge-envelope panel (the headline) ──────────────────────────────
        planner_ms = getattr(result, "planner_time_ms", 0.0) or 0.0
        navigator_ms = getattr(result, "navigator_time_ms", 0.0) or 0.0
        verifier_ms = getattr(result, "verifier_time_ms", 0.0) or 0.0
        _print_envelope_panel(
            total_ms=total_ms,
            planner_ms=planner_ms,
            navigator_ms=navigator_ms,
            verifier_ms=verifier_ms,
            peak_rss_mb=sampler.peak_mb,
            cold_start=cold_start,
        )
        print(f"\n{bold('=' * 78)}\n")

        # ── One-shot SUMMARY CARD (the screenshot target — paper Figure 1) ───
        conf_val = getattr(result, "confidence", None)
        conf_str = getattr(conf_val, "value", conf_val) if conf_val is not None else "n/a"
        _print_summary_card(
            question=question,
            answer=result.answer or "(no answer)",
            confidence=str(conf_str),
            total_ms=total_ms,
            planner_ms=planner_ms,
            navigator_ms=navigator_ms,
            verifier_ms=verifier_ms,
            peak_rss_mb=sampler.peak_mb,
            dataset=args.dataset,
            cold_start=cold_start,
        )

    def _run(question: str, cold_start: bool) -> None:
        """Dispatch one question to compare mode or the standard runner."""
        if args.compare:
            _run_compare(pipeline, retriever, capture, question)
        else:
            _run_one_question(question, cold_start=cold_start)

    # ── First question (pays the one-time warm-up) ───────────────────────────
    first_question = args.question
    if first_question is None:  # --interactive without --question
        try:
            first_question = input(bold("Question> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not first_question or first_question.lower() in ("exit", "quit"):
            return
    _run(first_question, cold_start=True)

    # ── Interactive REPL: 2nd+ questions run warm (steady state) ────────────
    if args.interactive:
        print(dim("  Interactive mode — pipeline stays warm. Enter the next "
                  "question (empty line or 'exit' to quit)."))
        while True:
            try:
                nxt = input(bold("\nQuestion> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not nxt or nxt.lower() in ("exit", "quit"):
                break
            _run(nxt, cold_start=False)


if __name__ == "__main__":
    main()
