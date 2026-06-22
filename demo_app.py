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

Usage (PowerShell / Windows: prefix `python -X utf8`):
    python -X utf8 demo_app.py --question "Who directed the film that won Best Picture in 1994?"
    python -X utf8 demo_app.py --question "..." --dataset hotpotqa
    python -X utf8 demo_app.py --question "..." --config config/settings.yaml   # non-frozen
    python -X utf8 demo_app.py --question "..." --skip-llm                      # retrieval only

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
                          verifier_ms: float, peak_rss_mb: float) -> None:
    """Print the headline edge-envelope readout for the demo."""
    section("EDGE ENVELOPE  (the headline property)")

    # Latency vs the 60 s budget.
    total_s = total_ms / 1000.0
    lat_ok = total_s <= _LATENCY_BUDGET_S
    lat_mark = green("WITHIN") if lat_ok else red("OVER")
    field("Total latency",
          f"{total_s:6.2f} s  /  {_LATENCY_BUDGET_S:.0f} s budget   [{lat_mark}]")
    if not lat_ok:
        # The first query of a session pays a one-time warm-up (model load,
        # cold GLiNER/cross-encoder weights, lazy BM25 index build). Reported
        # budget compliance is steady-state; flag this so a cold first run is
        # not misread as a per-query failure.
        field("", yellow("(likely cold start — re-run the same query for the "
                         "warm, steady-state timing the paper reports)"))
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
    verdict = (green(bold("OK  fits the edge envelope")) if (total_s <= _LATENCY_BUDGET_S
               and (peak_rss_mb == 0 or peak_rss_mb <= _MEMORY_BUDGET_MB))
               else red(bold("XX  exceeds the edge envelope")))
    print(f"  Verdict: {verdict}")


# ---------------------------------------------------------------------------
# One-shot summary card (the screenshot target for the demo paper / Figure 1)
# ---------------------------------------------------------------------------

def _print_summary_card(question: str, answer: str, confidence: str,
                        total_ms: float, planner_ms: float, navigator_ms: float,
                        verifier_ms: float, peak_rss_mb: float,
                        dataset: str) -> None:
    """Print a single, high-contrast, screenshot-able summary of the whole run.

    Designed to be captured as one image (paper Figure 1 / screencast end
    card): a boxed panel with the question, the answer, confidence, and the
    two edge-envelope verdicts (latency + memory) in contrasting colours.
    """
    total_s = total_ms / 1000.0
    lat_ok = total_s <= _LATENCY_BUDGET_S
    mem_ok = (peak_rss_mb == 0) or (peak_rss_mb <= _MEMORY_BUDGET_MB)
    all_ok = lat_ok and mem_ok

    width = 72
    bar = "═" * width
    # Colour the frame by the overall verdict so the screenshot reads at a glance.
    frame = (green if all_ok else red)

    def row(text: str) -> str:
        return "  " + frame("║ ") + text

    print("\n  " + frame("╔" + bar + "╗"))
    title = bold(cyan("  EDGE-RAG  ·  one-question summary"))
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
    lat_mark = green("WITHIN 60 s") if lat_ok else red("OVER 60 s")
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
    verdict = (green(bold("  ✔  FITS THE EDGE ENVELOPE  (CPU-only · no GPU · no cloud)"))
               if all_ok else
               red(bold("  ✘  EXCEEDS THE EDGE ENVELOPE")))
    print(row(verdict))
    print("  " + frame("╚" + bar + "╝") + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-query edge-RAG demo (S_P -> S_N -> S_V) with live "
                    "latency + peak-RSS against the edge envelope.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--question", "-q", type=str, required=True,
                        help="The question to run end-to-end (free text).")
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
    args = parser.parse_args()

    if args.no_color:
        # diagnose_verbose reads its module-level USE_COLOR; flip it there too.
        import diagnose_verbose as dv
        dv.USE_COLOR = False

    # Resolve config: explicit --config wins; else frozen; else settings.
    if args.config:
        config_path = Path(args.config)
    elif _FROZEN_CONFIG.exists():
        config_path = _FROZEN_CONFIG
    else:
        config_path = _DEFAULT_CONFIG

    import yaml
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ── Retrieval-mode override (the graph-rescue toggle) ───────────────────
    # --no-graph is a shortcut for --retrieval-mode vector. Mutating cfg["rag"]
    # here is the single reproducible toggle the paper describes: re-run the
    # same question with mode=vector (graph OFF) vs mode=hybrid (graph ON).
    mode_override = "vector" if args.no_graph else args.retrieval_mode
    if mode_override:
        cfg.setdefault("rag", {})["retrieval_mode"] = mode_override

    # ── Header ──────────────────────────────────────────────────────────────
    print("\n" + bold("=" * 78))
    print(f"  {bold(cyan('EDGE-RAG DEMO'))}  —  S_P -> S_N -> S_V on a single question")
    print(bold("=" * 78))
    field("Question", args.question)
    field("Dataset",  args.dataset)
    field("Config",   f"{config_path.name}")
    field("Retrieval", cfg.get("rag", {}).get("retrieval_mode", "hybrid")
          + ("  (graph lane OFF)" if mode_override == "vector" else ""))
    if not _PSUTIL_OK:
        print(f"  {yellow('* psutil not installed — peak-RSS readout disabled '
                          '(pip install psutil)')}")

    # ── Build pipeline (reuses the production builder) ──────────────────────
    print(f"\n{dim('Loading pipeline...')} ", end="", flush=True)
    t0 = time.time()
    pipeline, planner, verifier, store, retriever = build_pipeline(cfg, dataset=args.dataset)
    print(f"{green('OK')}  {dim(f'({(time.time() - t0) * 1000:.0f} ms)')}")

    # ── Attach the per-stage trace hooks (same as the diagnostic) ───────────
    patch_planner(planner)
    patch_retriever(retriever)
    patch_navigator(pipeline.navigator)
    patch_controller_bridge()
    if not args.skip_llm:
        patch_verifier(verifier)

    # ── Run, sampling peak RSS for the duration ─────────────────────────────
    with _PeakRSSSampler() as sampler:
        t_start = time.time()
        result = pipeline.process(args.question)
        total_ms = (time.time() - t_start) * 1000

    # ── Answer + confidence ─────────────────────────────────────────────────
    section("ANSWER")
    field("Answer", result.answer or "(no answer)")
    conf = getattr(result, "confidence", None)
    if conf is not None:
        # PipelineResult.confidence may be an enum or a float depending on path.
        conf_str = getattr(conf, "value", conf)
        field("Confidence", conf_str)

    # ── Edge-envelope panel (the headline) ──────────────────────────────────
    planner_ms = getattr(result, "planner_time_ms", 0.0) or 0.0
    navigator_ms = getattr(result, "navigator_time_ms", 0.0) or 0.0
    verifier_ms = getattr(result, "verifier_time_ms", 0.0) or 0.0
    _print_envelope_panel(
        total_ms=total_ms,
        planner_ms=planner_ms,
        navigator_ms=navigator_ms,
        verifier_ms=verifier_ms,
        peak_rss_mb=sampler.peak_mb,
    )
    print(f"\n{bold('=' * 78)}\n")

    # ── One-shot SUMMARY CARD (the screenshot target — paper Figure 1) ───────
    conf_val = getattr(result, "confidence", None)
    conf_str = getattr(conf_val, "value", conf_val) if conf_val is not None else "n/a"
    _print_summary_card(
        question=args.question,
        answer=result.answer or "(no answer)",
        confidence=str(conf_str),
        total_ms=total_ms,
        planner_ms=planner_ms,
        navigator_ms=navigator_ms,
        verifier_ms=verifier_ms,
        peak_rss_mb=sampler.peak_mb,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
