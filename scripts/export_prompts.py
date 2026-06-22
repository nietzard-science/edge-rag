"""Export every LLM prompt template verbatim from source into docs/prompts/PROMPTS.md.

B-2 reproducibility artifact. A reviewer must be able to read the *exact*
prompt the SLM receives without reading the Python source. To guarantee the
published prompts never drift from the code, this doc is GENERATED from the
live class attributes rather than hand-copied.

Design facts this artifact also documents (consistency with the paper):
  * The Planner (S_P) and Navigator (S_N) make NO LLM calls — they are
    algorithmic (SpaCy dependency parse, GLiNER NER, RRF fusion). Every
    generative prompt in the system lives on the Verifier (S_V).
  * All prompts are `str.format`-style templates; the placeholder names
    ({context}, {query}, {bridge_chain}, {violations}, {previous_answer})
    are the wiring contract between the pipeline and the model.

Run:
    python -X utf8 scripts/export_prompts.py
    python -X utf8 scripts/export_prompts.py --check   # CI: fail if stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.logic_layer.verifier import Verifier  # noqa: E402

# Ordered (group, attribute, one-line role) so the doc reads top-to-bottom in
# pipeline order rather than alphabetically.
_PROMPT_SPEC = [
    ("Primary answer prompts (one fires per query type)", [
        ("ANSWER_PROMPT", "Single-hop / default answer prompt."),
        ("BRIDGE_PROMPT", "Multi-hop bridge questions; injects the resolved reasoning chain."),
        ("COMPARISON_PROMPT", "Comparison questions (HotpotQA's second official type)."),
    ]),
    ("Self-correction & abstention", [
        ("CORRECTION_PROMPT", "Re-prompt after the Verifier flags unverified claims (agentic loop)."),
        ("INSUFFICIENT_EVIDENCE_PROMPT", "Partial-answer prompt when evidence is judged insufficient."),
    ]),
    ("Context shaping & format recovery", [
        ("DISTILL_PROMPT", "RECOMP-style context distillation run before the answer prompt."),
        ("FORMAT_RETRY_PROMPT", "One-shot retry when a wh-question gets a yes/no or abstention answer."),
    ]),
    ("Bounded self-correction retries (one call per query, keep-best guarded)", [
        ("BRIDGE_EXCLUSION_RETRY_PROMPT", "Re-prompt when the answer is a bridge entity, not the final answer."),
        ("ANTI_ABSTENTION_RETRY_PROMPT", "Re-prompt when the model abstains but query entities are in context."),
        ("GROUNDING_RETRY_PROMPT", "Re-prompt when the QA-conditioned NLI grounding gate sub-threshold-fires."),
    ]),
    ("Structured chain-of-thought variants (config.enable_structured_cot)", [
        ("ANSWER_PROMPT_COT", "Slot-filling CoT variant of ANSWER_PROMPT."),
        ("BRIDGE_PROMPT_COT", "Slot-filling CoT variant of BRIDGE_PROMPT."),
        ("COMPARISON_PROMPT_COT", "Slot-filling CoT variant of COMPARISON_PROMPT."),
    ]),
]

_HEADER = """# LLM Prompt Templates

> **Generated file — do not edit by hand.** Produced verbatim from
> `src/logic_layer/verifier.py` by `scripts/export_prompts.py`. Re-run that
> script after changing any prompt; CI runs it with `--check` to fail on drift.

All generative prompts in this system belong to the **Verifier (S_V)**. The
**Planner (S_P)** and **Navigator (S_N)** make no LLM calls — they are
algorithmic (SpaCy dependency parsing, GLiNER NER, Reciprocal Rank Fusion).
This is why "the model's prompts" and "the verifier's prompts" are the same
set.

Prompts are `str.format` templates. Placeholders:

| Placeholder | Filled with |
|---|---|
| `{context}` | The Navigator's retrieved + filtered chunks (≤ `max_docs`). |
| `{query}` | The user question (or rewritten sub-query). |
| `{bridge_chain}` | The Planner-derived reasoning chain (bridge prompts only). |
| `{violations}` | Verifier-flagged unverified claims (correction prompt only). |
| `{previous_answer}` | The prior turn's answer (format-retry prompt only). |

Model: `qwen2:1.5b` (Q4) via Ollama. Temperature 0.0 (deterministic decode).

---
"""


def _render() -> str:
    parts = [_HEADER]
    for group_title, entries in _PROMPT_SPEC:
        parts.append(f"\n## {group_title}\n")
        for attr, role in entries:
            text = getattr(Verifier, attr)
            parts.append(f"\n### `{attr}`\n\n{role}\n\n```text\n{text}\n```\n")
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="Exit non-zero if the on-disk doc differs from source (CI drift guard).")
    args = ap.parse_args()

    out_path = _ROOT / "docs" / "prompts" / "PROMPTS.md"
    rendered = _render()

    if args.check:
        if not out_path.exists():
            print(f"MISSING: {out_path} — run scripts/export_prompts.py", file=sys.stderr)
            return 1
        current = out_path.read_text(encoding="utf-8")
        if current != rendered:
            print("STALE: docs/prompts/PROMPTS.md differs from verifier.py source. "
                  "Re-run: python -X utf8 scripts/export_prompts.py", file=sys.stderr)
            return 1
        print("OK: PROMPTS.md is in sync with verifier.py")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    n = sum(len(e) for _, e in _PROMPT_SPEC)
    print(f"Wrote {out_path} ({n} prompt templates).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
