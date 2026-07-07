# LLM Prompt Templates

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

## Primary answer prompts (one fires per query type)

### `ANSWER_PROMPT`

Single-hop / default answer prompt.

```text
You are a factual QA assistant. Answer based ONLY on the context below.

Rules:
- Give the shortest possible answer: a name, place, date, number, or yes/no.
- Do NOT explain or add sentences beyond the direct answer.
- If the answer is a person, place, or thing: reply with just that name.
- If the answer is a number or statistic (e.g. population, count, year): reply with just the number.
- If the answer is yes/no: reply with just "yes" or "no".
- The answer is almost always present in the context — read every chunk carefully before concluding otherwise. Reply "I don't know" ONLY if the answer is genuinely absent after a careful read.

Context:
{context}

Question: {query}

Answer (as short as possible):
```

### `BRIDGE_PROMPT`

Multi-hop bridge questions; injects the resolved reasoning chain.

```text
You are a factual QA assistant. Answer based ONLY on the context below.

This is a multi-step question. Use the following reasoning chain to find the answer:
{bridge_chain}

Rules:
- Give the shortest possible answer: a name, place, date, number, or yes/no.
- Do NOT explain or add sentences beyond the direct answer.
- If the answer is a number or statistic (e.g. population, count, year): reply with just the number.
- The answer is almost always present in the context — read every chunk carefully before concluding otherwise. Reply "I don't know" ONLY if the answer is genuinely absent after a careful read.

Context:
{context}

Question: {query}

Answer (as short as possible):
```

### `COMPARISON_PROMPT`

Comparison questions (HotpotQA's second official type).

```text
You are a factual QA assistant. Answer based ONLY on the context below.

The question compares two people or things. Follow these steps:
1. Find the relevant fact for the FIRST person/thing in the context.
2. Find the relevant fact for the SECOND person/thing in the context.
3. Compare the two facts and give the answer.

Rules:
- For yes/no questions: reply with just "yes" or "no".
- For "which one" questions: reply with just the name.
- Do NOT explain beyond the direct answer.
- The answer is almost always present in the context — read both items' facts carefully before concluding otherwise. Reply "I don't know" ONLY if the answer is genuinely absent after a careful read.

Context:
{context}

Question: {query}

Answer (as short as possible):
```

## Self-correction & abstention

### `CORRECTION_PROMPT`

Re-prompt after the Verifier flags unverified claims (agentic loop).

```text
Your previous answer contained unverified claims.

Unverified claims:
{violations}

Context:
{context}

Question: {query}

Give the shortest correct answer (name, place, date, or yes/no only):
```

### `INSUFFICIENT_EVIDENCE_PROMPT`

Partial-answer prompt when evidence is judged insufficient.

```text
Based on the available context, I could not find sufficient evidence to fully answer your question.

Context:
{context}

Question: {query}

Please provide a partial answer based on the available evidence, clearly indicating what information is missing:
```

## Context shaping & format recovery

### `DISTILL_PROMPT`

RECOMP-style context distillation run before the answer prompt.

```text
You are a fact-extraction assistant. Read the context below and extract ONLY the facts directly relevant to answering the question.

Output rules:
- Bullet list of short atomic facts, one per line, prefixed with "- ".
- Each fact must be one sentence containing concrete details (names, dates, places, titles, numbers).
- Include the question's named entities verbatim where they appear in the context.
- Do NOT add opinions, transitions, or background narrative not directly supported by the context.
- If a fact is not stated in the context, do not invent it.

Question: {query}

Context:
{context}

Relevant facts:
```

### `FORMAT_RETRY_PROMPT`

One-shot retry when a wh-question gets a yes/no or abstention answer.

```text
Question: {query}

Context:
{context}

Your previous answer "{previous_answer}" does not match the format the question requires. The question asks for a specific name, place, date, title, or thing. Look in the context for that specific entity and provide it as the answer.

Final answer:
```

## Bounded self-correction retries (one call per query, keep-best guarded)

### `BRIDGE_EXCLUSION_RETRY_PROMPT`

Re-prompt when the answer is a bridge entity, not the final answer.

```text
Question: {query}

Context:
{context}

IMPORTANT INSTRUCTION: The following entities are intermediate bridges in the reasoning chain, NOT the final answer the question asks for: {exclusion_list}.
Find the FINAL answer that the question asks for. The final answer must be DIFFERENT from the bridge entities listed above. Look in the context for the attribute, title, name, date, or fact that the question is actually asking about.

Final answer:
```

### `ANTI_ABSTENTION_RETRY_PROMPT`

Re-prompt when the model abstains but query entities are in context.

```text
Question: {query}

Context:
{context}

The context above contains information about: {entity_list}. The answer to the question IS present in the context. Do NOT say you don't know. Read carefully and give the shortest specific answer (a name, place, date, number, or yes/no).

Answer:
```

### `GROUNDING_RETRY_PROMPT`

Re-prompt when the QA-conditioned NLI grounding gate sub-threshold-fires.

```text
Question: {query}

Context:
{context}

Your previous answer "{previous_answer}" may not be the specific thing the question asks for. Re-read the question and the context. The answer must be the exact entity, attribute, date, or number the question requests, and it must be stated in the context. Give only that answer.

Answer:
```

## Structured chain-of-thought variants (config.enable_structured_cot)

### `ANSWER_PROMPT_COT`

Slot-filling CoT variant of ANSWER_PROMPT.

```text
You are a factual QA assistant. Answer using ONLY the context. Work through the steps, then give the final answer on its own line.

Context:
{context}

Question: {query}

Step 1 — Named entities in the question:
Step 2 — The exact fact(s) in the context that answer the question:
FINAL ANSWER (shortest form — a name, place, date, number, or yes/no):
```

### `BRIDGE_PROMPT_COT`

Slot-filling CoT variant of BRIDGE_PROMPT.

```text
You are a factual QA assistant. This is a multi-step question. Answer using ONLY the context. Work through the steps, then give the final answer on its own line.

{bridge_chain}

Context:
{context}

Question: {query}

Step 1 — Named entities in the question:
Step 2 — Bridge entity (the intermediate that links the question parts), from the context:
Step 3 — The specific fact about the bridge entity that the question asks for:
FINAL ANSWER (shortest form, and DIFFERENT from the bridge entity):
```

### `COMPARISON_PROMPT_COT`

Slot-filling CoT variant of COMPARISON_PROMPT.

```text
You are a factual QA assistant. This is a comparison question. Answer using ONLY the context. Work through the steps, then give the final answer on its own line.

Context:
{context}

Question: {query}

Step 1 — The two (or more) items being compared:
Step 2 — The relevant attribute of each item, from the context:
Step 3 — The comparison result:
FINAL ANSWER (shortest form — the winning item, or yes/no):
```
