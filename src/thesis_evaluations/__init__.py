"""
Paper evaluation suite for:

    Edge-RAG: Empirical Characterization of When Knowledge-Graph Lanes
    Add Value in CPU-Only Hybrid Retrieval

This package produces the empirical evaluation for the paper. See
README.md in this directory for the script-to-section mapping and the
suggested execution order.

Modules
-------
- benchmark_datasets          core evaluation engine + metrics + loaders
                              (S_P -> S_N -> S_V pipeline + lane / component
                              ablation drivers)
- ir_metrics                  title-level IR metrics (Recall@k, nDCG@k, MRR)
                              over the per-question JSONL; rank-aware companion
                              to the set-based SF-F1 / SF-Recall
- modality_ablation           four-lane retrieval-modality ablation
                              (dense / BM25 / graph / hybrid) with IR metrics +
                              graph-rescued-case extraction
- make_dev_split              generate / verify the held-out dev band
                              (data/splits/dev_band.json) for train/dev/test
                              discipline; disjoint + reproducible
- quantization_sweep          quantization x model-size matrix; defends
                              the title's "Quantized" claim. Two modes:
                              --models (model-size sweep) and --model +
                              --bitwidths (Q4/Q8/fp16 sweep of ONE model, with
                              16 GB budget check + --smoke projection)
- agentic_ablation            5-row marginal-contribution table; defends
                              the "Agentic Verification" claim
- latency_memory_profile      per-stage timing + dual-process peak RSS;
                              defends the "Edge Devices" claim
- chunking_ablation           single-variable chunking sensitivity
                              (window x overlap), vector-side only
- verifier_cache_build        caches per-question retrieval so downstream
                              diagnostics can run without re-executing the
                              full pipeline
- verifier_failure_taxonomy   qualitative diagnostic that buckets every
                              wrong verifier answer by failure mode
                              (retrieval_miss / grounded_halluc / format /
                              abstention / close_miss)
- trust_eval                  selective-prediction metrics (risk-coverage,
                              abstention precision, ECE, AUROC) computed
                              from existing ablation JSONLs
- strategyqa_llm_only         parametric LLM-only reference for the
                              open-domain commonsense dataset (no retrieval
                              stores ingested by design)
- thesis_results_aggregator   capstone: turns all evaluator outputs
                              into LaTeX tables, figures, and an
                              overview.md / coverage_report.md bundle

No package-level imports or side effects: each module is imported on
demand by its CLI entry point (`python -m src.thesis_evaluations.<name>`)
so importing this package is cheap.

Last reviewed: 2026-06-03 (cleanup: removed obsolete verifier_only_ablation
and validate_reretrieval_loop scripts after agentic_ablation subsumed them)
"""
