# Run Summary — 20251005-sg-baseline

## Context
- Dataset: `sunstream_kb_2024q3_v1.3.jsonl` (50 support docs generated from evaluation prompts)
- Objective: Validate the automated baseline pipeline and ensure every logging artifact is produced end-to-end.

## Data Preparation
- Rows before/after: 50 → 50 (`0.0%` dedupe)
- Avg tokens per doc: 62 (whitespace tokenizer)
- Filters: language=en, status=published, min_tokens=50
- Runtime: < 0.1 s on local workstation

## Chunking
- Strategy: fixed 1,000-token windows (15% overlap)
- Output: 50 chunks (one per document)
- Token stats: min 51, max 93, mean 62
- Runtime: < 0.1 s

## Embedding & Ingestion
- Embedding surrogate: hashed MiniLM (384-dim)
- Vector store: FAISS HNSWFlat stub (M=32, efConstruction=200, efSearch=64)
- Vectors written: 50 (0 duplicates, 0 failures)
- Throughput: ~2,900 chunks/sec over 0.02 s

## Retrieval
- Retriever: lexical-overlap top-5 (hybrid filters off)
- Latency: mean 0.028 ms, p95 ≈ 0.033 ms
- Hit@5: 20/20 gold queries recovered at rank 1

## RAG Evaluation
- Generator: top-chunk template (first 80 tokens)
- Similarity (token overlap) avg: 0.95 → scored 5/5 on faithfulness, relevance, correctness for all gold cases
- Notes: Template mirrors retrieved chunk verbatim; replace with production LLM for qualitative analysis.

## Follow-Ups
- Swap in real embeddings/LLM once credentials are provisioned.
- Extend dataset beyond the synthetic 50-doc sample to stress-test chunking and latency.
- Add cost tracking once external APIs are introduced.
