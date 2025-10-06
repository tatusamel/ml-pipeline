# RAG Baseline Summary

We stood up a deterministic baseline pipeline on the bundled Sunstream support sample and captured audit logs for every stage.

## What We Completed
- Prepared the snapshot `sunstream_kb_2024q3_v1.3.jsonl` (50 docs) with the automated pipeline, filtering to English published articles and preserving 100% of records (0% dedupe) while averaging 62 tokens per document.
- Chunked the corpus into 50 fixed 1,000-token windows (15% overlap) so every document produces a reviewable slice (min 51, max 93 tokens).
- Embedded all chunks through the hashed MiniLM surrogate, persisted them to the FAISS-style index stub, and achieved ~2,900 chunks/sec with zero ingestion failures.
- Retrieved top-5 context via lexical-overlap scoring; all 20 gold-labelled test questions hit the correct chunk at rank 1, with mean latency 0.028 ms (p95 ≈ 0.033 ms).
- Generated answers by trimming the top chunk and auto-scored faithfulness/relevance/correctness via token overlap—all five sampled gold questions landed at 5/5.
- Logged each stage in `logs/*.csv`, refreshed `data/prepared/stats.md`, and captured the narrative in `reports/run_summaries/20251005-sg-baseline.md`.

## Run & Validate Locally
1. `python scripts/validate_dataset.py` — confirms the JSONL snapshot has the expected schema, unique IDs, and non-empty bodies.
2. `python scripts/run_baseline.py --fresh-logs` — regenerates prepared docs, chunk metadata, embeddings, retrieval results, evaluation scores, and rewrites every CSV log.
3. Inspect `logs/` and `data/prepared/prepared_docs.jsonl` for the regenerated artifacts; review console output for latency and hit@5 summaries.

## Repository Highlights
- `configs/baseline.md` & `configs/baseline.json` — canonical baseline settings for the deterministic run.
- `data/raw/sunstream_kb_2024q3_v1.3.jsonl` — bundled sample corpus used for automation tests.
- `scripts/run_baseline.py` — orchestrates preparation, chunking, ingestion, retrieval, and evaluation.
- `scripts/validate_dataset.py` — quick schema/uniqueness check for any JSONL snapshot before ingestion.
- `eval/testset/testset.csv` — 50-question evaluation set with gold answers for the first 20 items.
- `reports/run_summaries/20251005-sg-baseline.md` — snapshot of the automated baseline metrics and follow-ups.

With these assets, the project is ready for iterative improvements—swap in real embeddings, point to production data, or extend evaluation without touching the bookkeeping scaffolding.
