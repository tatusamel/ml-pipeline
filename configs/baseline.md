# Baseline Config

**run_id:** `20251005-sg-baseline`
**dataset snapshot:** `data/raw/sunstream_kb_2024q3_v1.3.jsonl`
**tokenizer for stats:** `whitespace` (internal utility)

## Chunking
- strategy: `fixed_tokens`
- chunk_size_tokens: `1000`
- overlap_pct: `15`

## Embeddings & Index
- embedding_model: `sentence-transformers/all-MiniLM-L6-v2` (dim: 384)
- vector_db: `FAISS`
- index_type/params: `HNSWFlat (M=32, efConstruction=200, efSearch=64)`

## Retrieval
- metric: `lexical_overlap`
- top_k: `5`
- hybrid_bm25: `off`
- reranker: `none`
- filters: `{}`
- context_budget_tokens: `4096`
- truncation_policy: `drop_oldest`

## Generator
- model: `template: top_chunk_trimmed`
- max_context_window: `1000`

## Evaluation
- metrics: `Retrieval hit@5`, `token-overlap faithfulness (1–5)`, `token-overlap relevance (1–5)`
- human_checks: `20 spot-checks`
- notes: `Stub generator mirrors the top chunk; replace with production LLM when credentials are available.`

## Rationale (short)
- Provide a deterministic baseline on the sample corpus that exercises every pipeline stage and populates the log artifacts end-to-end.
