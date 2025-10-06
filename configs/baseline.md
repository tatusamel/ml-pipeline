# Baseline Config (fill and commit)

**run_id:** `20251005-sg-baseline`
**dataset snapshot:** `gs://sunstream-support/kb_export_2024-09-28`
**tokenizer for stats:** `tiktoken cl100k_base`

## Chunking
- strategy: `fixed_tokens`
- chunk_size_tokens: `1000`
- overlap_pct: `15`

## Embeddings & Index
- embedding_model: `sentence-transformers/all-MiniLM-L6-v2` (dim: 384)  # swap if you prefer another
- vector_db: `FAISS`
- index_type/params: `HNSWFlat (M=32, efConstruction=200)`

## Retrieval
- metric: `cosine`
- top_k: `5`
- hybrid_bm25: `off`
- reranker: `none`
- filters: `{}`
- context_budget_tokens: `4096`
- truncation_policy: `drop_oldest`

## Generator
- model: `openai/gpt-4o-mini`
- max_context_window: `128000`

## Evaluation
- metrics: `Recall@5`, `LLM-judge faithfulness (1–5)`, `relevance (1–5)`
- human_checks: `20 spot-checks`
- notes: `Solar inverter KB has firmware-specific steps; double-check retrieved snippets reference matching controller versions.`

## Rationale (short)
- Keep it simple; establish a traceable baseline before sweeps.
