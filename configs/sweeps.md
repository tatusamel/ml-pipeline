# Planned Sweeps

| variant | chunk_size | overlap | k | hybrid | reranker | hypothesis |
|--------|------------|---------|---|--------|----------|------------|
| baseline | 1000 | 15% | 5 | off | none | starting point |
| smaller-chunks | 600 | 15% | 5 | off | none | more granular retrieval may help recall |
| larger-chunks | 1500 | 15% | 5 | off | none | better coherence, may reduce truncation |
| hybrid-on | 1000 | 15% | 5 | on | none | BM25+vector may help lexical queries |
| rerank | 1000 | 15% | 5 | off | cross-encoder-xyz | improve precision at top ranks |
