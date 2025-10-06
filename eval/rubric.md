# Evaluation Rubric

## Faithfulness (1–5)
1: contradicts evidence · 3: partially grounded · 5: fully supported by retrieved snippets

## Relevance (1–5)
1: off-topic · 3: partially addresses · 5: directly answers all parts

## Correctness (1–5)
1: mostly incorrect · 3: mixed · 5: factually correct given ground truth

**Pass criteria:** Faithfulness ≥4 and Relevance ≥4.  
On failure, tag cause: {missing_context | bad_retrieval | hallucination | truncation}.
