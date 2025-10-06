# RAG Pipeline Setup Guide

This repository scaffolds the documentation and tracking needed to stand up a baseline Retrieval-Augmented Generation pipeline for the Sunstream support corpus. Follow the checklist below before running any experiments.

## Prerequisites
- Access to the prepared snapshot `sunstream_kb_2024q3_v1.3` (stored at `gs://sunstream-support/kb_export_2024-09-28`).
- Ability to run data preparation scripts with tokenizer `tiktoken cl100k_base` and seed `42`.
- Team members assigned to each workflow area (see `OWNER_MATRIX.md`).

## Setup Steps
1. **Configure baseline run**: Update `configs/baseline.md` with run ID `20251005-sg-baseline`, confirm chunking, embedding, retrieval, generator, and evaluation settings.
2. **Document data prep**: After preparing the dataset, log the stats in `data/prepared/stats.md`, capturing row counts, dedupe %, filters, and timing.
3. **Seed evaluation queries**: Populate `eval/testset/testset.csv` with 50–100 representative support questions; include gold answers for at least the first 20 (`q001–q020`).
4. **Confirm ownership**: Ensure `OWNER_MATRIX.md` lists the correct responsible owners for prep, chunking, ingestion, retrieval, evaluation, and reporting.
5. **Prepare logging**: Keep the CSV templates under `logs/` ready to append results (`data_prep.csv`, `chunking.csv`, `ingestion.csv`, `retrieval.csv`, `rag_eval.csv`).
6. **Plan baseline execution**: When the pipeline run is executed, append metrics to the log files and draft a summary in `reports/run_summaries/` derived from the template provided.

With the above in place, the baseline pipeline run can be executed and audited consistently across the team.
