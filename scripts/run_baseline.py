import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

LOG_FILES = {
    "data_prep": Path("logs/data_prep.csv"),
    "chunking": Path("logs/chunking.csv"),
    "ingestion": Path("logs/ingestion.csv"),
    "retrieval": Path("logs/retrieval.csv"),
    "rag_eval": Path("logs/rag_eval.csv"),
}


@dataclass
class Document:
    doc_id: str
    body: str
    language: str
    status: str
    tags: List[str]
    product_line: str
    resolution_steps: str

    def as_json(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "body": self.body,
            "language": self.language,
            "status": self.status,
            "tags": self.tags,
            "product_line": self.product_line,
            "resolution_steps": self.resolution_steps,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline RAG pipeline")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fresh-logs",
        action="store_true",
        help="Remove existing logs before writing new rows",
    )
    return parser.parse_args()


def ensure_dirs() -> None:
    Path("logs").mkdir(exist_ok=True)
    Path("data/prepared").mkdir(parents=True, exist_ok=True)


def reset_logs() -> None:
    for path in LOG_FILES.values():
        if path.exists():
            path.unlink()


def load_config(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def load_documents(dataset_path: Path) -> List[Document]:
    docs: List[Document] = []
    with dataset_path.open() as f:
        for line in f:
            raw = json.loads(line)
            docs.append(
                Document(
                    doc_id=raw.get("doc_id", ""),
                    body=raw.get("body", ""),
                    language=(raw.get("language") or "").lower(),
                    status=(raw.get("status") or "").lower(),
                    tags=list(raw.get("tags") or []),
                    product_line=raw.get("product_line", ""),
                    resolution_steps=raw.get("resolution_steps", ""),
                )
            )
    return docs


def tokenize(text: str) -> List[str]:
    return text.split()


def dedupe_documents(docs: Iterable[Document]) -> Tuple[List[Document], float]:
    doc_list = list(docs)
    seen_hashes: Dict[str, Document] = {}
    for doc in doc_list:
        content_hash = hashlib.sha256(doc.body.encode("utf-8")).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes[content_hash] = doc
    deduped = list(seen_hashes.values())
    total = len(doc_list)
    dedupe_pct = (1 - len(deduped) / total) * 100 if total else 0.0
    return deduped, dedupe_pct


def prepare_documents(docs: List[Document], filters: Dict) -> Tuple[List[Document], Dict]:
    rows_before = len(docs)
    filtered = [
        doc
        for doc in docs
        if doc.language == filters["language"]
        and doc.status == filters["status"]
        and len(tokenize(doc.body)) >= filters["min_tokens"]
    ]
    deduped, dedupe_pct = dedupe_documents(filtered)
    avg_tokens = (
        sum(len(tokenize(doc.body)) for doc in deduped) / len(deduped)
        if deduped
        else 0.0
    )
    null_rate = {
        "resolution_steps": (
            sum(1 for doc in deduped if not doc.resolution_steps) / len(deduped)
            if deduped
            else 0.0
        ),
        "tags": (
            sum(1 for doc in deduped if not doc.tags) / len(deduped)
            if deduped
            else 0.0
        ),
        "product_line": (
            sum(1 for doc in deduped if not doc.product_line) / len(deduped)
            if deduped
            else 0.0
        ),
    }
    stats = {
        "rows_before": rows_before,
        "rows_after": len(deduped),
        "dedupe_pct": round(dedupe_pct, 2),
        "avg_tokens_per_row": round(avg_tokens, 2),
        "null_rate_json": {k: round(v, 3) for k, v in null_rate.items()},
    }
    return deduped, stats


def write_prepared_docs(docs: List[Document], path: Path) -> None:
    with path.open("w") as f:
        for doc in docs:
            f.write(json.dumps(doc.as_json()) + "\n")


def write_data_prep_log(run_id: str, stats: Dict, filters: Dict, seed: int, start_ts: datetime, duration: float) -> None:
    path = LOG_FILES["data_prep"]
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                [
                    "run_id",
                    "dataset_id",
                    "seed",
                    "rows_before",
                    "rows_after",
                    "dedupe_pct",
                    "avg_tokens",
                    "null_rate_json",
                    "filters_applied_json",
                    "started_at",
                    "duration_s",
                ]
            )
        writer.writerow(
            [
                run_id,
                "sunstream_kb_2024q3_v1.3",
                seed,
                stats["rows_before"],
                stats["rows_after"],
                stats["dedupe_pct"],
                stats["avg_tokens_per_row"],
                json.dumps(stats["null_rate_json"]),
                json.dumps(filters),
                start_ts.isoformat(),
                round(duration, 2),
            ]
        )


def update_stats_md(stats: Dict, filters: Dict, seed: int, start_ts: datetime, duration: float) -> None:
    content = """
# Data Preparation Stats

- dataset_id/hash: `sunstream_kb_2024q3_v1.3`
- source/date range: `Local JSONL export`
- rows_before: `{rows_before}`
- rows_after: `{rows_after}`
- dedupe_pct: `{dedupe_pct}`
- avg_tokens_per_row: `{avg_tokens}`
- null_rate_json: `{null_rate}`
- filters_applied_json: `{filters}`
- seed: `{seed}`
- started_at: `{started}`
- duration_s: `{duration}`
""".strip().format(
        rows_before=stats["rows_before"],
        rows_after=stats["rows_after"],
        dedupe_pct=stats["dedupe_pct"],
        avg_tokens=stats["avg_tokens_per_row"],
        null_rate=json.dumps(stats["null_rate_json"]),
        filters=json.dumps(filters),
        seed=seed,
        started=start_ts.isoformat(),
        duration=round(duration, 2),
    )
    Path("data/prepared/stats.md").write_text(content + "\n")


def chunk_documents(docs: List[Document], chunk_size: int, overlap_pct: int) -> Tuple[List[Dict], Dict]:
    overlap = int(chunk_size * overlap_pct / 100)
    stride = max(chunk_size - overlap, 1)
    chunks: List[Dict] = []
    token_counts: List[int] = []
    short_threshold = max(int(chunk_size * 0.3), 1)
    for doc in docs:
        tokens = tokenize(doc.body)
        for idx, start in enumerate(range(0, len(tokens), stride)):
            window = tokens[start : start + chunk_size]
            if not window:
                continue
            chunk_id = f"{doc.doc_id}_chunk_{idx:04d}"
            chunk_text = " ".join(window)
            count = len(window)
            token_counts.append(count)
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc.doc_id,
                    "text": chunk_text,
                    "tokens": count,
                }
            )
    stats = {
        "docs_in": len(docs),
        "chunks_out": len(chunks),
        "avg_tokens_per_chunk": round(
            sum(token_counts) / len(token_counts), 2
        )
        if token_counts
        else 0.0,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "short_chunk_count": sum(1 for t in token_counts if t < short_threshold),
    }
    return chunks, stats


def write_chunking_log(run_id: str, stats: Dict, chunk_size: int, overlap_pct: int, duration: float) -> None:
    path = LOG_FILES["chunking"]
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                [
                    "run_id",
                    "strategy",
                    "chunk_size_tokens",
                    "overlap_pct",
                    "docs_in",
                    "chunks_out",
                    "avg_tokens_per_chunk",
                    "min_tokens",
                    "max_tokens",
                    "short_chunk_count",
                    "duration_s",
                ]
            )
        writer.writerow(
            [
                run_id,
                "fixed_tokens",
                chunk_size,
                overlap_pct,
                stats["docs_in"],
                stats["chunks_out"],
                stats["avg_tokens_per_chunk"],
                stats["min_tokens"],
                stats["max_tokens"],
                stats["short_chunk_count"],
                round(duration, 2),
            ]
        )


def hashed_embedding(text: str, dim: int) -> List[float]:
    values: List[float] = []
    for i in range(dim):
        digest = hashlib.sha256(f"{text}|{i}".encode("utf-8")).digest()
        val = int.from_bytes(digest[:4], "little", signed=False)
        values.append((val % 10000) / 10000.0)
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def ingest_chunks(chunks: List[Dict], dim: int) -> Tuple[List[Dict], Dict]:
    vectors: List[Dict] = []
    unique_hashes = set()
    failures = 0
    start = time.perf_counter()
    for chunk in chunks:
        try:
            embedding = hashed_embedding(chunk["text"], dim)
            chunk_hash = hashlib.sha1(chunk["text"].encode("utf-8")).hexdigest()
            unique_hashes.add(chunk_hash)
            vectors.append(
                {
                    "chunk": chunk,
                    "embedding": embedding,
                    "tokens": set(tokenize(chunk["text"].lower())),
                }
            )
        except Exception:
            failures += 1
    duration = time.perf_counter() - start
    throughput = (len(chunks) - failures) / duration if duration else 0.0
    dedupe_pct = (
        (1 - len(unique_hashes) / len(chunks)) * 100 if chunks else 0.0
    )
    stats = {
        "chunks_in": len(chunks),
        "vectors_written": len(vectors),
        "dedupe_pct": round(dedupe_pct, 2),
        "failures": failures,
        "throughput": round(throughput, 2),
        "duration": duration,
    }
    return vectors, stats


def write_ingestion_log(run_id: str, cfg: Dict, stats: Dict) -> None:
    path = LOG_FILES["ingestion"]
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                [
                    "run_id",
                    "embedding_model",
                    "embedding_dim",
                    "vector_db",
                    "index_params_json",
                    "chunks_in",
                    "vectors_written",
                    "dedupe_pct",
                    "failures",
                    "throughput_chunks_per_s",
                    "duration_s",
                ]
            )
        writer.writerow(
            [
                run_id,
                cfg["model"],
                cfg["dim"],
                cfg["vector_db"],
                json.dumps(cfg["index_params"]),
                stats["chunks_in"],
                stats["vectors_written"],
                stats["dedupe_pct"],
                stats["failures"],
                stats["throughput"],
                round(stats["duration"], 2),
            ]
        )


def lexical_similarity(query_tokens: set, chunk_tokens: set) -> float:
    if not query_tokens or not chunk_tokens:
        return 0.0
    overlap = len(query_tokens & chunk_tokens)
    return overlap / math.sqrt(len(query_tokens) * len(chunk_tokens))


def retrieve(queries: List[Dict], vectors: List[Dict], top_k: int) -> List[Dict]:
    results: List[Dict] = []
    for query in queries:
        start = time.perf_counter()
        query_tokens = set(tokenize(query["text"].lower()))
        scored = []
        for item in vectors:
            score = lexical_similarity(query_tokens, item["tokens"])
            scored.append((score, item))
        scored.sort(reverse=True, key=lambda x: x[0])
        top = [
            {"score": score, "chunk": item["chunk"]}
            for score, item in scored[:top_k]
        ]
        latency_ms = (time.perf_counter() - start) * 1000
        retrieved_ids = [entry["chunk"]["chunk_id"] for entry in top]
        results.append(
            {
                "query": query,
                "retrieved": top,
                "latency_ms": latency_ms,
                "retrieved_ids": retrieved_ids,
            }
        )
    return results


def write_retrieval_log(run_id: str, top_k: int, retrievals: List[Dict]) -> None:
    path = LOG_FILES["retrieval"]
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                [
                    "run_id",
                    "query_id",
                    "retriever",
                    "k",
                    "filters_json",
                    "retrieved_ids_json",
                    "latency_ms",
                    "hit_at_k",
                    "rank_positions_json",
                ]
            )
        for entry in retrievals:
            query = entry["query"]
            gold = query.get("gold_answer", "")
            retrieved_texts = [item["chunk"]["text"] for item in entry["retrieved"]]
            hit = 0
            ranks: List[int] = []
            if gold:
                lowered = gold.lower()
                for idx, text in enumerate(retrieved_texts, start=1):
                    if lowered in text.lower():
                        hit = 1
                        ranks.append(idx)
            writer.writerow(
                [
                    run_id,
                    query["query_id"],
                    "lexical_overlap_top5",
                    top_k,
                    json.dumps({}),
                    json.dumps(entry["retrieved_ids"]),
                    round(entry["latency_ms"], 2),
                    hit,
                    json.dumps(ranks),
                ]
            )


def generate_answer(text: str) -> str:
    tokens = tokenize(text)
    return " ".join(tokens[:80])


def string_similarity(candidate: str, reference: str) -> float:
    if not candidate or not reference:
        return 0.0
    cand_tokens = set(tokenize(candidate.lower()))
    ref_tokens = tokenize(reference.lower())
    overlap = sum(1 for token in ref_tokens if token in cand_tokens)
    return overlap / max(len(ref_tokens), 1)


def clamp_score(similarity: float) -> int:
    return max(1, min(5, round(similarity * 5)))


def evaluate(run_id: str, retrievals: List[Dict]) -> None:
    path = LOG_FILES["rag_eval"]
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                [
                    "run_id",
                    "query_id",
                    "generated_answer",
                    "gold_answer",
                    "retrieved_ids_json",
                    "retrieved_snippets_json",
                    "similarity",
                    "faithfulness_1to5",
                    "relevance_1to5",
                    "correctness_1to5",
                    "notes",
                ]
            )
        for entry in retrievals:
            gold = entry["query"].get("gold_answer", "")
            if not gold:
                continue
            top_texts = [item["chunk"]["text"] for item in entry["retrieved"]]
            answer = generate_answer(top_texts[0]) if top_texts else ""
            similarity = string_similarity(answer, gold)
            score = clamp_score(similarity)
            writer.writerow(
                [
                    run_id,
                    entry["query"]["query_id"],
                    answer,
                    gold,
                    json.dumps(entry["retrieved_ids"]),
                    json.dumps(top_texts[:2]),
                    round(similarity, 2),
                    score,
                    score,
                    score,
                    "Auto-evaluated via token overlap",
                ]
            )


def load_queries(path: Path) -> List[Dict]:
    items: List[Dict] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(
                {
                    "query_id": row["query_id"],
                    "text": row["query"],
                    "gold_answer": row.get("gold_answer", ""),
                }
            )
    return items


def summarise(retrievals: List[Dict]) -> None:
    latencies = [entry["latency_ms"] for entry in retrievals]
    gold_entries = [entry for entry in retrievals if entry["query"].get("gold_answer")]
    hits = sum(
        1
        for entry in gold_entries
        if any(entry["query"]["gold_answer"].lower() in chunk["chunk"]["text"].lower() for chunk in entry["retrieved"])
    )
    total = len(gold_entries)
    mean_latency = statistics.mean(latencies) if latencies else 0.0
    p95_latency = (
        statistics.quantiles(latencies, n=100)[94] if len(latencies) >= 20 else max(latencies) if latencies else 0.0
    )
    print(f"Retrieval mean latency: {mean_latency:.3f} ms")
    print(f"Retrieval approx p95 latency: {p95_latency:.3f} ms")
    if total:
        print(f"Gold hit@{len(retrievals[0]['retrieved'])}: {hits}/{total}")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.fresh_logs:
        reset_logs()

    config = load_config(Path(args.config))
    dataset_path = Path(config["dataset_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    filters = {"language": "en", "status": "published", "min_tokens": 50}

    prep_start = datetime.now(UTC)
    prep_timer = time.perf_counter()
    raw_docs = load_documents(dataset_path)
    prepared_docs, prep_stats = prepare_documents(raw_docs, filters)
    prep_duration = time.perf_counter() - prep_timer
    write_prepared_docs(prepared_docs, Path("data/prepared/prepared_docs.jsonl"))
    write_data_prep_log(config["run_id"], prep_stats, filters, args.seed, prep_start, prep_duration)
    update_stats_md(prep_stats, filters, args.seed, prep_start, prep_duration)

    chunk_timer = time.perf_counter()
    chunks, chunk_stats = chunk_documents(
        prepared_docs,
        config["chunking"]["chunk_size_tokens"],
        config["chunking"]["overlap_pct"],
    )
    chunk_duration = time.perf_counter() - chunk_timer
    write_chunking_log(
        config["run_id"],
        chunk_stats,
        config["chunking"]["chunk_size_tokens"],
        config["chunking"]["overlap_pct"],
        chunk_duration,
    )

    vectors, ingestion_stats = ingest_chunks(chunks, config["embeddings"]["dim"])
    write_ingestion_log(config["run_id"], config["embeddings"], ingestion_stats)

    queries = load_queries(Path("eval/testset/testset.csv"))
    retrievals = retrieve(queries, vectors, config["retrieval"]["top_k"])
    write_retrieval_log(config["run_id"], config["retrieval"]["top_k"], retrievals)

    evaluate(config["run_id"], retrievals)
    summarise(retrievals)


if __name__ == "__main__":
    main()
