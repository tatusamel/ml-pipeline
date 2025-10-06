import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


REQUIRED_KEYS = {"doc_id", "title", "body", "language", "status", "tags", "product_line"}


def tokenize(text: str) -> List[str]:
    return text.split()


def validate(dataset_path: Path) -> Dict:
    doc_ids = Counter()
    total_tokens = 0
    languages = Counter()
    statuses = Counter()
    issues: List[str] = []

    with dataset_path.open() as f:
        for line_num, line in enumerate(f, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(f"Line {line_num}: invalid JSON ({exc})")
                continue

            missing = REQUIRED_KEYS - record.keys()
            if missing:
                issues.append(f"Line {line_num}: missing keys {sorted(missing)}")

            doc_id = record.get("doc_id")
            if not doc_id:
                issues.append(f"Line {line_num}: empty doc_id")
            else:
                doc_ids[doc_id] += 1

            body = record.get("body", "")
            if not body.strip():
                issues.append(f"Line {line_num}: empty body")
            tokens = tokenize(body)
            total_tokens += len(tokens)

            languages[record.get("language", "")] += 1
            statuses[record.get("status", "")] += 1

    duplicate_ids = [doc_id for doc_id, count in doc_ids.items() if count > 1]
    if duplicate_ids:
        issues.append(f"Duplicate doc_ids detected: {duplicate_ids}")

    docs_count = sum(doc_ids.values())
    avg_tokens = total_tokens / docs_count if docs_count else 0

    summary = {
        "documents": docs_count,
        "avg_tokens_per_doc": round(avg_tokens, 2),
        "languages": dict(languages),
        "statuses": dict(statuses),
        "issues": issues,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset JSONL file")
    parser.add_argument(
        "--dataset",
        default="data/raw/sunstream_kb_2024q3_v1.3.jsonl",
        help="Path to dataset JSONL",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    summary = validate(dataset_path)
    print(f"Documents: {summary['documents']}")
    print(f"Average tokens/doc: {summary['avg_tokens_per_doc']}")
    print(f"Languages: {summary['languages']}")
    print(f"Statuses: {summary['statuses']}")

    if summary["issues"]:
        print("Issues detected:")
        for issue in summary["issues"]:
            print(f" - {issue}")
        raise SystemExit(1)

    print("Dataset validation passed.")


if __name__ == "__main__":
    main()
