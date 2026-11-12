"""Split a personal QA dataset into train/validation JSONL at document level.

Input: one JSONL file where each line is {"context", "question", "answer"}.
Grouping is by `context` (the document), so every question about the same
document lands entirely in train or entirely in validation — never split
byte-wise or per-row across sides.

Usage:
    python scripts/split_qa_dataset.py \
        --input data/raw/my_qa.jsonl \
        --output-dir data/raw \
        --validation-ratio 0.1 \
        --seed 42
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from scripts.download_qa import format_stats, record_key, write_jsonl


DEFAULT_INPUT = Path("data/raw/my_qa.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/raw")
DEFAULT_VALIDATION_RATIO = 0.1
MIN_TRAIN_DOCUMENTS = 1


def group_by_document(records: list[dict]) -> dict[str, list[dict]]:
    """Group records by their context string (document-level buckets)."""
    documents: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        documents[record["context"]].append(record)
    return dict(documents)


def split_documents(
    records: list[dict],
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Shuffle documents and split them so validation holds ~validation_ratio of records.

    The split is at document level: whole contexts move together. With very
    small datasets one document may overshoot the ratio slightly; that is
    accepted to guarantee MIN_TRAIN_DOCUMENTS in train.
    """
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be strictly between 0 and 1")

    documents = group_by_document(records)
    contexts = sorted(documents)  # sorted for reproducibility before shuffling
    rng = random.Random(seed)
    rng.shuffle(contexts)

    total_records = len(records)
    validation_records: list[dict] = []
    train_records: list[dict] = []

    for context in contexts:
        bucket = documents[context]
        if (
            len(train_records) >= MIN_TRAIN_DOCUMENTS
            and (len(validation_records) + len(bucket)) / total_records <= validation_ratio
        ):
            validation_records.extend(bucket)
        else:
            train_records.extend(bucket)

    return train_records, validation_records


def split_qa_dataset(
    input_path: Path,
    output_dir: Path,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
    seed: int = 42,
) -> dict[str, int]:
    """Clean, deduplicate, document-split and write train.jsonl / validation.jsonl."""
    raw_records: list[dict] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {input_path}") from error

    stats = Counter(empty=0, duplicate=0)
    cleaned: list[dict] = []
    seen: set[str] = set()

    for record in raw_records:
        # Empty-field check.
        if not all(str(record.get(field, "")).strip() for field in ("context", "question", "answer")):
            stats["empty"] += 1
            continue
        cleaned.append({field: str(record[field]).strip() for field in ("context", "question", "answer")})

    # Global deduplication BEFORE splitting, so identical rows can never leak
    # across the train/validation boundary.
    deduplicated: list[dict] = []
    for record in cleaned:
        key = record_key(record)
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)
        deduplicated.append(record)

    if not deduplicated:
        raise RuntimeError(f"No usable QA examples found in {input_path}")

    train_records, validation_records = split_documents(
        deduplicated, validation_ratio=validation_ratio, seed=seed
    )

    counts: dict[str, int] = {}
    for name, records in (("train.jsonl", train_records), ("validation.jsonl", validation_records)):
        write_jsonl(records, output_dir / name)
        print(format_stats(name, stats, records))
        counts[name] = len(records)

    return counts


def positive_fraction(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number in (0, 1)") from error
    if not 0 < number < 1:
        raise argparse.ArgumentTypeError("must be a number in (0, 1)")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a personal QA JSONL file at document level.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source JSONL file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--validation-ratio",
        type=positive_fraction,
        default=DEFAULT_VALIDATION_RATIO,
        help="Target fraction of examples for validation.jsonl (document-level).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    counts = split_qa_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    print(f"Wrote {counts['train.jsonl']} train and {counts['validation.jsonl']} validation examples.")


if __name__ == "__main__":
    main()
