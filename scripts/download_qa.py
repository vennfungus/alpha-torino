"""Download SQuAD question-answer data into structured JSONL files.

Each example is written as a single JSON line: {"context", "question", "answer"}.
Splits follow the dataset's own document/example-level train and validation
splits; no byte-level or positional split is performed.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Mapping

from datasets import load_dataset


OUTPUT_DIR = Path("data/qa")
TRAIN_FILENAME = "train.jsonl"
VALIDATION_FILENAME = "validation.jsonl"
DATASET_NAME = "rajpurkar/squad"
DEFAULT_VALIDATION_RATIO = 0.1  # used only when --max-examples is given


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error

    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def normalize_example(item: Mapping) -> dict | None:
    """Validate and clean one dataset row.

    Returns a {"context", "question", "answer"} dict, or None when any required
    field is missing or empty after stripping.
    """
    context = item.get("context")
    question = item.get("question")
    answers = item.get("answers")

    if (
        not isinstance(context, str)
        or not isinstance(question, str)
        or not isinstance(answers, Mapping)
    ):
        raise ValueError("Dataset rows must contain context, question, and answers.")

    answer_texts = answers.get("text")
    if not isinstance(answer_texts, list):
        raise ValueError("Dataset answers must contain a list in the 'text' field.")

    context = context.strip()
    question = question.strip()
    answer = next(
        (text.strip() for text in answer_texts if isinstance(text, str) and text.strip()),
        "",
    )

    # Empty-field check: drop rows without context, question or answer.
    if not context or not question or not answer:
        return None

    return {"context": context, "question": question, "answer": answer}


def record_key(record: Mapping) -> str:
    """Deduplication key built from the three fields.

    \\x1f (ASCII unit separator) avoids collisions when values contain
    newlines or tabs.
    \x1f (ASCII unit separator) is unambiguous: values are escaped by doubling
    any embedded \x1f and fields are joined with three separators, so keys
    cannot collide even when values contain the separator, newlines or tabs.
    """
    separator = "\x1f\x1f\x1f"
    escaped = (value.replace("\x1f", "\x1f\x1f") for value in record.values())
    return separator.join(escaped)


def collect_split(
    split: str,
    limit: int | None,
    seed: int,
) -> tuple[list[dict], Counter]:
    """Stream one split, drop empty rows and duplicates, and return the records.

    Streaming keeps memory bounded; records are accumulated because the split is
    written to disk in a second pass after statistics are computed.
    """
    dataset = load_dataset(DATASET_NAME, split=split, streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=1000)

    stats = Counter(empty=0, duplicate=0, kept=0)
    records: list[dict] = []
    seen: set[str] = set()

    for item in dataset:
        if limit is not None and stats["kept"] >= limit:
            break

        try:
            record = normalize_example(item)
        except ValueError:
            stats["empty"] += 1
            continue

        if record is None:
            stats["empty"] += 1
            continue

        key = record_key(record)
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)

        records.append(record)
        stats["kept"] += 1

    if not records:
        raise RuntimeError(f"No question-answer examples were downloaded from split '{split}'.")

    return records, stats


def write_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_stats(name: str, stats: Counter, records: list[dict]) -> str:
    """Human-readable data statistics for one split."""
    total = stats["kept"]
    avg_context = sum(len(r["context"]) for r in records) / total
    avg_question = sum(len(r["question"]) for r in records) / total
    avg_answer = sum(len(r["answer"]) for r in records) / total
    avg_words = sum(len(r["answer"].split()) for r in records) / total
    return (
        f"[{name}]\n"
        f"  Records written : {total}\n"
        f"  Empty/invalid   : {stats['empty']}\n"
        f"  Duplicates      : {stats['duplicate']}\n"
        f"  Avg chars       : context={avg_context:.1f}, question={avg_question:.1f}, answer={avg_answer:.1f}\n"
        f"  Avg answer words: {avg_words:.1f}"
    )


def prepare_dataset(
    output_dir: Path,
    max_examples: int | None = None,
    seed: int = 42,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
) -> dict[str, int]:
    """Download both SQuAD splits and write train.jsonl / validation.jsonl.

    The split follows SQuAD's own example/document-level train and validation
    partitions; no byte-level split is performed.
    """
    counts: dict[str, int] = {}
    for split, filename in (("train", TRAIN_FILENAME), ("validation", VALIDATION_FILENAME)):
        limit = None
        if max_examples is not None:
            if split == "train":
                limit = max_examples
            else:
                limit = max(1, int(max_examples * validation_ratio))

        records, stats = collect_split(split, limit, seed)
        output_path = output_dir / filename
        write_jsonl(records, output_path)
        print(format_stats(filename, stats, records))
        print(f"  Saved to        : {output_path.resolve()}\n")
        counts[filename] = len(records)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SQuAD into structured JSONL splits.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory that receives train.jsonl and validation.jsonl.",
    )
    parser.add_argument(
        "--max-examples",
        type=positive_int,
        default=None,
        help="Max train examples (validation gets this * --validation-ratio). Omit for full splits.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=DEFAULT_VALIDATION_RATIO,
        help="Fraction of --max-examples used for validation.jsonl.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataset(
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        seed=args.seed,
        validation_ratio=args.validation_ratio,
    )


if __name__ == "__main__":
    main()
