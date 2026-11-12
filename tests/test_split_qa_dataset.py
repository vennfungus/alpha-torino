"""Tests for scripts/split_qa_dataset.py (personal dataset splitting)."""

import json
from pathlib import Path

import pytest

from scripts import split_qa_dataset


def doc(context: str, n_questions: int) -> list[dict]:
    return [
        {"context": context, "question": f"Question {i} for {context}?", "answer": f"Answer {i}"}
        for i in range(n_questions)
    ]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_input(tmp_path: Path, records: list[dict]) -> Path:
    source = tmp_path / "my_qa.jsonl"
    source.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8"
    )
    return source


# ---------------------------------------------------------------------------
# split_documents (document-level logic)
# ---------------------------------------------------------------------------


def test_split_documents_keeps_context_whole_and_respects_ratio() -> None:
    records = doc("Doc A", 6) + doc("Doc B", 4)
    train, validation = split_qa_dataset.split_documents(records, validation_ratio=0.4, seed=42)

    train_contexts = {r["context"] for r in train}
    validation_contexts = {r["context"] for r in validation}

    # No document appears on both sides.
    assert not train_contexts & validation_contexts
    # Ratio respected at record level (4 of 10 records -> validation).
    assert len(validation) == 4 and len(train) == 6


def test_split_documents_is_reproducible_with_same_seed() -> None:
    records = [r for i in range(20) for r in doc(f"Doc {i}", 2)]
    train_a, val_a = split_qa_dataset.split_documents(records, 0.2, seed=7)
    train_b, val_b = split_qa_dataset.split_documents(records, 0.2, seed=7)

    assert (train_a, val_a) == (train_b, val_b)


def test_split_documents_rejects_invalid_ratio() -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        split_qa_dataset.split_documents(doc("Doc", 2), validation_ratio=0)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        split_qa_dataset.split_documents(doc("Doc", 2), validation_ratio=1.0)


def test_split_documents_guarantees_train_not_empty() -> None:
    # One huge document would exceed the ratio; it must still stay in train.
    records = doc("Only document", 100)
    train, validation = split_qa_dataset.split_documents(records, validation_ratio=0.9, seed=1)

    assert len(train) == 100 and validation == []


# ---------------------------------------------------------------------------
# Full pipeline: split_qa_dataset
# ---------------------------------------------------------------------------


def test_split_qa_dataset_end_to_end_cleans_and_splits(tmp_path: Path) -> None:
    records = doc("Doc A", 4)
    records += doc("Doc B", 4)
    records += [
        {"context": "Doc A", "question": "Question 0 for Doc A?", "answer": "Answer 0"},  # duplicate
        {"context": "   ", "question": "Empty context?", "answer": "Answer"},  # empty
    ]
    source = write_input(tmp_path, records)

    counts = split_qa_dataset.split_qa_dataset(source, tmp_path, validation_ratio=0.25, seed=42)

    assert counts == {"train.jsonl": 4, "validation.jsonl": 4}
    train_records = read_jsonl(tmp_path / "train.jsonl")
    val_records = read_jsonl(tmp_path / "validation.jsonl")

    assert all(set(r) == {"context", "question", "answer"} for r in train_records + val_records)
    # Document-level split: identical contexts never straddle the boundary.
    assert not {r["context"] for r in train_records} & {r["context"] for r in val_records}
    # Deduplication happened before splitting, so no leaked duplicates.
    keys = [tuple(sorted(r.items())) for r in train_records + val_records]
    assert len(keys) == len(set(keys))


def test_split_qa_dataset_skips_blank_lines_and_rejects_bad_json(tmp_path: Path) -> None:
    source = tmp_path / "my_qa.jsonl"
    good = json.dumps({"context": "c", "question": "q?", "answer": "a"}, ensure_ascii=False)
    source.write_text(f"\n{good}\n\n{{bad json}}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON on line 4"):
        split_qa_dataset.split_qa_dataset(source, tmp_path)


def test_split_qa_dataset_raises_when_nothing_usable(tmp_path: Path) -> None:
    source = write_input(tmp_path, [{"context": "", "question": "", "answer": ""}])

    with pytest.raises(RuntimeError, match="No usable QA examples"):
        split_qa_dataset.split_qa_dataset(source, tmp_path)


def test_split_qa_dataset_rejects_single_document_without_validation(tmp_path: Path) -> None:
    source = write_input(tmp_path, doc("Only context", 3))

    with pytest.raises(ValueError, match="at least two distinct contexts"):
        split_qa_dataset.split_qa_dataset(source, tmp_path)


def test_positive_fraction_rejects_out_of_range() -> None:
    with pytest.raises(Exception):
        split_qa_dataset.positive_fraction("0")
    with pytest.raises(Exception):
        split_qa_dataset.positive_fraction("1.5")
    assert split_qa_dataset.positive_fraction("0.25") == 0.25
