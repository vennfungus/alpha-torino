"""Tests for the full scripts/download_qa.py pipeline."""

import argparse
import json
from pathlib import Path

import pytest

from scripts import download_qa


class FakeDataset(list):
    def shuffle(self, seed: int, buffer_size: int) -> "FakeDataset":
        assert seed == 42
        assert buffer_size == 1000
        return self


def make_rows() -> list[dict]:
    return [
        {
            "context": "  A short context.  ",
            "question": " What happened? ",
            "answers": {"text": ["  An event. ", "Another valid answer."]},
        },
        {
            "context": "Another context.",
            "question": "Second question?",
            "answers": {"text": ["Second answer."]},
        },
        # Invalid shape -> counted as invalid.
        {"context": "Broken context.", "question": "Broken?", "answers": None},
        # Empty answer after strip -> dropped.
        {
            "context": "Context with no answer.",
            "question": "Unanswered?",
            "answers": {"text": ["   "]},
        },
        # Duplicate of the first row -> dropped.
        {
            "context": "A short context.",
            "question": "What happened?",
            "answers": {"text": ["An event."]},
        },
    ]


@pytest.fixture()
def patch(monkeypatch: pytest.MonkeyPatch):
    """Patch load_dataset per split; tests add rows with returned setter."""
    data: dict[str, list] = {}
    calls: list[tuple[str, str, bool]] = []

    def fake_load_dataset(dataset_name: str, split: str, streaming: bool) -> FakeDataset:
        calls.append((dataset_name, split, streaming))
        return FakeDataset(data.get(split, []))

    monkeypatch.setattr(download_qa, "load_dataset", fake_load_dataset)
    return data, calls


# ---------------------------------------------------------------------------
# normalize_example
# ---------------------------------------------------------------------------


def test_normalize_example_strips_fields_and_picks_first_valid_answer() -> None:
    record = download_qa.normalize_example(make_rows()[0])

    assert record == {
        "context": "A short context.",
        "question": "What happened?",
        "answer": "An event.",
    }


def test_normalize_example_returns_none_on_empty_field() -> None:
    row = {
        "context": "Context.",
        "question": "   ",
        "answers": {"text": ["Answer."]},
    }

    assert download_qa.normalize_example(row) is None


def test_normalize_example_raises_on_missing_or_malformed_fields() -> None:
    with pytest.raises(ValueError, match="context, question, and answers"):
        download_qa.normalize_example({"question": "?", "answers": {"text": ["a"]}})

    with pytest.raises(ValueError, match="'text' field"):
        download_qa.normalize_example(
            {"context": "c", "question": "q", "answers": {"text": "not-a-list"}}
        )


# ---------------------------------------------------------------------------
# record_key
# ---------------------------------------------------------------------------


def test_record_key_is_order_sensitive() -> None:
    base = {"context": "A", "question": "B", "answer": "C"}
    swapped = {"context": "B", "question": "A", "answer": "C"}

    assert download_qa.record_key(base) == download_qa.record_key(dict(base))
    assert download_qa.record_key(base) != download_qa.record_key(swapped)


def test_record_key_does_not_collide_on_embedded_separator() -> None:
    first = {"context": "a\x1fb", "question": "c", "answer": "d"}
    second = {"context": "a", "question": "b\x1fc", "answer": "d"}

    assert download_qa.record_key(first) != download_qa.record_key(second)


# ---------------------------------------------------------------------------
# write_jsonl
# ---------------------------------------------------------------------------


def test_write_jsonl_writes_one_json_object_per_line(tmp_path: Path) -> None:
    records = [
        {"context": "ctx", "question": "q?", "answer": "Trả lời có dấu."},
        {"context": "ctx\nmulti-line", "question": "q\ttab?", "answer": "a"},
    ]
    output = tmp_path / "nested" / "train.jsonl"

    download_qa.write_jsonl(records, output)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == records


# ---------------------------------------------------------------------------
# collect_split
# ---------------------------------------------------------------------------


def test_collect_split_drops_empty_and_duplicate_rows(patch) -> None:
    data, calls = patch
    data["train"] = make_rows()

    records, stats = download_qa.collect_split("train", limit=None, seed=42)

    assert [r["question"] for r in records] == ["What happened?", "Second question?"]
    assert stats["kept"] == 2
    assert stats["empty"] == 2  # malformed answers + whitespace-only answer
    assert stats["duplicate"] == 1
    assert calls == [("rajpurkar/squad", "train", True)]


def test_collect_split_respects_limit_and_raises_when_empty(patch) -> None:
    data, _ = patch
    data["train"] = make_rows()

    records, stats = download_qa.collect_split("train", limit=1, seed=42)
    assert stats["kept"] == 1 and len(records) == 1

    data["train"] = [{"context": "", "question": "q", "answers": {"text": ["a"]}}]
    with pytest.raises(RuntimeError, match="No question-answer examples"):
        download_qa.collect_split("train", limit=None, seed=42)


# ---------------------------------------------------------------------------
# Full pipeline: prepare_dataset
# ---------------------------------------------------------------------------


def test_prepare_dataset_writes_both_splits_from_document_level_splits(patch, tmp_path: Path) -> None:
    data, calls = patch
    data["train"] = make_rows()
    data["validation"] = [
        {
            "context": "Validation context.",
            "question": "Validation question?",
            "answers": {"text": ["Validation answer."]},
        }
    ] * 2  # one valid row + one duplicate

    counts = download_qa.prepare_dataset(output_dir=tmp_path)

    assert counts == {"train.jsonl": 2, "validation.jsonl": 1}
    assert calls == [
        ("rajpurkar/squad", "train", True),
        ("rajpurkar/squad", "validation", True),
    ]

    train_lines = (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    val_lines = (tmp_path / "validation.jsonl").read_text(encoding="utf-8").splitlines()
    train_records = [json.loads(line) for line in train_lines]
    val_records = [json.loads(line) for line in val_lines]

    assert all(set(r) == {"context", "question", "answer"} for r in train_records + val_records)
    # Document-level split: no example appears in both sides.
    assert not ({download_qa.record_key(r) for r in train_records} & {download_qa.record_key(r) for r in val_records})


def test_prepare_dataset_max_examples_scales_validation_by_ratio(patch, tmp_path: Path) -> None:
    data, _ = patch
    data["train"] = [
        {
            "context": f"Context {i}",
            "question": f"Question {i}?",
            "answers": {"text": [f"Answer {i}"]},
        }
        for i in range(10)
    ]
    data["validation"] = [
        {
            "context": f"Val context {i}",
            "question": f"Val question {i}?",
            "answers": {"text": [f"Val answer {i}"]},
        }
        for i in range(10)
    ]

    counts = download_qa.prepare_dataset(output_dir=tmp_path, max_examples=5, validation_ratio=0.2)

    assert counts == {"train.jsonl": 5, "validation.jsonl": 1}


def test_format_stats_reports_counts_and_averages(patch, tmp_path: Path) -> None:
    data, _ = patch
    data["train"] = make_rows()

    records, stats = download_qa.collect_split("train", limit=None, seed=42)
    report = download_qa.format_stats("train.jsonl", stats, records)

    assert "Records written : 2" in report
    assert "Empty/invalid   : 2" in report
    assert "Duplicates      : 1" in report
    assert "Avg chars" in report and "Avg answer words" in report


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def test_positive_int_rejects_invalid_values() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        download_qa.positive_int("0")

    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        download_qa.positive_int("not-a-number")
