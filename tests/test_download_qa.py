import argparse
from pathlib import Path

import pytest

from scripts import download_qa


class FakeDataset(list):
    def shuffle(self, seed: int, buffer_size: int) -> "FakeDataset":
        assert seed == 42
        assert buffer_size == 1000
        return self


def test_download_qa_dataset_formats_examples_and_respects_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = FakeDataset(
        [
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
        ]
    )
    load_calls: list[tuple[str, str, bool]] = []

    def fake_load_dataset(
        dataset_name: str, split: str, streaming: bool
    ) -> FakeDataset:
        load_calls.append((dataset_name, split, streaming))
        return rows

    monkeypatch.setattr(download_qa, "load_dataset", fake_load_dataset)
    output = tmp_path / "nested" / "qa.txt"

    count = download_qa.download_qa_dataset(output, max_examples=1)

    assert count == 1
    assert output.read_text(encoding="utf-8") == (
        "Context: A short context.\n"
        "Question: What happened?\n"
        "Answer: An event.\n\n"
    )
    assert load_calls == [("rajpurkar/squad", "train", True)]


def test_download_qa_dataset_rejects_non_positive_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        download_qa.download_qa_dataset(tmp_path / "qa.txt", max_examples=0)


def test_positive_int_rejects_invalid_values() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        download_qa.positive_int("0")

    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        download_qa.positive_int("not-a-number")
