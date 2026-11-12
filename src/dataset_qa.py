"""JSONL loading and supervised answer-only dataset utilities."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .format_qa import build_training_sequence


class QADataset(Dataset):
    """Create byte-level examples whose loss is limited to the answer."""

    def __init__(
        self,
        records: Sequence[Mapping[str, str]],
        block_size: int,
        max_answer_bytes: int = 128,
    ) -> None:
        if not records:
            raise ValueError("records must contain at least one QA example")
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        if max_answer_bytes < 1:
            raise ValueError("max_answer_bytes must be positive")
        self.records = records
        self.block_size = block_size
        self.max_answer_bytes = max_answer_bytes

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        token_ids, prompt_length = build_training_sequence(
            record,
            block_size=self.block_size,
            max_answer_bytes=self.max_answer_bytes,
        )
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        targets = torch.tensor(token_ids[1:], dtype=torch.long)
        targets[: prompt_length - 1] = -100
        return input_ids, targets


def collate_qa_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad variable-length examples and mask all padded target positions."""
    if not batch:
        raise ValueError("batch must contain at least one QA example")
    max_length = max(len(input_ids) for input_ids, _ in batch)
    input_batch = torch.zeros((len(batch), max_length), dtype=torch.long)
    target_batch = torch.full((len(batch), max_length), -100, dtype=torch.long)
    for row, (input_ids, targets) in enumerate(batch):
        length = len(input_ids)
        input_batch[row, :length] = input_ids
        target_batch[row, :length] = targets
    return input_batch, target_batch


def load_qa_jsonl(path: str | Path) -> list[dict[str, str]]:
    """Load and validate UTF-8 JSONL records with context, question, and answer."""
    source = Path(path)
    records: list[dict[str, str]] = []
    with source.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {source}") from error
            if not isinstance(record, Mapping):
                raise ValueError(f"Expected a JSON object on line {line_number} of {source}")
            fields = {}
            for name in ("context", "question", "answer"):
                value = record.get(name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Field '{name}' must be a non-empty string on line "
                        f"{line_number} of {source}"
                    )
                fields[name] = value.strip()
            records.append(fields)

    if not records:
        raise ValueError(f"No QA examples found in {source}")
    return records
