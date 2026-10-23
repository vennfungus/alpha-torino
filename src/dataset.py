"""Dataset utilities for next-token language modelling."""

from pathlib import Path

import torch
from torch.utils.data import Dataset


def encode_text(text: str) -> list[int]:
    """Encode text as UTF-8 byte IDs in the range 0-255."""
    return list(text.encode("utf-8"))


def decode_tokens(token_ids: list[int] | torch.Tensor) -> str:
    """Decode UTF-8 byte IDs, replacing incomplete or invalid byte sequences."""
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return bytes(token_ids).decode("utf-8", errors="replace")


def load_raw_corpus(raw_dir: str | Path) -> tuple[str, list[Path]]:
    """Read and join all UTF-8 .txt files beneath a raw-data directory."""
    raw_path = Path(raw_dir)
    files = sorted(raw_path.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {raw_path}")

    documents = [file.read_text(encoding="utf-8").strip() for file in files]
    documents = [document for document in documents if document]
    if not documents:
        raise ValueError(f"All .txt files in {raw_path} are empty")
    return "\n\n".join(documents), files


class CausalLMDataset(Dataset):
    """Create fixed-length, next-token prediction examples from token IDs."""

    def __init__(self, token_ids: list[int] | torch.Tensor, block_size: int) -> None:
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        if len(token_ids) <= block_size:
            raise ValueError("token_ids must contain more tokens than block_size")
        self.token_ids = torch.as_tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.token_ids) - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = self.token_ids[index : index + self.block_size]
        target_ids = self.token_ids[index + 1 : index + self.block_size + 1]
        return input_ids, target_ids
