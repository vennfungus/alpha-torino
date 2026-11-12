"""UTF-8 byte tokenizer shared by text generation and QA."""

import torch


def encode_text(text: str) -> list[int]:
    """Encode text as UTF-8 byte IDs in the range 0-255."""
    return list(text.encode("utf-8"))


def decode_tokens(token_ids: list[int] | torch.Tensor) -> str:
    """Decode UTF-8 byte IDs, replacing incomplete or invalid byte sequences."""
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return bytes(token_ids).decode("utf-8", errors="replace")
