"""Evaluation utilities for causal language models."""

import torch

from .model import CausalLanguageModel


@torch.no_grad()
def evaluate(
    model: CausalLanguageModel,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | str = "cpu",
) -> float:
    """Return the mean next-token cross-entropy over the supplied batches."""
    if not batches:
        raise ValueError("batches must contain at least one batch")

    model.to(device)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for input_ids, targets in batches:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        _, loss = model(input_ids, targets)
        if loss is None:
            raise RuntimeError("Model did not return a loss for labelled input")
        token_count = targets.numel()
        total_loss += loss.item() * token_count
        total_tokens += token_count
    return total_loss / total_tokens
