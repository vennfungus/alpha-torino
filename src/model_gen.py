"""A compact decoder-only Transformer for causal language modelling."""

import torch
from torch import nn


class CausalLanguageModel(nn.Module):
    """Predict the next token using a decoder-only Transformer."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if vocab_size < 1 or block_size < 1:
            raise ValueError("vocab_size and block_size must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        _, sequence_length = input_ids.shape
        if sequence_length > self.block_size:
            raise ValueError("input sequence exceeds block_size")

        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.transformer(hidden, mask=causal_mask)
        logits = self.lm_head(self.final_norm(hidden))

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss
