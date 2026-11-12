"""Autoregressive text-token generation."""

import argparse
from pathlib import Path

import torch

from .dataset import decode_tokens, encode_text
from .model import CausalLanguageModel


@torch.no_grad()
def generate(
    model: CausalLanguageModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Append sampled tokens to a batch of input token IDs."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape (batch, sequence)")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    model.eval()
    for _ in range(max_new_tokens):
        context = input_ids[:, -model.block_size :]
        logits, _ = model(context)
        next_token_logits = logits[:, -1, :] / temperature
        next_token = torch.multinomial(
            torch.softmax(next_token_logits, dim=-1),
            num_samples=1,
        )
        input_ids = torch.cat((input_ids, next_token), dim=1)
    return input_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained causal language model.")
    parser.add_argument("--prompt", required=True, help="Starting text for generation")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model.pt"))
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint.get("tokenizer") != "utf8_bytes":
        raise ValueError("Checkpoint does not use the supported UTF-8 byte tokenizer")
    model = CausalLanguageModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    prompt_ids = torch.tensor([encode_text(args.prompt)], dtype=torch.long, device=device)
    if prompt_ids.shape[1] == 0:
        raise ValueError("prompt must not be empty")
    output = generate(
        model,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(decode_tokens(output[0]))


if __name__ == "__main__":
    main()
