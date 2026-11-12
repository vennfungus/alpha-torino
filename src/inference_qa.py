"""Generate answers from context and question using a trained QA checkpoint."""

import argparse
from pathlib import Path

import torch

from .format_qa import ANSWER_TERMINATOR, build_prompt
from .model_qa import QALanguageModel
from .tokenizer import decode_tokens, encode_text


@torch.no_grad()
def generate_answer(
    model: QALanguageModel,
    context: str,
    question: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    """Generate an answer, stopping when the learned answer delimiter appears."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature < 0:
        raise ValueError("temperature must not be negative")

    prompt = build_prompt(context, question, model.block_size)
    prompt_ids = encode_text(prompt)
    if not prompt_ids:
        raise ValueError("context and question must produce a non-empty prompt")

    device = next(model.parameters()).device
    generated = list(prompt_ids)
    answer_ids: list[int] = []
    terminator = list(ANSWER_TERMINATOR)
    model.eval()

    for _ in range(max_new_tokens):
        input_ids = torch.tensor(
            [generated[-model.block_size :]], dtype=torch.long, device=device
        )
        logits, _ = model(input_ids)
        next_token_logits = logits[:, -1, :]
        if temperature == 0:
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        else:
            probabilities = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
        token = int(next_token.item())
        generated.append(token)
        answer_ids.append(token)
        if answer_ids[-len(terminator) :] == terminator:
            del answer_ids[-len(terminator) :]
            break

    return decode_tokens(answer_ids).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer a question using a trained QA model.")
    parser.add_argument("--context", required=True, help="Reference passage for the answer")
    parser.add_argument("--question", required=True, help="Question about the context")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model_qa.pt"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Use 0 for deterministic greedy generation.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint.get("tokenizer") != "utf8_bytes":
        raise ValueError("Checkpoint does not use the supported UTF-8 byte tokenizer")
    model = QALanguageModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    answer = generate_answer(
        model,
        args.context,
        args.question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(answer)


if __name__ == "__main__":
    main()
