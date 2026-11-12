"""Evaluate answer generation on a held-out QA JSONL file."""

import argparse
import random
from pathlib import Path

import torch

from .dataset_qa import load_qa_jsonl
from .inference_qa import generate_answer
from .metrics_qa import exact_match, token_f1
from .model_qa import QALanguageModel


def evaluate_records(
    model: QALanguageModel,
    records: list[dict[str, str]],
    max_new_tokens: int = 128,
) -> tuple[dict[str, float], list[dict[str, str]]]:
    if not records:
        raise ValueError("records must contain at least one QA example")

    predictions: list[dict[str, str]] = []
    exact_scores = []
    f1_scores = []
    for record in records:
        prediction = generate_answer(
            model,
            record["context"],
            record["question"],
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        exact_scores.append(exact_match(prediction, record["answer"]))
        f1_scores.append(token_f1(prediction, record["answer"]))
        predictions.append({**record, "prediction": prediction})

    count = len(records)
    return {
        "exact_match": 100.0 * sum(exact_scores) / count,
        "token_f1": 100.0 * sum(f1_scores) / count,
        "examples": float(count),
    }, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the supervised QA model.")
    parser.add_argument("--data", type=Path, default=Path("data/raw/validation.jsonl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model_qa.pt"))
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_examples < 1:
        raise ValueError("max-examples must be positive")
    if args.max_new_tokens < 0:
        raise ValueError("max-new-tokens must be non-negative")

    records = load_qa_jsonl(args.data)
    random.Random(args.seed).shuffle(records)
    records = records[: args.max_examples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint.get("tokenizer") != "utf8_bytes":
        raise ValueError("Checkpoint does not use the supported UTF-8 byte tokenizer")
    model = QALanguageModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics, predictions = evaluate_records(
        model, records, max_new_tokens=args.max_new_tokens
    )
    print(f"Examples     : {int(metrics['examples'])}")
    print(f"Exact match  : {metrics['exact_match']:.2f}")
    print(f"Token F1     : {metrics['token_f1']:.2f}")
    for item in predictions[:5]:
        print(
            f"\nQuestion: {item['question']}\n"
            f"Reference: {item['answer']}\n"
            f"Prediction: {item['prediction']}"
        )


if __name__ == "__main__":
    main()
