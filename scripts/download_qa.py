import argparse
from pathlib import Path
from typing import Mapping

from datasets import load_dataset


OUTPUT_PATH = Path("data/raw/squad_qa.txt")
MAX_EXAMPLES = 50000
DATASET_NAME = "rajpurkar/squad"


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error

    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def download_qa_dataset(
    output_path: Path,
    max_examples: int = MAX_EXAMPLES,
    split: str = "train",
    seed: int = 42,
) -> int:
    if max_examples <= 0:
        raise ValueError("max_examples must be a positive integer")

    dataset = load_dataset(DATASET_NAME, split=split, streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=1000)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with output_path.open("w", encoding="utf-8") as file:
        for item in dataset:
            context = item.get("context")
            question = item.get("question")
            answers = item.get("answers")

            if (
                not isinstance(context, str)
                or not isinstance(question, str)
                or not isinstance(answers, Mapping)
            ):
                raise ValueError("Dataset rows must contain context, question, and answers.")

            answer_texts = answers.get("text")
            if not isinstance(answer_texts, list):
                raise ValueError("Dataset answers must contain a list in the 'text' field.")

            context = context.strip()
            question = question.strip()
            answer = next(
                (text.strip() for text in answer_texts if isinstance(text, str) and text.strip()),
                "",
            )
            if not context or not question or not answer:
                continue

            file.write(f"Context: {context}\n")
            file.write(f"Question: {question}\n")
            file.write(f"Answer: {answer}\n\n")

            count += 1
            if count >= max_examples:
                break

    if count == 0:
        raise RuntimeError("No question-answer examples were downloaded.")

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a SQuAD question-answer dataset.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--max-examples", type=positive_int, default=MAX_EXAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    count = download_qa_dataset(
        output_path=args.output,
        max_examples=args.max_examples,
        split=args.split,
        seed=args.seed,
    )
    print(f"Saved {count} question-answer examples to {args.output.resolve()}")


if __name__ == "__main__":
    main()
