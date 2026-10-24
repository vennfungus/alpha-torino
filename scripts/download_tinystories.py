
from pathlib import Path

from datasets import load_dataset


OUTPUT_PATH = Path("data/raw/tinystories_sample.txt")
MAX_STORIES = 300


def main() -> None:
    dataset = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=True,
    )

    dataset = dataset.shuffle(
        seed=42,
        buffer_size=1000,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for item in dataset:
            text = item.get("text", "").strip()

            if not text:
                continue

            file.write(text)
            file.write("\n\n")

            count += 1

            if count >= MAX_STORIES:
                break

    if count == 0:
        raise RuntimeError("No stories were downloaded.")

    print(f"Saved {count} stories to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
