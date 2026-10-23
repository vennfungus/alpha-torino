"""Training loop for a causal language model."""

import argparse
from collections.abc import Iterable
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import CausalLMDataset, encode_text, load_raw_corpus
from .model import CausalLanguageModel


def train_model(
    model: CausalLanguageModel,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device | str = "cpu",
) -> list[float]:
    """Train the model and return the mean loss for each epoch."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    model.to(device)
    model.train()
    epoch_losses = []
    for _ in range(epochs):
        total_loss = 0.0
        batch_count = 0
        for input_ids, targets in batches:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(input_ids, targets)
            if loss is None:
                raise RuntimeError("Model did not return a loss for labelled input")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        if batch_count == 0:
            raise ValueError("batches must contain at least one batch")
        epoch_losses.append(total_loss / batch_count)
    return epoch_losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level causal language model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model_config.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train_config.yaml"))
    args = parser.parse_args()

    model_config = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    train_config = yaml.safe_load(args.train_config.read_text(encoding="utf-8"))
    if not isinstance(model_config, dict) or not isinstance(train_config, dict):
        raise ValueError("Configuration files must contain YAML mappings")

    if model_config.get("vocab_size", 256) != 256:
        raise ValueError("The UTF-8 byte tokenizer requires vocab_size: 256")
    block_size = model_config["block_size"]
    batch_size = train_config["batch_size"]
    if block_size < 1 or batch_size < 1:
        raise ValueError("block_size and batch_size must be at least 1")

    text, source_files = load_raw_corpus(args.data_dir)
    token_ids = torch.tensor(encode_text(text), dtype=torch.long)
    if len(token_ids) <= block_size:
        raise ValueError(
            f"Corpus has {len(token_ids)} bytes; it must contain more than "
            f"block_size ({block_size}) bytes"
        )

    split_index = int(len(token_ids) * 0.9)
    has_validation = split_index > block_size and len(token_ids) - split_index > block_size
    if has_validation:
        train_tokens = token_ids[:split_index]
        validation_tokens = token_ids[split_index:]
    else:
        train_tokens = token_ids
        validation_tokens = None

    train_dataset = CausalLMDataset(train_tokens, block_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device_setting = train_config.get("device", "auto")
    if device_setting == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_setting)

    architecture = {
        "vocab_size": 256,
        "block_size": block_size,
        "d_model": model_config["d_model"],
        "n_heads": model_config["n_heads"],
        "n_layers": model_config["n_layers"],
        "dropout": model_config["dropout"],
    }
    model = CausalLanguageModel(**architecture)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config.get("weight_decay", 0.0),
    )

    print(f"Training on {len(source_files)} text file(s), {len(token_ids)} UTF-8 bytes, device={device}")
    losses = train_model(
        model,
        train_loader,
        optimizer,
        epochs=train_config["epochs"],
        device=device,
    )

    validation_loss = None
    if validation_tokens is not None:
        validation_dataset = CausalLMDataset(validation_tokens, block_size)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for input_ids, targets in validation_loader:
                _, loss = model(input_ids.to(device), targets.to(device))
                if loss is None:
                    raise RuntimeError("Model did not return a validation loss")
                count = targets.numel()
                total_loss += loss.item() * count
                total_tokens += count
        validation_loss = total_loss / total_tokens

    for epoch, loss in enumerate(losses, start=1):
        print(f"Epoch {epoch}/{len(losses)} - train loss: {loss:.4f}")
    if validation_loss is not None:
        print(f"Validation loss: {validation_loss:.4f}")
    else:
        print("Validation skipped: corpus is too small for a separate validation split.")

    checkpoint_path = Path(train_config["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": architecture,
            "tokenizer": "utf8_bytes",
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
