"""Training loop for a causal language model."""

import argparse
from collections.abc import Iterable
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import BufferedShuffleSampler, MappedByteCorpus, MappedCausalLMDataset
from .model import CausalLanguageModel


def split_tokens(
    token_ids: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split tokens in order into 90% training and 10% validation data."""
    split_index = int(len(token_ids) * 0.9)
    if split_index <= block_size or len(token_ids) - split_index <= block_size:
        raise ValueError(
            "Corpus is too small for a 90/10 train/validation split with "
            f"block_size ({block_size})"
        )
    return token_ids[:split_index], token_ids[split_index:]


def train_model(
    model: CausalLanguageModel,
    train_batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    epochs: int,
    validation_batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    patience: int = 2,
    min_delta: float = 0.0,
    device: torch.device | str = "cpu",
) -> list[tuple[float, float]]:
    """Train and validate each epoch, returning train and validation losses."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if patience < 1:
        raise ValueError("patience must be at least 1")
    if min_delta < 0:
        raise ValueError("min_delta must not be negative")

    model.to(device)
    epoch_losses = []
    best_validation_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        train_progress = tqdm(
            train_batches,
            desc=f"Epoch {epoch}/{epochs} [train]",
            leave=False,
        )
        for input_ids, targets in train_progress:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(input_ids, targets)
            if loss is None:
                raise RuntimeError("Model did not return a loss for labelled input")
            loss.backward()
            optimizer.step()
            token_count = targets.numel()
            total_loss += loss.item() * token_count
            total_tokens += token_count
            train_progress.set_postfix(loss=f"{total_loss / total_tokens:.4f}")
        if total_tokens == 0:
            raise ValueError("train_batches must contain at least one batch")
        train_loss = total_loss / total_tokens

        model.eval()
        validation_loss_total = 0.0
        validation_tokens = 0
        validation_progress = tqdm(
            validation_batches,
            desc=f"Epoch {epoch}/{epochs} [valid]",
            leave=False,
        )
        with torch.no_grad():
            for input_ids, targets in validation_progress:
                targets = targets.to(device)
                _, loss = model(input_ids.to(device), targets)
                if loss is None:
                    raise RuntimeError("Model did not return a validation loss")
                token_count = targets.numel()
                validation_loss_total += loss.item() * token_count
                validation_tokens += token_count
                validation_progress.set_postfix(
                    loss=f"{validation_loss_total / validation_tokens:.4f}"
                )
        if validation_tokens == 0:
            raise ValueError("validation_batches must contain at least one batch")
        validation_loss = validation_loss_total / validation_tokens

        epoch_losses.append((train_loss, validation_loss))
        print(
            f"Epoch {epoch}/{epochs} - train loss: {train_loss:.4f} "
            f"- validation loss: {validation_loss:.4f}"
        )
        if validation_loss < best_validation_loss - min_delta:
            best_validation_loss = validation_loss
            best_model_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping after {epoch} epoch(s); "
                    f"best validation loss: {best_validation_loss:.4f}"
                )
                break
    if best_model_state is None:
        raise RuntimeError("Training completed without a best model checkpoint")
    model.load_state_dict(best_model_state)
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

    with MappedByteCorpus(args.data_dir) as corpus:
        split_index = int(corpus.byte_length * 0.9)
        if (
            split_index <= block_size
            or corpus.byte_length - split_index <= block_size
        ):
            raise ValueError(
                "Corpus is too small for a 90/10 train/validation split with "
                f"block_size ({block_size})"
            )

        train_dataset = MappedCausalLMDataset(
            corpus,
            start=0,
            end=split_index,
            block_size=block_size,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=BufferedShuffleSampler(train_dataset),
        )
        validation_dataset = MappedCausalLMDataset(
            corpus,
            start=split_index,
            end=corpus.byte_length,
            block_size=block_size,
        )
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

        print(
            f"Training on {len(corpus.files)} text file(s), "
            f"{corpus.byte_length} UTF-8 bytes, device={device}"
        )
        train_model(
            model,
            train_loader,
            optimizer,
            epochs=train_config["epochs"],
            validation_batches=validation_loader,
            patience=train_config.get("early_stopping_patience", 2),
            min_delta=train_config.get("early_stopping_min_delta", 0.0),
            device=device,
        )

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
