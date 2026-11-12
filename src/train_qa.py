"""Supervised training entry point for context-and-question QA."""

import argparse
import random
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm.auto import tqdm

from .dataset_qa import QADataset, collate_qa_batch, load_qa_jsonl
from .format_qa import ANSWER_TERMINATOR
from .model_qa import QALanguageModel


def _autocast_context(device: torch.device, use_bfloat16: bool):
    if use_bfloat16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _run_epoch(
    model: QALanguageModel,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    use_bfloat16: bool,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0
    progress = tqdm(batches, desc="train" if training else "valid", leave=False)
    for input_ids, targets in progress:
        input_ids = input_ids.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with _autocast_context(device, use_bfloat16):
                _, loss = model(input_ids, targets)
            if loss is None:
                raise RuntimeError("QA model did not return a loss for labelled input")
            if not torch.isfinite(loss):
                raise RuntimeError("QA model produced a non-finite loss")
            if training:
                loss.backward()
                optimizer.step()

        token_count = int(targets.ne(-100).sum().item())
        if token_count == 0:
            raise ValueError("QA batches must contain at least one answer target")
        total_loss += loss.item() * token_count
        total_tokens += token_count
        progress.set_postfix(loss=f"{total_loss / total_tokens:.4f}")

    if total_tokens == 0:
        raise ValueError("QA data loader must contain at least one batch")
    return total_loss / total_tokens


def train_qa(
    model: QALanguageModel,
    train_batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    validation_batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    epochs: int,
    patience: int = 3,
    min_delta: float = 0.0,
    device: torch.device | str = "cpu",
    use_bfloat16: bool = False,
) -> list[tuple[float, float]]:
    """Train with answer-only cross entropy and restore the best validation weights."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if patience < 1:
        raise ValueError("patience must be at least 1")
    if min_delta < 0:
        raise ValueError("min_delta must not be negative")

    target_device = torch.device(device)
    model.to(target_device)
    history: list[tuple[float, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(
            model, train_batches, target_device, use_bfloat16, optimizer
        )
        validation_loss = _run_epoch(
            model, validation_batches, target_device, use_bfloat16
        )
        history.append((train_loss, validation_loss))
        print(
            f"Epoch {epoch}/{epochs} - train loss: {train_loss:.4f} "
            f"- validation loss: {validation_loss:.4f}"
        )

        if validation_loss < best_loss - min_delta:
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping after {epoch} epoch(s); "
                f"best validation loss: {best_loss:.4f}"
            )
            break

    if best_state is None:
        raise RuntimeError("QA training completed without a best model checkpoint")
    model.load_state_dict(best_state)
    return history


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the supervised QA model.")
    parser.add_argument("--model-config", type=Path, default=Path("configs/model_config_qa.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train_config_qa.yaml"))
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    args = parser.parse_args()

    model_config = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    train_config = yaml.safe_load(args.train_config.read_text(encoding="utf-8"))
    if not isinstance(model_config, dict) or not isinstance(train_config, dict):
        raise ValueError("Configuration files must contain YAML mappings")

    vocab_size = model_config.get("vocab_size", 256)
    if vocab_size != 256:
        raise ValueError("The UTF-8 byte tokenizer requires vocab_size: 256")
    block_size = _positive_int(model_config.get("block_size"), "block_size")
    batch_size = _positive_int(train_config.get("batch_size"), "batch_size")
    num_workers = train_config.get("num_workers", 0)
    if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")
    max_answer_bytes = _positive_int(
        train_config.get("max_answer_bytes", 128), "max_answer_bytes"
    )

    train_path = args.train_file or Path(train_config.get("train_file", "data/raw/train.jsonl"))
    validation_path = args.validation_file or Path(
        train_config.get("validation_file", "data/raw/validation.jsonl")
    )
    train_records = load_qa_jsonl(train_path)
    validation_records = load_qa_jsonl(validation_path)
    train_dataset = QADataset(train_records, block_size, max_answer_bytes)
    validation_dataset = QADataset(validation_records, block_size, max_answer_bytes)

    seed = train_config.get("seed", 42)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    validation_samples = train_config.get("validation_samples_per_epoch")
    if validation_samples is not None:
        validation_samples = _positive_int(
            validation_samples, "validation_samples_per_epoch"
        )
        validation_sampler = SubsetRandomSampler(
            random.sample(
                range(len(validation_dataset)),
                min(validation_samples, len(validation_dataset)),
            ),
            generator=generator,
        )
    else:
        validation_sampler = None

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_qa_batch,
        "pin_memory": train_config.get("device", "auto") != "cpu",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_options["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        sampler=validation_sampler,
        **loader_options,
    )

    device_setting = train_config.get("device", "auto")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_setting == "auto"
        else torch.device(device_setting)
    )
    mixed_precision = train_config.get("mixed_precision", True)
    if not isinstance(mixed_precision, bool):
        raise ValueError("mixed_precision must be a boolean")
    use_bfloat16 = (
        mixed_precision and device.type == "cuda" and torch.cuda.is_bf16_supported()
    )

    architecture = {
        "vocab_size": 256,
        "block_size": block_size,
        "d_model": _positive_int(model_config.get("d_model"), "d_model"),
        "n_heads": _positive_int(model_config.get("n_heads"), "n_heads"),
        "n_layers": _positive_int(model_config.get("n_layers"), "n_layers"),
        "dropout": model_config.get("dropout", 0.1),
    }
    model = QALanguageModel(**architecture)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.get("learning_rate", 0.0001),
        weight_decay=train_config.get("weight_decay", 0.0),
    )
    history = train_qa(
        model,
        train_loader,
        validation_loader,
        optimizer,
        epochs=_positive_int(train_config.get("epochs"), "epochs"),
        patience=_positive_int(
            train_config.get("early_stopping_patience", 3),
            "early_stopping_patience",
        ),
        min_delta=train_config.get("early_stopping_min_delta", 0.0),
        device=device,
        use_bfloat16=use_bfloat16,
    )

    checkpoint_path = args.checkpoint_path or Path(
        train_config.get("checkpoint_path", "checkpoints/model_qa.pt")
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": architecture,
            "tokenizer": "utf8_bytes",
            "answer_terminator": ANSWER_TERMINATOR.decode("ascii"),
            "completed_epoch": len(history),
            "epoch_losses": history,
        },
        temporary_path,
    )
    temporary_path.replace(checkpoint_path)
    print(f"Saved QA checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
