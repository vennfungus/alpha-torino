# src.measure_gen

"""Standalone metrics logger: 
    validation loss, perplexity, checkpoint size, inference speed."""

import math
import time
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

from src.dataset_gen import MappedByteCorpus, MappedCausalLMDataset
from src.inference_gen import generate
from src.model_gen import CausalLanguageModel

import csv
from datetime import datetime

def main() -> None:
    model_config = yaml.safe_load(Path("configs/model_config_gen.yaml").read_text(encoding="utf-8"))
    train_config = yaml.safe_load(Path("configs/train_config_gen.yaml").read_text(encoding="utf-8"))

    checkpoint_path = Path(train_config["checkpoint_path"])
    data_dir = Path("data/raw")
    batch_size = train_config["batch_size"]

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = CausalLanguageModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --- Checkpoint size (both file .pt and real-size on disk) ---
    checkpoint_bytes = checkpoint_path.stat().st_size
    checkpoint_mb = checkpoint_bytes / (1024 * 1024)
    param_count = sum(p.numel() for p in model.parameters())

    # --- Validation loss + perplexity (splitting into 90/10 as the same as train.py) ---
    block_size = model_config["block_size"]
    with MappedByteCorpus(data_dir) as corpus:
        split_index = int(corpus.byte_length * 0.9)
        validation_dataset = MappedCausalLMDataset(
            corpus, start=split_index, end=corpus.byte_length, block_size=block_size
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            num_workers=train_config.get("num_workers", 0),
        )

        loss_total = 0.0
        token_total = 0
        with torch.no_grad():
            for input_ids, targets in validation_loader:
                input_ids = input_ids.to(device)
                targets = targets.to(device)
                _, loss = model(input_ids, targets)
                loss_total += loss.item() * targets.numel()
                token_total += targets.numel()
    validation_loss = loss_total / token_total
    perplexity = math.exp(validation_loss)

    # --- Inference speed (tokens/s, generating autoregressive batch=1) ---
    prompt = torch.randint(0, 256, (1, block_size), device=device)
    max_new_tokens = 200

    # Onene-time warming up, not calc compile/lazy-init
    generate(model, prompt, max_new_tokens=10)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    generate(model, prompt, max_new_tokens=max_new_tokens)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tokens_per_second = max_new_tokens / elapsed

    print("=" * 50)
    print(f"Checkpoint file       : {checkpoint_path}")
    print(f"Checkpoint size       : {checkpoint_bytes:,} bytes ({checkpoint_mb:.2f} MB)")
    print(f"Parameters            : {param_count:,}")
    print(f"Validation loss       : {validation_loss:.4f}")
    print(f"Perplexity            : {perplexity:.4f}")
    print(f"Inference speed       : {tokens_per_second:.2f} tokens/s (batch=1, {max_new_tokens} tokens)")

    # Saving into csv 
    csv_path = Path("results/alphametrics.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_mb": round(checkpoint_mb, 2),
        "parameters": param_count,
        "validation_loss": validation_loss,
        "perplexity": perplexity,
        "tokens_per_second": tokens_per_second,
        "batch_size": batch_size,
        "vocab_size": model_config["vocab_size"],
        "block_size": model_config["block_size"],
        "d_model": model_config["d_model"],
        "n_heads": model_config["n_heads"],
        "n_layers": model_config["n_layers"],
        "dropout": model_config["dropout"],
    }

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

if __name__ == "__main__":
    main()