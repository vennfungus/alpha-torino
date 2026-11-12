from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset_qa import QADataset, collate_qa_batch
from src.model_qa import QALanguageModel
from src.train_qa import _resolve_data_dir, train_qa


def test_train_qa_optimises_answer_only_loss_and_reports_history() -> None:
    records = [
        {
            "context": "The capital of France is Paris.",
            "question": "What is the capital of France?",
            "answer": "Paris",
        },
        {
            "context": "The capital of Italy is Rome.",
            "question": "What is the capital of Italy?",
            "answer": "Rome",
        },
    ]
    dataset = QADataset(records, block_size=96, max_answer_bytes=16)
    batches = DataLoader(dataset, batch_size=2, collate_fn=collate_qa_batch)
    model = QALanguageModel(
        vocab_size=256,
        block_size=96,
        d_model=8,
        n_heads=2,
        n_layers=1,
        dropout=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    history = train_qa(
        model,
        batches,
        batches,
        optimizer,
        epochs=1,
    )

    assert len(history) == 1
    assert all(torch.isfinite(torch.tensor(history[0])))


def test_train_qa_rejects_invalid_early_stopping_settings() -> None:
    model = QALanguageModel(256, 8, d_model=8, n_heads=2, n_layers=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    try:
        train_qa(model, [], [], optimizer, epochs=1, patience=0)
    except ValueError as error:
        assert "patience" in str(error)
    else:
        raise AssertionError("Expected invalid patience to be rejected")


def test_default_data_dir_uses_existing_qa_dataset_compatibility_path(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    legacy_dir = tmp_path / "data" / "qa_dataset"
    legacy_dir.mkdir(parents=True)

    assert _resolve_data_dir(Path("data/qa"), None) == Path("data/qa_dataset")


def test_explicit_data_dir_overrides_compatibility_path(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "qa_dataset").mkdir(parents=True)

    assert _resolve_data_dir(Path("data/qa"), Path("custom/qa")) == Path("custom/qa")
