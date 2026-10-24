import pytest
import torch

from src.model import CausalLanguageModel
from src.train import split_tokens, train_model


def test_split_tokens_uses_ninety_ten_ratio() -> None:
    train_tokens, validation_tokens = split_tokens(torch.arange(100), block_size=4)

    assert len(train_tokens) == 90
    assert len(validation_tokens) == 10
    assert train_tokens[-1].item() == 89
    assert validation_tokens[0].item() == 90


def test_split_tokens_rejects_corpus_too_small_for_both_splits() -> None:
    with pytest.raises(ValueError, match="too small for a 90/10"):
        split_tokens(torch.arange(20), block_size=2)


def test_train_model_reports_training_and_validation_loss_each_epoch(capsys) -> None:
    model = CausalLanguageModel(
        vocab_size=16,
        block_size=4,
        d_model=8,
        n_heads=2,
        n_layers=1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    train_batches = [(torch.tensor([[1, 2, 3, 4]]), torch.tensor([[2, 3, 4, 5]]))]
    validation_batches = [(torch.tensor([[5, 6, 7, 8]]), torch.tensor([[6, 7, 8, 9]]))]

    losses = train_model(
        model,
        train_batches,
        optimizer,
        epochs=2,
        validation_batches=validation_batches,
    )

    output = capsys.readouterr().out
    assert len(losses) == 2
    assert all(torch.isfinite(torch.tensor(losses)).flatten())
    assert output.count("validation loss:") == 2
