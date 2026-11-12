import pytest
import torch

from src.model import CausalLanguageModel
from src.train import split_tokens, train_model


class FixedValidationLossModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if targets[0, 0].item() == 0:
            loss = self.weight * 0 + 1
        else:
            loss = (self.weight - 1) ** 2
        return torch.empty(0), loss


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


def test_train_model_stops_after_patience_and_restores_best_weights(capsys) -> None:
    def run(epochs: int, patience: int) -> tuple[FixedValidationLossModel, list[tuple[float, float]]]:
        model = FixedValidationLossModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        train_batches = [(torch.ones((1, 1), dtype=torch.long), torch.ones((1, 1), dtype=torch.long))]
        validation_batches = [
            (torch.zeros((1, 1), dtype=torch.long), torch.zeros((1, 1), dtype=torch.long))
        ]
        losses = train_model(
            model,
            train_batches,
            optimizer,
            epochs=epochs,
            validation_batches=validation_batches,
            patience=patience,
        )
        return model, losses

    one_epoch_model, _ = run(epochs=1, patience=1)
    stopped_model, losses = run(epochs=5, patience=1)

    assert len(losses) == 2
    assert torch.equal(stopped_model.weight, one_epoch_model.weight)
    assert "Early stopping after 2 epoch(s)" in capsys.readouterr().out
