import pytest
import torch

from src.model import CausalLanguageModel
from src.train import train_model
from src.visualize import plot_loss_curves


def test_plot_loss_curves_saves_a_png(tmp_path) -> None:
    losses = [(1.0, 1.2), (0.8, 0.9), (0.6, 0.7)]

    plot_path = plot_loss_curves(losses, tmp_path / "loss_curve.png")

    assert plot_path.exists()
    assert plot_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_loss_curves_rejects_empty_history(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least one epoch"):
        plot_loss_curves([], tmp_path / "loss_curve.png")


def test_train_model_returns_history_for_plotting() -> None:
    model = CausalLanguageModel(
        vocab_size=16, block_size=4, d_model=8, n_heads=2, n_layers=1
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    batches = [(torch.tensor([[1, 2, 3, 4]]), torch.tensor([[2, 3, 4, 5]]))]

    losses = train_model(
        model,
        batches,
        optimizer,
        epochs=2,
        validation_batches=batches,
    )

    assert len(losses) == 2
    assert all(torch.isfinite(torch.tensor(pair)).all() for pair in losses)
