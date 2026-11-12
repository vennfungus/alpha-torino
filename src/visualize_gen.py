"""Plot training and validation loss curves from :func:`src.train.train_model`.

The training loop in ``src/train.py`` already returns one ``(train_loss,
validation_loss)`` pair per epoch. This module turns that history into a PNG
chart so overfitting (validation loss rising while training loss falls) or
underfitting (both losses flat and high) can be judged at a glance.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Render to files without needing a display or GUI loop.

import matplotlib.pyplot as plt


def plot_loss_curves(
    epoch_losses: list[tuple[float, float]],
    output_path: str | Path,
    title: str = "Training and validation loss",
) -> Path:
    """Save a line chart of per-epoch training and validation loss.

    ``epoch_losses`` is the list returned by ``train_model``: index 0 holds the
    training loss of epoch 1, index 1 that of epoch 2, and so on; each entry is
    a ``(train_loss, validation_loss)`` pair. The plot is written to
    ``output_path`` (a ``.png`` is recommended) and the path is returned.
    """
    if not epoch_losses:
        raise ValueError("epoch_losses must contain at least one epoch")

    epochs = range(1, len(epoch_losses) + 1)
    train_losses = [train for train, _ in epoch_losses]
    validation_losses = [validation for _, validation in epoch_losses]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, train_losses, marker="o", label="train loss")
    axis.plot(epochs, validation_losses, marker="s", label="validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_title(title)
    axis.set_xticks(list(epochs))
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
