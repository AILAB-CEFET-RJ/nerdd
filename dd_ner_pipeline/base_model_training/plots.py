from pathlib import Path

import matplotlib.pyplot as plt


def save_loss_plot(training_losses, validation_losses, output_dir, fold, trial, lr, weight_decay):
    """Persist train/validation loss curve for a trial."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, max(len(training_losses), len(validation_losses)) + 1))

    plt.figure(figsize=(10, 6))
    if training_losses:
        plt.plot(epochs[: len(training_losses)], training_losses, marker="o", label="Train Loss")
    if validation_losses:
        plt.plot(epochs[: len(validation_losses)], validation_losses, marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Fold {fold} - Trial {trial} - LR={lr:.7f}, WD={weight_decay:.6f}")
    if epochs:
        plt.xticks(epochs)
        plt.xlim(0.8, epochs[-1] + 0.2)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path / f"loss_fold{fold}_trial{trial}.png")
    plt.close()
