import logging

import torch
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    """Run one training epoch and return mean training loss."""
    model.train()
    running_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch} (Train)", leave=False):
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        running_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    return running_loss / len(train_loader)


def validate_one_epoch(model, val_loader, device, epoch):
    """Run one validation epoch and return mean validation loss."""
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch} (Val)", leave=False):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            val_loss += outputs.loss.item()

    return val_loss / len(val_loader)


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    patience,
    metric_fn,
    stage_label=None,
):
    """Train model with early stopping based on metric_fn output."""
    training_losses = []
    validation_losses = []
    best_metric = -1.0
    best_epoch = None
    best_state_dict = None
    patience_counter = 0
    stage_prefix = f"[{stage_label}] " if stage_label else ""

    for epoch_idx in range(num_epochs):
        epoch = epoch_idx + 1
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_loss = validate_one_epoch(model, val_loader, device, epoch)
        metric = metric_fn(model)

        training_losses.append(train_loss)
        validation_losses.append(val_loss)

        improved = metric > best_metric
        if improved:
            best_metric = metric
            best_epoch = epoch
            best_state_dict = clone_model_state(model)
            patience_counter = 0
        else:
            patience_counter += 1

        LOGGER.info(
            "%sEpoch %s: Train Loss=%.4f | Val Loss=%.4f | F1-macro (Val)=%.4f | Patience=%s/%s",
            stage_prefix,
            epoch,
            train_loss,
            val_loss,
            metric,
            patience_counter,
            patience,
        )

        if not improved:
            if patience_counter >= patience:
                LOGGER.info(
                    "%sEarly stopping triggered at epoch %s based on F1 score.",
                    stage_prefix,
                    epoch,
                )
                break

    if best_state_dict is not None:
        restore_model_state(model, best_state_dict, device)

    return {
        "training_losses": training_losses,
        "validation_losses": validation_losses,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
    }


def clone_model_state(model):
    """Clone model state to CPU tensors for lightweight checkpointing."""
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def restore_model_state(model, state_dict, device):
    """Load a cloned state dict into a model on target device."""
    model.load_state_dict(state_dict)
    model.to(device)
    return model
