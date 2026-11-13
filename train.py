"""Check small-batch learning and train a bike rental network by epoch."""

import argparse
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader

from bikes_data import TARGET, make_rentals
from model import BikeNet
from prepare import (
    DEFAULT_BATCH_SIZE,
    FEATURES,
    SEED,
    BikeDataset,
    make_loaders,
    make_preprocessor,
    make_splits,
)

BATCH_SIZE = 16
HIDDEN_SIZES = (32, 16)
LEARNING_RATE = 0.01
TRAINING_STEPS = 500
MAX_LOSS_RATIO = 0.01
MAX_FINAL_MSE = 1.0
SHORT_EPOCHS = 5
LONG_EPOCHS = 30


def fit_one_batch(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    steps: int = TRAINING_STEPS,
) -> list[float]:
    """Fit a fixed batch and return its initial and post-update losses.

    Entry zero is measured before training; entry k is measured after update
    k. Reject prediction/target shape mismatches before computing any loss,
    and fail immediately if a measured loss is not finite.
    """
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    model.train()
    with torch.no_grad():
        predictions = model(features)
        if predictions.shape != targets.shape:
            raise ValueError("prediction and target shapes must match")
        initial_loss = loss_function(predictions, targets)
    if not torch.isfinite(initial_loss).item():
        raise ValueError("loss must be finite")
    losses = [initial_loss.item()]

    for _ in range(steps):
        optimizer.zero_grad()
        predictions = model(features)
        if predictions.shape != targets.shape:
            raise ValueError("prediction and target shapes must match")
        loss = loss_function(predictions, targets)
        if not torch.isfinite(loss).item():
            raise ValueError("loss must be finite")
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predictions = model(features)
            if predictions.shape != targets.shape:
                raise ValueError("prediction and target shapes must match")
            updated_loss = loss_function(predictions, targets)
        if not torch.isfinite(updated_loss).item():
            raise ValueError("loss must be finite")
        losses.append(updated_loss.item())

    return losses


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Return sample-weighted mean MSE and MAE for one complete epoch.

    An optimizer enables training; otherwise use evaluation mode and disable
    gradients. Predictions and rental targets must both have shape (N, 1).
    Training metrics describe predictions before each batch's update.
    """
    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()
    loss_function = nn.MSELoss()
    total_squared_error = 0.0
    total_absolute_error = 0.0
    sample_count = 0
    gradient_context = torch.enable_grad() if is_training else torch.no_grad()
    with gradient_context:
        for features, targets in loader:
            if optimizer is not None:
                optimizer.zero_grad()
            predictions = model(features)
            if targets.ndim != 2 or targets.shape[1] != 1:
                raise ValueError("targets must have shape (N, 1)")
            if predictions.shape != targets.shape:
                raise ValueError("prediction and target shapes must match")
            loss = loss_function(predictions, targets)
            if not torch.isfinite(loss).item():
                raise ValueError("loss must be finite")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch_size = len(targets)
            total_squared_error += loss.item() * batch_size
            total_absolute_error += (predictions.detach() - targets).abs().sum().item()
            sample_count += batch_size
    if sample_count == 0:
        raise ValueError("loader must contain at least one sample")
    return (
        total_squared_error / sample_count,
        total_absolute_error / sample_count,
    )


def train_model(
    model: nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> tuple[list[dict[str, int | float]], dict[str, int | float]]:
    """Log each epoch, restore the lowest-validation-MSE weights, and report it.

    Epoch numbers start at one. best_val_mae is the MAE at the selected
    lowest-loss epoch, rather than the minimum MAE across all epochs.
    The saved state is an independent deep copy, not a live weight reference.
    """
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    history = []
    best_information = {
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "best_val_mae": float("inf"),
    }
    best_weights = deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        training_loss, _ = run_epoch(model, training_loader, optimizer)
        validation_loss, validation_mae = run_epoch(model, validation_loader)
        history.append(
            {
                "epoch": epoch,
                "train_loss": training_loss,
                "val_loss": validation_loss,
                "val_mae": validation_mae,
            }
        )
        if validation_loss < best_information["best_val_loss"]:
            best_weights = deepcopy(model.state_dict())
            best_information = {
                "best_epoch": epoch,
                "best_val_loss": validation_loss,
                "best_val_mae": validation_mae,
            }
        print(
            f"Epoch {epoch:02d}: train_loss={training_loss:.6f}, "
            f"val_loss={validation_loss:.6f}, val_mae={validation_mae:.6f}"
        )
    model.load_state_dict(best_weights)
    return history, best_information


def run_training_checks() -> None:
    """Run fresh short and longer training checks without using the test loader."""
    frame = make_rentals(n=6000, seed=SEED)
    training_rows, validation_rows, test_rows = make_splits(frame, seed=SEED)
    preprocessor = make_preprocessor(training_rows[FEATURES])
    datasets = tuple(
        BikeDataset(preprocessor.transform(rows[FEATURES]), rows[TARGET].to_numpy())
        for rows in (training_rows, validation_rows, test_rows)
    )
    for run_name, epochs in (("Short", SHORT_EPOCHS), ("Longer", LONG_EPOCHS)):
        torch.manual_seed(SEED)
        training_loader, validation_loader, _test_loader = make_loaders(
            *datasets, batch_size=DEFAULT_BATCH_SIZE, seed=SEED
        )
        model = BikeNet(datasets[0].features.shape[1], hidden_sizes=HIDDEN_SIZES)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        print(f"{run_name} run: {epochs} epochs, batch_size={DEFAULT_BATCH_SIZE}")
        _, best = train_model(
            model, training_loader, validation_loader, optimizer, epochs
        )
        print(
            f"Best epoch: {best['best_epoch']}, "
            f"validation MSE: {best['best_val_loss']:.6f}, "
            f"validation MAE: {best['best_val_mae']:.6f}"
        )


def main() -> None:
    """Prepare training-only inputs and report the small-batch learning check."""
    torch.manual_seed(SEED)
    frame = make_rentals(n=6000, seed=SEED)
    training_rows, _, _ = make_splits(frame, seed=SEED)
    preprocessor = make_preprocessor(training_rows[FEATURES])
    training_features = preprocessor.transform(training_rows[FEATURES])
    dataset = BikeDataset(training_features, training_rows[TARGET].to_numpy())
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
        drop_last=False,
    )
    features, targets = next(iter(loader))
    model = BikeNet(training_features.shape[1], hidden_sizes=HIDDEN_SIZES)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    losses = fit_one_batch(model, features, targets, loss_function, optimizer)

    initial_loss, final_loss = losses[0], losses[-1]
    loss_ratio = final_loss / initial_loss if initial_loss > 0 else 0.0
    print(f"Batch features: {tuple(features.shape)}, {features.dtype}")
    print(f"Batch targets: {tuple(targets.shape)}, {targets.dtype}")
    print(f"Initial MSE: {initial_loss:.10f}")
    for step in range(100, len(losses), 100):
        print(f"Step {step} MSE: {losses[step]:.10f}")
    print(f"Final MSE: {final_loss:.10f}")
    print(f"Final/initial ratio: {loss_ratio:.10f}")
    print(f"Completed steps: {len(losses) - 1}")
    if final_loss > initial_loss * MAX_LOSS_RATIO or final_loss > MAX_FINAL_MSE:
        raise RuntimeError("small-batch loss reduction targets were not met")
    print("Small-batch learning check: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, choices=(4, 5), default=4)
    arguments = parser.parse_args()
    if arguments.step == 4:
        main()
    else:
        run_training_checks()
