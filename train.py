"""Verify a hand-written training loop by memorizing one training batch."""

import torch
from torch import nn
from torch.utils.data import DataLoader

from bikes_data import TARGET, make_rentals
from model import BikeNet
from prepare import FEATURES, SEED, BikeDataset, make_preprocessor, make_splits

BATCH_SIZE = 16
HIDDEN_SIZES = (32, 16)
LEARNING_RATE = 0.01
TRAINING_STEPS = 500
MAX_LOSS_RATIO = 0.01
MAX_FINAL_MSE = 1.0


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
    main()
