"""Check small-batch learning and train a bike rental network by epoch."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
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
PROJECT_EPOCHS = 60
REPORTS_DIRECTORY = Path(__file__).resolve().parent / "reports"
RUN_LEARNING_RATES = {"main": LEARNING_RATE, "low_lr": 0.0001, "high_lr": 0.1}


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


def fit_baselines(
    training_features: np.ndarray, training_targets: np.ndarray
) -> tuple[float, LinearRegression]:
    """Fit a target mean and linear regression using training rows only.

    Use the existing training-fitted preprocessor's NumPy output. Targets
    contain unscaled rental counts and are aligned to one dimension.
    """
    training_targets = np.asarray(training_targets).reshape(-1)
    training_mean = float(training_targets.mean())
    linear_model = LinearRegression()
    linear_model.fit(training_features, training_targets)
    return training_mean, linear_model


def evaluate_baselines(
    training_mean: float,
    linear_model: LinearRegression,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
) -> dict[str, float]:
    """Return validation MAE in rentals for both fitted baselines.

    Align predictions and targets to one dimension without rounding or
    clipping predictions. This function never fits either baseline.
    """
    validation_targets = np.asarray(validation_targets).reshape(-1)
    mean_predictions = np.full(len(validation_features), training_mean)
    linear_predictions = linear_model.predict(validation_features).reshape(-1)
    return {
        "training_mean_val_mae": float(
            mean_absolute_error(validation_targets, mean_predictions)
        ),
        "linear_regression_val_mae": float(
            mean_absolute_error(validation_targets, linear_predictions)
        ),
    }


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


def run_experiment(
    run_id: str,
    learning_rate: float,
    datasets: tuple[BikeDataset, BikeDataset, BikeDataset],
) -> dict:
    """Train one fresh seeded network and retain its restored best weights.

    Runs share prepared datasets and differ only in learning rate. Create
    independent loaders each time and never iterate over the test loader.
    """
    torch.manual_seed(SEED)
    training_loader, validation_loader, _test_loader = make_loaders(
        *datasets, batch_size=DEFAULT_BATCH_SIZE, seed=SEED
    )
    model = BikeNet(datasets[0].features.shape[1], hidden_sizes=HIDDEN_SIZES)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(
        f"Run {run_id}: epochs={PROJECT_EPOCHS}, hidden_sizes={HIDDEN_SIZES}, "
        f"batch_size={DEFAULT_BATCH_SIZE}, learning_rate={learning_rate}, "
        f"loss=MSE, optimizer=Adam, seed={SEED}"
    )
    history, best = train_model(
        model, training_loader, validation_loader, optimizer, PROJECT_EPOCHS
    )
    print(
        f"Run {run_id} best: epoch={best['best_epoch']}, "
        f"val_loss={best['best_val_loss']:.6f}, "
        f"val_mae={best['best_val_mae']:.6f}"
    )
    return {
        "run_id": run_id,
        "model": model,
        "best_weights": deepcopy(model.state_dict()),
        "test_loader": _test_loader,
        "history": history,
        "best": best,
        "hidden_sizes": HIDDEN_SIZES,
        "learning_rate": learning_rate,
        "batch_size": DEFAULT_BATCH_SIZE,
        "epochs": PROJECT_EPOCHS,
        "seed": SEED,
        "curve_file": (
            "learning_curves.png"
            if run_id == "main"
            else f"learning_curves_{run_id}.png"
        ),
    }


def save_learning_curves(
    history: list[dict[str, int | float]],
    best_epoch: int,
    output_path: Path,
    title: str = "Bike Rental Learning Curves",
) -> Path:
    """Save every epoch's MSE curves and mark the lowest-validation-loss epoch."""
    if not history:
        raise ValueError("history must contain at least one epoch")
    best_row = min(history, key=lambda row: row["val_loss"])
    if best_epoch != best_row["epoch"]:
        raise ValueError("best_epoch must match the lowest validation loss epoch")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 6), layout="constrained")
    try:
        epochs = [row["epoch"] for row in history]
        axis.plot(epochs, [row["train_loss"] for row in history], label="Training Loss")
        axis.plot(epochs, [row["val_loss"] for row in history], label="Validation Loss")
        axis.axvline(
            best_epoch, color="black", linestyle="--", label=f"Best Epoch: {best_epoch}"
        )
        axis.scatter([best_epoch], [best_row["val_loss"]], color="black", zorder=3)
        axis.set(title=title, xlabel="Epoch", ylabel="Loss (MSE)")
        axis.legend()
        axis.grid(alpha=0.25)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)
    return output_path


def save_exploratory_plot(frame: pd.DataFrame, output_path: Path) -> Path:
    """Save the prescribed exploratory scatter plot using all generated rows."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 5.6), layout="constrained")
    try:
        axis.scatter(frame["hour"], frame[TARGET], s=12, alpha=0.25)
        axis.set(
            title="Bike Rentals by Hour (6,000 Observations)",
            xlabel="Hour of Day",
            ylabel="Bike Rentals",
            xticks=range(0, 24, 2),
        )
        axis.grid(alpha=0.25)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)
    return output_path


def save_reports(
    results: list[dict], selected_run: str, output_directory: Path
) -> None:
    """Write actual epoch histories and run summaries without any test metrics.

    best_val_mae belongs to the lowest-loss epoch; min_val_mae is the
    minimum MAE over all epochs. Final losses belong to the last epoch.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    history_rows = []
    experiment_rows = []
    for result in results:
        history = result["history"]
        history_rows.extend({"run_id": result["run_id"], **row} for row in history)
        experiment_rows.append(
            {
                "run_id": result["run_id"],
                "hidden_sizes": str(result["hidden_sizes"]),
                "learning_rate": result["learning_rate"],
                "batch_size": result["batch_size"],
                "epochs": result["epochs"],
                "seed": result["seed"],
                **result["best"],
                "min_val_mae": min(row["val_mae"] for row in history),
                "final_train_loss": history[-1]["train_loss"],
                "final_val_loss": history[-1]["val_loss"],
                "curve_file": result["curve_file"],
                "selected": result["run_id"] == selected_run,
            }
        )
    pd.DataFrame(history_rows).to_csv(output_directory / "history.csv", index=False)
    pd.DataFrame(experiment_rows).to_csv(
        output_directory / "experiments.csv", index=False
    )


def evaluate_test_models(
    training_mean: float,
    linear_model: LinearRegression,
    selected_model: nn.Module,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    test_loader: DataLoader,
) -> dict[str, float]:
    """Measure MAE for three fitted models on the same ordered held-out rows.

    Never fit or update a model. Iterate the neural network loader once,
    checking its feature/target alignment with the original NumPy arrays.
    Preserve continuous predictions, including any negative values.
    """
    test_features = np.asarray(test_features)
    test_targets = np.asarray(test_targets).reshape(-1)
    if test_features.ndim != 2 or len(test_features) != len(test_targets):
        raise ValueError("test features and targets must have aligned rows")
    if len(test_targets) == 0:
        raise ValueError("test data must contain at least one sample")
    selected_model.eval()
    prediction_batches = []
    sample_count = 0
    with torch.no_grad():
        for features, targets in test_loader:
            batch_end = sample_count + len(targets)
            expected_features = torch.as_tensor(
                test_features[sample_count:batch_end], dtype=features.dtype
            )
            expected_targets = torch.as_tensor(
                test_targets[sample_count:batch_end].copy(), dtype=targets.dtype
            ).reshape(-1, 1)
            if not torch.equal(features, expected_features) or not torch.equal(
                targets, expected_targets
            ):
                raise ValueError("test loader rows must match the ordered test arrays")
            predictions = selected_model(features)
            if predictions.shape != targets.shape:
                raise ValueError("prediction and target shapes must match")
            prediction_batches.append(predictions.reshape(-1).cpu())
            sample_count = batch_end
    if sample_count != len(test_targets):
        raise ValueError("test loader must contain every test row exactly once")
    neural_predictions = torch.cat(prediction_batches).numpy()
    mean_predictions = np.full(len(test_targets), training_mean)
    linear_predictions = linear_model.predict(test_features).reshape(-1)
    return {
        "training_mean": float(mean_absolute_error(test_targets, mean_predictions)),
        "linear_regression": float(
            mean_absolute_error(test_targets, linear_predictions)
        ),
        "neural_network": float(mean_absolute_error(test_targets, neural_predictions)),
    }


def save_metrics(metrics: dict, output_path: Path) -> Path:
    """Serialize already-computed metrics as finite, reproducible JSON."""
    serialized = json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized, encoding="utf-8")
    return output_path


def run_project() -> dict:
    """Generate all reports and evaluate the validation-selected model last.

    Prepare data once, fit baselines on training only, and select by best
    validation MSE. Only the restored selected network reaches test evaluation.
    """
    print("Project run: select using validation, then evaluate test rows last.")
    frame = make_rentals(n=6000, seed=SEED)
    training_rows, validation_rows, test_rows = make_splits(frame, seed=SEED)
    preprocessor = make_preprocessor(training_rows[FEATURES])
    feature_arrays = tuple(
        preprocessor.transform(rows[FEATURES])
        for rows in (training_rows, validation_rows, test_rows)
    )
    target_arrays = tuple(
        rows[TARGET].to_numpy() for rows in (training_rows, validation_rows, test_rows)
    )
    datasets = tuple(
        BikeDataset(features, targets)
        for features, targets in zip(feature_arrays, target_arrays, strict=True)
    )
    training_mean, linear_model = fit_baselines(feature_arrays[0], target_arrays[0])
    baseline_metrics = evaluate_baselines(
        training_mean, linear_model, feature_arrays[1], target_arrays[1]
    )
    print(f"Training target mean: {training_mean:.12f}")
    for name, value in baseline_metrics.items():
        print(f"{name}: {value:.12f} rentals")
    results = [
        run_experiment(run_id, learning_rate, datasets)
        for run_id, learning_rate in RUN_LEARNING_RATES.items()
    ]
    selected = min(results, key=lambda result: result["best"]["best_val_loss"])
    for result in results:
        save_learning_curves(
            result["history"],
            result["best"]["best_epoch"],
            REPORTS_DIRECTORY / result["curve_file"],
            title=f"Bike Rental Learning Curves: {result['run_id']}",
        )
    save_exploratory_plot(frame, REPORTS_DIRECTORY / "hour_vs_rentals.png")
    save_reports(results, selected["run_id"], REPORTS_DIRECTORY)
    print(
        f"Selected run: {selected['run_id']}, "
        f"best epoch: {selected['best']['best_epoch']}, "
        f"validation MSE: {selected['best']['best_val_loss']:.6f}, "
        f"validation MAE: {selected['best']['best_val_mae']:.6f}"
    )
    best_row = min(selected["history"], key=lambda row: row["val_loss"])
    if selected["best"]["best_epoch"] != best_row["epoch"]:
        raise RuntimeError("selected best epoch must minimize validation loss")
    if not all(
        torch.equal(value, selected["best_weights"][name])
        for name, value in selected["model"].state_dict().items()
    ):
        raise RuntimeError("selected model weights differ from restored best weights")
    test_metrics = evaluate_test_models(
        training_mean,
        linear_model,
        selected["model"],
        feature_arrays[2],
        target_arrays[2],
        selected["test_loader"],
    )
    metrics = {
        "seed": SEED,
        "split_sizes": {
            "training": len(training_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
        },
        "mae_unit": "bike rentals",
        "selection": {
            "criterion": "lowest_validation_mse",
            "selected_run": selected["run_id"],
            "best_epoch": selected["best"]["best_epoch"],
            "best_validation_mse": selected["best"]["best_val_loss"],
            "settings": {
                "hidden_sizes": list(selected["hidden_sizes"]),
                "learning_rate": selected["learning_rate"],
                "batch_size": selected["batch_size"],
                "epochs": selected["epochs"],
                "loss": "MSE",
                "optimizer": "Adam",
            },
        },
        "models": {
            "training_mean": {
                "validation_mae": baseline_metrics["training_mean_val_mae"],
                "test_mae": test_metrics["training_mean"],
            },
            "linear_regression": {
                "validation_mae": baseline_metrics["linear_regression_val_mae"],
                "test_mae": test_metrics["linear_regression"],
            },
            "neural_network": {
                "validation_mae": selected["best"]["best_val_mae"],
                "test_mae": test_metrics["neural_network"],
            },
        },
    }
    save_metrics(metrics, REPORTS_DIRECTORY / "metrics.json")
    for name, value in test_metrics.items():
        print(f"{name} test MAE: {value:.12f} rentals")
    return {
        "results": results,
        "selected_run": selected["run_id"],
        "selected_model": selected["model"],
        "baseline_metrics": baseline_metrics,
        "metrics": metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, choices=(4, 5))
    arguments = parser.parse_args()
    if arguments.step == 4:
        main()
    elif arguments.step == 5:
        run_training_checks()
    else:
        run_project()
