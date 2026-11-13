"""Verify learning, fault detection, sample metrics and independent best weights."""

import re
from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import train


def _run_two_updates(training_case, training_function):
    """Exercise the two public training paths with the same fixed batch twice."""
    model, optimizer, features, targets = training_case
    if training_function == "fit_one_batch":
        train.fit_one_batch(model, features, targets, nn.MSELoss(), optimizer, steps=2)
    else:
        dataset = TensorDataset(features.repeat(2, 1), targets.repeat(2, 1))
        train.run_epoch(model, DataLoader(dataset, batch_size=len(features)), optimizer)


def _assert_all_linear_weights_changed(model, original):
    """Require an actual update in every Linear layer, not just a step call."""
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            assert not torch.equal(layer.weight, original[name]), (
                f"Linear weight did not change: {name}"
            )


def test_optimizer_step_changes_every_linear_weight(training_case):
    """One real epoch batch must update every Linear weight tensor."""
    model, optimizer, features, targets = training_case
    original = {
        name: layer.weight.detach().clone()
        for name, layer in model.named_modules()
        if isinstance(layer, nn.Linear)
    }
    loader = DataLoader(TensorDataset(features, targets), batch_size=len(features))
    train.run_epoch(model, loader, optimizer)
    _assert_all_linear_weights_changed(model, original)


def test_fit_one_batch_memorizes_training_batch(training_case):
    """Meet the existing 500-step learning criteria on actual training rows."""
    model, optimizer, features, targets = training_case
    losses = train.fit_one_batch(model, features, targets, nn.MSELoss(), optimizer)
    assert len(losses) == train.TRAINING_STEPS + 1
    assert torch.isfinite(torch.tensor(losses)).all()
    assert losses[-1] <= losses[0] * train.MAX_LOSS_RATIO
    assert losses[-1] <= train.MAX_FINAL_MSE


@pytest.mark.parametrize("training_function", ["fit_one_batch", "run_epoch"])
def test_training_clears_existing_gradients(training_case, training_function):
    """Check cleared gradients immediately before each differentiable forward."""
    model, _, _, _ = training_case
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 123.0)
    observed = []

    def check_gradients(module, _inputs):
        if torch.is_grad_enabled():
            assert all(
                parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
                for parameter in module.parameters()
            ), "stale gradients before training forward"
            observed.append(True)

    handle = model.register_forward_pre_hook(check_gradients)
    try:
        _run_two_updates(training_case, training_function)
    finally:
        handle.remove()
    assert len(observed) == 2


@pytest.mark.parametrize("training_function", ["fit_one_batch", "run_epoch"])
@pytest.mark.parametrize("missing_operation", ["zero_grad", "step"])
def test_training_checks_detect_missing_optimizer_operations(
    training_case, training_function, missing_operation, monkeypatch
):
    """Fault injection must fail the intended gradient or weight assertion."""
    model, optimizer, _, _ = training_case
    monkeypatch.setattr(optimizer, missing_operation, lambda: None)
    if missing_operation == "zero_grad":
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 123.0)

        def reject_stale_gradients(module, _inputs):
            if torch.is_grad_enabled():
                assert all(
                    parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
                    for parameter in module.parameters()
                ), "stale gradients before training forward"

        handle = model.register_forward_pre_hook(reject_stale_gradients)
        try:
            with pytest.raises(
                AssertionError, match="stale gradients before training forward"
            ):
                _run_two_updates(training_case, training_function)
        finally:
            handle.remove()
    else:
        original = {
            name: layer.weight.detach().clone()
            for name, layer in model.named_modules()
            if isinstance(layer, nn.Linear)
        }
        _run_two_updates(training_case, training_function)
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Linear):
                assert torch.equal(layer.weight, original[name])
        with pytest.raises(AssertionError, match="Linear weight did not change"):
            _assert_all_linear_weights_changed(model, original)


def test_run_epoch_uses_sample_weighted_metrics():
    """A final singleton batch must contribute one sample, not one full batch."""
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    features = torch.zeros(5, 2)
    targets = torch.tensor([0.0, 1.0, 2.0, 3.0, 10.0]).reshape(-1, 1)
    loader = DataLoader(TensorDataset(features, targets), batch_size=2)
    mse, mae = train.run_epoch(model, loader)
    assert mse == pytest.approx(22.8)
    assert mae == pytest.approx(3.2)


def test_validation_epoch_preserves_weights(training_case, prepared_datasets):
    """Evaluation disables gradients, switches mode and preserves all parameters."""
    model, _, _, _ = training_case
    model.train()
    original = deepcopy(model.state_dict())
    validation = prepared_datasets[1]
    loader = DataLoader(
        TensorDataset(validation.features[:17], validation.targets[:17]), batch_size=8
    )
    observed = []

    def check_evaluation(module, _inputs):
        assert not module.training
        assert not torch.is_grad_enabled()
        observed.append(True)

    handle = model.register_forward_pre_hook(check_evaluation)
    try:
        train.run_epoch(model, loader)
    finally:
        handle.remove()
    assert len(observed) == 3
    assert all(
        torch.equal(value, original[key]) for key, value in model.state_dict().items()
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_train_model_returns_complete_history(training_case, prepared_datasets):
    """Real short training returns finite metrics and consistent best information."""
    model, optimizer, features, targets = training_case
    training_loader = DataLoader(
        TensorDataset(features, targets), batch_size=len(features)
    )
    validation = prepared_datasets[1]
    validation_loader = DataLoader(
        TensorDataset(validation.features[:17], validation.targets[:17]), batch_size=8
    )
    history, best = train.train_model(
        model, training_loader, validation_loader, optimizer, 3
    )
    assert len(history) == 3
    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert all(
        set(row) == {"epoch", "train_loss", "val_loss", "val_mae"} for row in history
    )
    assert all(
        torch.isfinite(torch.tensor(list(row.values()))).all() for row in history
    )
    best_row = min(history, key=lambda row: row["val_loss"])
    assert best == {
        "best_epoch": best_row["epoch"],
        "best_val_loss": best_row["val_loss"],
        "best_val_mae": best_row["val_mae"],
    }
    restored_loss, restored_mae = train.run_epoch(model, validation_loader)
    assert restored_loss == best["best_val_loss"]
    assert restored_mae == best["best_val_mae"]


def test_train_model_restores_independent_best_weights(training_case, monkeypatch):
    """Select the middle minimum-MSE epoch, not the last or minimum-MAE epoch."""
    model, optimizer, features, targets = training_case
    loader = DataLoader(TensorDataset(features, targets), batch_size=len(features))
    epoch = 0
    snapshots = []
    validation_losses = [3.0, 1.0, 2.0]
    validation_maes = [0.7, 0.8, 0.1]

    def controlled_epoch(network, _loader, optimizer=None):
        nonlocal epoch
        if optimizer is not None:
            epoch += 1
            with torch.no_grad():
                for parameter in network.parameters():
                    parameter.fill_(float(epoch))
            return 10.0 - epoch, 1.0
        snapshots.append(deepcopy(network.state_dict()))
        return validation_losses[epoch - 1], validation_maes[epoch - 1]

    monkeypatch.setattr(train, "run_epoch", controlled_epoch)
    history, best = train.train_model(model, loader, loader, optimizer, 3)
    assert best == {"best_epoch": 2, "best_val_loss": 1.0, "best_val_mae": 0.8}
    assert min(history, key=lambda row: row["val_mae"])["epoch"] == 3
    for name, value in model.state_dict().items():
        assert torch.equal(value, snapshots[1][name])
        assert not torch.equal(value, snapshots[2][name])


@pytest.mark.parametrize(
    "training_function,mismatch",
    [("fit_one_batch", "target"), ("run_epoch", "target"), ("run_epoch", "prediction")],
)
def test_training_rejects_prediction_target_shape_mismatch(
    training_case, training_function, mismatch, monkeypatch
):
    """Reject broadcasting or wrong output columns before any optimizer update."""
    model, optimizer, features, targets = training_case
    if mismatch == "target":
        targets = targets[:, 0]
    else:
        model = nn.Linear(features.shape[1], 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=train.LEARNING_RATE)

    def unexpected_step():
        raise AssertionError("optimizer step reached despite invalid shapes")

    monkeypatch.setattr(optimizer, "step", unexpected_step)
    message = (
        "targets must have shape (N, 1)"
        if training_function == "run_epoch" and mismatch == "target"
        else "prediction and target shapes must match"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        if training_function == "fit_one_batch":
            train.fit_one_batch(
                model, features, targets, nn.MSELoss(), optimizer, steps=1
            )
        else:
            loader = DataLoader(
                TensorDataset(features, targets), batch_size=len(features)
            )
            train.run_epoch(model, loader, optimizer)
