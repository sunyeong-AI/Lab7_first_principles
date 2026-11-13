"""Test row identity, training-only preparation and repeatable batch ordering."""

import re

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from prepare import (
    NUMERIC_FEATURES,
    BikeDataset,
    make_loaders,
    make_preprocessor,
    make_splits,
)


@pytest.mark.parametrize("row_count,feature_count", [(1, 3), (7, 8)])
@pytest.mark.parametrize("column_targets", [False, True])
def test_dataset_returns_float32_rows(row_count, feature_count, column_targets):
    """Accept vector/column labels and retain row, batch, dtype and value alignment."""
    features = np.arange(row_count * feature_count).reshape(row_count, feature_count)
    targets = np.arange(row_count)
    dataset = BikeDataset(features, targets[:, None] if column_targets else targets)
    assert len(dataset) == row_count
    item_features, item_target = dataset[0]
    assert item_features.shape == (feature_count,)
    assert item_target.shape == (1,)
    assert item_features.dtype == item_target.dtype == torch.float32
    batch_features, batch_targets = next(
        iter(DataLoader(dataset, batch_size=row_count))
    )
    assert batch_features.shape == (row_count, feature_count)
    assert batch_targets.shape == (row_count, 1)
    assert batch_features.dtype == batch_targets.dtype == torch.float32
    torch.testing.assert_close(
        batch_features, torch.tensor(features, dtype=torch.float32)
    )
    torch.testing.assert_close(batch_targets[:, 0], torch.tensor(targets).float())


@pytest.mark.parametrize("seed", [2026, 17])
def test_splits_partition_original_rows(rentals_frame, seed):
    """Retain every original observation exactly once and reproduce its split."""
    splits = make_splits(rentals_frame, seed=seed)
    repeated = make_splits(rentals_frame, seed=seed)
    assert [len(split) for split in splits] == [3600, 1200, 1200]
    indices = [set(split.index) for split in splits]
    assert indices[0] | indices[1] | indices[2] == set(rentals_frame.index)
    assert not indices[0] & indices[1]
    assert not indices[0] & indices[2]
    assert not indices[1] & indices[2]
    for split, again in zip(splits, repeated, strict=True):
        pd.testing.assert_frame_equal(split, rentals_frame.loc[split.index])
        pd.testing.assert_frame_equal(split, again)


def test_preprocessor_fits_training_only():
    """Detect contaminated scaler statistics and validation-only encoder categories."""
    training = pd.DataFrame(
        {
            name: np.arange(6, dtype=float) + offset
            for offset, name in enumerate(NUMERIC_FEATURES)
        }
    )
    training["is_weekend"] = [0, 1, 0, 1, 0, 1]
    training["station_type"] = ["downtown", "residential"] * 3
    validation = training.iloc[:2].copy()
    validation[NUMERIC_FEATURES] += 10
    validation["is_weekend"] = 1
    validation["station_type"] = "park"
    preprocessor = make_preprocessor(training)
    scaler = preprocessor.named_transformers_["numeric"]
    encoder = preprocessor.named_transformers_["categorical"]
    assert scaler.n_samples_seen_ == len(training)
    np.testing.assert_allclose(scaler.mean_, training[NUMERIC_FEATURES].mean())
    np.testing.assert_allclose(scaler.var_, training[NUMERIC_FEATURES].var(ddof=0))
    assert encoder.categories_[0].tolist() == ["downtown", "residential"]
    original_mean, original_variance = scaler.mean_.copy(), scaler.var_.copy()
    original_categories = encoder.categories_[0].copy()
    train_values = preprocessor.transform(training)
    validation_values = preprocessor.transform(validation)
    numeric_count = len(NUMERIC_FEATURES)
    np.testing.assert_allclose(
        train_values[:, :numeric_count].mean(axis=0), 0, atol=1e-12
    )
    assert np.all(np.abs(validation_values[:, :numeric_count].mean(axis=0)) > 0.1)
    np.testing.assert_array_equal(validation_values[:, numeric_count:], 0)
    np.testing.assert_array_equal(scaler.mean_, original_mean)
    np.testing.assert_array_equal(scaler.var_, original_variance)
    np.testing.assert_array_equal(encoder.categories_[0], original_categories)


@pytest.mark.parametrize("batch_size", [4, 7])
@pytest.mark.parametrize("seed", [2026, 17])
def test_loaders_shuffle_repeatably_and_preserve_eval_order(batch_size, seed):
    """Inspect actual row orders and final partial batches without prediction."""
    datasets = tuple(
        BikeDataset(np.arange(size).reshape(-1, 1), np.arange(size))
        for size in (17, 7, 5)
    )
    loaders = make_loaders(*datasets, batch_size=batch_size, seed=seed)
    repeated = make_loaders(*datasets, batch_size=batch_size, seed=seed)
    orders = [
        torch.cat([features[:, 0] for features, _ in loader]) for loader in loaders
    ]
    repeated_order = torch.cat([features[:, 0] for features, _ in repeated[0]])
    assert torch.equal(orders[0], repeated_order)
    assert not torch.equal(orders[0], torch.arange(17).float())
    assert torch.equal(orders[0].sort().values, torch.arange(17).float())
    assert torch.equal(orders[1], torch.arange(7).float())
    assert torch.equal(orders[2], torch.arange(5).float())


@pytest.mark.parametrize(
    "features,targets,message",
    [
        (np.zeros(3), np.zeros(3), "features must have shape (N, feature_count)"),
        (np.zeros((3, 2)), np.zeros((3, 2)), "targets must have shape (N,) or (N, 1)"),
        (
            np.zeros((3, 2)),
            np.zeros(2),
            "features and targets must have the same row count",
        ),
    ],
)
def test_dataset_rejects_invalid_shapes(features, targets, message):
    """Match existing validation errors instead of adding new production checks."""
    with pytest.raises(ValueError, match=re.escape(message)):
        BikeDataset(features, targets)
