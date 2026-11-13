"""Provide deterministic data and fresh training state without writing reports."""

import pytest
import torch

from bikes_data import TARGET, make_rentals
from model import BikeNet
from prepare import (
    FEATURES,
    SEED,
    BikeDataset,
    make_loaders,
    make_preprocessor,
    make_splits,
)
from train import BATCH_SIZE, HIDDEN_SIZES, LEARNING_RATE


@pytest.fixture(autouse=True)
def isolated_torch_seed():
    """Restore the caller's CPU random state after each seeded test."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        yield


@pytest.fixture(scope="session")
def rentals_frame():
    """Generate the unchanged professor data once for read-only test use."""
    return make_rentals(n=6000, seed=SEED)


@pytest.fixture(scope="session")
def prepared_datasets(rentals_frame):
    """Prepare all split datasets; test rows are never passed to a model."""
    rows = make_splits(rentals_frame, seed=SEED)
    preprocessor = make_preprocessor(rows[0][FEATURES])
    return tuple(
        BikeDataset(preprocessor.transform(split[FEATURES]), split[TARGET].to_numpy())
        for split in rows
    )


@pytest.fixture
def training_case(prepared_datasets):
    """Return a fresh network, Adam optimizer and the verified training batch."""
    loader, _, _ = make_loaders(*prepared_datasets, batch_size=BATCH_SIZE, seed=SEED)
    features, targets = next(iter(loader))
    model = BikeNet(features.shape[1], hidden_sizes=HIDDEN_SIZES)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return model, optimizer, features, targets
