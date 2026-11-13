"""Split bike rental rows and prepare reproducible PyTorch data loaders."""

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

SEED = 2026
DEFAULT_BATCH_SIZE = 64
TRAIN_FRACTION = 0.6
NUMERIC_FEATURES = [
    "hour",
    "temperature_c",
    "rain_mm",
    "wind_kph",
    "is_weekend",
]
CATEGORICAL_FEATURES = ["station_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_splits(
    frame: pd.DataFrame, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return raw 60/20/20 splits, retaining original row indices.

    Split row positions before any preprocessing. Equal feature values do
    not identify duplicate observations and are never removed here.
    """
    positions = np.arange(len(frame))
    train_positions, remaining_positions = train_test_split(
        positions, train_size=TRAIN_FRACTION, random_state=seed
    )
    validation_positions, test_positions = train_test_split(
        remaining_positions, test_size=0.5, random_state=seed
    )
    return (
        frame.iloc[train_positions].copy(),
        frame.iloc[validation_positions].copy(),
        frame.iloc[test_positions].copy(),
    )


def make_preprocessor(training_features: pd.DataFrame) -> ColumnTransformer:
    """Fit and return a transformer using only the supplied training features.

    Standardize all five numeric inputs, including is_weekend. Learn station
    categories from training only; unseen categories transform to all zeros.
    Call transform, never fit, on validation and test rows.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    preprocessor.fit(training_features[FEATURES])
    return preprocessor


class BikeDataset(Dataset):
    """Store prepared features and unscaled rental targets as float32 tensors.

    Each item returns features of shape (feature_count,) and a target of
    shape (1,). A batch therefore has shapes (batch_size, feature_count)
    and (batch_size, 1), matching a network with one output column.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Convert aligned rows to tensors after checking their shapes."""
        feature_array = np.asarray(features, dtype=np.float32)
        target_array = np.asarray(targets, dtype=np.float32)
        if feature_array.ndim != 2:
            raise ValueError("features must have shape (N, feature_count)")
        if target_array.ndim == 1:
            target_array = target_array.reshape(-1, 1)
        if target_array.ndim != 2 or target_array.shape[1] != 1:
            raise ValueError("targets must have shape (N,) or (N, 1)")
        if len(feature_array) != len(target_array):
            raise ValueError("features and targets must have the same row count")
        self.features = torch.from_numpy(np.ascontiguousarray(feature_array))
        self.targets = torch.from_numpy(np.ascontiguousarray(target_array))

    def __len__(self) -> int:
        """Return the number of observations."""
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one aligned feature row and one rental target."""
        return self.features[index], self.targets[index]


def make_loaders(
    train_dataset: BikeDataset,
    validation_dataset: BikeDataset,
    test_dataset: BikeDataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = SEED,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return loaders that shuffle training only and retain every observation.

    New loaders with the same seed reproduce training order. The training
    generator advances between epochs, allowing a new shuffle each epoch.
    Zero workers keeps iteration straightforward on Windows.
    """
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train_loader, validation_loader, test_loader
