"""Check configurable network outputs and existing configuration errors."""

import re

import pytest
import torch

from model import BikeNet


@pytest.mark.parametrize("hidden_sizes", [(32, 16), (16, 8), (64, 32, 16)])
@pytest.mark.parametrize("batch_size", [1, 7])
def test_bikenet_output_shape(prepared_datasets, hidden_sizes, batch_size):
    """Return one float32 value per observation, including single-row batches."""
    features = prepared_datasets[0].features[:batch_size]
    model = BikeNet(features.shape[1], hidden_sizes=hidden_sizes)
    predictions = model(features)
    assert predictions.shape == (batch_size, 1)
    assert predictions.dtype == torch.float32


@pytest.mark.parametrize(
    "input_dim,hidden_sizes,message",
    [
        (0, (32, 16), "input_dim must be a positive integer"),
        (True, (32, 16), "input_dim must be a positive integer"),
        (8, (16,), "hidden_sizes must contain at least two layer sizes"),
        (8, (32, 0), "each hidden layer size must be a positive integer"),
        (8, (32, -1), "each hidden layer size must be a positive integer"),
        (8, (32, True), "each hidden layer size must be a positive integer"),
    ],
)
def test_bikenet_rejects_invalid_configuration(input_dim, hidden_sizes, message):
    """Reject unsupported dimensions and match the actual error messages."""
    with pytest.raises(ValueError, match=re.escape(message)):
        BikeNet(input_dim, hidden_sizes=hidden_sizes)
