"""Define the neural network for bike rental count regression."""

import torch
from torch import nn


class BikeNet(nn.Module):
    """Predict one rental count per row using ReLU hidden layers.

    Derive input_dim from the prepared training features. Outputs have shape
    (batch_size, 1) and no final activation, matching BikeDataset targets.
    """

    def __init__(
        self, input_dim: int, hidden_sizes: tuple[int, ...] = (32, 16)
    ) -> None:
        """Build at least two positive-width hidden layers and one output.

        Raise ValueError for invalid dimensions or fewer than two hidden
        layers. Boolean values are not accepted as integer dimensions.
        """
        super().__init__()
        if type(input_dim) is not int or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if not isinstance(hidden_sizes, (tuple, list)) or len(hidden_sizes) < 2:
            raise ValueError("hidden_sizes must contain at least two layer sizes")
        if any(type(size) is not int or size <= 0 for size in hidden_sizes):
            raise ValueError("each hidden layer size must be a positive integer")

        layers = []
        previous_size = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.ReLU())
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return predictions of shape (batch_size, 1) for a feature batch."""
        return self.network(features)
