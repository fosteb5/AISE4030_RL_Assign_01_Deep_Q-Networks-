"""
Double Dueling DQN network architecture.
"""

from typing import Tuple

import torch
import torch.nn as nn


class D3QNNetwork(nn.Module):
    """
    Double Dueling DQN backbone with shared convolutional layers
    and separate value and advantage streams.
    """

    def __init__(self, input_shape: Tuple[int, int, int], num_actions: int) -> None:
        """
        Initializes the network.

        Args:
            input_shape (Tuple[int, int, int]): Input observation shape (C, H, W).
            num_actions (int): Number of discrete actions.
        """
        super().__init__()
        channels, _, _ = input_shape

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        feature_dim = self._get_feature_dim(input_shape)

        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def _get_feature_dim(self, input_shape: Tuple[int, int, int]) -> int:
        """
        Computes the flattened feature size after the convolutional backbone.

        Args:
            input_shape (Tuple[int, int, int]): Input observation shape (C, H, W).

        Returns:
            int: Flattened feature dimension after convolutions.
        """
        with torch.no_grad():
            sample = torch.zeros(1, *input_shape, dtype=torch.float32)
            features = self.feature_extractor(sample)
            return int(features.reshape(1, -1).size(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes Q-values for the input batch.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) or (C, H, W).

        Returns:
            torch.Tensor: Q-values of shape (B, num_actions).
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = x.float()
        features = self.feature_extractor(x)
        features = features.reshape(features.size(0), -1)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
