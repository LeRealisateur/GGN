import torch
from torch import nn


class TemporalCNN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        """
        Temporal CNN for processing temporal features.

        Args:
            in_channels (int): Number of input channels (e.g., 1 for single-channel input).
            hidden_channels (int): Number of hidden/output channels.
        """
        super(TemporalCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Forward pass for TemporalCNN.

        Args:
            x: (batch_size, num_nodes, feature_dim)

        Returns:
            Flattened feature tensor: (batch_size, hidden_channels)
        """
        # Reshape input: (batch_size, num_nodes, feature_dim) -> (batch_size, feature_dim, num_nodes)
        x = x.transpose(1, 2)

        # Apply convolutional layers
        x = self.relu(self.conv1(x))  # (batch_size, hidden_channels, num_nodes)
        x = self.relu(self.conv2(x))  # (batch_size, hidden_channels, num_nodes)

        # Adaptive pooling to reduce spatial dimension to 1
        x = self.pool(x)  # (batch_size, hidden_channels, 1)

        # Flatten to (batch_size, hidden_channels)
        x = x.squeeze(-1)
        return x
