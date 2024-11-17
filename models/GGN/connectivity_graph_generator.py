from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class ConnectivityGraphGenerator(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, temperature=0.5):
        super(ConnectivityGraphGenerator, self).__init__()
        self.temperature = temperature
        self.para_learner = ParaLearner(in_channels, hidden_channels, out_channels)
        self.num_nodes = 64

    def forward(self, x):
        x = x.float()
        batch_size, num_nodes, in_channels = x.size()
        x_flat = x.view(batch_size * num_nodes, in_channels)
        #mean, variance, mixing = self.para_learner(x_flat, edge_index_batch)

    def compute_edge_probabilities_batch(self, mean, variance):
        # mean and variance: [batch_size, num_nodes, out_channels]
        delta_mean = mean.unsqueeze(2) - mean.unsqueeze(1)  # [batch_size, num_nodes, num_nodes, out_channels]
        sigma_sum = variance.unsqueeze(2) + variance.unsqueeze(1)  # Same shape
        exponent = - (delta_mean ** 2) / (2 * sigma_sum ** 2 + 1e-8)
        p = torch.exp(exponent).mean(dim=-1)  # Mean over out_channels
        return p  # [batch_size, num_nodes, num_nodes]


class ParaLearner(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ParaLearner, self).__init__()

        self.gnn_mean = GNNLayer(in_channels, hidden_channels)
        self.gnn_variance = GNNLayer(in_channels, hidden_channels)
        self.gnn_mixing = GNNLayer(in_channels, hidden_channels)

        self.mean_out = nn.Linear(hidden_channels, out_channels)
        self.variance_out = nn.Linear(hidden_channels, out_channels)
        self.mixing_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        mean = self.mean_out(self.gnn_mean(x, edge_index))
        variance = self.variance_out(self.gnn_variance(x, edge_index))
        mixing = self.mixing_out(self.gnn_mixing(x, edge_index))
        return mean, variance, mixing


class GNNLayer(MessagePassing, ABC):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.propagate(edge_index, x=x)
        return self.relu(self.linear1(x))

    def message(self, x_j):
        return x_j
