from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from contextlib import contextmanager


class ConnectivityGraphGenerator(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, temperature=0.5):
        super(ConnectivityGraphGenerator, self).__init__()
        self.temperature = temperature
        self.para_learner = ParaLearner(in_channels, hidden_channels, out_channels)
        self.num_nodes = 64

    def forward(self, x):
        x = x.float()
        edge_index_batch = self.generate_edge_index_batch(x)
        batch_size, num_nodes, in_channels = x.size()
        x_flat = x.view(batch_size * num_nodes, in_channels)

        mean, variance = self.para_learner(x_flat, edge_index_batch)

        edge_probs = self.compute_edge_probabilities_batch(mean.view(batch_size, num_nodes, -1),
                                                           variance.view(batch_size, num_nodes, -1))

        # Step 3: Sample edges using Gumbel-Softmax
        sampled_edges = self.gumbel_softmax(edge_probs)

        return sampled_edges

    def generate_edge_index_batch(self, x):
        batch_size, num_nodes, _ = x.shape
        device = x.device

        # Generate edge index for a single graph (fully connected)
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        # Number of edges in a single graph
        num_edges = edge_index.size(1)

        # Create node index offsets for batching
        node_offsets = torch.arange(batch_size).repeat_interleave(num_edges) * num_nodes

        # Repeat edge_index for the batch and add offsets
        edge_index_batch = edge_index.repeat(1, batch_size) + node_offsets.unsqueeze(0)

        return edge_index_batch.to(device)

    def compute_edge_probabilities_batch(self, mean, variance):
        # mean and variance: [batch_size, num_nodes, out_channels]
        delta_mean = mean.unsqueeze(2) - mean.unsqueeze(1)  # [batch_size, num_nodes, num_nodes, out_channels]
        sigma_sum = variance.unsqueeze(2) + variance.unsqueeze(1)  # Same shape
        exponent = - (delta_mean ** 2) / (2 * sigma_sum ** 2 + 1e-8)
        p = torch.exp(exponent).mean(dim=-1)  # Mean over out_channels
        return p  # [batch_size, num_nodes, num_nodes]

    def gumbel_softmax(self, edge_probs):
        """
        Sample edges using Gumbel-Softmax.
        """
        # Add Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(edge_probs) + 1e-8) + 1e-8)
        logits = torch.log(edge_probs + 1e-8) + gumbel_noise

        # Apply softmax with temperature
        sampled_edges = F.softmax(logits / self.temperature, dim=-1)

        # Convert to binary adjacency matrix
        sampled_edges = (sampled_edges > 0.5).float()
        return sampled_edges
    
    @contextmanager
    def evaluation_mode(self):
        original_mode = self.training
        self.eval()
        try:
            yield self
        finally:
            if original_mode:
                self.train()


class ParaLearner(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ParaLearner, self).__init__()

        self.gnn_mean = GNNLayer(in_channels, hidden_channels)
        self.gnn_variance = GNNLayer(in_channels, hidden_channels)

        self.mean_out = nn.Linear(hidden_channels, out_channels)
        self.variance_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        mean = self.mean_out(self.gnn_mean(x, edge_index))
        variance = self.variance_out(self.gnn_variance(x, edge_index))
        return mean, variance


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
