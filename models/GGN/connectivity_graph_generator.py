from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from contextlib import contextmanager


def gumbel_softmax(logits, temperature=0.5):
    """
    Applies the Gumbel-Softmax trick to make sampling differentiable.

    Args:
        logits (torch.Tensor): Logits for the edges.
        temperature (float): Temperature for the Gumbel-Softmax distribution.

    Returns:
        torch.Tensor: Probabilities after Gumbel-Softmax sampling.
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    return F.softmax((logits + gumbel_noise) / temperature, dim=-1)


class ConnectivityGraphGenerator(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.para_learner = ParaLearner(in_channels, hidden_channels, out_channels)

    def generate_edge_index(self, num_nodes):
        """
        Generate upper triangular edge indices for a fully connected graph without self-loops.

        Args:
            num_nodes (int): Number of nodes in the graph.

        Returns:
            torch.Tensor: Tensor of shape [2, num_edges] representing edge indices.
        """
        row, col = torch.triu_indices(num_nodes, num_nodes, offset=1)
        return torch.stack([row, col])

    def forward(self, x_topology, x_temporal):
        """
        Generate enhanced adjacency matrix from input connection matrix.

        Args:
            x (torch.Tensor): Input connection matrix of shape [batch_size, num_electrodes, num_electrodes].

        Returns:
            torch.Tensor: Enhanced adjacency matrix of shape [batch_size, num_electrodes, num_electrodes].
            :param x_temporal:
            :param x_topology:
        """

        batch_size, num_electrodes, _ = x_topology.shape
        device = x_topology.device

        # Generate edge indices for a fully connected graph (upper triangular)
        edge_index = self.generate_edge_index(num_electrodes).to(device)  # Shape: [2, num_edges_per_graph]

        # Repeat edge_index for each graph in the batch
        edge_index = edge_index.repeat(1, batch_size)  # Shape: [2, num_edges_total]

        # Compute node offsets for batched processing to ensure unique node indexing
        node_offsets = torch.arange(batch_size, device=device).repeat_interleave(
            edge_index.size(1) // batch_size) * num_electrodes
        edge_index = edge_index + node_offsets.unsqueeze(0)  # Shape: [2, num_edges_total]

        # Compute edge parameters using ParaLearner
        mean, variance, weight = self.para_learner(x_topology, x_temporal, edge_index)

        gumbel_weights = gumbel_softmax(weight, temperature=self.temperature)

        # Compute the enhanced adjacency matrix
        enhanced_adj = self._compute_adjacency(mean, variance, gumbel_weights, num_electrodes, batch_size, device)

        return enhanced_adj

    def _compute_adjacency(self, mean, variance, weight, num_electrodes, batch_size, device):
        """
        Compute enhanced adjacency matrix from edge parameters.

        Args:
            original_x (torch.Tensor): Original connection matrix of shape [batch_size, num_electrodes, num_electrodes].
            mean (torch.Tensor): Edge means of shape [batch_size * num_edges_per_graph, out_channels].
            variance (torch.Tensor): Edge variances of shape [batch_size * num_edges_per_graph, out_channels].
            weight (torch.Tensor): Edge weights of shape [batch_size * num_edges_per_graph].
            num_electrodes (int): Number of electrodes (nodes) in each graph.
            batch_size (int): Number of graphs in the batch.

        Returns:
            torch.Tensor: Enhanced adjacency matrix of shape [batch_size, num_electrodes * num_electrodes].
        """
        total_edges = mean.size(0)
        edges_per_graph = total_edges // batch_size  # e.g., 2016 for 64 nodes: 64*63/2 = 2016

        # Generate edge indices once (upper triangular indices for a fully connected graph)
        row, col = self.generate_edge_index(num_electrodes).to(device)  # Shape: [2, num_edges_per_graph]

        # Repeat edge indices for each graph in the batch
        edge_index_list = []
        weights_list = []
        for b in range(batch_size):
            # Offset the indices for batch separation
            offset = b * num_electrodes
            edge_index = torch.stack([row + offset, col + offset])  # Shape: [2, num_edges_per_graph]

            # Compute weights for the current batch graph
            delta_mean = mean[b * edges_per_graph: (b + 1) * edges_per_graph]  # [num_edges_per_graph, out_channels]
            delta_var = variance[b * edges_per_graph: (b + 1) * edges_per_graph]  # [num_edges_per_graph, out_channels]
            batch_weights = weight[b * edges_per_graph: (b + 1) * edges_per_graph]  # [num_edges_per_graph]

            # Compute similarity (optional, adjust based on need)
            similarity = torch.exp(
                -0.5 * torch.mean((delta_mean ** 2) / (delta_var + 1e-8), dim=-1))  # [num_edges_per_graph]
            edge_weights = similarity * batch_weights  # Combine similarity with learned weights

            edge_index_list.append(edge_index)
            weights_list.append(edge_weights)

        # Concatenate all graphs' edge indices
        edge_index = torch.cat(edge_index_list, dim=1)  # Shape: [2, num_edges]

        return edge_index

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
        super().__init__()
        # Single GNNLayer to compute node embeddings
        self.gnn = GNNLayer(in_channels, hidden_channels)

        # Linear layers to compute edge parameters based on concatenated node embeddings
        self.mean_out = nn.Linear(hidden_channels * 2, out_channels)
        self.variance_out = nn.Linear(hidden_channels * 2, out_channels)
        self.weight_out = nn.Linear(hidden_channels * 2, 1)

    def forward(self, x_topology, x_temporal, edge_index):
        """
        Compute edge-level parameters for mean, variance, and weight.

        Args:
            x_topology (torch.Tensor): Node features from topology (connection matrix) of shape [batch_size, num_electrodes, feature_dim].
            x_temporal (torch.Tensor): Temporal node features of shape [batch_size, num_electrodes, temporal_dim].
            edge_index (torch.Tensor): Graph connectivity of shape [2, num_edges_total].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: mean, variance, weight for edges.
        """
        batch_size, num_electrodes, topo_dim = x_topology.shape
        _, _, temp_dim = x_temporal.shape

        # Concatenate topology and temporal features along the feature dimension
        combined_features = torch.cat([x_topology, x_temporal], dim=-1)  # Shape: [3200, 64]
        x = combined_features.view(batch_size * num_electrodes,
                                   topo_dim + temp_dim)  # Shape: [num_nodes, topo_dim + temp_dim]

        # Compute node embeddings using the GNN layer
        node_embeddings = self.gnn(x, edge_index)  # Shape: [3200, hidden_channels]

        # Extract source and target node indices for each edge
        src_indices = edge_index[0]  # Shape: [100800]
        dst_indices = edge_index[1]  # Shape: [100800]

        # Gather source and target node embeddings
        src_embeddings = node_embeddings[src_indices]  # Shape: [100800, hidden_channels]
        dst_embeddings = node_embeddings[dst_indices]  # Shape: [100800, hidden_channels]

        # Concatenate source and target embeddings to form edge features
        edge_features = torch.cat([src_embeddings, dst_embeddings], dim=1)  # Shape: [100800, hidden_channels * 2]

        # Compute edge parameters
        mean = self.mean_out(edge_features)  # Shape: [100800, out_channels]
        variance = F.softplus(self.variance_out(edge_features)) + 1e-6  # Shape: [100800, out_channels]
        weight = torch.sigmoid(self.weight_out(edge_features)).squeeze(-1)  # Shape: [100800]

        return mean, variance, weight


class GNNLayer(MessagePassing, ABC):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')  # Use mean aggregation.
        self.linear = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        """
        Forward pass for the GNN layer.

        Args:
            x (torch.Tensor): Node features of shape [num_nodes, in_channels].
            edge_index (torch.Tensor): Graph connectivity of shape [2, num_edges].

        Returns:
            torch.Tensor: Updated node embeddings of shape [num_nodes, out_channels].
        """
        x = x.float()  # Ensure input is float32
        out = self.propagate(edge_index, x=x)  # Shape: [num_nodes, in_channels]
        out = self.linear(out)  # Shape: [num_nodes, out_channels]
        out = self.relu(out)  # Shape: [num_nodes, out_channels]
        return out

    def message(self, x_j):
        """
        Message passing function.

        Args:
            x_j (torch.Tensor): Neighboring node features of shape [num_edges, in_channels].

        Returns:
            torch.Tensor: Messages to be aggregated.
        """
        return x_j  # Directly pass the neighbor features. Modify if additional transformations are needed.
