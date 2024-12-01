from abc import ABC

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import torch.nn.functional as F
from contextlib import contextmanager


class SpatialDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, device):
        """
        Args:
            in_channels: Number of input node feature channels.
            hidden_channels: Number of hidden channels.
            out_channels: Number of output node feature channels.
            num_layers: Number of attentive graph convolution layers.
            device: Device to run computations on (e.g., 'cuda' or 'cpu').
        """
        super(SpatialDecoder, self).__init__()
        layers = [AttentiveGraphConvLayer(in_channels, hidden_channels, device)]
        for _ in range(num_layers - 2):
            layers.append(AttentiveGraphConvLayer(hidden_channels, hidden_channels, device))
        layers.append(AttentiveGraphConvLayer(hidden_channels, out_channels, device))
        self.layers = nn.ModuleList(layers)
        self.activation = nn.ELU()

    def forward(self, sampled_edge_indices, temporal_features, edge_features=None):
        """
        Args:
            sampled_edge_indices: (batch_size, num_nodes, num_nodes) Adjacency matrix (sampled edges).
            temporal_features: (batch_size, num_nodes, feature_dim) Node features.
            edge_features: (batch_size, num_nodes, edge_dim) Optional edge-specific features.

        Returns:
            Node-level representations.
        """
        batch_size, num_nodes, _ = temporal_features.size()

        # Convert sampled_edge_indices to edge_index format (COO)
        edge_index_list = []
        edge_batch_list = []

        for b in range(batch_size):
            edges = torch.nonzero(sampled_edge_indices[b], as_tuple=False).t()  # (2, num_edges)
            edge_index_list.append(edges)
            edge_batch_list.append(torch.full((edges.size(1),), b, dtype=torch.long, device=temporal_features.device))

        edge_index = torch.cat(edge_index_list, dim=1)  # (2, total_edges)
        batch_indices = torch.cat(edge_batch_list, dim=0)  # (total_edges,)

        # Flatten temporal features for batched graph processing
        x = temporal_features.view(-1, temporal_features.size(-1))  # (batch_size * num_nodes, feature_dim)

        for layer in self.layers:
            x = layer(x, edge_index, batch=batch_indices, edge_features=edge_features)
            x = self.activation(x)

        # Reshape back to (batch_size, num_nodes, out_channels)
        x = x.view(batch_size, num_nodes, -1)
        return x
    
    @contextmanager
    def evaluation_mode(self):
        original_mode = self.training
        self.eval()
        try:
            yield self
        finally:
            if original_mode:
                self.train()


class AttentiveGraphConvLayer(MessagePassing, ABC):
    def __init__(self, in_channels, hidden_channels, device):
        super(AttentiveGraphConvLayer, self).__init__(aggr='mean')
        self.linear = nn.Linear(in_channels, hidden_channels, bias=False)
        self.attention = nn.Parameter(torch.Tensor(2 * hidden_channels, 1).to(device))
        nn.init.xavier_uniform_(self.attention.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, batch=None, edge_features=None):
        """
        Args:
            x: (total_nodes, feature_dim) Flattened node features.
            edge_index: (2, total_edges) Edge indices in COO format.
            batch: (total_edges,) Batch indices for each edge.
            edge_features: Optional additional edge features.

        Returns:
            Updated node features.
        """
        x = self.linear(x)  # Transform node features
        return self.propagate(edge_index, x=x, batch=batch, edge_features=edge_features)

    def message(self, x_i, x_j, edge_index_i, batch, edge_features=None):
        """
        Compute attention-weighted messages.
        Args:
            x_i: (num_edges, hidden_dim) Features of source nodes.
            x_j: (num_edges, hidden_dim) Features of target nodes.
            edge_index_i: Indices of the target nodes.
            batch: Batch indices for edges.
            edge_features: (num_edges, edge_dim), optional.

        Returns:
            Attention-weighted messages.
        """
        x_cat = torch.cat([x_i, x_j], dim=-1)  # Concatenate source and target features
        alpha = (x_cat @ self.attention).squeeze(-1)  # Compute attention scores
        alpha = self.leaky_relu(alpha)
        alpha = softmax(alpha, edge_index_i)  # Apply softmax for normalization

        # Optional edge feature integration
        if edge_features is not None:
            alpha = alpha * edge_features

        return x_j * alpha.unsqueeze(-1)  # Weight messages by attention scores

    def update(self, aggr_out):
        return aggr_out
    
    @contextmanager
    def evaluation_mode(self):
        original_mode = self.training
        self.eval()
        try:
            yield self
        finally:
            if original_mode:
                self.train()
