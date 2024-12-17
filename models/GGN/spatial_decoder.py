from abc import ABC

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool, GATConv


class SpatialDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads=4, dropout=0.6):
        """
        Args:
            in_channels (int): Number of input node feature channels.
            hidden_channels (int): Number of hidden channels.
            out_channels (int): Number of output node feature channels.
            num_layers (int): Number of GATConv layers.
            heads (int): Number of attention heads in GATConv.
            dropout (float): Dropout rate for GATConv layers.
        """
        super(SpatialDecoder, self).__init__()

        self.pre_transform = nn.Linear(in_channels, hidden_channels)

        self.convs = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.cached_attention = []

        self.convs.append(
            GATConv(hidden_channels, hidden_channels, heads=heads, concat=True, dropout=dropout, add_self_loops=False))
        self.activations.append(nn.ReLU())

        # Hidden GATConv layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=dropout,
                                      add_self_loops=False))
            self.activations.append(nn.ReLU())

        # Last GATConv layer
        self.convs.append(
            GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout, add_self_loops=False))
        self.activations.append(nn.ReLU())

        # Pooling layer to aggregate node features into graph-level features
        self.pool = global_mean_pool

    def forward(self, edge_index, x, batch):
        """
        Args:
            edge_index (Tensor): Graph connectivity in COO format with shape [2, num_edges].
            x (Tensor): Node feature matrix with shape [num_nodes, in_channels].
            batch (Tensor): Batch vector assigning each node to a specific graph in the batch with shape [num_nodes].

        Returns:
            Tensor: Graph-level representations with shape [num_graphs, out_channels].
        """
        batch_size, num_nodes, in_channels = x.shape
        edges_per_graph = (num_nodes * (num_nodes - 1)) // 2
        # Reshape node features from [50, 64, 128] to [3200, 128] 3200 because 64 nodes * batch size
        x = x.view(-1, in_channels)
        x = self.pre_transform(x)

        # Pass through GATConv layers
        for i, (conv, activation) in enumerate(zip(self.convs, self.activations)):
            x, (edge_index, attention_weights) = conv(x, edge_index, return_attention_weights=True)
            if i == len(self.convs) - 1:
                # Reshape attention weights to [batch_size, num_edges, 1]
                attention_weights = attention_weights.view(batch_size, edges_per_graph, -1)
                mean_alpha = attention_weights.mean(dim=0)  # Average across batch dimension
                self.cached_attention.append(mean_alpha)
            x = activation(x)

        x = self.pool(x, batch)
        return x
