import torch
from torch import nn
from torch_geometric.graphgym import GATConv


class SpatialDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SpatialDecoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(AttentiveGraphConvolution(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(AttentiveGraphConvolution(hidden_channels, hidden_channels))
        self.layers.append(AttentiveGraphConvolution(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv in self.layers:
            x = torch.relu(conv(x, edge_index))
        return x


class AttentiveGraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super(AttentiveGraphConvolution, self).__init__()
        self.gat_conv = GATConv(in_channels, out_channels, heads=heads, concat=True)

    def forward(self, x, edge_index):
        x = self.gat_conv(x, edge_index)
        return x
