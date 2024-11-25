import torch
from torch import nn
from .connectivity_graph_generator import ConnectivityGraphGenerator
from .spatial_decoder import SpatialDecoder
from .temporal_cnn import TemporalCNN
from .temporal_encoder import TemporalEncoder
from .classifier import GGNClassifier


class GGN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, device):
        super(GGN, self).__init__()
        self.in_channels = in_channels
        self.connectivity_graph = ConnectivityGraphGenerator(in_channels, hidden_channels, out_channels)
        self.temporal_encoder = TemporalEncoder(65, 128, 2)
        self.temporal_cnn = TemporalCNN(in_channels=128, hidden_channels=64)

        self.spatial_decoder = SpatialDecoder(hidden_channels, hidden_channels, out_channels, 4, device)

        self.classifier = GGNClassifier(hidden_channels, out_channels)

    def forward(self, x_temporal, x_topology):
        batch_size = x_temporal.size(0)
        num_nodes = self.in_channels

        # Step 1: Generate adjacency matrices and edge indices
        sampled_edge_indices = self.connectivity_graph(x_topology)

        # Step 2: Obtain node features from Temporal Encoder
        temporal_features = self.temporal_encoder(x_temporal)
        temporal_features = temporal_features.unsqueeze(1).repeat(1, num_nodes, 1)

        spatial_features = self.spatial_decoder(sampled_edge_indices, temporal_features)
        spatial_features = spatial_features.mean(dim=-1)

        temporal_features = self.temporal_cnn(temporal_features)

        cat_features = torch.cat((temporal_features, spatial_features), 1)

        output = self.classifier(cat_features)

        return output

