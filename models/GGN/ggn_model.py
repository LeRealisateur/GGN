from torch import nn
from .connectivity_graph_generator import ConnectivityGraphGenerator
from .temporal_cnn import TemporalCNN
from .temporal_encoder import TemporalEncoder
from .classifier import GGNClassifier


class GGN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GGN, self).__init__()
        #self.connectivity_graph = ConnectivityGraphGenerator(in_channels, hidden_channels, out_channels)
        self.temporal_encoder = TemporalEncoder(in_channels, 128, 2)
        self.temporal_cnn = TemporalCNN(in_channels=1, hidden_channels=64)
        self.classifier = GGNClassifier(64, out_channels)

    def forward(self, x_temporal, x_topology):
        #mean, variance, weight = self.connectivity_graph(x_topology)

        hn = self.temporal_encoder(x_temporal)
        hn = hn.unsqueeze(1).unsqueeze(2)
        x_temporal = self.temporal_cnn(hn)
        output = self.classifier(x_temporal)
        return output

