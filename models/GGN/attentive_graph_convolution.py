from torch import nn


class AttentiveGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(AttentiveGraphConvolution, self).__init__()
        
