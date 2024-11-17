from torch import nn


class TemporalEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TemporalEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        return hn[-1]
