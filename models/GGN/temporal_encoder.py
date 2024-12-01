from torch import nn
from contextlib import contextmanager


class TemporalEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TemporalEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        return hn[-1]

    @contextmanager
    def evaluation_mode(self):
        original_mode = self.training
        self.eval()
        try:
            yield self
        finally:
            if original_mode:
                self.train()