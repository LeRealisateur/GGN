import torch
from torch import nn


class GGNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(GGNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
