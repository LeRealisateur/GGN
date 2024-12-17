import torch
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=65, 
                 dropoutRate=0.5, kernLength=64, F1=8, 
                 D=2, F2=None, norm_rate=0.25, dropoutType='Dropout'):
        """
        PyTorch Implementation of EEGNet
        Arguments:
        - nb_classes: Number of output classes
        - Chans: Number of EEG channels
        - Samples: Number of time points
        - dropoutRate: Dropout fraction
        - kernLength: Length of temporal convolution kernel
        - F1: Number of temporal filters
        - D: Depth multiplier for depthwise convolution
        - F2: Number of pointwise filters (default: F1 * D)
        - norm_rate: Max norm constraint for Dense layer
        - dropoutType: 'Dropout' or 'SpatialDropout2D'
        """
        super(EEGNet, self).__init__()
        
        if F2 is None:
            F2 = F1 * D
        
        # Dropout type
        self.dropout = nn.Dropout if dropoutType == 'Dropout' else nn.Dropout2d

        # Block 1: Temporal Convolution + Depthwise Convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (1, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            self.dropout(dropoutRate)
        )

        # Block 2: Separable Convolution
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            self.dropout(dropoutRate)
        )

        # Fully Connected Layer
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(self._get_flatten_size(Chans, Samples, F1, D, F2), nb_classes),
            nn.Softmax(dim=1)
        )
        
    def _get_flatten_size(self, Chans, Samples, F1, D, F2):
        """
        Calcule la taille de la sortie aplatie après les blocs convolutifs.
        """
        with torch.no_grad():
            x = torch.zeros((1,1, Chans, Samples))
            x = self.block1(x)
            x = self.block2(x)
            return x.numel()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

