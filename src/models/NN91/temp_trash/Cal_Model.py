import torch
import torch.nn as nn

class CalNet(nn.Module):
    def __init__ (self, in_features, n_classes):
        super(CalNet, self).__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.densenet = nn.Sequential(
            nn.Linear(self.in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.in_features, self.n_classes)
        )
    def forward (self, x):
        x = self.mlp(x)
        return x