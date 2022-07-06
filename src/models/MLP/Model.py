import torch
import torch.nn as nn

class DNAMLP(nn.Module):
    def __init__ (self, in_features, n_classes):
        super(DNAMLP, self).__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.densenet = nn.Sequential(
            nn.Linear(self.in_features, n_classes),
        )
        
    def forward (self, x):
        x = self.densenet(x)
        return x