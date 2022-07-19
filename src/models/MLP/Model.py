import torch
import torch.nn as nn

class DNAMLP(nn.Module):
    def __init__ (self, in_features, n_classes, dropout_rate = 0.0):
        super(DNAMLP, self).__init__()
        self.densenet = nn.Sequential(
            nn.Linear(in_features, n_classes),
            nn.Dropout(p = dropout_rate)
        )
        
    def forward (self, x):
        x = self.densenet(x)
        return x