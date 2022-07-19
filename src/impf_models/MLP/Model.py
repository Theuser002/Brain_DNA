import torch
import torch.nn as nn

class Impf_DNAMLP(nn.Module):
    def __init__ (self, in_features, n_classes):
        super(Impf_DNAMLP, self).__init__()
        self.densenet = nn.Sequential(
            nn.Linear(in_features, n_classes),
        )
        
    def forward (self, x):
        x = self.densenet(x)
        return x

class Impf_GlioMLP(nn.Module):
    def __init__ (self, in_features, n_classes, dropout_rate = 0):
        super(Impf_GlioMLP, self).__init__()        
        self.densenet = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(p = dropout_rate),
            nn.Linear(64, n_classes)
        )
        
    def forward (self, x):
        x = self.densenet(x)
        return x