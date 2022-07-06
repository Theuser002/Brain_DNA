import torch
from torch.utils.data import Dataset

class CNS (Dataset):
    def __init__(self, features, labels, mode = 'train', split = None):
        self.mode = mode
        self.split = split
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        if self.mode.lower() == 'predict':
            return torch.tensor(feature).float()
        else:
            return torch.tensor(feature).float(), torch.tensor(label).long()