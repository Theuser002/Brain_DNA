import torch
from torch.utils.data import Dataset

class Cal_Dataset(Dataset):
    def __init__(self, probs, labels, mode = 'Train', split = None):
        self.mode = mode
        self.split = split
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        prob = probs[idx]
        label = labels[idx]
        if self.mode.lower() == 'test':
            return torch.tensor(prob).float()
        else:
            return torch.tensor(prob).float(), torch.tensor(label).long()