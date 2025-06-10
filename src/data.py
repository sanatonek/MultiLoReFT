import torch
from torch.utils.data import Dataset

class MultimodalDataset(Dataset):
    """Dataset class for multimodal data."""
    def __init__(self, h1, h2, x1, x2, labels):
        self.h1 = torch.tensor(h1, dtype=torch.float32)
        self.h2 = torch.tensor(h2, dtype=torch.float32)
        self.x1 = torch.tensor(x1, dtype=torch.float32)
        self.x2 = torch.tensor(x2, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.h1[idx], self.h2[idx], self.x1[idx], self.x2[idx], self.labels[idx]
