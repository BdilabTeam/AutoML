from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import pickle
import numpy as np
import torch.optim as optim


class JstDataset(Dataset):

    def __init__(self, data_path, transform=None):
        super(JstDataset).__init__()
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.length = len(self.data[0])
        self.transform = transform

    def __getitem__(self, index):
        return self.data[0][index], self.data[1][index]

    def __len__(self):
        return self.length
