from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import pickle
import numpy as np
import torch.optim as optim


class SaDataset(Dataset):

    def __init__(self, data_path, transform=None):
        super(SaDataset).__init__()
        if data_path:
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)
        self.length = len(self.data[0])
        self.transform = transform
        self.data_path = data_path

    def __getitem__(self, index):
        return self.data[0][index], self.data[1][index]

    def __len__(self):
        return self.length

    def add_sample(self, state, epoch):
        self.data[0] = np.append(self.data[0], state.reshape((1, -1)), axis=0)
        self.data[1] = np.append(self.data[1], epoch)


    def sample_save(self):
        with open(self.data_path, 'wb') as f:
            pickle.dump(self.data, f)