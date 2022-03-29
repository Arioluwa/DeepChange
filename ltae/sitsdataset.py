import torch
from torch.utils import data
from torchvision import transforms
import numpy as np

from utils import load_npz, read_ids

import os
import time
import random

n_channel = 10

class SITSData(data.Dataset):
    def __init__(self, sits, seed, partition='train', transform=None):
        """
        Args:
            sits (path):
            seed (int: within 0-9):
            partition:
            transform:
        return:
        """
        self.sits = sits
        self.seed = seed
        self.transform = transform
        
        # get partition ids using the read_id() func
        self.train_ids, self.val_ids, self.test_ids = read_ids(self.seed)

        # select partition
        if partition == 'train':
            self.ids = self.train_ids
        elif partition == 'val':
            self.ids = self.val_ids
        elif partition == 'test':
            self.ids = self.test_ids
        else:
            raise ValueError('Invalid partition: {}'.format(partition))

        X, y, block_ids = load_npz(self.sits)
        
        # concatenate the data
        data_ = np.concatenate((X, y[:, None], block_ids[:, None]), axis=1)

        # filter by block_id
        data_ = data_[np.isin(data_[:, -1], self.ids)]
        
        self.X_ = data_[:, :-2]
        self.y_ = data_[:, -2]
        
        del X
        del y
        del block_ids
        del data_
        
    def __len__(self):
        return len(self.y_)

    def __getitem__(self, idx):
        self.X = self.X_[idx]
        self.y = self.y_[idx]

        self.X = np.array(self.X).astype('float32')
        self.y = np.array(self.y).astype('float32')
        
        self.X = self.X.reshape(int(self.X.shape[0]/n_channel), n_channel)

        # transform
        if self.transform:
            self.X = self.transform(self.X)
            self.X = self.X.transpose(0, 2, 1)
        
        torch_x = torch.from_numpy(self.X)
        torch_y = torch.from_numpy(self.y)
        return torch_x, torch_y