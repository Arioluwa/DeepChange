import torch
from torch.utils import data
from torchvision import transforms
import numpy as np

from utils import load_npz, read_ids

import os
import time
import datetime
import random

n_channel = 10

class SITSData(data.Dataset):
    def __init__(self, sits, seed, date_, partition='train', transform=None):
        """
        Args:
            sits (path): .npy file
            seed (int: within 0-9):
            date_ (path): gapfilled date paths
            partition:
            transform:
        return:
        """
        self.sits = sits
        self.seed = seed
        self.transform = transform
        self.date_ = date_
        
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
        
        y = np.unique(y, return_inverse=True)[1] # reassigning label [1,23] to [0,18]
        
        # concatenate the data
        data_ = np.concatenate((X, y[:, None], block_ids[:, None]), axis=1)

        # filter by block_id
        data_ = data_[np.isin(data_[:, -1], self.ids)]
        
        self.X_ = data_[:, :-2]
        self.y_ = data_[:, -2]
        
        self.date_positions = date_positions(date_)
        
        del X
        del y
        del block_ids
        del data_
        
    def __len__(self):
        return len(self.y_)

    def __getitem__(self, idx):
        self.X = self.X_[idx]
        self.y = self.y_[idx]

        self.X = np.array(self.X, dtype = float)
        self.y = np.array(self.y, dtype = int)
        
        self.X = self.X.reshape(int(self.X.shape[0]/n_channel), n_channel)

        # transform
        if self.transform:
            self.X = self.transform(self.X)
            # self.X = self.X.transpose(0, 2, 1) #why?
        
        torch_x = torch.from_numpy(self.X)
        torch_y = torch.from_numpy(self.y)
        return torch_x, torch_y
    
def date_positions(gfdate_path):
    with open(gfdate_path, "r") as f:
        date_list = f.readlines()
    date_list = [x.strip() for x in date_list]
    date_list = [datetime.datetime.strptime(x, "%Y%m%d").timetuple().tm_yday for x in date_list]
    date_ = [x for x in date_list]
    return date_