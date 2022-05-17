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
    def __init__(self, sits, date_, transform=None):
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
        self.transform = transform
        self.date_ = date_
        
        with np.load(self.sits) as f:
            self.X_ = f['X']
            self.y_ = f['y']
        
        self.date_positions = date_positions(date_)
        
        
    def __len__(self):
        return len(self.y_)

    def __getitem__(self, idx):
        # print("getitem begins...")
        
        self.X = self.X_[idx]
        self.y = self.y_[idx]
        
        # del self.X_
        # del self.y_

        self.X = np.array(self.X, dtype = "float16")
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