import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

# import os
#from utils.utils import load_npz, read_ids

L = 33
n_channel = 10

def standardize(X):
    m = X.mean(axis=1)
    s = X.std(axis=1)
    X = (X - m) / s
    return X


def load_npz(file_path):
    """
    Load data from a .npz file
    """
    with np.load(file_path) as data:
        X = data["X"]
        y = data["y"]
        # polygon_ids = data["polygon_ids"]
        block_ids = data["block_id"]
    return X, y, block_ids#, polygon_ids

######### Read Train, Validation, and Test ids #########

def read_ids(file_path):
    """
    Read ids from file
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
        Train_ids = eval(lines[0].split(":")[1])
        test_ids = eval(lines[1].split(":")[1])
        # Eval_ids = eval(lines[2].split(":")[1])
    return Train_ids, test_ids#, Eval_ids

class SITSData(Dataset):
    def __init__(self, case: int, source_sits, target_sits, train_val_eval, set= 'trainval', transform=None):
        self.source_sits = source_sits
        self.target_sits = target_sits
        self.train_val_eval = train_val_eval
        self.transform = transform
        self.case = case
        self.set = set
        
        # read the set ids
        self.train_ids, self.test_ids = read_ids(self.train_val_eval)
        
        # case selection
        if self.set == 'trainval':
            ids = self.train_ids
        elif self.set == 'test':
            ids = self.test_ids
        else:
            raise ValueError("Please choose a set between trainval and test")

        # read the data 

        X_source, y_source, block_ids_source = load_npz(self.source_sits)
        X_target, y_target, block_ids_target = load_npz(self.target_sits)

        _source = np.concatenate((X_source, y_source[:, None], block_ids_source[:, None]), axis=1)
        _target = np.concatenate((X_target, y_target[:, None], block_ids_target[:, None]), axis=1)

        _source = _source[np.isin(_source[:, -1], ids)]
        _target = _target[np.isin(_target[:, -1], ids)]

        self.Xtrain_source = _source[:, :-2]
        self.ytrain_source = _source[:, -2]

        self.Xtrain_target = _target[:, :-2]
        self.ytrain_target = _target[:, -2]

        if self.case == 1:
            self.Xtrain = np.concatenate((self.Xtrain_source, self.Xtrain_target), axis=0)
            self.ytrain = np.concatenate((self.ytrain_source, self.ytrain_target), axis=0)
        elif self.case == 2:
            self.Xtrain = self.Xtrain_source
            self.ytrain = self.ytrain_source
        elif self.case == 3:
            self.Xtrain = self.Xtrain_target
            self.ytrain = self.ytrain_target
        else:
            raise ValueError("Please choose a case between 1 and 3")        

        print("The number of training samples is: ", self.Xtrain.shape)
        print("The number of training samples is: ", self.ytrain.shape)  

        def __len__(self):
            return len(self.ytrain)

        def __getitem__(self, idx):
            X = self.Xtrain[idx]
            y = self.ytrain[idx]
            if self.transform:
                X = self.transform(X)
            return X, y

if __name__ == "__main__":
    source_sits = "../../../data/theiaL2A_zip_img/output/2018/2018_SITS_data.npz"
    target_sits = "../../../data/theiaL2A_zip_img/output/2019/2019_SITS_data.npz"
    train_val_eval = "train_val_eval_rf.txt"
    case = 2
    set = "trainval"
    transform = None
    dataset = SITSData(case, source_sits, target_sits, train_val_eval, set, transform)
    dl = DataLoader(dataset, batch_size=1, shuffle=True)
    print(next(iter(dl)))

