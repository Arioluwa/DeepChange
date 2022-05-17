import numpy as np


######### Read and load sits npz file #########

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

######### Read Train, Validation, and Evaluation ids #########

def read_ids(seed_value):
    """
    Read ids from file
    """
    assert seed_value >= 0 and seed_value <= 10
    
    with open("./ids/train_val_eval_seed_" + str(seed_value)+".txt", "r") as f:
        lines = f.readlines()
        Train_ids = eval(lines[0].split(":")[1])
        Val_ids = eval(lines[1].split(":")[1])
        test_ids = eval(lines[2].split(":")[1])
    return Train_ids, Val_ids, test_ids

class standardize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        return (sample - self.mean) / self.std