import numpy as np


_train_val_eval = "train_val_eval_rf.txt"
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

# def read_ids(file_path):
#     """
#     Read ids from file
#     """
#     with open(file_path, "r") as f:
#         lines = f.readlines()
#         Train_ids = eval(lines[0].split(":")[1])
#         test_ids = eval(lines[1].split(":")[1])
#         Eval_ids = eval(lines[2].split(":")[1])
#     return Train_ids, test_ids, Eval_ids