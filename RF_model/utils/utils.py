import numpy as np


_train_val_eval = "train_val_eval_rf.txt"
######### Read and load sits npz file #########

def load_npz(file_path):
    """
    Load data from a .npz file
    """
    start_time = time.time()
    with np.load(file_path) as data:
        X = data["X"]
        y = data["y"]
        # polygon_ids = data["polygon_ids"]
        block_ids = data["block_id"]
    print("load npz time: ", time.time() - start_time)
    return X, y, block_ids#, polygon_ids

######### Read Train, Validation, and Evaluation ids #########

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

######### Load different sets of data #########

# train_ids, val_ids = read_ids(_train_val_eval)

# X, Y, polygon_ids, block_ids = load_npz("2018_SITS_data.npz")

# total_set = np.concatenate((X, Y[:, None], block_ids[:, None]), axis=1)
# def load_set(set = "train", total_set):
#     """
#     Load training, validation and evaluation set
#     """
#     if set == "train":
#         Trainingset = total_set[np.isin(total_set[:, -1], train_ids)]
#         Xtrain = Trainingset[:, :-2]
#         Ytrain = Trainingset[:, -2]
#         return Xtrain, Ytrain
#     elif set == "val":
#         Validset = total_set[np.isin(total_set[:, -1], val_ids)]
#         Xval = Validset[:, :-2]
#         Yval = Validset[:, -2]
#         return Xval, Yval
#     # elif set == "eval":
#     #     Evalset = total_set[np.isin(total_set[:, -1], eval_ids)]
#     #     Xeval = Evalset[:, :-2]
#     #     Yeval = Evalset[:, -2]
#     #     return Xeval, Yeval
#     else:
#         raise ValueError("set must be either 'train' or 'val'")