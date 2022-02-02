import numpy as np



######### Read and load sits npz file #########

def load_npz(file_path):
    """
    Load data from a .npz file
    """
    with np.load(file_path) as data:
        X = data["X"]
        y = data["y"]
        polygon_ids = data["polygon_ids"]
        block_ids = data["block_id"]
    return X, y, polygon_ids, block_ids

######### Read Train, Validation, and Evaluation ids #########

def read_ids(file_path):
    """
    Read ids from file
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
        Train_ids = eval(lines[0].split(":")[1])
        Val_ids = eval(lines[1].split(":")[1])
        Eval_ids = eval(lines[2].split(":")[1])
    return Train_ids, Val_ids, Eval_ids

######### Load different sets of data #########

def load_set(set = "train"):
    """
    Load training, validation and evaluation set
    """
    train_ids, val_ids, eval_ids = read_ids("train_val_eval.txt")

    X, Y, polygon_ids, block_ids = load_npz("2018_SITS_data.npz")

    total_set = np.concatenate((X, Y[:, None], block_ids[:, None]), axis=1)

    if set == "train":
        Trainingset = total_set[np.isin(total_set[:, -1], train_ids)]
        Xtrain = Trainingset[:, :-2]
        Ytrain = Trainingset[:, -2]
        return Xtrain, Ytrain
    elif set == "val":
        Validset = total_set[np.isin(total_set[:, -1], val_ids)]
        Xval = Validset[:, :-2]
        Yval = Validset[:, -2]
        return Xval, Yval
    elif set == "eval":
        Evalset = total_set[np.isin(total_set[:, -1], eval_ids)]
        Xeval = Evalset[:, :-2]
        Yeval = Evalset[:, -2]
        return Xeval, Yeval
    else:
        raise ValueError("set must be either 'train', 'val' or 'eval'")