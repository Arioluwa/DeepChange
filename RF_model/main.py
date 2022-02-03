import os
import argparse
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score
#from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='RF model')

parser.add_argument('-f', '--file_path', type=str, help='path to .npz file', required=True)
parser.add_argument('-t', '--train_ids', type=str, help='path to train ids file', required=True)

args = parser.parse_args()

npz_file = args.file_path
train_ids_file = args.train_ids

year = os.path.basename(npz_file).split('_')[0]
# RF Model Script
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


X, Y, polygon_ids, block_ids = load_npz(npz_file)

train_ids, val_ids, eval_ids = read_ids(train_ids_file)

total_set = np.concatenate((X, Y[:, None], block_ids[:, None]), axis=1)

def load_set(set = "train"):
    """
    Load training, validation and evaluation set
    """
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

def savereport(report, output_path):
    """
    Save classification report to file
    """
    with open(output_path, "w") as f:
        f.write(report)
start_time = time.time()
# Train dataset
Xtrain, ytrain = load_set("train")
print("Loaded training set:", time.time() - start_time)

start_time = time.time()
# Random Forest model
rf = RandomForestClassifier(n_estimators=100, max_depth = 25, n_jobs=20, max_features= "sqrt", min_samples_leaf=10, random_state=42)

# Fit the model
rf = rf.fit(Xtrain, ytrain)
print("Fitted model:", time.time() - start_time)

# Validation dataset
Xval, yval = load_set("val")

print("Validation set:", Xval.shape)
print("Validation set:", yval.shape)


start_time = time.time()
# predict on validation set
y_pred = rf.predict(Xval)
print("Predicted on validation set:", time.time() - start_time)

label = ['Sunflower', 'Corn', 'Rice', 'Tubers/roots', 'Soy', 'Straw cereals', 'Protein crops', 'Oilseeds', 'Grasslands', 'Vineyards', 'Hardwood forest', 'Softwood forest', 'Natural grasslands and pastures', 'Woody moorlands', 'Dense built-up area', 'Diffuse built-up area', 'Industrial and commercial areas', 'Roads', 'Glaciers and eternal snows']
validation_report = classification_report(yval, y_pred, target_names=label)
print(validation_report)
# save validation report as txt file with year
savereport(validation_report, "%s_validation_report.txt" % year)

# Evaluation dataset
Xeval, yeval = load_set("eval")

start_time = time.time()
# predict on evaluation set
eval_predictions = rf.predict(Xeval)
print("Predicted on evaluation set:", time.time() - start_time)

evaluation_report = classification_report(yeval, eval_predictions, target_names=label)
print(evaluation_report)
# save evaluation report as txt file
savereport(evaluation_report, "%s_evaluation_report.txt" % year)


