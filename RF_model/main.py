import os
<<<<<<< HEAD
import argparse
=======
>>>>>>> 1d364dbd00097303b2fd14b44f76d0b1770c0eed
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
<<<<<<< HEAD
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
rf = RandomForestClassifier(n_estimators=100, max_depth = 25, n_jobs=20, max_features= "sqrt", min_samples_leaf=10, random_state=42, oob_score=True)

# Fit the model
rf = rf.fit(Xtrain, ytrain)
print("Fitted model:", time.time() - start_time)
print("OOB Error: ",rf.oob_score_)
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


=======
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
import joblib
from utils.utils import load_npz, read_ids

_2018_SITS_data = "../../../data/theiaL2A_zip_img/output/2018/2018_SITS_data.npz"
_2019_SITS_data = "../../../data/theiaL2A_zip_img/output/2019/2019_SITS_data.npz"
_train_val_eval = "train_val_eval_rf.txt"


class RFmodel:
    def __init__(
        self,
        case: int,
        source_sits=_2018_SITS_data,
        target_sits=_2019_SITS_data,
        train_val_eval=_train_val_eval,
    ):
        super().__init__()
        self.case = case
        self.source_sits = source_sits
        self.target_sits = target_sits
        self.train_val_eval = train_val_eval

    # useful functions
    def load_set(self, set_name: str, total_set):
        """
        Load a set of data:
        to be used in prepare_data(), it organise dataset according to the set_name based block_ids
        - set_name: train or test (reads from the block_ids using read_ids() utils) 
        - total_set: the concatenated set of data for source or target
        - returns X and Y:
            X is the data, Y is the label
        """
        self.train_ids, self.test_ids = read_ids(self.train_val_eval)

        if set_name == "train":
            ids = self.train_ids
        elif set_name == "test":
            ids = self.test_ids
        else:
            raise ValueError("Please choose a set between train and test")

        set = total_set[np.isin(total_set[:, -1], ids)]
        X = set[:, :-2]
        Y = set[:, -2]
        
        return X, Y

    def prepare_data(self):
        """
        Prepare data:
        - load data from npz files(load_npz() in utils): returns X, Y and block_ids
        - concatenate the data from source and target [X, Y, block_ids] == total_set
        - split the data into train and test sets for source and target
        - returns Xtrain and Ytrain base on CASE value
        CASE: 
        - 1: Train on Soucre and target, test on source and target
        - 2: Train on Source only, test on Source and target
        - 3: Train on Target only, test on Source and target
        - 4: Train on Source only, test on Source only
        - 5: Train on Target only, test on Target only
        """
        print("Preparing data.........")

        X_s, Y_s, block_ids_s = load_npz(self.source_sits)
        X_t, Y_t, block_ids_t = load_npz(self.target_sits)
        print("Loading npz files done.........")

        # self.train_ids, self.test_ids = read_ids(self.train_val_eval)

        self.total_set_s = np.concatenate(
            (X_s, Y_s[:, None], block_ids_s[:, None]), axis=1
        )
        self.total_set_t = np.concatenate(
            (X_t, Y_t[:, None], block_ids_t[:, None]), axis=1
        )
        print("Concatenating data done.........")

        # Training set for target and source
        self.Xtrain_s, self.Ytrain_s = self.load_set("train", self.total_set_s)
        self.Xtrain_t, self.Ytrain_t = self.load_set("train", self.total_set_t)
        print("Loading train set done.........")

        # Validation set for target and source
        self.Xtest_s, self.Ytest_s = self.load_set("test", self.total_set_s)
        self.Xtest_t, self.Ytest_t = self.load_set("test", self.total_set_t)
        print("Loading test set done.........")

        if self.case == 1:
            # concatenate training set for target and source
            self.Xtrain = np.concatenate((self.Xtrain_s, self.Xtrain_t), axis=0)
            self.Ytrain = np.concatenate((self.Ytrain_s, self.Ytrain_t), axis=0)

        elif self.case == 2:
            # Xtrain is the training set for source only
            self.Xtrain = self.Xtrain_s
            self.Ytrain = self.Ytrain_s

        elif self.case == 3:
            # Xtrain is the training set for target only
            self.Xtrain = self.Xtrain_t
            self.Ytrain = self.Ytrain_t

        elif self.case == 4:
            self.Xtrain = self.Xtrain_s
            self.Ytrain = self.Ytrain_s

        elif self.case == 5:
            self.Xtrain = self.Xtrain_t
            self.Ytrain = self.Ytrain_t

        else:
            raise ValueError("Please choose a case between 1 and 5")
        
        self.Xtrain, self.Ytrain = shuffle(self.Xtrain, self.Ytrain)
        print("Preparing data completed.........")
        return self.Xtrain, self.Ytrain

    def train_model(self):
        """
        Train the model
        - train the model with Xtrain and Ytrain
        - save the model to a file
        """
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=25,
            n_jobs=20,
            max_features="sqrt",
            min_samples_leaf=10,
            random_state=42,
            oob_score=True,
        )
        start_time = time.time()
        print("Training model.........")
        self.model.fit(self.Xtrain, self.Ytrain)
        print("Training model done.........")
        print("Training time: %s minutes" % ((time.time() - start_time)/60))
        # create folder to save model if it doesn't exist
        if not os.path.exists("models"):
            os.makedirs("models")
        # save model as pickle with the name of the case
        joblib.dump(self.model, "models/rf_model_" + str(self.case) + ".pkl", compress=3)
        print("Model saved.........")

    # def savereport(rep):
    #     with open("reports/report.txt", "a") as f:
    #         f.write(rep)
    def test_model(self):
        """
        Test the model
        - predict the labels for Xval
        - print and write the classification report to txt file
        """
        if not os.path.exists("reports"):
                os.makedirs("reports")
        label = ["Dense built-up area", "Diffuse built-up area", "Industrial and commercial areas", "Roads", "Oilseeds (Rapeseed)", "Straw cereals (Wheat, Triticale, Barley)", "Protein crops (Beans / Peas)", "Soy", "Sunflower", "Corn",  "Tubers/roots", "Grasslands", "Orchards and fruit growing", "Vineyards", "Hardwood forest", "Softwood forest", "Natural grasslands and pastures", "Woody moorlands", "Water"]
        print("Testing model.........")
        start_time = time.time()
        self.Xtest_s, self.Ytest_s = shuffle(self.Xtest_s, self.Ytest_s)
        if self.case == 4:
            """Test the model on the source only"""
            self.Ypred = self.model.predict(self.Xtest_s)
            self.report = classification_report(self.Ytest_s, self.Ypred, target_names=label)
            print(self.report)
        
            # save report as txt with case value
            with open("reports/rf_report_" + str(self.case) + ".txt", "w") as f:
                f.write(self.report)
                f.write("\n")
                f.write("OOB score: " + str(self.model.oob_score_))
                f.close()

        elif self.case == 5:
            """Test the model on the target only"""
            self.Ypred = self.model.predict(self.Xtest_t)
            self.report = classification_report(self.Ytest_t, self.Ypred, target_names=label)
            print(self.report)

            # save report as txt with case value
            with open("reports/rf_report_" + str(self.case) + ".txt", "w") as f:
                f.write(self.report)
                f.write("\n")
                f.write("OOB score: " + str(self.model.oob_score_))
                f.close()
        else:
            """Test the model on the source and target"""
            self.predictions_s = self.model.predict(self.Xtest_s)
            self.predictions_t = self.model.predict(self.Xtest_t)
            self.report_t = classification_report(self.Ytest_t, self.predictions_t, target_names=label)
            self.report_s = classification_report(self.Ytest_s, self.predictions_s, target_names=label)
            print("Target report: \n", self.report_t)
            print("Source report: \n", self.report_s)
            print("Writing report to txt file.........")
        
            # save report as txt with case value
            with open("reports/rf_report_" + str(self.case) + ".txt", "w") as f:
                f.write(self.report_s)
                f.write("\n")
                f.write(self.report_t)
                f.write("\n")
                f.write("OOB score: " + str(self.model.oob_score_))
                f.close()

        print("Writing report to txt file done.........")
        print("Testing model done.........")
        print("Testing time: %s minutes" % ((time.time() - start_time)/60))       
    

    # def get_predictions(self):
    #     """
    #     Get predictions
    #     """
    #     return self.predictions


if __name__ == "__main__":
    # case 1: train on both source and target
    # case 2: train on source only
    # case 3: train on target only
    # case 4: train on both source and target, but use validation set for target only
    # compute time in minutes
    start_time = time.time()
    case_ = 5
    model = RFmodel(case=case_)
    model.prepare_data()
    # check if case model file exits, skip training if it does
    if not os.path.exists("models/rf_model_" + str(case_) + ".pkl"):
        model.train_model()
    else:
        model.model = joblib.load("models/rf_model_" + str(case_) + ".pkl")
    
    model.test_model()
    # print time in minutes
    print("Total time taken: --- %s minutes ---" % ((time.time() - start_time) / 60))
>>>>>>> 1d364dbd00097303b2fd14b44f76d0b1770c0eed
