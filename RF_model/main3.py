import os
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
import joblib
from utils.utils import load_npz, read_ids

_2018_SITS_data = "../../../data/theiaL2A_zip_img/output/2018/2018_SITS_data.npz"
_2019_SITS_data = "../../../data/theiaL2A_zip_img/output/2019/2019_SITS_data.npz"
_train_val_eval = "train_val_eval_rf.txt"

np.random.seed(0)
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

        else:
            raise ValueError("Please choose a case between 1 and 3")
        
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
        
        self.predictions_s = self.model.predict(self.Xtest_s)
        self.predictions_t = self.model.predict(self.Xtest_t)
        self.report_s = classification_report(self.Ytest_s, self.predictions_s, target_names=label)
        self.report_t = classification_report(self.Ytest_t, self.predictions_t, target_names=label)
        print("Target report: \n", self.report_t)
        print("Source report: \n", self.report_s)
        print("Writing report to txt file.........")
        # self.Xtrain, self.Ytrain = shuffle(self.Xtrain, self.Ytrain)
        unique, count = np.unique(self.Ytrain, return_counts = True)
        n_sample = dict(zip(label,count))
        print("Number of samples in each class: \n", n_sample)
            # save report as txt with case value
        with open("reports/rf_report_" + str(self.case) + ".txt", "w") as f:
                f.write("Source report: \n")
                f.write(self.report_s)
                f.write("\n")
                f.write("Target report: \n")
                f.write(self.report_t)
                f.write("\n")
                f.write("OOB score: " + str(self.model.oob_score_))
                f.write("\n")
                f.write("Number of samples in each class: \n")
                f.write(str(n_sample))
                f.close()

        print("Writing report to txt file done.........")
        print("Testing model done.........")
        print("Testing time: %s minutes" % ((time.time() - start_time)/60))       
    




if __name__ == "__main__":
    # case 1: train on both source and target
    # case 2: train on source only
    # case 3: train on target only
    # case 4: train on both source and target, but use validation set for target only
    # compute time in minutes
    start_time = time.time()
    case_ = 1
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

