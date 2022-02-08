import os
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from RF_model.main import X
from utils.utils import load_npz, read_ids

_2018_SITS_data = "2018_SITS_data.npz"
_2019_SITS_data = "2019_SITS_data.npz"
_train_val_eval = "train_val_eval_rf.txt"


class RFmodel:
    def __init__(
        self,
        case: int,
        first_sits=_2018_SITS_data,
        second_sits=_2019_SITS_data,
        train_val_eval=_train_val_eval,
    ):
        super().__init__()
        self.case = case
        self.first_sits = first_sits
        self.second_sits = second_sits
        self.train_val_eval = train_val_eval

    # useful functions
    def load_set(self, set_name: str, total_set):
        """
        Load a set of data
        """
        self.train_ids, self.test_ids = read_ids(self.train_val_eval)
        
        if set_name == "train":
            ids = self.train_ids
        elif set_name == "val":
            ids = self.test_ids
        else:
            raise ValueError("Please choose a set between train and val")

        set = total_set[np.isin(total_set[:, -1], ids)]
        X = set[:, :-2]
        Y = set[:, -2]

        return X, Y

    def prepare_data(self):
        """
        Prepare data
        """
        X_s, Y_s, block_ids_s = load_npz(self.first_sits)
        X_t, Y_t, block_ids_t = load_npz(self.second_sits)

        # self.train_ids, self.test_ids = read_ids(self.train_val_eval)

        self.total_set_s = np.concatenate(
            (X_s, Y_s[:, None], block_ids_s[:, None]), axis=1
        )
        self.total_set_t = np.concatenate(
            (X_t, Y_t[:, None], block_ids_t[:, None]), axis=1
        )

        # Training set for target and source
        self.Xtrain_s, self.Ytrain_s = self.load_set("train", self.total_set_s)
        self.Xtrain_t, self.Ytrain_t = self.load_set("train", self.total_set_t)

        # Validation set for target and source
        self.Xval_s, self.Yval_s = self.load_set("val", self.total_set_s)
        self.Xval_t, self.Yval_t = self.load_set("val", self.total_set_t)

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
            pass
        else:
            raise ValueError("Please choose a case between 1 and 4")

    def train_model(self):
        """
        Train the model
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
        self.model.fit(self.Xtrain, self.Ytrain)

    def test_model(self):
        """
        Test the model
        """
        self.predictions = self.model.predict(self.Xval)
        self.report = classification_report(self.Yval, self.predictions)
        print(self.report)
        # save report as txt with case value
        with open(f"report_rf_{self.case}.txt", "w") as f:
            f.write(self.report)
            f.write("\n")
            f.write("OOB score:",self.model.oob_score_)
    # def get_predictions(self):
    #     """
    #     Get predictions
    #     """
    #     return self.predictions