import os
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import joblib
from utils.utils import load_npz, read_ids


class RFmodel:
    def __init__(self, case:int, seed:int, sits):
        
        super().__init__()
        self.case = case
        self.seed = seed
        self.sits = sits
        
    def load_set(self, partition:str, total_set):
        """
        """
        #read ids
        train_id, val_id, test_id = read_ids(self.seed)
        
        if partition == 'train':
            ids = train_id
        elif partition == 'test':
            ids = test_id
        else:
            raise ValueError('Please choose a partition train and test')
        
        
        data_ = total_set[np.isin(total_set[:, -1], ids)]

        X = data_[:, :-2]
        y = data_[:, -2]

        return X, y
        
                
    def prepare_data(self):
        """
        
        """
        X, y, block_ids = load_npz(self.sits)
        
        total_ = np.concatenate((X, y[:, None], block_ids[:, None]), axis=1)
        
        self.Xtrain, self.ytrain = self.load_set("train", total_)
        
        self.Xtest, self.ytest = self.load_set("test", total_)
        
        self.Xtrain, self.ytrain = shuffle(self.Xtrain, self.ytrain)


    def train_model(self):
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=25,
            n_jobs=10,
            max_features="sqrt",
            min_samples_leaf=10,
            random_state=42,
            oob_score=True)
        
        self.model.fit(self.Xtrain, self.ytrain)
        # if not os.path.exists("models"):
        #     os.makedirs("models")
        # save model as pickle with the name of the case
        joblib.dump(self.model, "models/rf_seed_" + str(self.seed)+"_case_" +str(self.case) + ".pkl", compress=3)
        
    def test_model(self):
        """
        """
        if not os.path.exists("reports"):
                os.makedirs("reports")
        label = ["Dense built-up area", "Diffuse built-up area", "Industrial and commercial areas", "Roads", "Oilseeds (Rapeseed)", "Straw cereals (Wheat, Triticale, Barley)", "Protein crops (Beans / Peas)", "Soy", "Sunflower", "Corn",  "Tubers/roots", "Grasslands", "Orchards and fruit growing", "Vineyards", "Hardwood forest", "Softwood forest", "Natural grasslands and pastures", "Woody moorlands", "Water"]
        
        self.Xtest, self.ytest = shuffle(self.Xtest, self.ytest)
        
        prediction = self.model.predict(self.Xtest)
        
        report = classification_report(self.ytest, prediction, target_names=label, digits=4)
        
        confusion_ = confusion_matrix(self.ytest, prediction)
        
        with open("reports/rf_report_case_" + str(self.case)+ "_seed_" + str(self.seed) + ".txt", "w") as f:
                f.write("Report: \n")
                f.write(report)
                f.write("\n")
                f.close()
        
        with open("reports/confusion" + str(self.case)+ "_seed_" + str(self.seed) + ".txt", "w") as f:
                f.write("Confusion: \n")
                f.write(confusion_)
                f.write("\n")
                f.close()
                
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--case', type=str, help='case')
    parser.add_argument('--sits', type=str, help='Path to the data folder')
    parser.add_argument('--seed', type=int, help='Seed')
    
    args = parser.parse_args()
    start_time = time.time()

    model = RFmodel(args.case, args.seed, args.sits)
    
    model.prepare_data()
    
    if not os.path.exists("models/rf_seed_" + str(args.seed) + "_case_"+str(args.case) + ".pkl"):
        model.train_model()
    else:
        print("model available..")
        model.model = joblib.load("models/rf_seed_" + str(args.seed) + "_case_" +str(args.case) + ".pkl")
    
    model.test_model()