import os
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import joblib
from utils.utils import load_npz, read_ids
import argparse


class RFmodel:
    def __init__(self, case:int, seed:int, sits, outdir):
        
        super().__init__()
        self.case = case
        self.seed = seed
        self.sits = sits
        self.outdir = outdir
        
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
        print("Preparing data.........")
        
        X, y, block_ids = load_npz(self.sits)
        print("Loading npz files done.........")
        
        total_ = np.concatenate((X, y[:, None], block_ids[:, None]), axis=1)
        print("Concatenating data done.........")
        print(total_.shape)
        
        del X
        del y
        del block_ids
        
        self.Xtrain, self.ytrain = self.load_set("train", total_)
        print("Loading train set done.........")
        print(self.Xtrain.shape)
        print(self.ytrain.shape)
        
        self.Xtest, self.ytest = self.load_set("test", total_)
        print("Loading test set done.........")
        print(self.Xtest.shape)
        print(self.ytest.shape)
        
        self.Xtrain, self.ytrain = shuffle(self.Xtrain, self.ytrain)
        
        del total_
        
        print("Preparing data completed.........")
        return self.Xtrain, self.ytrain


    def train_model(self):
        """
        """
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=25,
            n_jobs=10,
            max_features="sqrt",
            min_samples_leaf=10,
            random_state=42,
            oob_score=True,)
        
        print("Fitting model...")
        self.model.fit(self.Xtrain, self.ytrain)
        print("Fitting model done...")
        
        # if not os.path.exists("models"):
        #     os.makedirs("models")
        # save model as pickle with the name of the case
        if not os.path.exists(os.path.join(self.outdir, "Seed_{}".format(self.seed))):
            os.makedirs(os.path.join(self.outdir, "Seed_{}".format(self.seed)))
            
        joblib.dump(self.model, os.path.join(self.outdir, "Seed_{}".format(self.seed), "rf_case_" +str(self.case) + ".pkl"), compress=3)
        print("save model")
        
    def test_model(self):
        """
        """
        print("test model")
        if not os.path.exists(os.path.join(self.outdir, "Seed_{}".format(self.seed),"reports")):
                os.makedirs(os.path.join(self.outdir, "Seed_{}".format(self.seed), "reports"))
        label = ["Dense built-up area", "Diffuse built-up area", "Industrial and commercial areas", "Roads", "Oilseeds (Rapeseed)", "Straw cereals (Wheat, Triticale, Barley)", "Protein crops (Beans / Peas)", "Soy", "Sunflower", "Corn",  "Tubers/roots", "Grasslands", "Orchards and fruit growing", "Vineyards", "Hardwood forest", "Softwood forest", "Natural grasslands and pastures", "Woody moorlands", "Water"]
        
        self.Xtest, self.ytest = shuffle(self.Xtest, self.ytest)
        
        print("prediction... ")
        prediction = self.model.predict(self.Xtest)
        del self.Xtest
        print("prediction... done")
        
        report = classification_report(self.ytest, prediction, target_names=label, digits=4)
        
        confusion_ = confusion_matrix(self.ytest, prediction)
        
        with open(os.path.join(self.outdir, "Seed_{}".format(self.seed), "reports/rf_report_case_{}.txt".format(self.case)), "w") as f:
                f.write("Report: \n")
                f.write(report)
                # f.write("\n")
                f.close()
        
        with open(os.path.join(self.outdir, "Seed_{}".format(self.seed),"reports/confusion_{}.txt".format(self.case)), "w") as f:
                f.write("Confusion: \n")
                f.write(str(confusion_))
                # f.write("\n")
                f.close()
                
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--case', type=str, help='case')
    parser.add_argument('--sits', type=str, help='Path to the data folder')
    parser.add_argument('--seed', type=int, help='Seed')
    parser.add_argument('--outdir', type=str, help='Seed')
    
    args = parser.parse_args()
    start_time = time.time()

    model = RFmodel(args.case, args.seed, args.sits, args.outdir)
    
    model.prepare_data()
    
    if not os.path.exists(os.path.join(args.outdir, "Seed_{}".format(args.seed),"rf_case_"+str(args.case) + ".pkl")):
        model.train_model()
    else:
        print("model available..")
        model.model = joblib.load(os.path.join(args.outdir, "Seed_{}".format(args.seed),"rf_case_" +str(args.case) + ".pkl"))
    
    model.test_model()
    
    
    # python main2.py --case "1" --sits "../../../data/theiaL2A_zip_img/output/2019/2019_SITS_data.npz" --seed 0 --outdir ../../../results/ltae/model/2018
    
    # /share/projects/erasmus/deepchange/codebase/DeepChange/RF_model$ python main2.py --case 1 --sits "../../../data/theiaL2A_zip_img/output/2018/2018_SITS_data.npz" --seed 0 --outdir ../../../results/ltae/model/2018 && python main2.py --case 2 --sits "../../../data/theiaL2A_zip_img/output/2019/2019_SITS_data.npz" --seed 0 --outdir ../../../results/ltae/model/2019