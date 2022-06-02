import os
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
import joblib
import argparse
import glob
import sys

def test_model(config):
        """
        Test the trained RF model on the 
        """
        # print("test model")
        # check the report folder already exist
        if not os.path.exists(os.path.join(config['model_folder'], "reports")):
                os.makedirs(os.path.join(config['model_folder'], "reports"))
        label = ["Dense built-up area", "Diffuse built-up area", "Industrial and commercial areas", "Roads", "Oilseeds (Rapeseed)", "Straw cereals (Wheat, Triticale, Barley)", "Protein crops (Beans / Peas)", "Soy", "Sunflower", "Corn",  "Tubers/roots", "Grasslands", "Orchards and fruit growing", "Vineyards", "Hardwood forest", "Softwood forest", "Natural grasslands and pastures", "Woody moorlands", "Water"]
        
        # This is need to revert the y from [0to18] to [1to23]
        dict_ = {0:1, 
                1:2, 
                2:3, 
                3:4, 
                4:5, 
                5:6, 
                6:7,
                7:8,
                8:9,
                9:10,
                10:12,
                11:13,
                12:14,
                13:15,
                14:16,
                15:17,
                16:18,
                17:19,
                18:23}
        
        # read the test dataset
        sits_data = glob.glob(os.path.join(config['dataset_folder'], '*.npz'))[0]
        
        print('read dataset')
        with np.load(sits_data) as f:
            X = f['X']
            y = f['y']
        
        y = [dict_[k] for k in y]
        X, y = shuffle(X, y)
        
        ## read model file .pkl
        
        model_file = glob.glob(os.path.join(config['model_folder'], '*.pkl'))[0]
        model = joblib.load(model_file)
        
        print("prediction... ")
        prediction = model.predict(X)
        print("prediction... done")
        
        report = classification_report(y, prediction, target_names=label, digits=4)
        
        # confusion_ = confusion_matrix(self.ytest, prediction)
        kappa = cohen_kappa_score(y, prediction)
        
        # Year of the test dataset, to be used to save the metrics
        dataset_name = os.path.basename(config['dataset_folder']).split('.')[0]
        
        #save metrics
        with open(os.path.join(config['model_folder'], "reports/{}_report.txt".format(dataset_name)), "w") as f:
                f.write(report)
                f.close()
        
        with open(os.path.join(config['model_folder'], "reports/{}_kappa.txt".format(dataset_name)), "w") as f:
                f.write(str(kappa))
                f.close()
                

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_folder', '-m', type=str, help='Path to the model file .Pth.')
    parser.add_argument('--dataset_folder', '-d', type=str, help='Path to the dataset folder.')
    
    config = parser.parse_args()
    config = vars(config)
    
    test_model(config)
    
    # python test.py -m ../../../results/RF/model/2019/third/Seed_0 -d ../../../data/theiaL2A_zip_img/output/2018; python test.py -m ../../../results/RF/model/2019/third/Seed_0 -d ../../../data/theiaL2A_zip_img/output/2019; python test.py -m ../../../results/RF/model/2019/second/Seed_0 -d ../../../data/theiaL2A_zip_img/output/2018; python test.py -m ../../../results/RF/model/2019/second/Seed_0 -d ../../../data/theiaL2A_zip_img/output/2019; python test.py -m ../../../results/RF/model/2019/fourth/Seed_0 -d ../../../data/theiaL2A_zip_img/output/2018; python test.py -m ../../../results/RF/model/2019/fourth/Seed_0 -d ../../../data/theiaL2A_zip_img/output/2019
    
    # python test.py -m ../../../results/RF/model/2018/third/Seed_0 -d ../../../data/theiaL2A_zip_img/output/2018; python test.py -m ../../../results/RF/model/2018/third/Seed_0 -d ../../../data/theiaL2A_zip_img/output/2019; python test.py -m ../../../results/RF/model/2018/second/Seed_0 -d ../../../data/theiaL2A_zip_img/output/2018; python test.py -m ../../../results/RF/model/2018/second/Seed_0 -d ../../../data/theiaL2A_zip_img/output/2019; python test.py -m ../../../results/RF/model/2018/fourth/Seed_0 -d ../../../data/theiaL2A_zip_img/output/2018; python test.py -m ../../../results/RF/model/2018/fourth/Seed_0 -d ../../../data/theiaL2A_zip_img/output/2019