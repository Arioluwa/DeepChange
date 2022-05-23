import sys
import os
import joblib
import json
import time
import csv
import argparse
import pprint
import time
import datetime

# from yaml import load
from models.stclassifier import dLtae
import torch
import numpy as np
from models.stclassifier import dLtae
# from models.ltae import LTAE

import gdal, osr
from gdalconst import *

def class_mapping(y_label):
    """
    """
    unique_class = np.unique(y_label)
    nclass = len(unique_class)
    max_ylabel = np.unique(y_label)[-1]+1 #-- +1 to take into account the case where y=0	

    class_map = [0]*max_ylabel
    revert_class_map = unique_class.tolist()
    #-- Insert in class_map values from 1 to c, with c the number of classes
    n = nclass
    while n>0:
        insert_val = revert_class_map[n-1]
        class_map[insert_val] = n
        n = n-1	
    return class_map, revert_class_map

#--------------------------------------------------------------------------------------------------------
def read_class_map(file):
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        class_map = next(reader)
        revert_class_map = next(reader)
    class_map = [int(k) for k in class_map]
    revert_class_map = [int(k) for k in revert_class_map]
    return class_map, revert_class_map

def save_class_map(file, class_map, revert_class_map):
    with open(file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(class_map)
        writer.writerow(revert_class_map)

def load_npz(file):
    """
    """
    with np.load(file) as data:
        return data['X'], data['y']

def reshape_data(X, n_channel):
    """
    """
    X = X.reshape(X.shape[0], int(X.shape[1]/n_channel), n_channel)
    return X

def standardize_data(X, mean, std):
    return (X - mean) / std

def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]

######Read mean and std from the path as in the new train, using the basename from the path
# mean = np.loadtxt('../ltae/mean_std/source_mean.txt')
# std = np.loadtxt('../ltae/mean_std/source_std.txt')
##add n_channel as an argument

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

def date_positions(gfdate_path):
    with open(gfdate_path, "r") as f:
        date_list = f.readlines()
    date_list = [x.strip() for x in date_list]
    date_list = [datetime.datetime.strptime(x, "%Y%m%d").timetuple().tm_yday for x in date_list]
    date_ = [x for x in date_list]
    return date_

def main(args):
    
    pprint.pprint(vars(args))

    out_path = args.output
    model_file = args.model
    in_img = args.in_img
    ref_file = args.ref_file
    str_model = ['rf', 'LTAE']
    m = args.flag
    n_channel = args.n_channel
    case = args.case
    config = args.config
    date_ = args.date_
    
    image_name = in_img.split('/')
    image_name = image_name[-1].split('_')[0]
    device = args.device
    print("device=", device)
    
    mean = np.loadtxt('../ltae/mean_std/' + image_name +'_mean.txt')
    std = np.loadtxt('../ltae/mean_std/'+image_name +'_std.txt')

    out_map = out_path + '/' + image_name + '_' + str_model[m-1] + "_case_" + str(case)+ '_map' + '.tif'
    out_soft_pred = out_path + "/" + image_name + '_' +str_model[m-1]+ "_case_"+str(case)+'.npy'

    print("out_map: ", out_map)
    print("out_npy: ", out_soft_pred)
    if os.path.exists(out_map):
        print("out_map ",out_map,"already exists => exit")
        sys.exit("\n*** not overwriting out_map ***\n")


    # select model 
    if m==1: # RF
        model = joblib.load(model_file)
    else: # LTAE
        config = json.load(open(config))
        stat_dict = torch.load(model_file)['state_dict']
        model = dLtae(in_channels = config['in_channels'], n_head = config['n_head'], d_k= config['d_k'], n_neurons=config['n_neurons'], dropout=config['dropout'], d_model= config['d_model'],
                    mlp = config['mlp4'], T =config['T'], len_max_seq = config['len_max_seq'], 
                positions=date_positions(date_), return_att=False)
        
        print("device=", device)
        model = model.to(device)
        model = model.double()
        model.load_state_dict(stat_dict)
        
        model.eval() # disable your dropout and layer norm putting the model in evaluation mode.

    flag_del = False #-- deleting the training data
    class_map_file = '.'.join(ref_file.split('.')[0:-1])
    class_map_file = class_map_file + '_classMap.txt'

    print("class_map_file: ", class_map_file)
    # if not os.path.exists(class_map_file):
    #     X_train, y_train = load_npz(ref_file)
    #     class_map, revert_class_map = class_mapping(y_train)
    #     save_class_map(class_map_file, class_map, revert_class_map)
    #     flag_del = True
    # else:
    class_map, revert_class_map = read_class_map(class_map_file)

    print("class_map: ", class_map)
    print("revert_class_map: ", revert_class_map)

    if flag_del:
        del X_train
        del y_train

    # get image info about gps coordinates for origin plus size pixels
    image = gdal.Open(in_img, gdal.GA_ReadOnly) #, NUM_THREADS=8
    geotransform = image.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    spacingX = geotransform[1]
    spacingY = geotransform[5]
    r, c = image.RasterYSize, image.RasterXSize
    out_raster_SRS = osr.SpatialReference()
    out_raster_SRS.ImportFromWkt(image.GetProjectionRef())

    print("r=", r, " -- c=", c)
    print("originX: ", originX)
    print("originY: ", originY)
    print("spacingX: ", spacingX)
    print("spacingY: ", spacingY)
    print("geotransform: ", geotransform)

    #-- Set up the characteristics of the output image
    driver = gdal.GetDriverByName('GTiff')
    out_map_raster = driver.Create(out_map, c, r, 1, gdal.GDT_Byte)
    out_map_raster.SetGeoTransform([originX, spacingX, 0, originY, 0, spacingY])
    out_map_raster.SetProjection(out_raster_SRS.ExportToWkt())
    out_map_band = out_map_raster.GetRasterBand(1)


    size_areaX = 10980
    size_areaY = 10
    x_vec = list(range(int(c/size_areaX)))
    x_vec = [x*size_areaX for x in x_vec]
    y_vec = list(range(int(r/size_areaY)))
    y_vec = [y*size_areaY for y in y_vec]
    x_vec.append(c)
    y_vec.append(r)

    soft_prediction = []
    for x in range(len(x_vec)-1):
        for y in range(len(y_vec)-1):

            xy_top_left = (x_vec[x], y_vec[y])
            xy_bottom_right = (x_vec[x+1], y_vec[y+1])

            print('top_left=', xy_top_left, 'to bottom_right=', xy_bottom_right)

            #now loading associated data
            xoff = xy_top_left[0]
            yoff = xy_top_left[1]
            xsize = xy_bottom_right[0] - xy_top_left[0]
            ysize = xy_bottom_right[1] - xy_top_left[1]

            X_test = image.ReadAsArray(xoff=xoff, yoff=yoff, xsize=xsize, ysize=ysize) #, gdal.GDT_Float32

            X_test = X_test.transpose((1,2,0))
            sX = X_test.shape[0]
            sY = X_test.shape[1]
            X_test = X_test.reshape(X_test.shape[0]*X_test.shape[1], X_test.shape[2])

            if m == 2:
                # LTAE
                X_test = X_test.astype(np.int16)
                X_test = reshape_data(X_test, n_channel)
                X_test = standardize_data(X_test, mean, std) # confirm if this is needed
                X_test = torch.from_numpy(X_test)

                with torch.no_grad(): # disable the autograd engine (which you probably don't want during inference)
                    pred = model(X_test)
                
                if args.with_softmax:
                    soft_pred = torch.nn.functional.softmax(pred, dim=-1)
                    soft_pred = soft_pred.cpu().numpy().astype("float16")
                else:
                    soft_pred = pred
                    soft_pred = soft_pred.cpu().numpy().astype("float16")
                
                hard_pred = pred.argmax(dim=-1).cpu().numpy()
                hard_pred = [dict_[k] for k in hard_pred]
                hard_pred = np.array(hard_pred, dtype=np.uint8)
                del pred
                
            else:
                # RF
                soft_pred = model.predict_proba(X_test)
                hard_pred = soft_pred.argmax(axis=1)

                hard_pred = [revert_class_map[k] for k in hard_pred]
                hard_pred = np.array(hard_pred, dtype=np.uint8)
                soft_pred = soft_pred.astype("float16")
            
            ### generate the output classification map
            pred_array = hard_pred.reshape(sX, sY)
            out_map_band.WriteArray(pred_array, xoff=xoff, yoff=yoff)
            out_map_band.FlushCache()

            # append soft prediction and save
            soft_prediction.append(soft_pred)
            del soft_pred
            del hard_pred
            del pred_array
            del X_test

    proba_distribtion = np.concatenate(soft_prediction)
    np.save(out_soft_pred, proba_distribtion)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--flag', dest='flag', type=int, default=1, help='the Model type: 1 for RF, 2 for LTAE')
    parser.add_argument('--model', dest='model', type=str, help='model path')
    parser.add_argument('--ref_file', dest='ref_file', type=str, help='data npz file')
    parser.add_argument('--in_img', dest='in_img', type=str, help='input SITS image')
    parser.add_argument('--output', dest='output', type=str, help='output classification map')
    parser.add_argument('--case', dest='case', type=str, help='case name')
    parser.add_argument('--config', dest='config', type=str, help='Json config file')
    parser.add_argument('--with_softmax', dest='with_softmax', type=bool, default=False, help='flag for softmax')
    parser.add_argument('--n_channel', dest='n_channel', type=int, default=10, help='number of channels')
    parser.add_argument('--device', dest='device', type=str, default='cpu', help='device')
    parser.add_argument('--date_', dest='date_', type=str, help='Gapfilled dates')

    args = parser.parse_args()

    main(args)
    # python classification.py --model ../RF_model/models/rf_seed_0_case_1.pkl --ref_file ../../../data/theiaL2A_zip_img/output/2018/2018_SITS_data.npz --in_img ../../../data/theiaL2A_zip_img/output/2018/2018_GapFilled_Image.tif --output ../../../results/RF/classificationmap --flag 1 --case 1 --with_softmax True
    # python classification.py --model ../RF_model/models/rf_seed_0_case_3.pkl --ref_file ../../../data/theiaL2A_zip_img/output/2019/2019_SITS_data.npz --in_img ../../../data/theiaL2A_zip_img/output/2019/2019_GapFilled_Image.tif --output ../../../results/RF/classificationmap --flag 1 --case 3
    
#     python classificationviz.py --model ../../../results/RF/model/2019/Seed_0/rf_case_2.pkl --ref_file ../../../data/theiaL2A_zip_img/output/2019/2019_SITS_data.npz --in_img ../../../data/theiaL2A_zip_img/output/2019/2019_GapFilled_Image.tif --output ../../../results/RF/classificationmap --flag 1 --case 4

# python classificationviz.py --model ../../../results/ltae/model/2018/Seed_0/model.pth.tar --ref_file ../../../data/theiaL2A_zip_img/output/2018/2018_SITS_data.npz --in_img ../../../data/theiaL2A_zip_img/output/2018/2018_GapFilled_Image.tif --output ../../../results/RF/classificationmap --flag 2 --case 1 --config ../../../results/ltae/model/2018/Seed_0/conf.json --with_softmax True

# python classificationviz2.py --model ../../../results/ltae/model/2019/Seed_0/model.pth.tar --ref_file ../../../data/theiaL2A_zip_img/output/2019/2019_SITS_data.npz --in_img ../../../data/theiaL2A_zip_img/output/2019/2019_GapFilled_Image.tif --output ../../../results/ltae/classificationmap --flag 2 --case 2 --config ../../../results/ltae/model/2019/Seed_0/conf.json >> ../logs/log202205202055.txt; python classificationviz2.py --model ../../../results/ltae/model/2019/Seed_0/model.pth.tar --ref_file ../../../data/theiaL2A_zip_img/output/2018/2018_SITS_data.npz --in_img ../../../data/theiaL2A_zip_img/output/2018/2018_GapFilled_Image.tif --output ../../../results/ltae/classificationmap --flag 2 --case 2 --config ../../../results/ltae/model/2019/Seed_0/conf.json >> ../logs/log202205202056.txt