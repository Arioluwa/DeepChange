model_file= '../RF_model/models/rf_model_3.pkl'
ref_file = '../../../data/theiaL2A_zip_img/output/2018/2018_SITS_data.npz'
in_img = '../../../data/theiaL2A_zip_img/output/2018/2018_GapFilled_Image.tif'
out_path = '../../../results/RF/simliarity_measure'

import sys
import os
import joblib
import time
import csv
import optparse

import numpy as np
try:
    import gdal, osr
    from gdalconst import *
except:
    from osgeo import gdal, osr
    from osgeo.gdalconst import *

def load_npz(file_path):
    """
    Load data from a .npz file
    """
    with np.load(file_path) as data:
        X = data["X"]
        y = data["y"]
       
    return X, y

model_name = model_file.split('/')[-1]
model_name = model_name.split('.')[0]

image_name = in_img.split('/')
image_name = image_name[-1].split('_')[0]

start_time_ = time.time()
start_time = time.time()
model = joblib.load(model_file)
print("Read model: ", time.time()-start_time)

start_time = time.time()
image = gdal.Open(in_img, gdal.GA_ReadOnly) #, NUM_THREADS=8
geotransform = image.GetGeoTransform()
originX = geotransform[0]
originY = geotransform[3]
spacingX = geotransform[1]
spacingY = geotransform[5]
r, c = image.RasterYSize, image.RasterXSize
# r = 1000
out_raster_SRS = osr.SpatialReference()
out_raster_SRS.ImportFromWkt(image.GetProjectionRef())

size_areaX = 10980
size_areaY = 200
x_vec = list(range(int(c/size_areaX)))
x_vec = [x*size_areaX for x in x_vec]
y_vec = list(range(int(r/size_areaY)))
y_vec = [y*size_areaY for y in y_vec]
x_vec.append(c)
y_vec.append(r)
print("Initiation: ", time.time() - start_time)

proba_dist = []
start_time_ri = time.time()
for x in range(len(x_vec)-1):
	for y in range(len(y_vec)-1):
        	
		xy_top_left = (x_vec[x],y_vec[y])
		xy_bottom_right = (x_vec[x+1],y_vec[y+1])
		
		print('top_left=',xy_top_left,' to bottom_right=',xy_bottom_right)

		#now loading associated data
		xoff = xy_top_left[0]
		yoff = xy_top_left[1]
		xsize = xy_bottom_right[0]-xy_top_left[0]
		ysize = xy_bottom_right[1]-xy_top_left[1]
		start_time = time.time()
		X_test = image.ReadAsArray(xoff=xoff, yoff=yoff, xsize=xsize, ysize=ysize) #, gdal.GDT_Float32
		print("Reading: ", time.time()-start_time)
        
        #-- reshape the cube in a column vector
		X_test = X_test.transpose((1,2,0))
		sX = X_test.shape[0]
		sY = X_test.shape[1]
		X_test = X_test.reshape(X_test.shape[0]*X_test.shape[1],X_test.shape[2])
		print("Reading image in loop: ", time.time()-start_time_ri)

		start_time = time.time()
		p_img = model.predict_proba(X_test)
		print("predict: ", time.time()-start_time)
    # proba_dist.append(p_img)
		start_time = time.time()
		proba_dist.append(p_img)
		print("append: ", time.time()-start_time)
start_time = time.time()
probability_distribution = np.concatenate(proba_dist)
print("Concatenate: ", time.time()-start_time)
print('done........')
start_time = time.time()
np.save(os.path.join(out_path, image_name + '_' + model_name + '.npy'), probability_distribution)
print("Saving npy: ", time.time()-start_time)
print("Total taken: ", time.time()-start_time_)