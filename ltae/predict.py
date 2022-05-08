import sys
import os
import joblib
import json
import time
import csv
import optparse
import torch
import numpy as np
from models.stclassifier import dLtae
# from models.ltae import LTAE

import gdal, osr
from gdalconst import *

class OptionParser (optparse.OptionParser):

	def check_required(self, opt):
		option = self.get_option(opt)
		# Assumes the option's 'default' is set to None!
		if getattr(self.values, option.dest) is None:
			self.error("%s option not supplied" % option)


# ==================
# parse command line
# ==================
if len(sys.argv) == 1:
	prog = os.path.basename(sys.argv[0])
	print ("       " + sys.argv[0] + " [option]")
	print ("     Aide: ", prog, "--help")
	print ("         ou: ", prog, "  -h")
	print ("example command: python classif_francoisComputer.py -f 2 -m model -t 54HWE_train.sqlite  -i 54HWE_img.tif  -o map")
	# print "or"
	# print ""
else:
	usage = " usage: %prog [options] "
	parser = OptionParser(usage=usage)
	parser.add_option("-f", "--flag", dest="flag", action="store", type="int", help="the Model type: 1 for RF, 2 for LTAE.", default="2")
	parser.add_option("-m", "--model", dest="model", action="store", type="string", help="The model algorithm.")
	parser.add_option("-r", "--ref", dest="ref_file", action="store", type="string", help="The reference data.")
	parser.add_option("-i", "--input", dest="in_img", action="store", type="string", help="The image to classify.")
	parser.add_option("-o", "--output", dest="output", action="store", type="string", help="The directory of model and statistics.")
	parser.add_option("-c", "--case", dest="case", action="store", type="string", help="Json config file path")
	parser.add_option("-g", "--config", dest="config", action="store", type="string", help="Json config file path")
	parser.add_option("-d", "--device", dest="device", action="store", type="string", help="Json config file path")
	# parser.add_option("-b", "--nchannels", dest="nchannels", action="store", type="int", help="The number of channels in the image")
	(options, args) = parser.parse_args()

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

#-----------------------------------------------------------------------
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
def load_npz(file_path):
    """
    Load data from a .npz file
    """
    with np.load(file_path) as data:
        X = data["X"]
        y = data["y"]
       
    return X, y

def reshape_data(X, n_channel):
    """
    Reshape data to fit the model
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
# def read_mean_std()

# rewrite a function

mean = np.loadtxt('../ltae/mean_std/source_mean.txt')
std = np.loadtxt('../ltae/mean_std/source_std.txt')

nchannels = 10

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

# out_path = options.output
# model_file = options.model
# in_img = options.in_img # -i 54HWE_img.tif
# ref_file = options.ref_file #-t 54HWE_train.sqlite


out_path = options.output
model_file = options.model
in_img = options.in_img
ref_file = options.ref_file
str_model =['rf', 'LTAE']
m = options.flag
case = options.case
config = options.config

if case ==2:
    mean = np.loadtxt('../ltae/mean_std/source_mean.txt')
    std = np.loadtxt('../ltae/mean_std/source_std.txt')
elif case == 3:
    mean = np.loadtxt('../ltae/mean_std/target_mean.txt')
    std = np.loadtxt('../ltae/mean_std/target_std.txt')
else:
    mean = np.loadtxt('../ltae/mean_std/source_mean.txt')
    std = np.loadtxt('../ltae/mean_std/source_std.txt')

m = options.flag
image_name = in_img.split('/')
image_name = image_name[-1].split('_')[0]
device = options.device
print("device=", device)

out_map = out_path + '/' + image_name + '_' + str_model[m-1] + "_case_" + str(case)+ '_map' + '.tif'
out_npy = out_path + "/" + image_name + '_' +str_model[m-1]+ "_case_"+str(case)+'.npy'
print("out_map: ", out_map)
print("out_npy: ", out_npy)
if os.path.exists(out_map):
	print("out_map ",out_map,"already exists => exit")
	sys.exit("\n*** not overwriting out_map ***\n")

# out_confmap = out_path + '/' + image_name + '_' + str_model[m-1] + '_proba' + '.tif'


if m==1:
    model = joblib.load(model_file)
else:
    config = json.load(open(config))
    stat_dict = torch.load(model_file)['state_dict']
    model = dLtae(in_channels = config['in_channels'], n_head = config['n_head'], d_k= config['d_k'], n_neurons=config['n_neurons'], dropout=config['dropout'], d_model= config['d_model'],
                 mlp = config['mlp4'], T =config['T'], len_max_seq = config['len_max_seq'], 
              positions=None, return_att=False)
    
    print("device=", device)
    model = model.to(device)
    model = model.double()
    model.load_state_dict(stat_dict)
    
    model.eval() # disable your dropout and layer norm putting the model in evaluation mode.


flag_del = False #-- deleting the training data
class_map_file = '.'.join(ref_file.split('.')[0:-1])
class_map_file = class_map_file + '_classMap.txt'
print("class_map_file: ", class_map_file)
if not os.path.exists(class_map_file): 
	X_train, y_train = load_npz(ref_file)
	class_map, revert_class_map = class_mapping(y_train)
	save_class_map(class_map_file, class_map, revert_class_map)
	flag_del = True
else:
	class_map, revert_class_map = read_class_map(class_map_file)

print("class_map: ", class_map)
print("revert_class_map: ", revert_class_map)

if flag_del:
	del X_train
	del y_train

#get image info about gps coordinates for origin plus size pixels
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

# out_confmap_raster = driver.Create(out_confmap, c, r, 1, gdal.GDT_Float32)
# out_confmap_raster.SetGeoTransform([originX, spacingX, 0, originY, 0, spacingY])
# out_confmap_raster.SetProjection(out_raster_SRS.ExportToWkt())
# out_confmap_band = out_confmap_raster.GetRasterBand(1)


size_areaX = 10980
size_areaY = 50
x_vec = list(range(int(c/size_areaX)))
x_vec = [x*size_areaX for x in x_vec]
y_vec = list(range(int(r/size_areaY)))
y_vec = [y*size_areaY for y in y_vec]
x_vec.append(c)
y_vec.append(r)

soft_prediction = []
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
		# X_test = X_test.astype("int16")
        
		if m == 2:
			X_test = X_test.astype("int16")
			X_test = reshape_data(X_test, nchannels)
			X_test = standardize_data(X_test, mean, std)
			X_test = torch.from_numpy(X_test)
			# X_test = recursive_todevice(X_test, device)
			with torch.no_grad(): # disable the autograd engine (which you probably don't want during inference)
				prediction = model(X_test)
			# print("soft max....")
			soft_pred = torch.nn.functional.softmax(prediction, dim=-1)

			soft_pred = soft_pred.cpu().numpy().astype("float16")
			hard_pred = prediction.argmax(dim=1).cpu().numpy()
			hard_pred = [dict_[k] for k in hard_pred]
			del prediction
			del X_test

		else:
			soft_pred = model.predict_proba(X_test)
			hard_pred = soft_pred.argmax(axis=1)
			hard_pred = [revert_class_map[k] for k in hard_pred]
		# break            
		hard_pred = np.array(hard_pred, dtype=np.uint8)    
		pred_array = hard_pred.reshape(sX,sY)

		start_time = time.time()
		out_map_band.WriteArray(pred_array, xoff=xoff, yoff=yoff)
		out_map_band.FlushCache()
		print("write map: ", time.time()-start_time)

		start_time = time.time()
		soft_prediction.append(soft_pred)
		print("Writing array: ", time.time()-start_time)
		del soft_pred
		del pred_array
		del hard_pred
		del X_test
        

probability_distribution = np.concatenate(soft_prediction)

np.save(out_npy, probability_distribution)