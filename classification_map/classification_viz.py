import sys
import os
import joblib
import time
import csv
import optparse

import numpy as np

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
	# parser.add_option("-f", "--flag", dest="flag", action="store", type="int", help="the Model type: 1 for CNN, 2 for RF, 3 for RNN.", default="2")
	parser.add_option("-m", "--model", dest="model", action="store", type="string", help="The model algorithm.")
	parser.add_option("-t", "--ref", dest="ref_file", action="store", type="string", help="The reference data.")
	parser.add_option("-i", "--input", dest="in_img", action="store", type="string", help="The image to classify.")
	parser.add_option("-o", "--output", dest="output", action="store", type="string", help="The directory of model and statistics.")
	# parser.add_option("-b", "--nchannels", dest="nchannels", action="store", type="int", help="The number of channels in the image")
	(options, args) = parser.parse_args()


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

#-----------------------------------------------------------------------
def reshape_data(X, nchannels):
	"""
		Reshaping (feature format (3 bands): d1.b1 d1.b2 d1.b3 d2.b1 d2.b2 d2.b3 ...)
		INPUT:
			-X: original feature vector ()
			-feature_strategy: used features (options: SB, NDVI, SB3feat)
			-nchannels: number of channels
		OUTPUT:
			-new_X: data in the good format for Keras models
	"""
	
	return X.reshape(X.shape[0],int(X.shape[1]/nchannels),nchannels) # x: row, y: time, z: band

#-----------------------------------------------------------------------
def read_minMaxVal(file):
	
	with open(file, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		min_per = next(reader)
		max_per = next(reader)
	min_per = [float(k) for k in min_per]
	min_per = np.array(min_per)
	max_per = [float(k) for k in max_per]
	max_per = np.array(max_per)
	return min_per, max_per

#-----------------------------------------------------------------------
def save_minMaxVal(file, min_per, max_per):
	
	with open(file, 'w') as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerow(min_per)
		writer.writerow(max_per)

#-----------------------------------------------------------------------
def computingMinMax(X, per=2):
	min_per = np.percentile(X, per, axis=(0,1))
	max_per = np.percentile(X, 100-per, axis=(0,1))
	return min_per, max_per

#-----------------------------------------------------------------------
def normalizingData(X, min_per, max_per):
	return (X-min_per)/(max_per-min_per)

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

nchannels = 10

out_path = options.output
model_file = options.model
in_img = options.in_img # -i 54HWE_img.tif
ref_file = options.ref_file #-t 54HWE_train.sqlite

model_name = model_file.split('/')[-1]
model_name = model_name.split('.')[0]

image_name = in_img.split('/')
image_name = image_name[-1].split('_')[0]

out_map = out_path + '/' + image_name + '_' + model_name + '_map' + '.tif'
print("out_map: ", out_map)
if os.path.exists(out_map):
	print("out_map ",out_map,"already exists => exit")
	sys.exit("\n*** not overwriting out_map ***\n")

out_confmap = out_path + '/' + image_name + '_' + model_name + '_confmap' + '.tif'


model = joblib.load(model_file)

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


#convert gps corners into image (x,y)
def gps_2_image_xy(x,y):
	return (x-originX)/spacingX,(y-originY)/spacingY
def gps_2_image_p(point):
	return gps_2_image_xy(point[0],point[1])

size_areaX = 10980
size_areaY = 500
x_vec = list(range(int(c/size_areaX)))
x_vec = [x*size_areaX for x in x_vec]
y_vec = list(range(int(r/size_areaY)))
y_vec = [y*size_areaY for y in y_vec]
x_vec.append(c)
y_vec.append(r)

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

		start_time = time.time()
		p_img = model.predict_proba(X_test)
		print("Prediction: ", time.time()-start_time)

		y_test = p_img.argmax(axis=1)
		y_prob = p_img.max(axis=1)

		y_test = [revert_class_map[k] for k in y_test]
		y_test = np.array(y_test, dtype=np.uint8)
		pred_array = y_test.reshape(sX,sY)
		
		start_time = time.time()
		out_map_band.WriteArray(pred_array, xoff=xoff, yoff=yoff)
		print("Writing array: ", time.time()-start_time)

		start_time = time.time()
		out_map_band.FlushCache()
		print("Writing disk: ", time.time()-start_time)

    

