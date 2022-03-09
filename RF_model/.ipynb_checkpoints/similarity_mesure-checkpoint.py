import os
import numpy as np
import rasterio
import argparse
import time

parser = argparse.ArgumentParser()

parser.add_argument('--source', '-s', type=str, dest='source_', action='store', required=True)
parser.add_argument('--target', '-t', type=str, dest='target_', action='store', required=True)
parser.add_argument('--image', '-i', type=str, dest='image_', action='store', required=True)
parser.add_argument('--outdir', '-o', type=str, dest='outdir_', action='store', required=True)
opt = parser.parse_args()

source_ = opt.source_
target_ = opt.target_
image_ = opt.image_
outdir_ = opt.outdir_

model_name = os.path.basename(source_).split('.')[-2]
model_name = model_name.split('_')[-2:]
model_name = '_'.join(model_name)
model_name

start_time = time.time()
# read image and get the dimensions, crs and transform
with rasterio.open(image_) as src:
    # get the dimensions
    height, width = src.shape
    # get the crs and transform
    crs = src.crs
#     transform = src.transform
    profile = src.profile
# print(profile)
 # update profile
profile.update({'dtype': 'float32','count': 1})
# print(profile)
print('read image: {} seconds'.format(time.time() - start_time))
print('height: {}, width: {}, crs: {}'.format(height, width, crs))

start_time = time.time()
source_array = np.load(source_)
print('load source: {} seconds'.format(time.time() - start_time))
start_time = time.time()
target_array = np.load(target_)
print('load target: {} seconds'.format(time.time() - start_time))

start_time = time.time()
euc_dist = np.linalg.norm(source_array - target_array, axis=1)
print('euclidean distance computed: {} seconds'.format(time.time() - start_time))
print(euc_dist.shape)
euc_dist = euc_dist.astype(np.float16)
print(euc_dist[:5])
# euc_dist to 2d
euc_dist = euc_dist.reshape(width, height)
print(euc_dist.shape)

# start_time = time.time()
# # create a new raster
with rasterio.open(os.path.join(outdir_, model_name + '_similarity_measure.tif'), 'w', **profile) as dst:
    dst.write(euc_dist, 1)
print('raster writing completed: {} seconds'.format(time.time() - start_time))