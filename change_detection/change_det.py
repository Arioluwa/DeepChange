# this need revision, I later used otb compute change detection cli command.
#check cmd.sh

import time
import rasterio
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import argparse

starttime = time.time()
title_ = " "
# paths to the reference data
gt_source_path = '../development/2018_rasterizedImage.tif'
gt_target_path = '../development/2019_rasterizedImage.tif'

# read the reference data as a numpy array
gt_source = rasterio.open(gt_source_path).read(1)
gt_target = rasterio.open(gt_target_path).read(1)
# paths to the predicted map produced on source(2018) and target(2019) 

# case 1:
# pred_source_path = r'C:\Users\Adebowale\Documents\theia\RF_maps\2018_rf_model_1_map.tif'
# pred_target_path = r'C:\Users\Adebowale\Documents\theia\RF_maps\2019_rf_model_1_map.tif'

# case 2:
# pred_source_path = r'C:\Users\Adebowale\Documents\theia\RF_maps\2018_rf_model_2_map.tif'
# pred_target_path = r'C:\Users\Adebowale\Documents\theia\RF_maps\2019_rf_model_2_map.tif'

# case 3:
pred_source_path = r'C:\Users\Adebowale\Documents\theia\RF_maps\2018_rf_model_3_map.tif'
pred_target_path = r'C:\Users\Adebowale\Documents\theia\RF_maps\2019_rf_model_3_map.tif'

# case normal situation:
# pred_source_path = r'C:\Users\Adebowale\Documents\theia\RF_maps\2018_rf_model_2_map.tif'
# pred_target_path = r'C:\Users\Adebowale\Documents\theia\RF_maps\2019_rf_model_3_map.tif'


# read the predicted map as a numpy array
pred_source = rasterio.open(pred_source_path).read(1)
# a profile is need to write the final change map
with rasterio.open(pred_target_path) as src:
    pred_target = src.read(1)
    profile = src.profile
    profile['nodata'] = 0.0

# Note
# 0 = nodata
# 1 = no change
# 2 = change

# first mask nodata values from both gt dataset
gt_mask_nodata = (gt_source != 0) & (gt_target != 0)

# change binary map between the gt dataset based on the nodata mask
gt_binary = np.where(gt_mask_nodata, np.where(gt_source == gt_target, 1, 2), 0) # [0,1,2] unique values

#predicted binary map
pred_binary = np.where(pred_source == pred_target, 1, 2)

# Change matrix
# In order to produce a change matrix, we need to mask where the nodata (0) values from gt_binary in predicted binary maps.
# This gives an equal shape both.

gt_binary_mask = np.ma.masked_array(gt_binary, mask=True) # returns all true mask
gt_binary_mask.mask[gt_binary_mask.data != 0] = False # mask all non-zero mask

# mask the predicted binary map
pred_binary_mask = np.ma.masked_array(pred_binary, mask=True)
pred_binary_mask.mask[gt_binary_mask.data != 0] = False # mask all non-zero mask based on the gt_binary_mask

gt_binary_values = np.ma.compressed(gt_binary_mask) #[1,2] unique values
pred_binary_values = np.ma.compressed(pred_binary_mask) # [1,2] unique values

# change matrix
cm = confusion_matrix(gt_binary_values, pred_binary_values)

cm_per = cm.astype('float') / np.sum(cm)
# plot the change matrix
label = ['No change', 'Change']
cm_plot = sns.heatmap(cm_per, annot=True, fmt='.2%', cmap='Greens', xticklabels=label, yticklabels=label, cbar=False)
cm_plot.set(xlabel= "Predicted", ylabel= "Ground truth")
cm_plot.set_title(title_)
# save the plot
plt.savefig('change_matrix_percent_case3.png')

# change map
# Note:
# 0 = nodata
# 1 = (1 == 1) ~ No change = No change
# 2 = (1 == 2) ~ No change = Change
# 3 = (2 == 1) ~ Change = No change
# 4 = (2 == 2) ~ Change = Change

# change array - 
change_array = np.empty_like(gt_binary_values) # as the same dimension as gt_binary_values
change_array[(gt_binary_values == 1) & (pred_binary_values == 1)] = 1
change_array[(gt_binary_values == 1) & (pred_binary_values == 2)] = 2
change_array[(gt_binary_values == 2) & (pred_binary_values == 1)] = 3
change_array[(gt_binary_values == 2) & (pred_binary_values == 2)] = 4

# change map
change_map = np.empty_like(gt_binary)
change_map[~gt_binary_mask.mask] = change_array.ravel() # returns 

# write the change map
with rasterio.open('change_map_case_3.tif', 'w', **profile) as dst:
    dst.write(change_map, 1)
print("--- %s minutes ---" % ((time.time() - starttime) / 60))