import rasterio
import numpy as np
from rasterio.features import shapes
import pandas as pd

import matplotlib.pyplot as plt
import os
import time 
import pandas as pd
def changeErrorCheck(
    change_map,
    gt_source,
    gt_target,
    pred_source,
    pred_target, 
    outdir):

    """
    Base on the error in the change map and confusion matrix, the No Change to Change category is inspected to know the source of error.

    Input: raster files (change map, rasterized reference image(source and target), predicted image(source and target))

    Output: csv table with the error source (error occurences combination) and summarized error count.
        gt_source | gt_target | predicted_ source | predicted_target
    Note: Source and Target in the case are 2018 and 2019 respectively
    """

    # read rasters
    change_map_raster = rasterio.open(change_map).read(1)
    gt_source_raster = rasterio.open(gt_source).read(1)
    gt_target_raster = rasterio.open(gt_target).read(1)
    pred_source_raster = rasterio.open(pred_source).read(1)
    pred_target_raster = rasterio.open(pred_target).read(1)
    
    # from change_map_raster, return  where value == 2 
    change_mask = np.ma.masked_array(change_map_raster, mask=True)

    # write a function to optimize the above code
    def mask_and_extract(array, mask_array, mask_value):
        array = np.ma.masked_array(array, mask=True)
        array.mask[mask_array.data == mask_value] = False
        array_value = np.ma.compressed(array)
        # 
        return array_value

    gt_source_ = mask_and_extract(gt_source_raster, change_mask, 2)
    gt_target_ = mask_and_extract(gt_target_raster, change_mask, 2)
    pred_source_ = mask_and_extract(pred_source_raster, change_mask, 2)
    pred_target_ = mask_and_extract(pred_target_raster, change_mask, 2)

    # create a dataframe with gt_source_, gt_target_, pred_source_, pred_target_
    df = pd.DataFrame({'gt_source': gt_source_, 'gt_target': gt_target_, 'pred_source': pred_source_, 'pred_target': pred_target_})

    label = ['Dense built-up area', 'Diffuse built-up area', 'Industrial and commercial areas', 'Roads', 'Oilseeds (Rapeseed)', 'Straw cereals (Wheat, Triticale, Barley)', 'Protein crops (Beans / Peas)', 'Soy', 'Sunflower', 'Corn',  'Tubers/roots', 'Grasslands', 'Orchards and fruit growing', 'Vineyards', 'Hardwood forest', 'Softwood forest', 'Natural grasslands and pastures', 'Woody moorlands', 'Water']
    
    # rename class to label
    df['gt_source'] = df['gt_source'].map(dict(zip(range(1, len(label) + 1), label)))
    df['gt_target'] = df['gt_target'].map(dict(zip(range(1, len(label) + 1), label)))
    df['pred_source'] = df['pred_source'].map(dict(zip(range(1, len(label) + 1), label)))
    df['pred_target'] = df['pred_target'].map(dict(zip(range(1, len(label) + 1), label)))


    # check the error
    err_df = df.groupby(['pred_source', 'pred_target']).size()
    errorstat_df = err_df.to_frame(name= 'count').reset_index()
    errorstat_df.sort_values(by=['count'], ascending=False, inplace=True)
    errorstat_df = errorstat_df[errorstat_df['pred_source'] != errorstat_df['pred_target']]
    errorstat_df['error'] = errorstat_df.apply(lambda x: str(x['pred_source']) + ' -> ' + str(x['pred_target']), axis=1)
    
    # output name and dir
    output_name = os.path.basename(change_map).split('.')[-2]
    output_name = output_name.split('_')[-2:]
    output_name = '_'.join(output_name)
    
    # save the error table
    df.to_csv(os.path.join(outdir, 'error_table' + output_name +'.csv'), index=False)
    errorstat_df.to_csv(os.path.join(outdir, 'error_count_' + output_name +'.csv'), index=False)
    
    plt.figure(figsize=(35,15))
    errorstat_df = errorstat_df.head(20)
    errorstat_df.plot.bar(x='error', y='count', rot=90)
    # xlabel to prediction error
    plt.xlabel('Prediction error')
    # plt.title('First 20 class changes error count')
    plt.savefig(os.path.join(outdir, output_name +'.png'), bbox_inches="tight")
    # errorstat_df.to_csv(os.path.join(outdir, output_name +'.csv'))
    # return ncc_gdf

if __name__ == '__main__':
    gt_source = '../../../data/rasterized_samples/2018_rasterizedImage.tif'
    gt_target = '../../../data/rasterized_samples/2019_rasterizedImage.tif'

    for case in ['1', '2', '3']:
        start_time = time.time()
        change_map = '../../../results/RF/change_D/change_map_case_' + case + '.tif'
        pred_source = '../../../results/RF/2018_rf_model_' + case + '_map.tif'
        pred_target = '../../../results/RF/2019_rf_model_' + case + '_map.tif'
        outdir = '../../../results/RF/change_D/errorstats/new'
        # outdir = '../../../results/RF/change_D/errorstats'
        changeErrorCheck(change_map, gt_source, gt_target, pred_source, pred_target, outdir)
        print("Run time: {} minutes for case {}".format((time.time() - start_time)/60 , case))
