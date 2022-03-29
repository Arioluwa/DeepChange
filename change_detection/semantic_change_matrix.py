import os
import rasterio
import numpy as np
import pandas as pd

def changeErrorCheck(
    case,
    gt_source,
    gt_target,
    pred_source,
    pred_target,
    outdir
    ):
    # read rasters
    gt_source_ = rasterio.open(gt_source).read(1).flatten().astype('int')
    gt_target_ = rasterio.open(gt_target).read(1).flatten().astype('int')
    pred_source_ = rasterio.open(pred_source).read(1).flatten()
    pred_target_ = rasterio.open(pred_target).read(1).flatten()
    
    gt_mask_nodata = (gt_source_ != 0) & (gt_target_ != 0)
    
    def mask_and_extract(array, mask_array):
        array = np.ma.masked_array(array, mask=True)
        array.mask[mask_array] = False
        array_value = np.ma.compressed(array)
        return array_value

    gt_source_ = mask_and_extract(gt_source_, gt_mask_nodata)
    gt_target_ = mask_and_extract(gt_target_, gt_mask_nodata)
    pred_source_ = mask_and_extract(pred_source_, gt_mask_nodata)
    pred_target_ = mask_and_extract(pred_target_, gt_mask_nodata)
    
    # class label
    dict_={1: 'Dense built-up area',
    2: 'Diffuse built-up area',
     3: 'Industrial and commercial areas',
     4: 'Roads',
     5: 'Oilseeds (Rapeseed)',
     6: 'Straw cereals (Wheat, Triticale, Barley)',
     7: 'Protein crops (Beans / Peas)',
     8: 'Soy',
     9: 'Sunflower',
     10: 'Corn',
     12: 'Tubers/roots',
     13: 'Grasslands',
     14: 'Orchards and fruit growing',
     15: 'Vineyards',
     16: 'Hardwood forest',
     17: 'Softwood forest',
     18: 'Natural grasslands and pastures',
     19: 'Woody moorlands',
     23: 'Water'}
    
    df = pd.DataFrame({'gt_source': gt_source_, 'gt_target': gt_target_, 'pred_source': pred_source_, 'pred_target': pred_target_})
    # rename class to label
    df['gt_source'] = df['gt_source'].map(dict_)
    df['gt_target'] = df['gt_target'].map(dict_)
    df['pred_source'] = df['pred_source'].map(dict_)
    df['pred_target'] = df['pred_target'].map(dict_)
    
    
    df['gt_change']= df.apply(lambda x: str(x['gt_source']) + '-' + str(x['gt_target']), axis=1)
    df['pred_change']= df.apply(lambda x: str(x['pred_source']) + '-' + str(x['pred_target']), axis=1)
    
    # cm = pd.crosstab(df['gt_change'], df['pred_change'], rownames=['Ground Truth'], colnames=['Prediction'])
    cm = pd.crosstab(df['pred_change'], df['gt_change'], colnames=['Prediction'], rownames=['Ground Truth'])
    
    # https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2
    # https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1
    
    cm.to_csv(os.path.join(outdir, "semantic_change_matrix_case_" + case + '.csv'))

if __name__ == '__main__':
    gt_source_path = '../../../data/rasterized_samples/2018_rasterizedImage.tif'
    gt_target_path = '../../../data/rasterized_samples/2019_rasterizedImage.tif'
    
    # for case in ['1', '2', '3']:
    #     pred_source_path = '../../../results/RF/2018_rf_case_'+ case +'_map.tif'
    #     pred_target_path = '../../../results/RF/2019_rf_case_'+ case +'_map.tif'
    #     outdir = '../../../results/RF/binary_change_D/semantic_change_matrix'
    #     changeErrorCheck(case, gt_source_path, gt_target_path, pred_source_path, pred_target_path, outdir)
    
    # case 4
    case = '4'
    pred_source_path = '../../../results/RF/2018_rf_case_2_map.tif'
    pred_target_path = '../../../results/RF/2019_rf_case_3_map.tif'
    outdir = '../../../results/RF/binary_change_D/semantic_change_matrix'
    changeErrorCheck(case, gt_source_path, gt_target_path, pred_source_path, pred_target_path, outdir)
    