import os
import numpy as np
import rasterio
import time

def similarity_check(source_, target_, outdir_, gt_source_, gt_target_):
    """
    Descr:
    Input:
        source_ & target_: probability distribution for source & target of each case  (RF model). It's a npy file (N, no of classes).
        gt_source and gt_target: source and target rasterized reference dataset
    
    Descr: 
    
    """
    model_name = os.path.basename(source_).split('.')[-2]
    model_name = model_name.split('_')[-2:]
    model_name = '_'.join(model_name)


    start_time = time.time()

    with rasterio.open(gt_source_) as src:
        height, width = src.shape
        profile = src.profile
        gt_source_ = src.read(1)

    gt_target_ = rasterio.open(gt_target_).read(1)

    gt_mask_nodata = (gt_source_ == 0) & (gt_target_ == 0)
    print(profile)
    profile.update({'nodata':-999.})
    print(profile)
    print('read image: {} seconds'.format(time.time() - start_time))
    print('height: {}, width: {}'.format(height, width))

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
    # print(euc_dist[:5])
    # euc_dist to 2d
    euc_dist = euc_dist.reshape(width, height)

    euc_dist[gt_mask_nodata] = -999.

    start_time = time.time()
    # create a new raster
    with rasterio.open(os.path.join(outdir_, model_name + '_ref_mask_similarity_measure.tif'), 'w', **profile) as dst:
        dst.write(euc_dist, 1)
    print('raster writing completed: {} seconds'.format(time.time() - start_time))

if __name__ == '__main__':
    gt_source_ = '../../../data/rasterized_samples/2018_rasterizedImage.tif'
    gt_target_ = '../../../data/rasterized_samples/2019_rasterizedImage.tif'

    for case in ['3']:#['1', '2', '3']:
        start_time = time.time()
        source_ = '../../../results/RF/simliarity_measure/2018_rf_model_' + case + '.npy'
        target_ = '../../../results/RF/simliarity_measure/2019_rf_model_' + case + '.npy'
        outdir_ = '../../../results/RF/simliarity_measure'
        similarity_check(source_, target_, outdir_, gt_source_, gt_target_)