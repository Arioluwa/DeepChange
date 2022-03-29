import numpy as np
import rasterio
import os

def certainty(data, raster):
    probability_distr = np.load(data)
    probability_distr =np.sort(probability_distr, axis=1)[:,::-1]
    uncertainty = probability_distr[:,0] - probability_distr[:,1]
    with rasterio.open(raster) as src:
        profile = src.profile
        profile['nodata'] = -999.
        height, width = src.shape
    base_name = os.path.basename(data).split('.')[-2]
    base_name = base_name.split('_')[0]
    uncertainty = np.reshape(width, height)
    with rasterio.open(os.path.join(outdir, base_name + 'certainty' + case +'.tif'), 'w', **profile) as dst:
        dst.write(uncertainty, 1)
if __name__ == '__main__':
    _path  = '../../../data/rasterized_samples/2018_rasterizedImage.tif'
    for case in ['1', '2', '3']:
        proba_distr = '../../../results/RF/simliarity_measure/2018_rf_model_'+case+'.npy'
        certainty(proba_distr, _path)
        proba_distr = '../../../results/RF/simliarity_measure/2019_rf_model_'+case+'.npy'
        certainty(proba_distr, _path)
        