import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import time
def check_fiability(hard_pred, soft_pred, source_certainty, target_certainty, outdir):
    """
    
    """
    case_name  = os.path.basename(hard_pred)
    case_name = case_name.split('.')[0]
    case_name = case_name.split('_')[2:]
    case_name = '_'.join(case_name)
    
    with rasterio.open(hard_pred) as src:
        hard_ = src.read(1).astype('int')
        profile = src.profile
        height, width = src.shape
        
    soft_ = rasterio.open(soft_pred).read(1).astype('int')
    source_certainty = rasterio.open(source_certainty).read(1)
    target_certainty = rasterio.open(target_certainty).read(1)
    
    mask = hard_ != 0
    
    def mask_and_extract(array, mask_array):
        array = np.ma.masked_array(array, mask=True)
        array.mask[mask_array] = False
        array_value = np.ma.compressed(array)
        return array_value
    
    hard_ = mask_and_extract(hard_, mask)
    soft_ = mask_and_extract(soft_, mask)
    source_certainty = mask_and_extract(source_certainty, mask)
    target_certainty = mask_and_extract(target_certainty, mask)
    
    # # agreement between soft and hard detection and they are correct
    agree_ = np.zeros_like(hard_)
    agree_[(hard_ == 1) & (soft_ == 1) | (hard_ == 4) & (soft_ == 4)] = 1 #both correct
    agree_[(hard_ == 3) & (soft_ == 3) | (hard_ == 2) & (soft_ == 2)] = 2 # both incorrect 
    agree_[(hard_ == 1) & (soft_ == 2) | (hard_ == 4) & (soft_ == 3)] = 3 # hard correct and soft not
    agree_[(hard_ == 2) & (soft_ == 1) | (hard_ == 3) & (soft_ == 4)] = 4 # hard incorrect and soft is
    
    # fiability measure histogram
    def _mask(cert, agree, value):
        cert = np.ma.masked_array(cert, mask=True)
        cert.mask[agree == value] = False
        return cert.compressed()
    
    # plot fiability distribution
    def plot_hist(data, data2):
        plt.hist([data, data2], bins=10, label=['Source', 'Target'], ec='white', log=True, color=['#1f77b4', 'gray'])
        plt.xlabel('certainty')
        plt.ylabel('pixels')
        plt.legend(loc="upper left")
        plt.savefig(os.path.join(outdir, case_name+'_fiability_subcase_' +str(i)+'.eps'), format='eps')
        # plt.show()
        plt.close()
    
    for i in range(1, 5):
        _h = _mask(source_certainty, hard_, i)
        _s = _mask(target_certainty, soft_, i)
        plot_hist(_h, _s)
    
if __name__ == '__main__':
    
    # for i in ['1','2','3' ]:
    #     start_time = time.time()
    #     hard_pred = '../../../results/RF/binary_change_D/change_map_case_'+i+'.tif'  #4case
    #     soft_pred = '../../../results/RF/simliarity_measure/optimal_threshold/sim-change_map_case_'+i+'.tif'
    #     source_certainty = '../../../results/RF/simliarity_measure/certainty/2018_certainty_'+i+'.tif'
    #     target_certainty = '../../../results/RF/simliarity_measure/certainty/2019_certainty_'+i+'.tif'
    #     outdir = './charts'
    #     check_fiability(hard_pred, soft_pred, source_certainty, target_certainty, outdir)
    #     print("time taken: {}, is {}".format(i, time.time() - start_time))
    for i in ['1','2','3']:
        hard_pred = '../../../results/RF/binary_change_D/change_map_case_4.tif'  #4case
        soft_pred = '../../../results/RF/simliarity_measure/optimal_threshold/sim-change_map_case_4.tif'
        source_certainty = '../../../results/RF/simliarity_measure/certainty/2018_certainty_'+i+'.tif'
        target_certainty = '../../../results/RF/simliarity_measure/certainty/2019_certainty_'+i+'.tif'
        outdir = './charts'
        check_fiability(hard_pred, soft_pred, target_certainty, target_certainty, outdir)