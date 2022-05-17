from utils import *
import numpy as np
import os
def separate_data(sits, seed, outdir):

    train_ids, val_ids, test_ids = read_ids(seed)

    X, y, block_ids = load_npz(sits)
    print("npz....")

    y = np.unique(y, return_inverse=True)[1] # reassigning label [1,23] to [0,18]
    print("reaass....")

    # concatenate the data
    data_ = np.concatenate((X, y[:, None], block_ids[:, None]), axis=1)
    print("concatenating....")
    
    del X
    del y

    def pickk(partition, dt_):
        
        if partition == 'train':
            ids = train_ids
        elif partition == 'val':
            ids = val_ids
        elif partition == 'test':
            ids = test_ids
        else:
            raise ValueError('Invalid partition: {}'.format(partition))
        
        print("filter....")    
        dat = dt_[np.isin(dt_[:, -1], ids)]
        print("filter done....")    
        
        print("selecting....")    
        X_ = dat[:, :-2]
        y_ = dat[:, -2]
        print("selecting done....")    
        
        print("savinng....")    
        np.savez_compressed(os.path.join(outdir, "{}.npz".format(partition)), X=X_, y=y_)
        print("completed....")    
    pickk('train', data_)
    print("1 done....")    
    pickk('val', data_)
    print("2 done....")    
    pickk('test', data_)
    print("3 done....")    

if __name__ == '__main__':
    sits = "../../../data/theiaL2A_zip_img/output/2018/2018_SITS_data.npz"
    
    for seed in [1,2,3,4]:
        outdir = "../../../data/theiaL2A_zip_img/output/2018/Seed_{}".format(seed)
        separate_data(sits, seed, outdir)