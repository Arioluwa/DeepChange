import os
import time
import rasterio
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import f1_score 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import argparse
import pprint
starttime = time.time()


def main(args):
    """
    Objective:
            To evaluate the binary change detection(BCD) produced by the multitemporal classification model
            The BCD is compared with the ground truth dataset; 
    Input:
        Ground truth raster(source and target)
        Classification Map (source and target)
    """
    prepare_output(args)
    outdir =args.outdir
    
    pprint.pprint(vars(args))
    # read the reference data as a numpy array
    gt_source = rasterio.open(args.gt_source).read(1)
    gt_target = rasterio.open(args.gt_target).read(1)

    # read the predicted map as a numpy array
    pred_source = rasterio.open(args.pred_source).read(1)
    # a profile is need to write the final change map
    with rasterio.open(args.pred_target) as src:
        pred_target = src.read(1)
        profile = src.profile
        profile['nodata'] = 0.0

    # Note
    # 0 = nodata
    # 1 = no change
    # 2 = change

    # first mask nodata values from both gt dataset 
    gt_mask_nodata = (gt_source != 0) & (gt_target != 0)

    # Binary change map between the gt dataset based on the nodata mask
    # i.e. if gt_source == gt_target =1
    #           #else 2
    gt_binary = np.where(gt_mask_nodata, np.where(gt_source == gt_target, 1, 2), 0) # [0,1,2] unique values

    # Binary change mp for on the classification map.
    # if pred_source == pred_target = 1
            #else 2
    pred_binary = np.where(pred_source == pred_target, 1, 2)

    # Change matrix
    # In order to produce a change matrix, we need to mask where the nodata (0) values from gt_binary in predicted binary maps.
    ## This gives an equal shape both.

    gt_binary_mask = np.ma.masked_array(gt_binary, mask=True) # returns all true mask
    gt_binary_mask.mask[gt_binary_mask.data != 0] = False # mask all non-zero mask

    ## mask the predicted binary map
    pred_binary_mask = np.ma.masked_array(pred_binary, mask=True)
    pred_binary_mask.mask[gt_binary_mask.data != 0] = False # mask all non-zero mask based on the gt_binary_mask

    gt_binary_values = np.ma.compressed(gt_binary_mask) #[1,2] unique values
    pred_binary_values = np.ma.compressed(pred_binary_mask) # [1,2] unique values

    pred_binary_map = np.empty_like(gt_binary)
    pred_binary_map[~gt_binary_mask.mask] = pred_binary_values.ravel()
    # write the change map
    # HP Hard prediction
    # with rasterio.open(os.path.join(outdir_, 'HP_binary_change_map'+ args.case +'.tif'), 'w', **profile) as dst:
    #     dst.write(pred_binary_map, 1)
    # print("--- %s minutes ---" % ((time.time() - starttime) / 60))
#     # change matrix
    cm = confusion_matrix(gt_binary_values, pred_binary_values)
    label = ['No change', 'Change']
    # plot the change matrix with percent or not 
    # print(cm[0,1])
    if args.percent:
        cm = cm.astype('float') / np.sum(cm)
        cm_plot = sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=label, yticklabels=label, cbar=False, annot_kws={"size": 30})
        cm_plot.set(xlabel= "Predicted", ylabel= "Ground truth")
        # save the plot
        plt.savefig(os.path.join(outdir,'./charts','bcd_change_matrix_percent_case_' + args.case +'.png'), dpi = 500)
        plt.close()
    else:
        cm_plot = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label, yticklabels=label, cbar=False, annot_kws={"size": 30})
        cm_plot.set(xlabel= "Predicted", ylabel= "Ground truth")

        for t in cm_plot.texts:
            t.set_text('{:,d}'.format(int(t.get_text())))
        # print(cm_plot)
        # save the plot
        plt.savefig(os.path.join(outdir, './charts','bcd_change_matrix_case_' + args.case +'.png'), dpi = 500)
        plt.close()
        # Quality assurance
        f_score = f1_score(gt_binary_values, pred_binary_values)
        def quality_check():
            with open(os.path.join(outdir, './charts', 'QA_stats' + args.case + '.txt'), 'w') as f:
                f.write("Error matrix \n")
                f.write(str(cm))
                # f.write(classif_r)
                f.write("\n Total error: {}".format((cm[0,1] + cm[1,0])))
                f.write("\n OA: {}".format(((cm[0,0] + cm[1,1])/(cm[0,0] +cm[0,1]+cm[1,0]+cm[1,1]))))
                f.write("\n fscore: {}".format(f_score))
                f.close()
        quality_check()

    ## change map
    # Note:
    # 0 = nodata
    # 1 = (1 == 1) ~ No change = No change
    # 2 = (1 == 2) ~ No change = Change
    # 3 = (2 == 1) ~ Change = No change
    # 4 = (2 == 2) ~ Change = Change

    #change array - 
#     change_array = np.empty_like(gt_binary_values) # as the same dimension as gt_binary_values
#     change_array[(gt_binary_values == 1) & (pred_binary_values == 1)] = 1
#     change_array[(gt_binary_values == 1) & (pred_binary_values == 2)] = 2
#     change_array[(gt_binary_values == 2) & (pred_binary_values == 1)] = 3
#     change_array[(gt_binary_values == 2) & (pred_binary_values == 2)] = 4

#     # change map
#     change_map = np.empty_like(gt_binary)
#     change_map[~gt_binary_mask.mask] = change_array.ravel() # returns 

    # write the change map
    # with rasterio.open(os.path.join(outdir, 'change_map_case'+ args.case +'.tif'), 'w', **profile) as dst:
    #     dst.write(change_map, 1)
    print("--- %s minutes ---" % ((time.time() - starttime) / 60))
    
#     ## this is just to compute the fscore for pred_binary; this is needed to be compared with 
#     ## fscore from threshold of similarity measure
#     gt_binary_values[gt_binary_values ==1] = 0
#     gt_binary_values[gt_binary_values ==2] = 1
#     pred_binary_values[pred_binary_values ==1] = 0
#     pred_binary_values[pred_binary_values ==2] = 1
    
    # fscore = f1_score(gt_binary_values, pred_binary_values)
    # f = open(os.path.join('./charts/', 'delcase_' + args.case + '_fscore.txt'), 'w')
    # f.write('F1 score: {}'.format(fscore))
def prepare_output(args):
        os.makedirs(args.outdir, exist_ok=True)
        os.makedirs(os.path.join(args.outdir, './charts'), exist_ok=True)
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--outdir', '-o', default='../../../results/RF/binary_change_D', type=str, help='Path to save the BCD tif files.')
    parser.add_argument('--case', '-c', default=2, type=str, help='Cases 1 to 4')
    parser.add_argument('--gt_source', '-gs', default='../../../data/rasterized_samples/2018_rasterizedImage.tif', type=str, help='Source ground truth path')
    parser.add_argument('--gt_target', '-gp', default='../../../data/rasterized_samples/2019_rasterizedImage.tif', type=str, help='Target ground truth path')
    parser.add_argument('--pred_source', '-ps', default='../../../results/RF/2018_rf_case_2_map.tif', type=str, help='Source classification map')
    parser.add_argument('--pred_target', '-pt', default='../../../results/RF/2019_rf_case_3_map.tif', type=str, help='Target  classification map')
    parser.add_argument('--percent', '-p', default=False, type=bool, help='Cal percent in the confusion matrix')
    
    args = parser.parse_args()
    main(args)
    # args = vars(args) # if needed as a dict... call as args['case']
    
    # execute for RF
    # for case in range(1, 4):
    #     args.case = str(case)
    #     args.pred_source = '../../../results/RF/2018_rf_case_{}_map.tif'.format(case)
    #     args.pred_target = '../../../results/RF/2019_rf_case_{}_map.tif'.format(case)
    #     main(args)
    #     print('Case {} done'.format(case))
    # # case 4
    # args.case = "4"
    # # args.percent = True
    # args.pred_source = '../../../results/RF/2018_rf_case_2_map.tif'
    # args.pred_target = '../../../results/RF/2019_rf_case_3_map.tif'
    # main(args)
    # print('Case {} done'.format(args.case))
    
    #ltae
    # for case in range(2,4):
    #     args.case = str(case)
    #     args.pred_source = '../../../results/ltae/classificationmap/Seed_0/2018_LTAE_map_case_{}.tif'.format(case)
    #     args.pred_target = '../../../results/ltae/classificationmap/Seed_0/2019_LTAE_map_case_{}.tif'.format(case)
    #     args.outdir = "../../../results/ltae/Change_detection/bcd"
    #     main(args)
    #     print('Case {} done'.format(case))
    # # # case 4
    # args.case = "4"
    # # args.percent = True
    # args.pred_source = '../../../results/ltae/classificationmap/Seed_0/2018_LTAE_map_case_2.tif'
    # args.pred_target = '../../../results/ltae/classificationmap/Seed_0/2019_LTAE_map_case_3.tif'
    # args.outdir = "../../../results/ltae/Change_detection/bcd"
    # main(args)
    # print('Case {} done'.format(args.case))