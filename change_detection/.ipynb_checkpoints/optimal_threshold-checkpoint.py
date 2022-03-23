import os
import time
import rasterio
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score 
from sklearn.metrics import auc


def optm_threshold(
    similarity_map,
    case,
    gt_source,
    gt_target, 
    threshold: bool,
    outdir):
    
    """
    Desc: Change detection using similarity measure (Euclidean distance)
          It test binary change/no-change on the several thresholds, based on the optimal threshold with the highest F1-score performance the a final confusion matrix is provided.
    INPUT: 
        similarity_map: a raster file
        gt_source: rasterized ground truth source dataset
        gt_target: rasterized ground truth target dataset
        threshold: default = None, range btw the min and max value of the similarity_map
    OUTPUT: Charts and DataFrame - 
        precision-recall,
        ROC curve,
        Confusion matrix,
        and dataframe of fscore for each threshold
         
    """
    
    # read all images to array
    # similarity_array = rasterio.open(similarity_map).read(1)
    with rasterio.open(similarity_map) as src:
        similarity_array = src.read(1)
        profile = src.profile
        profile['nodata'] = 0.0
    
    gt_source_ = rasterio.open(gt_source).read(1)
    gt_target_ = rasterio.open(gt_target).read(1)
    
    # get gt mask (where there are value) and binary (change/no-chnage)
    gt_mask = (gt_source_ != 0) & (gt_target_ != 0)
    gt_binary = np.where(gt_mask, np.where(gt_source_ == gt_target_, 1, 2), 0) # gt change/no-change
    gt_binary_mask = np.ma.masked_array(gt_binary, mask=True)
    gt_binary_mask.mask[gt_binary_mask.data != 0] = False # mask all non-zero mask
    gt_binary_ = gt_binary_mask.compressed() # gt_binary with nodata value (0)
    gt_binary_[gt_binary_ == 1] = 0 # No-change
    gt_binary_[gt_binary_ == 2] = 1 # Change
    
    # mask similarity map 
    similarity_mask = np.ma.masked_array(similarity_array, mask= True)
    similarity_mask.mask[gt_binary_mask.data != 0] = False # extract according to the non-zero from ground truth data
    similarity_ = similarity_mask.compressed()
    # model_name = os.path.basename(similarity_map).split('.')[-2]
    # model_name = model_name.split('_')[:2]
    # model_name = '_'.join(model_name)
      # model_name
    
    if threshold == True:
        thresholds = np.linspace(similarity_.min(), similarity_.max(), 10)

        # initiate metrics
        fscore_ = np.zeros(shape=(len(thresholds)))
        precision_ = np.zeros(shape=(len(thresholds)))
        recall_ = np.zeros(shape=(len(thresholds)))
        specificity_ = np.zeros(shape=(len(thresholds)))
        sensitivity_ = np.zeros(shape=(len(thresholds)))
        avg_pre = np.zeros(shape=(len(thresholds)))
        fpr = []
        tpr = []
        start_time = time.time()
        # compute metrics for each threshold, at the same time compute the binary on the similarity using the provided threshold
            # np.where(similarity_ >= elem, 1, 0) # where similarity measure (euclidean distance) is greater than threshold == 1 otherwise 0.
        for index, elem in enumerate(thresholds):
            fscore_[index] = f1_score(gt_binary_, np.where(similarity_ >= elem, 1, 0))
            precision_[index] = precision_score(gt_binary_, np.where(similarity_ >= elem, 1, 0))
            recall_[index] = recall_score(gt_binary_, np.where(similarity_ >= elem, 1, 0))
            cm = confusion_matrix(gt_binary_, np.where(similarity_ >= elem, 1, 0)) # confusion matrix
            fpr.append(np.float16(cm[0,1]/(cm[0,1] + cm[1,1]))) # fp / (fp+tn)
            tpr.append(np.float16(cm[0,0]/(cm[0,0] + cm[1,0]))) # tp / (tp + fn)

        print('metrics computation completed: %s seconds' % (time.time()-start_time))  


        ## get the optimal threshold based on fscore
        opt_threshold_idx = np.argmax(fscore_)
        opt_threshold = thresholds[opt_threshold_idx]

        # plot precision-recall curve
        plt.figure()
        plt.plot(recall_, precision_, label='Precision-Recall curve')
        plt.plot(recall_[opt_threshold_idx], precision_[opt_threshold_idx], 'o', markersize=5, color='r', label='Optimal threshold: {0:0.4f}'.format(opt_threshold))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.title( 'AUC={0:0.2f}'.format(auc(recall_, precision_)))#plt.title(model_name +'_Precision-Recall curve, AUC={0:0.2f},'.format(auc(recall_, precision_)))
        plt.savefig(os.path.join(outdir, 'case_' + case + '_precision-recall.png'))
        plt.close()

        # plot ROC curve 
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel('False positive rate')
        plt.ylabel('True positi/ve rate')
        plt.title('case_' + case + '_ROC curve')
        plt.savefig(os.path.join(outdir, 'case_' + case + '_roc_curve.png'))
        plt.close()

    #     # get the optimal threshold based on fscore
        df = pd.DataFrame({'threshold':thresholds, 'fscore':fscore_})
        df.to_csv(os.path.join(outdir, 'case_' + case + '_fscore.csv'), index=False)

        #confusion
        cm_sim = confusion_matrix(gt_binary_, np.where(similarity_ >= opt_threshold, 1, 0))
        cm_sim_per = cm_sim.astype('float') / np.sum(cm_sim)

        label = ['No change', 'Change']
        cm_sim_plot = sns.heatmap(cm_sim_per, annot=True, fmt ='.2%', cmap='Greens', xticklabels=label, yticklabels=label, cbar=False)
        cm_sim_plot.set(xlabel= "Predicted", ylabel= "Ground truth")
        # cm_sim_plot.set_title("")
        # save the plot
        plt.savefig(os.path.join('./charts', 'similarity_confusion_matrix_case_' + case +'.png'))
        plt.close()

        # similarity histogram distribution
        plt.figure(figsize=(8,5))
        plt.hist(similarity_, bins=10, label='similarity distribution', ec='white', log=True)
        plt.axvline(x=thresholds[opt_threshold_idx], color='r', linestyle='--', label="optimal threshold: {0:0.3f}".format(thresholds[opt_threshold_idx]))
        # plt.yticks([])
        plt.legend()
        # save chart
        plt.savefig(os.path.join('./charts', 'similarity_distribution_case_' + case +'.png'))
        plt.close()

        # similarity binary change
        similarity_binary = np.where(similarity_ >= opt_threshold, 2, 1)
        similarity_binary_change = np.empty_like(gt_binary) # shape: 10980, 10980
        similarity_binary_change[~gt_binary_mask.mask] = similarity_binary.ravel()

        ## write to raster
        with rasterio.open(os.path.join(outdir, 'similarity_bin_change_case_' + case + '.tif'), 'w', **profile) as dst:
            dst.write(similarity_binary_change, 1)
            dst.close()


        ## change map
        # Note:
        # 0 = nodata
        # 1 = (0 == 1) ~ No change = No change
        # 2 = (0 == 2) ~ No change = Change
        # 3 = (1 == 1) ~ Change = No change
        # 4 = (1 == 2) ~ Change = Change

        ## change array
        change_array = np.empty_like(gt_binary_) # flatten shape
        change_array[(gt_binary_ == 0) & (similarity_binary == 1)] = 1
        change_array[(gt_binary_ == 0) & (similarity_binary == 2)] = 2
        change_array[(gt_binary_ == 1) & (similarity_binary == 1)] = 3
        change_array[(gt_binary_ == 1) & (similarity_binary == 2)] = 4

        change_map = np.empty_like(gt_binary) # shape: 10980, 10980
        change_map[~gt_binary_mask.mask] = change_array.ravel()

        print(np.unique(change_map))
        with rasterio.open(os.path.join(outdir, 'sim-change_map_case_' + case + '.tif'), 'w', **profile) as dst:
            dst.write(change_map, 1)
            dst.close()
    else:
        threshold = threshold_otsu(similarity_)
        otsu_binary = similarity_ > threshold
        
        # # plot precision-recall curve
        # plt.figure()
        # plt.plot(recall_, precision_, label='Precision-Recall curve')
        # plt.plot(recall_[opt_threshold_idx], precision_[opt_threshold_idx], 'o', markersize=5, color='r', label='Optimal threshold: {0:0.4f}'.format(opt_threshold))
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.legend()
        # plt.title( 'AUC={0:0.2f}'.format(auc(recall_, precision_)))#plt.title(model_name +'_Precision-Recall curve, AUC={0:0.2f},'.format(auc(recall_, precision_)))
        # plt.savefig(os.path.join(outdir, model_name + '_precision-recall.png'))
        # plt.close()

        # # plot ROC curve 
        # plt.figure()
        # plt.plot(fpr, tpr)
        # plt.xlabel('False positive rate')
        # plt.ylabel('True positi/ve rate')
        # plt.title(model_name + '_ROC curve')
        # plt.savefig(os.path.join(outdir, model_name + '_roc_curve.png'))
        # plt.close()
        fscore = f1_score(gt_binary_, otsu_binary)
        f = open(os.path.join('otsu_case_' + case + '_fscore.txt'), 'w')
        f.write('F1 score: {}'.format(fscore))

        #confusion
        cm_sim = confusion_matrix(gt_binary_, otsu_binary)
        cm_sim_per = cm_sim.astype('float') / np.sum(cm_sim)

        label = ['No change', 'Change']
        cm_sim_plot = sns.heatmap(cm_sim_per, annot=True, fmt ='.2%', cmap='Greens', xticklabels=label, yticklabels=label, cbar=False)
        cm_sim_plot.set(xlabel= "Predicted", ylabel= "Ground truth")
        # cm_sim_plot.set_title("")
        # save the plot
        plt.savefig(os.path.join('./charts', 'otsu_similarity_confusion_matrix_case_' + case +'.png'))
        plt.close()
        
        

if __name__ == '__main__':
    gt_source = '../../../data/rasterized_samples/2018_rasterizedImage.tif'
    gt_target = '../../../data/rasterized_samples/2019_rasterizedImage.tif'

    for case in ['4']:#['1', '2', '3', '4']:
        similarity_map = '../../../results/RF/simliarity_measure/case_'+ case +'_ref_mask_similarity_measure.tif'
        outdir = '../../../results/RF/simliarity_measure/optimal_threshold'
        optm_threshold(similarity_map, case, gt_source, gt_target, False, outdir)
        