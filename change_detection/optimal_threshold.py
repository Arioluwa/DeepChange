import os
import time
import rasterio
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
# from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score 
from sklearn.metrics import roc_curve
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc


def optm_threshold(
    similarity_map,
    gt_source,
    gt_target, 
    threshold,
    outdir):
    
    """
    Descr: Compare the 
    """
    
    # read all images to array
    similarity_array = rasterio.open(similarity_map).read(1)
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
    
    if threshold == None:
        thresholds = np.linspace(similarity_.min(), similarity_.max(), 10)
    else:
        thresholds = np.array(threshold)
        
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
        # precision_[index] = precision_score(gt_binary_, np.where(similarity_ >= elem, 1, 0))
        # recall_[index] = recall_score(gt_binary_, np.where(similarity_ >= elem, 1, 0))
        # # avg_pre[index] = average_precision_score(gt_binary_, np.where(similarity_ >= elem, 1, 0))
        # cm = confusion_matrix(gt_binary_, np.where(similarity_ >= elem, 1, 0)) # confusion matrix
        # # specificity_[index] = cm[1,1] / (cm[1,0] + cm[1,1])
        # # sensitivity_[index] = cm[0,0] / (cm[0,0] + cm[0,1]) # the same a recall
        # fpr.append(np.float16(cm[0,1]/(cm[0,1] + cm[1,1]))) # fp / (fp+tn)
        # tpr.append(np.float16(cm[0,0]/(cm[0,0] + cm[1,0]))) # tp / (tp + fn)

    print('metrics computation completed: %s seconds' % (time.time()-start_time))  
    model_name = os.path.basename(similarity_map).split('.')[-2]
    model_name = model_name.split('_')[:2]
    model_name = '_'.join(model_name)
    # model_name
    
    # get the optimal threshold based on fscore
    opt_threshold_idx = np.argmax(fscore_)
    opt_threshold = thresholds[opt_threshold_idx]
    
    # # plot precision-recall curve
    # plt.figure()
    # plt.plot(recall_, precision_, label='Precision-Recall curve')
    # plt.plot(recall_[opt_threshold_idx], precision_[opt_threshold_idx], 'o', markersize=5, color='r', label='Optimal threshold: {0:0.4f}'.format(opt_threshold))
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.legend()
    # plt.title(model_name +'_Precision-Recall curve, AUC={0:0.2f},'.format(auc(recall_, precision_)))
    # plt.savefig(os.path.join(outdir, model_name + '_precision-recall.png'))
    
    # # plot ROC curve 
    # plt.figure()
    # plt.plot(fpr, tpr)
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title(model_name + '_ROC curve')
    # plt.savefig(os.path.join(outdir, model_name + '_roc_curve.png'))
    
    # get the optimal threshold based on fscore
    df = pd.DataFrame({'threshold':thresholds, 'fscore':fscore_})
    df.to_csv(os.path.join(outdir, model_name + '_fscore.csv'), index=False)
    
    cm_sim = confusion_matrix(gt_binary_, np.where(similarity_ >= opt_threshold, 1, 0))
    cm_sim_per = cm_sim.astype('float') / np.sum(cm_sim)
    
    label = ['No change', 'Change']
    cm_sim_plot = sns.heatmap(cm_sim_per, annot=True, fmt ='.2%', cmap='Greens', xticklabels=label, yticklabels=label, cbar=False)
    cm_sim_plot.set(xlabel= "Predicted", ylabel= "Ground truth")
    cm_sim_plot.set_title("")
    # save the plot
    plt.savefig(os.path.join('./charts','similarity_confusion_matrix_case_' + case +'.png'))
    

if __name__ == '__main__':
    gt_source = '../../../data/rasterized_samples/2018_rasterizedImage.tif'
    gt_target = '../../../data/rasterized_samples/2019_rasterizedImage.tif'

    for case in ['1']:#, '2', '3']:
        similarity_map = '../../../results/RF/simliarity_measure/case_'+ case +'_ref_mask_similarity_measure.tif'
        outdir = '../../../results/RF/simliarity_measure/optimal_threshold'
        optm_threshold(similarity_map, gt_source, gt_target, None, outdir)