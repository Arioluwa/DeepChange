import os
import sys
import time
import argparse
import pprint
import rasterio
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import xlogy
import matplotlib.pyplot as plt

import torch
from skimage.filters import threshold_otsu
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score 
from sklearn.metrics import auc


def main(args):
    """
    Desc: Change detection using similarity measure (Cross-entropy)
          It estimates binary change/no-change threshold using the otsu's threshold and optimal thresholding technique. based on the optimal threshold with the highest F1-score performance the a final confusion matrix is provided.
    INPUT: 
        source_prob: source probability distribution matrix (N, k) - N is total number of samples and k number of classes
        target_prob: target probability distribution matrix (N, k) - N is total number of samples and k number of classes
        gt_source: rasterized ground truth source dataset
        gt_target: rasterized ground truth target dataset

    OUTPUT: Charts and DataFrame - 
        precision-recall,
        ROC curve,
        Confusion matrix,
        and dataframe of fscore for each threshold
        binary maps
    """
    
    pprint.pprint(vars(args))
    prepare_output(args)
    
    outdir = args.outdir
    
    # Groud truth data
    with rasterio.open(args.gt_source) as src:
        gt_source_ = src.read(1).flatten().astype('int')
        width, height = src.shape
        profile = src.profile        

    gt_target_ = rasterio.open(args.gt_target).read(1).flatten().astype('int')
    
    # get gt mask (where there are value) and binary (change/no-chnage)
    gt_mask = (gt_source_ != 0) & (gt_target_ != 0)
    gt_binary = np.where(gt_mask, np.where(gt_source_ == gt_target_, 1, 2), 0) # gt change/no-change
    gt_binary_mask = np.ma.masked_array(gt_binary, mask=True)
    gt_binary_mask.mask[gt_binary_mask.data != 0] = False # mask all non-zero mask
    gt_binary_ = gt_binary_mask.compressed() # gt_binary with nodata value (0)
    gt_binary_[gt_binary_ == 1] = 0 # No-change
    gt_binary_[gt_binary_ == 2] = 1 # Change
    
    start_time = time.time()
    # Dissimilarity measure = Cross entropy
    print("computing dissimilarity")

    if args.relu: # for LTAE
        if os.path.exists(os.path.join(outdir, 'case_' + args.case + '_similarity_measure.tif')):
            with rasterio.open(os.path.join(outdir, 'case_' + args.case + '_similarity_measure.tif')) as src:
                similarity_array = src.read(1).flatten()
        else:
            # read all images to probability distribution vectors
            source_ = np.load(args.source_prob)
            target_ = np.load(args.target_prob)
            
            # convert to tensor
            source_ = torch.nn.functional.softmax(torch.from_numpy(source_).float(), dim=-1)
            target_ = torch.nn.functional.log_softmax(torch.from_numpy(target_).float(), dim=-1)
            
            print('computation start...')
            similarity_array = np.array([cross_entropy_rl(source_[i],target_[i]) for i in range(len(source_))])
            print('computation done...')
            
            # save similarity_array as an image
            similarity_map = similarity_array.reshape(width, height)

            with rasterio.open(os.path.join(outdir, 'case_' + args.case + '_similarity_measure.tif'), 'w', **profile) as dst:
                dst.write(similarity_map, 1)
                dst.close()
            del similarity_map
            
        
    else: # for RF 
        source_ = np.clip(source_, args.epsilion, 1. - args.epsilion)
        target_ = np.clip(target_, args.epsilion, 1. - args.epsilion)
        
        similarity_array = np.array([cross_entropy(source_[i], target_[i]) for i in range(len(source_))])
        print('computation done...')
    print('Dissimilarity computation completed: %s mins' % ((time.time()-start_time)/60)) 
    
    
    # mask similarity map 
    similarity_mask = np.ma.masked_array(similarity_array, mask= True)
    similarity_mask.mask[gt_binary_mask.data != 0] = False # extract according to the non-zero from ground truth data
    similarity_ = similarity_mask.compressed()
    gt_binary_mask = gt_binary_mask.reshape(width, height)
    
    
    if args.optimal:
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
        plt.savefig(os.path.join(outdir, './charts', 'case_' + args.case + '_precision-recall.png'), dpi=500)
        plt.close()

        # plot ROC curve 
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel('False positive rate')
        plt.ylabel('True positi/ve rate')
        # plt.title('case_' + args.case + '_ROC curve')
        plt.savefig(os.path.join(outdir,'./charts', 'case_' + args.case + '_roc_curve.png'), dpi=500)
        plt.close()

        # get the optimal threshold based on fscore
        df = pd.DataFrame({'threshold':thresholds, 'fscore':fscore_})
        df.to_csv(os.path.join(outdir, 'case_' + args.case + '_fscore.csv'), index=False)
        pred_binary_v = np.where(similarity_ >= opt_threshold, 1, 0)
        #confusion
        cm_sim = confusion_matrix(gt_binary_, pred_binary_v)
        # cm_sim_per = cm_sim.astype('float') / np.sum(cm_sim)

        label = ['No change', 'Change']
        
        cm_sim_plot = sns.heatmap(cm_sim, annot=True, fmt ='d', cmap='Blues', xticklabels=label, yticklabels=label, cbar=False, annot_kws={"size": 30})
        cm_sim_plot.set(xlabel= "Predicted", ylabel= "Ground truth")
        for t in cm_sim_plot.texts:
            t.set_text('{:,d}'.format(int(t.get_text())))
        # cm_sim_plot.set_title("")
        # save the plot
        plt.savefig(os.path.join(outdir,'./charts', 'similarity_error_matrix_case_' + args.case +'.png'), dpi=500)
        plt.close()
        
        #percentage
        cm = cm_sim.astype('float') / np.sum(cm_sim)
        cm_plot = sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=label, yticklabels=label, cbar=False, annot_kws={"size": 30})
        cm_plot.set(xlabel= "Predicted", ylabel= "Ground truth")
        # save the plot
        plt.savefig(os.path.join(outdir,'./charts','similarity_error_matrix_percent_case_' + args.case +'.png'), dpi = 500)
        plt.close()

        # similarity histogram distribution
        plt.figure(figsize=(8,5))
        plt.hist(similarity_, bins=10, label='similarity distribution', ec='white', log=True)
        plt.axvline(x=thresholds[opt_threshold_idx], color='r', linestyle='--', label="optimal threshold: {0:0.3f}".format(thresholds[opt_threshold_idx]))
        plt.legend()
        
        # save chart
        plt.savefig(os.path.join(outdir,'./charts', 'similarity_distribution_case_' + args.case +'.png'), dpi = 500)
        plt.close()
        
        # Quality assurance
        f_score = f1_score(gt_binary_, pred_binary_v)
        quality_check(args, cm_sim, f_score, method_='opt')
        
        if args.map:
            # similarity binary change
            similarity_binary = np.where(similarity_ >= opt_threshold, 2, 1)
            similarity_binary_change = np.empty(shape=(width, height)) # shape: 10980, 10980
            similarity_binary_change[~gt_binary_mask.mask] = similarity_binary.ravel()

            ## write to raster
            with rasterio.open(os.path.join(outdir, 'similarity_binary_change_case_' + args.case + '.tif'), 'w', **profile) as dst:
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

            change_map = np.empty(shape=(width, height)) # shape: 10980, 10980
            change_map[~gt_binary_mask.mask] = change_array.ravel()

            with rasterio.open(os.path.join(outdir, 'similiarity-change_map_case_' + args.case + '.tif'), 'w', **profile) as dst:
                dst.write(change_map, 1)
                dst.close()
    
                
    if args.percentile:
        percent = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]
        thresholds = np.asarray([np.percentile(similarity_array, percent[i]) for i in range(len(percent))])

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
        plt.savefig(os.path.join(outdir, './charts', 'case_' + args.case + '_percentile_precision-recall.png'), dpi=500)
        plt.close()

        # plot ROC curve 
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel('False positive rate')
        plt.ylabel('True positi/ve rate')
        # plt.title('case_' + args.case + '_ROC curve')
        plt.savefig(os.path.join(outdir,'./charts', 'case_' + args.case + '_percentile_roc_curve.png'), dpi=500)
        plt.close()

        # get the optimal threshold based on fscore
        df = pd.DataFrame({'threshold':thresholds, 'fscore':fscore_})
        df.to_csv(os.path.join(outdir, 'case_' + args.case + '_percentile_fscore.csv'), index=False)
        pred_binary_v = np.where(similarity_ >= opt_threshold, 1, 0)
        #confusion
        cm_sim = confusion_matrix(gt_binary_, pred_binary_v)
        # cm_sim_per = cm_sim.astype('float') / np.sum(cm_sim)

        label = ['No change', 'Change']

        cm_sim_plot = sns.heatmap(cm_sim, annot=True, fmt ='d', cmap='Blues', xticklabels=label, yticklabels=label, cbar=False, annot_kws={"size": 30})
        cm_sim_plot.set(xlabel= "Predicted", ylabel= "Ground truth")
        for t in cm_sim_plot.texts:
            t.set_text('{:,d}'.format(int(t.get_text())))
        
        # save the plot
        plt.savefig(os.path.join(outdir,'./charts', 'similarity_error_percentile_matrix_case_' + args.case +'.png'), dpi=500)
        plt.close()
        
        #percentage
        cm = cm_sim.astype('float') / np.sum(cm_sim)
        cm_plot = sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=label, yticklabels=label, cbar=False, annot_kws={"size": 30})
        cm_plot.set(xlabel= "Predicted", ylabel= "Ground truth")
        # save the plot
        plt.savefig(os.path.join(outdir,'./charts','similarity_error_percentile_matrix_percent_case_' + args.case +'.png'), dpi = 500)
        plt.close()

        # similarity histogram distribution
        plt.figure(figsize=(8,5))
        plt.hist(similarity_, bins=10, label='similarity distribution', ec='white', log=True)
        plt.axvline(x=thresholds[opt_threshold_idx], color='r', linestyle='--', label="optimal threshold: {0:0.3f}".format(thresholds[opt_threshold_idx]))
        plt.legend()
        
        # save chart
        plt.savefig(os.path.join(outdir,'./charts', 'similarity_distribution_percentile__case_' + args.case +'.png'), dpi = 500)
        plt.close()

        # Quality assurance
        f_score = f1_score(gt_binary_, pred_binary_v)
        quality_check(args, cm_sim, f_score, method_='opt')

        if args.map:
            # similarity binary change
            similarity_binary = np.where(similarity_ >= opt_threshold, 2, 1)
            similarity_binary_change = np.empty(shape=(width, height)) # shape: 10980, 10980
            similarity_binary_change[~gt_binary_mask.mask] = similarity_binary.ravel()

            ## write to raster
            with rasterio.open(os.path.join(outdir, 'similarity_binary_percentile_change_case_' + args.case + '.tif'), 'w', **profile) as dst:
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

            change_map = np.empty(shape=(width, height)) # shape: 10980, 10980
            change_map[~gt_binary_mask.mask] = change_array.ravel()

            with rasterio.open(os.path.join(outdir, 'similiarity-change_map_percentile_case_' + args.case + '.tif'), 'w', **profile) as dst:
                dst.write(change_map, 1)
                dst.close()
                

    if args.otsu: #otsu threshold
        
        ##""" Otsu's threshold on the similarity masked matrix """##
        otsu_threshold = threshold_otsu(similarity_)
        otsu_binary = similarity_ > otsu_threshold
        
        # similarity histogram distribution
        plt.figure(figsize=(8,5))
        plt.hist(similarity_, bins=10, label='similarity distribution', ec='white')
        plt.axvline(x=thresholds[opt_threshold_idx], color='r', linestyle='--', label="optimal threshold: {0:0.3f}".format(thresholds[opt_threshold_idx]))
        plt.axvline(x=otsu_threshold, color='r', linestyle='--', label="otsu threshold: {0:0.3f}".format(otsu_threshold))
        plt.legend()
        # save chart
        plt.savefig(os.path.join(outdir,'./charts', 'otsu_similarity_distribution_case_' + args.case +'.png'), dpi = 500)
        plt.close()
        
        # save otsu f1score
        fscore = f1_score(gt_binary_, otsu_binary)
        f = open(os.path.join(outdir, 'otsu_case_' + args.case + '_fscore.txt'), 'w')
        f.write('F1 score: {}'.format(fscore))
        f.write('Otsu threshold: {}'.format(otsu_threshold))

        #confusion matrix
        cm_sim = confusion_matrix(gt_binary_, otsu_binary)

        label = ['No change', 'Change']
        
        # Error matrix in Percentage
        cm = cm_sim.astype('float') / np.sum(cm_sim)
        cm_plot = sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=label, yticklabels=label, cbar=False, annot_kws={"size": 30})
        cm_plot.set(xlabel= "Predicted", ylabel= "Ground truth")
            # save the plot
        plt.savefig(os.path.join(outdir,'./charts','otsu_bcd_change_matrix_percent_case_' + args.case +'.png'), dpi = 500)
        plt.close()
        
        # Error matrix in Figures
        cm_sim_plot = sns.heatmap(cm_sim, annot=True, fmt ='d', cmap='Blues', xticklabels=label, yticklabels=label, cbar=False, annot_kws={"size": 30})
        cm_sim_plot.set(xlabel= "Predicted", ylabel= "Ground truth")
            # cm_sim_plot.set_title("")
        for t in cm_sim_plot.texts:
            t.set_text('{:,d}'.format(int(t.get_text())))
            # save the plot
        plt.savefig(os.path.join(outdir, './charts', 'otsu_similarity_confusion_matrix_case_' + args.case +'.png'), dpi =500)
        plt.close()
        
        # Quality assurance
        f_score = f1_score(gt_binary_, otsu_binary)
        quality_check(args, cm_sim, f_score, method_='otsu')
        
        
        ##""" Otsu's on the whole similarity matrix##
        otsu_threshold = threshold_otsu(similarity_array)
        otsu_binary = similarity_array > otsu_threshold
        
        # Otsu's similarity histogram distribution
        plt.figure(figsize=(8,5))
        plt.hist(similarity_array, bins=10, label='similarity distribution', ec='white')
        plt.axvline(x=otsu_threshold, color='orange', linestyle='--', label="otsu's threshold: {0:0.3f}".format(otsu_threshold))
        plt.legend()
        # save chart
        plt.savefig(os.path.join(outdir,'./charts', 'otsu_whole_similarity_distribution_case_' + args.case +'.png'), dpi = 500)
        plt.close()
        
        if args.map:
            # similarity binary change
            similarity_binary = np.where(similarity_ >= opt_threshold, 2, 1)
            similarity_binary_change = np.empty(shape=(width, height)) # shape: 10980, 10980
            similarity_binary_change[~gt_binary_mask.mask] = similarity_binary.ravel()

            ## write to raster
            with rasterio.open(os.path.join(outdir, 'otsu_similarity_binary_change_case_' + args.case + '.tif'), 'w', **profile) as dst:
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

            change_map = np.empty(shape=(width, height)) # shape: 10980, 10980
            change_map[~gt_binary_mask.mask] = change_array.ravel()

            # print(np.unique(change_map))
            with rasterio.open(os.path.join(outdir, 'otsu_sim-change_map_case_' + args.case + '.tif'), 'w', **profile) as dst:
                dst.write(change_map, 1)
                dst.close()

def prepare_output(args):
        os.makedirs(args.outdir, exist_ok=True)
        os.makedirs(os.path.join(args.outdir, './charts'), exist_ok=True)

def quality_check(args, cm, f_score, method_):
    """
    method_: thresh approach used; otsu or optimal threshing
    """
    if method_=='otsu':
        title = 'otsu_QA_stats'
    else:
        title = 'QA_stats'
    with open(os.path.join(args.outdir, './charts', title + args.case + '.txt'), 'w') as f:
                f.write("Error matrix \n")
                f.write(str(cm))
                # f.write(classif_r)
                f.write("\n Total error: {}".format((cm[0,1] + cm[1,0])))
                f.write("\n OA: {}".format(((cm[0,0] + cm[1,1])/(cm[0,0] +cm[0,1]+cm[1,0]+cm[1,1]))))
                f.write("\n fscore: {}".format(f_score))
                f.close()
    
def cross_entropy_rl(p, q):
    """
    Desc: cross-entropy for LTAE (RELU output (max(0,z)); before softmax)
        p & q: (k) probability distribution of each sample vector. 
            p - source and q - target
    """
    return -(torch.matmul(p, q)).sum()
  

def cross_entropy(p, q):
    """
    Desc: cross-entropy for RF & LTAE (with softmax) == probability matrix btw 0, 1.
        p & q: (k) probability distribution of each sample vector. 
            p - source and q - target
    """
    return -(xlogy(p,q)).sum()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--outdir', '-o', default='../../../results/RF/simliarity_measure/optimal_threshold', type=str, help='Path to save files.')
    parser.add_argument('--case', '-c', default=2, type=str, help='Cases 1 to 4')
    parser.add_argument('--gt_source', '-gs', default='../../../data/rasterized_samples/2018_rasterizedImage.tif', type=str, help='Source ground truth path')
    parser.add_argument('--gt_target', '-gp', default='../../../data/rasterized_samples/2019_rasterizedImage.tif', type=str, help='Target ground truth path')
    parser.add_argument('--source_prob', '-s', type=str, help='class probability distribution')
    parser.add_argument('--target_prob', '-t', type=str, help='class probability distribution')
    parser.add_argument('--otsu', '-ot', default=False, type=bool, help='Compute optimal threshold using otsu-threshold')
    parser.add_argument('--map', '-m', default=False, type=bool, help='generate maps')
    # parser.add_argument('--percent', '-p', default=False, type=bool, help='Cal percent in the confusion matrix')
    parser.add_argument('--relu', '-r', default=False, type=bool, help='Cal percent in the confusion matrix')
    parser.add_argument('--epsilion', '-e', default=1e-7, type=float, help='')
    parser.add_argument('--optimal', '-opt', default=False, type=bool, help='')
    parser.add_argument('--percentile', '-per', default=False, type=bool, help='')
    
    args = parser.parse_args()
    main(args)


#RF
# python ce_optimal_threshold.py -o ../../../results/RF/simliarity_measure/new_optimal_threshold -c 4 -s ../../../results/RF/classificationmap/2018_rf_case_1.npy -t ../../../results/RF/classificationmap/2019_rf_case_2.npy

# python ce_optimal_threshold.py -o ../../../results/ltae/Change_detection/similarity_measure-CE -c 4 -s ../../../results/ltae/classificationmap/2018_LTAE_case_1.npy -t ../../../results/ltae/classificationmap/2019_LTAE_case_2.npy -ot True -r True
