import os
import rasterio
import numpy as np
import pandas as pd

def changeErrorCheck(args):
    """
    
    """    
    # read rasters
    gt_source_ = rasterio.open(args.gt_source).read(1).flatten().astype('int')
    gt_target_ = rasterio.open(args.gt_target).read(1).flatten().astype('int')
    pred_source_ = rasterio.open(args.pred_source).read(1).flatten()
    pred_target_ = rasterio.open(args.pred_target).read(1).flatten()
    
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
    
    #compute confusion matrix
    cm = pd.crosstab(df['gt_change'], df['pred_change'], rownames=['Ground Truth'], colnames=['Prediction'])
    # cm = pd.crosstab(df['pred_change'], df['gt_change'], colnames=['Prediction'], rownames=['Ground Truth'])
    
    # https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2
    # https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1
    
    # cm.to_csv(os.path.join(outdir, "semantic_change_matrix_case_" + args.case + '.csv'))
    
    cm['recall'] = 0
    cm.loc['precision'] = 0
    cm['f1'] = 0
    
    # compute f1, recall and precision, only for similar changes in the ground truth
    for rname in cm.index:
        for cname in cm.columns:
            if rname == cname:
                cm['recall'].loc[rname] = (cm.loc[rname, cname] / cm.sum()[rname]) * 100 
                cm[cname].loc['precision'] = cm.loc[rname, cname] / cm.sum(axis = 1)[cname] * 100
                cm['f1'].loc[rname] = 2 * (cm.loc[rname, 'recall'] * cm.loc['precision', cname]) / (cm.loc[rname, 'recall'] + cm.loc['precision', cname])
    cm_ = cm[cm['recall'] != 0]
    prec = cm.iloc[[-1]]
    prec = prec.loc[:, (prec !=0).any(axis=0)]
    t = prec.T
    t.index.name = None
    t.columns.name = None
    t.reset_index(drop=True, inplace=True)
    def check_semantic_change(change_str):
    change_str = change_str.split("-")
    if change_str[0] == change_str[1]:
        return "no change"
    else:
        return "change"
    
    t['cat'] = t['changes'].apply(check_semantic_change)
    t_change = t[t['cat'] == 'change'].sort_values(by=['f1'], ascending=False)
    t_nochange = t[t['cat'] == 'no change'].sort_values(by=['f1'], ascending=False)
    # bar plot
    t_nochange.plot.bar(rot=0, figsize=(10,5), fontsize=10, color=['gray', '#1F77B4', '#FF7F0E'])
    plt.xticks(range(len(t_nochange)), t_nochange['changes'], rotation=90)
    plt.title('No-change in semantic class')
    plt.savefig(os.path.join(outdir, "./charts", "semantic-nochange"+args.case+".png"), dpi=3000)
    plt.show()

if __name__ == '__main__':
    gt_source_path = '../../../data/rasterized_samples/2018_rasterizedImage.tif'
    gt_target_path = '../../../data/rasterized_samples/2019_rasterizedImage.tif'
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--outdir', '-o', default='../../../results/RF/binary_change_D', type=str, help='Path to save the BCD tif files.')
    parser.add_argument('--case', '-c', default=2, type=str, help='Cases 1 to 4')
    parser.add_argument('--gt_source', '-gs', default='../../../data/rasterized_samples/2018_rasterizedImage.tif', type=str, help='Source ground truth path')
    parser.add_argument('--gt_target', '-gp', default='../../../data/rasterized_samples/2019_rasterizedImage.tif', type=str, help='Target ground truth path')
    parser.add_argument('--pred_source', '-ps', default='../../../results/RF/2018_rf_case_2_map.tif', type=str, help='Source classification map')
    parser.add_argument('--pred_target', '-pt', default='../../../results/RF/2019_rf_case_3_map.tif', type=str, help='Target  classification map')
    
    args = parser.parse_args()
    # args = vars(args) # if needed as a dict... call as args['case']
    pprint.pprint(vars(args))
    # main(args)
    
    for case in range(1, 4):
        args.case = str(case)
        args.pred_source = '../../../results/RF/2018_rf_case_{}_map.tif'.format(case)
        args.pred_target = '../../../results/RF/2019_rf_case_{}_map.tif'.format(case)
        main(args)
        print('Case {} done'.format(case))
    # case 4
    args.case = "4"
    # args.percent = True
    args.pred_source = '../../../results/RF/2018_rf_case_2_map.tif'
    args.pred_target = '../../../results/RF/2019_rf_case_3_map.tif'
    main(args)
    print('Case {} done'.format(args.case))
    
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
    