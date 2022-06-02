import torch
import torch.utils.data as data
from torchvision import transforms
import torchnet as tnt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import os
import json
import glob
import pickle as pkl
import argparse
import pprint
import time
import sys

from utils import *
from altdataset import SITSData
# from altdataset import date_positions
from models.stclassifier import dLtae
from models.ltae import LTAE
from learning.focal_loss import FocalLoss
from learning.weight_init import weight_init
from learning.metrics import mIou, confusion_matrix_analysis

def evaluation(model, criterion, loader, device, config, mode='val'):
    y_true = []
    y_pred = []

    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    loss_meter = tnt.meter.AverageValueMeter()
    label = ["Dense built-up area", "Diffuse built-up area", "Industrial and commercial areas", "Roads", "Oilseeds (Rapeseed)", "Straw cereals (Wheat, Triticale, Barley)", "Protein crops (Beans / Peas)", "Soy", "Sunflower", "Corn",  "Tubers/roots", "Grasslands", "Orchards and fruit growing", "Vineyards", "Hardwood forest", "Softwood forest", "Natural grasslands and pastures", "Woody moorlands", "Water"]
    for (x, y) in loader:
        start_time = time.time()
        y_true.extend(list(map(int, y)))
        x = recursive_todevice(x, device)
        y = y.to(device)

        with torch.no_grad():
            prediction = model(x)
            loss = criterion(prediction, y)

        acc_meter.add(prediction, y)
        loss_meter.add(loss.item())

        y_p = prediction.argmax(dim=1).cpu().numpy()
        y_pred.extend(list(y_p))
        
        print("evaluation iteration completed in {:.4f} seconds".format(time.time() - start_time))
    metrics = {'{}_accuracy'.format(mode): acc_meter.value()[0],
               '{}_loss'.format(mode): loss_meter.value()[0],
               '{}_IoU'.format(mode): mIou(y_true, y_pred, config['num_classes'])}
    
    if mode == 'val':
        return metrics
    elif mode == 'test':
        return metrics, confusion_matrix(y_true, y_pred, labels=list(range(config['num_classes']))), classification_report(y_true, y_pred, target_names=label, digits=4), cohen_kappa_score(y_true, y_pred)
    
def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]
    
# def prepare_output(config):
#     os.makedirs(config['res_dir'], exist_ok=True)
#     os.makedirs(os.path.join(config['res_dir'], 'Seed_{}'.format(config['seed'])), exist_ok=True)
    
def save_results(metrics, conf_mat, report, config, kappa, vars_):
    # save the name
    dataset_name = os.path.basename(vars_['dataset_folder']).split('.')[0]
    
    with open(os.path.join(config['res_dir'], 'Seed_{}'.format(config['seed']), '{}_test_metrics.json'.format(dataset_name)), 'w') as outfile:
        json.dump(metrics, outfile, indent=4)
    pkl.dump(conf_mat, open(os.path.join(config['res_dir'], 'Seed_{}'.format(config['seed']), '{}_conf_mat.pkl'.format(dataset_name)), 'wb'))

    with open(os.path.join(config['res_dir'], 'Seed_{}'.format(config['seed']),'{}_report.txt'.format(dataset_name)), 'w') as f:
        f.write(report)
        f.write('\n kappa coefficient \n')
        f.write(str(kappa))
        f.close()
    # with open(os.path.join(config['res_dir'], 'Seed_{}'.format(config['seed']),'{}_kappa.txt'.format(dataset_name)), 'w') as f:
    #     f.write(str(kappa))
    #     f.close()

def main(vars_):
    
    config = json.load(open(vars_['config']))
    model_path = vars_['model']
    dataset_folder = vars_['dataset_folder']
    
    state_dict = torch.load(model_path)['state_dict']
    print('loading completed')
    
    ## mean and std
    mean_ = np.loadtxt(glob.glob(dataset_folder + '/*mean.txt')[0])
    std_ = np.loadtxt(glob.glob(dataset_folder + '/*std.txt')[0])
    transform = transforms.Compose([standardize(mean_, std_)])
    
    ## get datasets and dates
    sits_data = glob.glob(os.path.join(dataset_folder, 'Seed_{}'.format(config['seed'])  + '/*.npz'))
    doy = glob.glob(dataset_folder + '/gapfilled*.txt')[0]
    
    ## dataset
    print('start...')
    test_dt = SITSData(sits_data[0], doy,  transform = transform)
    print('done...')
    ## dataloader
    test_loader = data.DataLoader(test_dt, batch_size=config['batch_size'],
                                       num_workers=config['num_workers'], 
                                        shuffle=True,
                                         pin_memory=True)
    
    # state_dict = torch.load(model_path)['state_dict']
    
    model = dLtae(in_channels = config['in_channels'], n_head = config['n_head'], d_k= config['d_k'], n_neurons=config['n_neurons'], dropout=config['dropout'], d_model= config['d_model'],
                 mlp = config['mlp4'], T =config['T'], len_max_seq = config['len_max_seq'], 
              positions=test_dt.date_positions if config['positions'] == 'bespoke' else None, return_att=False)
    
    # device = config['device']
    device = 'cpu'
    model = model.to(device)
    model = model.double()
    model.load_state_dict(state_dict)
    
    model.eval()
    
    criterion = FocalLoss(config['gamma'])
    
    ## evaluation
    start_time = time.time()
    test_metrics, conf_mat, report_, kappa = evaluation(model, criterion, test_loader, device=device, mode='test', config=config)
    print("testing completed in: ", (time.time() - start_time))
    save_results(test_metrics, conf_mat, report_, config, kappa, vars_)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', '-m', type=str, help='Path to the model file .Pth.')
    parser.add_argument('--dataset_folder', '-d', type=str, help='Path to the dataset folder.')
    parser.add_argument('--config', '-c', type=str, help='Path to config file.')
    
    vars_ = parser.parse_args()
    vars_ = vars(vars_)
    
    main(vars_)
    
# python test.py -m ../../../results/ltae/model/2018/Seed_0/model.pth.tar -d ../../../data/theiaL2A_zip_img/output/2019 -c ../../../results/ltae/model/2018/Seed_0/conf.json