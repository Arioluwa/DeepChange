from matplotlib.pyplot import close
import torch
import torch.utils.data as data
from torchvision import transforms
import torchnet as tnt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
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
from models.stclassifier import dLtae
from models.ltae import LTAE
from learning.focal_loss import FocalLoss
from learning.weight_init import weight_init
from learning.metrics import mIou, confusion_matrix_analysis
import wandb


def train_epoch(model, optimizer, criterion, data_loader, device, config):
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    loss_meter = tnt.meter.AverageValueMeter()
    y_true = []
    y_pred = []

    for i, (x, y) in enumerate(data_loader):
        start_time = time.time()
        y_true.extend(list(map(int, y)))

        x = recursive_todevice(x, device)
        y = y.to(device)
        # print(x.is_cuda)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y.long())
        loss.backward()
        # if config['scheduler_']:
        #     optimizer.step()
        #     scheduler.step()
        # else:
        optimizer.step()
            

        pred = out.detach()
        y_p = pred.argmax(dim=1).cpu().numpy()
        y_pred.extend(list(y_p))
        acc_meter.add(pred, y)
        loss_meter.add(loss.item())

        if (i + 1) % config['display_step'] == 0:
            print('Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}'.format(i + 1, len(data_loader), loss_meter.value()[0], acc_meter.value()[0]))
        
        print("Iteration {} completed in {:.4f} second".format(i + 1, time.time() - start_time))
        # if i +1 == int(len(data_loader)/config['factor']):  break
        # if i +1 >= config['factor']:  break
    epoch_metrics = {'train_loss': loss_meter.value()[0],
                     'train_accuracy': acc_meter.value()[0],
                     'train_IoU': mIou(y_true, y_pred, n_classes=config['num_classes'])}
    wandb.log({"train_loss": epoch_metrics['train_loss'], "train_accuracy": epoch_metrics['train_accuracy'], "train_IoU": epoch_metrics['train_IoU']})
    
    return epoch_metrics

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
        return metrics, confusion_matrix(y_true, y_pred, labels=list(range(config['num_classes']))), classification_report(y_true, y_pred, target_names=label, digits=4)

def get_loader(train_dt, val_dt, test_dt, config):
    
    loader_seq = []
    train_loader = data.DataLoader(train_dt, batch_size=config['batch_size'],
                                       num_workers=config['num_workers'],
                                       shuffle=True,
                                          pin_memory=True)
    validation_loader = data.DataLoader(val_dt, batch_size=config['batch_size'],
                                       num_workers=config['num_workers'],
                                        shuffle=True,
                                       pin_memory=True)
    test_loader = data.DataLoader(test_dt, batch_size=config['batch_size'],
                                       num_workers=config['num_workers'], 
                                        shuffle=True,
                                         pin_memory=True)
    loader_seq.append((train_loader, validation_loader, test_loader))
    return loader_seq

def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]
    
def prepare_output(config):
    os.makedirs(config['res_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['res_dir'], 'Seed_{}'.format(config['seed'])), exist_ok=True)

def checkpoint(log, config):
    with open(os.path.join(config['res_dir'], 'Seed_{}'.format(config['seed']), 'seed_{}_trainlog.json'.format(config['seed'])), 'w') as outfile:
        json.dump(log, outfile, indent=4)

def save_results(metrics, conf_mat, report, config):
    with open(os.path.join(config['res_dir'], 'Seed_{}'.format(config['seed']), 'seed_{}_test_metrics.json'.format(config['seed'])), 'w') as outfile:
        json.dump(metrics, outfile, indent=4)
    pkl.dump(conf_mat, open(os.path.join(config['res_dir'], 'Seed_{}'.format(config['seed']), 'seed_{}_conf_mat.pkl'.format(config['seed'])), 'wb'))

    with open(os.path.join(config['res_dir'], 'Seed_{}'.format(config['seed']),'seed_{}_report.txt'.format(config['seed'])), 'w') as f:
        f.write(report)
        f.close()
# def overall_performance(config):
#     cm = np.zeros((config['num_classes'], config['num_classes']))
#     # for seed in range(1, config['seed'] + 1):
#     #     cm += pkl.load(open(os.path.join(config['res_dir'], 'Seed_{}'.format(config['seed']), 'conf_mat.pkl'), 'rb'))
#     cm = pkl.load(open(os.path.join(config['res_dir'], 'Seed_{}'.format(config['seed']), 'conf_mat.pkl'), 'rb'))

#     _, perf = confusion_matrix_analysis(cm)

#     print('Overall performance:')
#     print('Acc: {},  IoU: {}'.format(perf['Accuracy'], perf['MACRO_IoU']))

#     with open(os.path.join(config['res_dir'], 'overall.json'), 'w') as file:
#         file.write(json.dumps(perf, indent=4))
    ####suggestion: KFold is the same as seed, read from the ids folder each id text

def main(config):
    # np.random.seed(config['seed'])
    # torch.manual_seed(config['seed'])
    prepare_output(config)
    wandb.login()
    mean_ = np.loadtxt(glob.glob(config['dataset_folder'] + '/*mean.txt')[0])
    std_ = np.loadtxt(glob.glob(config['dataset_folder'] + '/*std.txt')[0])
    transform = transforms.Compose([standardize(mean_, std_)])
    
    sits_data = glob.glob(os.path.join(config['dataset_folder'], 'Seed_{}'.format(config['seed'])  + '/*.npz'))
    doy = glob.glob(config['dataset_folder'] + '/gapfilled*.txt')[0]
    
    train_dt = SITSData(sits_data[2], doy, transform = transform)
    print("train dataset completed")
    val_dt = SITSData(sits_data[1], doy,  transform = transform)
    print("val dataset completed")
    test_dt = SITSData(sits_data[0], doy,  transform = transform)
    print("test dataset completed")
    
    device = torch.device(config['device'])
    
    loaders = get_loader(train_dt, val_dt, test_dt, config)
    print("Loader gotten completed")
    
    for train_loader, val_loader, test_loader in loaders:
        print('Train {}, Val {}, Test {}'.format(len(train_loader), len(val_loader), len(test_loader)))#, int(len(train_loader)/config['factor'])))

        model_config = dict(in_channels=config['in_channels'], n_head=config['n_head'], d_k=config['d_k'],
                            n_neurons=config['n_neurons'], dropout=config['dropout'], d_model=config['d_model'], mlp= config['mlp4'], T=config['T'], len_max_seq=config['len_max_seq'],
                            positions=train_dt.date_positions if config['positions'] == 'bespoke' else None)
        
        model = dLtae(**model_config)
        config['N_params'] = model.param_ratio()
        config['Train_loader_size'] = len(train_loader)
        config['Val_loader_size'] = len(val_loader)
        config['Test_loader_size'] = len(test_loader)
        
        # config['scheduler_'] = config['scheduler_']
        wandb.init(config = config)
        
        with open(os.path.join(config['res_dir'], 'Seed_{}'.format(config['seed']), 'conf.json'), 'w') as file:
            file.write(json.dumps(config, indent=4))
        # break
        model = model.to(device)
        model.apply(weight_init)
        steps_per_epoch = len(train_loader)
        
        # if config['scheduler_']:
            # optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config['weight_decay'])
            # optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'] * steps_per_epoch, eta_min=0) #T_max (int) – Maximum number of iterations.. eta_min (float) – Minimum learning rate. Default: 0.
        # else:
        optimizer = torch.optim.Adam(model.parameters())
            
        criterion = FocalLoss(config['gamma'])
        
        model = model.double() #RuntimeError: expected scalar type Double but found Float 
        
        trainlog = {}
        
        best_mIoU = 0
        st_ = time.time()
        for epoch in range(1, config['epochs'] + 1):
            print('EPOCH {}/{}'.format(epoch, config['epochs']))
            st__ = time.time()
            model.train()
            
            start_time = time.time()
            # print(torch.get_num_threads())
            # sys.exit()
            train_metrics = train_epoch(model, optimizer, criterion, train_loader, device=device, config=config)
            print("Training time for {} is {}".format(epoch, (time.time() - start_time)/60))
            
            print('Validation . . . ')
            start_time = time.time()
            model.eval()
            val_metrics = evaluation(model, criterion, val_loader, device=device, config=config, mode='val')

            print('Loss {:.4f},  Acc {:.2f},  IoU {:.4f}'.format(val_metrics['val_loss'], val_metrics['val_accuracy'],
                                                                 val_metrics['val_IoU']))
            print("Validation time for {} is {}".format(epoch, (time.time() - start_time)/60))
            wandb.log({"val_loss": val_metrics['val_loss'], "val_acc": val_metrics['val_accuracy'], "val_IoU": val_metrics['val_IoU']})
            wandb.log({"epoch": epoch})

            trainlog[epoch] = {**train_metrics, **val_metrics}
            checkpoint(trainlog, config)

            if val_metrics['val_IoU'] >= best_mIoU:
                best_mIoU = val_metrics['val_IoU']
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                           os.path.join(config['res_dir'], 'Seed_{}'.format(config['seed']), 'model.pth.tar'))
            print("total time taken for {} epoch: {:.3f} mins.".format(epoch, (time.time() - st__)/60))

        print('Testing best epoch . . .') #test on the best model only and that should be once.
        model.load_state_dict(
                torch.load(os.path.join(config['res_dir'], 'Seed_{}'.format(config['seed']), 'model.pth.tar'))['state_dict'])
        start_time = time.time()
        model.eval()

        test_metrics, conf_mat, report_ = evaluation(model, criterion, test_loader, device=device, mode='test', config=config)

        print('Loss {:.4f},  Acc {:.2f},  IoU {:.4f}'.format(test_metrics['test_loss'], test_metrics['test_accuracy'], test_metrics['test_IoU']))
        print("Test time for {} is {}".format(epoch, (time.time() - start_time)/60))
        wandb.log({"test_loss": test_metrics['test_loss'], "test_accuracy": test_metrics['test_accuracy'], "test_IoU": test_metrics['test_IoU']})
        
        
        save_results(test_metrics, conf_mat, report_, config)
        
        print("total time taken for all {} epochs: {:.3f} mins.".format(config['epochs'], (time.time() - st_)/60))

    # overall_performance(config)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Set-up parameters
    parser.add_argument('--dataset_folder', default='../../../data/theiaL2A_zip_img/output/2019', type=str, help='Path to the data folder.') #move npy and date into a folder
    parser.add_argument('--res_dir', default='../../../results/ltae/trials', help='Path to the folder where the results should be stored')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=100, type=int, help='Interval in batches between display of training metrics')
    # parser.add_argument('--scheduler_', default=False, type=bool, help='Enable scheduler')
    # parser.add_argument('--preload', dest='preload', action='store_true', help='If specified, the whole dataset is loaded to RAM at initialization')
    parser.set_defaults(preload=False)
    

    # Training parameters
    parser.add_argument('--epochs', default=2, type=int, help='Number of epochs per fold')
    parser.add_argument('--batch_size', default=2048, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--gamma', default=1, type=float, help='Gamma parameter of the focal loss')
    # parser.add_argument('--weight_decay', default=0, type=float, help='Weight decay rate')

    ## L-TAE 
    parser.add_argument('--in_channels', default=10, type=int, help='Number of channels of the input embeddings')
    parser.add_argument('--n_head', default=16, type=int, help='Number of attention heads')
    parser.add_argument('--d_k', default=8, type=int, help='Dimension of the key and query vectors')
    parser.add_argument('--n_neurons', default=[256,128], type=str, help='Number of neurons in the layers of n_neurons')
    parser.add_argument('--T', default=1000, type=int, help='Maximum period for the positional encoding')
    parser.add_argument('--positions', default='None', type=str,
                        help='Positions to use for the positional encoding (bespoke / order)')
    parser.add_argument('--len_max_seq', default=53, type=int,
                        help='Maximum sequence length for positional encoding (only necessary if positions == order)')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout probability')
    parser.add_argument('--d_model', default=256, type=int,
                        help="size of the embeddings (E), if input vectors are of a different size, a linear layer is used to project them to a d_model-dimensional space")

    ## Classifier
    parser.add_argument('--num_classes', default=19, type=int, help='Number of classes')
    parser.add_argument('--mlp4', default='[128, 64, 32, 19]', type=str, help='Number of neurons in the layers of MLP (Decoder)')
    
    config = parser.parse_args()
    config = vars(config)
    for k, v in config.items():
        if 'mlp' in k or k == 'nker':
            v = v.replace('[', '')
            v = v.replace(']', '')
            config[k] = list(map(int, v.split(',')))

    pprint.pprint(config)
    main(config)
    # python train.py --dataset_folder ../../../data/theiaL2A_zip_img/output/2018 --res_dir ../../../results/ltae/model/2018 --epochs 1