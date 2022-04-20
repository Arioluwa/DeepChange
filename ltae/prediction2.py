import torch
import torch.utils.data as data
from torchvision import transforms
import torchnet as tnt
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import json
import pickle as pkl

from utils import *
from dataset import SITSData
from models.stclassifier import dLtae
from models.ltae import LTAE
from learning.focal_loss import FocalLoss
from learning.weight_init import weight_init
from learning.metrics import mIou, confusion_matrix_analysis

try:
    import gdal, osr
    from gdalconst import *
except:
    from osgeo import gdal, osr
    from osgeo.gdalconst import *

def evaluation(model, criterion, loader, device, config):
    y_true = []
    y_pred = []

    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    loss_meter = tnt.meter.AverageValueMeter()

    for (x, y) in loader:
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

    metrics = {'{}_accuracy'.format(mode): acc_meter.value()[0],
               '{}_loss'.format(mode): loss_meter.value()[0],
               '{}_IoU'.format(mode): mIou(y_true, y_pred, config['num_classes'])}

    
    return metrics, confusion_matrix(y_true, y_pred, labels=list(range(config['num_classes'])))
    
