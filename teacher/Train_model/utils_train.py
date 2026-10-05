# functions for calculating metrics, and save result
import os
import torch
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score


# calculate confusion metrics
def cal_metrics(output, labels, method='macro'):
    if output.get_device() != 'cpu':
        output = output.detach().cpu()
    if labels.get_device() != 'cpu':
        labels = labels.detach().cpu()
        
    pred = torch.argmax(output, dim=1)
    
    acc = accuracy_score(pred, labels)
    f1 = f1_score(pred, labels, average=method, zero_division=0)
    rec = recall_score(pred, labels, average=method, zero_division=0)
    prec = precision_score(pred, labels, average=method, zero_division=0)
    
    return acc, f1, rec, prec

# save model to path
def save_checkpoint(save_path, model):
    if save_path == None:
        return
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to ==> {save_path} at {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')

# load model from path
def load_checkpoint(load_path, model, device):
    if load_path==None:
        return
    state_dict = torch.load(load_path, map_location=device)
    print(f'\nModel loaded from <== {load_path}')

    model.load_state_dict(state_dict)
    return model

# save training result
def save_result(res_fname, content, save_dir='./result/'):

    if '.pt' in res_fname:
        res_fname = res_fname[:-3]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(save_dir + res_fname + '.txt', 'w') as f:
        f.write(content)
    
    print('\n===>   Result are save to ', save_dir + res_fname + '.txt')

