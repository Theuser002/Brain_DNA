import sys
sys.path.append('../../../src/')

import sys
import os
import torch
import config
import train_MLP
import numpy as np

from utils import make_ndarray_from_csv, get_int_label
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from Dataset import CNS
from Model import DNAMLP
from Calibrate.MLP.Cal_Model import CalNet
from torch.nn.functional import softmax, one_hot
from utils import brier_score_tensor
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def val_with_cal (fold, val_loader, criterion, config):
    correct = 0
    total = 0
    total_loss = 0
    total_bs = 0
    device = config['device']
    class_in_features = config['n_features']
    cal_in_features = config['cal_n_features']
    class_model = DNAMLP(class_in_features, config['n_classes'])
    cal_model = CalNet(cal_in_features, config['cal_n_classes'])
    
    CLASS_BEST_STATE_PATH = os.path.join(config['MLP_BEST_STATES_DIR'], f'{fold}_best_state.pth')
    CAL_BEST_STATE_PATH = os.path.join(config['MLP_CAL_BEST_STATES_DIR'], f'{fold}_best_cal_state.pth')
    print(CLASS_BEST_STATE_PATH, CAL_BEST_STATE_PATH)
    class_model.load_state_dict(torch.load(CLASS_BEST_STATE_PATH))
    cal_model.load_state_dict(torch.load(CAL_BEST_STATE_PATH))
    
    class_model.to(device)
    cal_model.to(device)
    
    with torch.no_grad():
        all_labels = []
        all_logits = []
        for features, labels in tqdm(val_loader):
            # Move tensors to device:
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass:
            logits = class_model(features)
            probs = softmax(logits, dim = 1)
            cal_logits = cal_model(probs)
            
            # Evaluation and batch loss
            loss = criterion(cal_logits, labels)
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            
            # Batch BS
            batch_bs = brier_score_tensor(cal_logits, labels)
            total_bs += batch_bs
            
            # Save logits and labels to calculate AUC
            for logit, label in zip(cal_logits, labels):
                all_labels.append(label.item())
                all_logits.append(np.array(logit.detach().cpu().numpy()))
            
        # Epoch's average LL
        val_loss = total_loss / len(val_loader)
        
        # Epoch's average acc & ME
        val_acc = (correct / total) * 100
        val_me = (100 - val_acc)
        
        # Epoch's average BS
        val_bs = total_bs / len(val_loader)
        
        # Epoch's AUC
        all_labels_one_hot = one_hot(torch.Tensor(np.array(all_labels)).long())
        all_probs = softmax(torch.Tensor(np.array(all_logits)), dim = 1)
        val_auc = roc_auc_score(all_labels_one_hot, all_probs)
        
    return val_loss, val_acc, val_me, val_bs, val_auc
            

if __name__ == "__main__":
    cfg = config.config_dict
    print('Validate on outerfold test sets with calibration model')
    
    for i in range(1, 6):
        outer_fold = f'{i}.0'
        print(f'Fold {i}.0')
        
        # Create dataset(s) and DataLoader(s)
        val_features, val_labels = make_ndarray_from_csv(outer_fold, mode = 'test')
        val_labels_int = np.array([get_int_label(label) for label in val_labels])
        
        val_dataset = CNS(val_features, val_labels_int, mode = 'val')
        val_loader = DataLoader(val_dataset, batch_size = cfg['val_batch_size'], shuffle = False)
        
        # Init model object
        in_features = cfg['n_features']
        model = DNAMLP(in_features, cfg['n_classes'])
        model = model.to(cfg['device'])
        # Load model based on fold
        BEST_STATE_PATH = os.path.join(cfg['MLP_BEST_STATES_DIR'], f'{outer_fold}_best_state.pth')
        model.load_state_dict(torch.load(BEST_STATE_PATH))
        
        # Define validating hyperparams
        criterion = CrossEntropyLoss(weight = None)
        loss, acc, me, bs, auc = val_with_cal(outer_fold, val_loader, criterion, cfg)
        print('val_loss: %.5f | val_acc: %.3f | val_me: %3f | val_bs: %.3f | val_auc: %.3f' % (loss, acc, me, bs, auc))
        