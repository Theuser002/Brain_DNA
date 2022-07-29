import sys
sys.path.append('/media/data/hungnt/work/Datasets/BTMD/Code/src')

import os
import sys
import torch
import numpy as np
import cal_train
import utils
import config
import argparse

from utils import get_int_label
from Cal_Dataset import Cal_Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from Cal_Model import CalNet
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type = str, default='no_save')
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    print('Training calibration model')
    args = parse_args()
    save = args.save
    cfg = config.config_dict
    
    if cfg['CAL_FIRST_TIME'] == True:
        print('dumping ')
        cal_train.dump_probs()
    
    for i in range(1, 6):
        outer_fold = f'{i}.0'
        
        probs, labels = cal_train.load_probs(outer_fold)
        int_labels = np.array([get_int_label(label) for label in labels])
        probs = np.array(probs)
        
        train_probs, val_probs, train_labels, val_labels = train_test_split(probs, int_labels, test_size = 0.3, random_state = 42, shuffle = True, stratify = int_labels)
        
        # Create datasets and Dataloaders
        cal_train_dataset = Cal_Dataset(train_probs, train_labels)
        cal_val_dataset = Cal_Dataset(val_probs, val_labels)
        cal_train_loader = DataLoader(cal_train_dataset)
        cal_val_loader = DataLoader(cal_val_dataset)
        
        print(len(probs), len(int_labels), len(cal_train_dataset), len(cal_val_dataset))
        
        # Init model object
        in_features = cfg['cal_n_features']
        cal_model = CalNet(in_features, cfg['cal_n_classes'])
        if cfg['CAL_FIRST_TIME'] == False:
            # Load model based on fold
            CAL_BEST_STATE_PATH = os.path.join(cfg['MLP_CAL_BEST_STATES_DIR'], f'{outer_fold}_best_cal_state.pth')
            cal_model.load_state_dict(torch.load(CAL_BEST_STATE_PATH))
        
        # Define training and validating hyperparams
        cal_criterion = CrossEntropyLoss(weight = None)
        cal_optimizer = Adam(cal_model.parameters(), lr = cfg['cal_lr'], weight_decay = cfg['weight_decay'])
        # for j in range(0, len(cal_val_dataset)):
        #     print(np.argmax(cal_train_dataset[j][0]), cal_train_dataset[j][1])
        
        if save == 'save':
            print('Running in save mode')
            best_accs = cal_train.cal_run(outer_fold, cal_train_loader, cal_val_loader, cal_model, cal_criterion, cal_optimizer, cfg)
        else:
            print('Running in no save mode')
            best_accs = cal_train.cal_run_no_save(outer_fold, cal_train_loader, cal_val_loader, cal_model, cal_criterion, cal_optimizer, cfg)