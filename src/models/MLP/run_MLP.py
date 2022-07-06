import sys
sys.path.append('/media/data/hungnt/work/SourceCode/Brain_DNA/src/')
import warnings
warnings.filterwarnings("ignore")


import os
import pandas as pd
import numpy as np
import torch
import pyreadr
import config
import Dataset
import time
import train_MLP
import argparse
import utils

from Dataset import CNS
from torch.utils.data import DataLoader
from Model import DNAMLP
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from utils import make_ndarray_from_csv, get_int_label

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type = str, default='no_save')
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    print("Running mlp classifiers")
    args = parse_args()
    save = args.save
    clf_cfg = config.classifier_config
    
    eval_file = open(clf_cfg['MLP_EVALUATION_RESULTS'], 'w')
    eval_file.write('EVALUATION RESULTS:\n')
    eval_file.close()
    
    print(f'root: {config.root_dir}')
    print(f"device: {clf_cfg['device']}")
    print(f'save mode: {save}')
    # All folds
    folds = ['1.0', '1.1', '1.2', '1.3', '1.4', '1.5', 
             '2.0', '2.1', '2.2', '2.3', '2.4', '2.5', 
             '3.0', '3.1', '3.2', '3.3', '3.4', '3.5',
             '4.0', '4.1', '4.2', '4.3', '4.4', '4.5',
             '5.0', '5.1', '5.2', '5.3', '5.4', '5.5']
    outer_folds = ['1.0', '2.0', '3.0', '4.0', '5.0']
    inner_folds = [x for x in folds if x not in outer_folds]
    # Remove some folds that are not going to be trained
    trained_folds = folds
    
    groups = utils.positive_groups
    
    # Train the inner folds
    for group in tqdm(groups, desc='Groups: ', position=0):
        for fold in tqdm(trained_folds, desc='Folds: ', position=1):
            # Read from csv to dataframe
            train_features, train_labels, val_features, val_labels = make_ndarray_from_csv(group, fold)
            
            # Encode the labels
            train_labels_int = np.array([get_int_label(label, group) for label in train_labels])
            val_labels_int = np.array([get_int_label(label, group) for label in val_labels])
            
            # Create datasets and Dataloaders
            train_dataset = CNS(train_features, train_labels_int, mode = 'train')
            val_dataset = CNS(val_features, val_labels_int, mode = 'val')
            train_loader = DataLoader(train_dataset, batch_size = clf_cfg['mlp_train_batch_size'], shuffle = True)
            val_loader = DataLoader(val_dataset, batch_size = clf_cfg['mlp_val_batch_size'], shuffle = False)
            
            # Init model object
            in_features = clf_cfg['n_features']
            model = DNAMLP(in_features, clf_cfg['n_classes'])
            if clf_cfg['MLP_FIRST_TIME'] == False:
                # Load model based on fold
                BEST_STATE_PATH = os.path.join(clf_cfg['MLP_BEST_STATES_DIR'], group, f'{fold}.pth')
                model.load_state_dict(torch.load(BEST_STATE_PATH))
            
            # Define training and validating hyperparams
            criterion = CrossEntropyLoss(weight=None)
            optimizer = Adam(model.parameters(), lr = clf_cfg['mlp_lr'], weight_decay = clf_cfg['mlp_weight_decay'])
            tqdm.write(f'Running in {save} mode')
            best_accs = train_MLP.run(group, fold, train_loader, val_loader, model, criterion, optimizer, clf_cfg, save)
        

        