import sys
sys.path.append('../../../src')
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
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from sklearn import preprocessing

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-use_smote', action = 'store_true')
    parser.add_argument('-use_weights', action = 'store_true')
    parser.add_argument('-scale_data', action = 'store_true')
    parser.add_argument('--single_fold', default = 'disabled')
    parser.add_argument('--save', type = str, default='no_save')
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    save = args.save
    use_SMOTE = args.use_smote
    use_weights = args.use_weights
    scale_data = args.scale_data
    single_fold = args.single_fold
    clf_cfg = config.classifier_config
    device = clf_cfg['device']
    
    eval_file = open(clf_cfg['MLP_EVALUATION_RESULTS'], 'w')
    eval_file.write('EVALUATION RESULTS:\n')
    eval_file.write(f'SMOTE: {use_SMOTE} | weights: {use_weights}\n')
    eval_file.close()
    
    print('Running mlp classifier')
    print(f'root: {config.root_dir}')
    print(f"device: {device}")
    print(f'save mode: {save}')
    print(f'SMOTE: {use_SMOTE} | weights: {use_weights}')
    print(f'single fold: {single_fold}')
    print('\n')

    folds = utils.folds
    outer_folds = utils.outer_folds
    inner_folds = utils.inner_folds
    if single_fold.lower() == 'disabled':
        trained_folds = folds
    else:
        trained_folds = [single_fold]
    
    # Train the inner folds
    for group in tqdm(groups, desc='Groups: ', position=0):
        for fold in tqdm(trained_folds, desc='Folds: ', position=1):

            # Read from csv to dataframe
            train_features, train_labels = make_ndarray_from_csv(group, fold, mode = 'train')
            val_features, val_labels = make_ndarray_from_csv(group, fold, mode = 'test')
            # SCALE DATA TO STANDARD DISTRIBUTION (MEAN 0. AND STD DEVIATION = 1.)
            if scale_data == True:
                scaler = preprocessing.StandardScaler().fit(train_features)
                train_features = scaler.transform(train_features)
                val_features = scaler.transform(val_features)

            value_counts = pd.Series(train_labels).value_counts()
            if use_SMOTE == True:
                smote = SMOTE(sampling_strategy = "auto", random_state = 42, k_neighbors = max(1, min(value_counts) - 1))
                train_features, train_labels = smote.fit_resample(train_features, train_labels)
            if use_weights == True:
                class_weights = torch.Tensor(compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y = train_labels )).to(device)
            else:
                class_weights = None
            
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
            model = DNAMLP(in_features, clf_cfg['n_classes'], clf_cfg['mlp_dropout_rate'])
            if clf_cfg['MLP_FIRST_TIME'] == False:
                # Load model based on fold
                BEST_STATE_PATH = os.path.join(clf_cfg['MLP_BEST_STATES_DIR'], group, f'{fold}.pth')
                model.load_state_dict(torch.load(BEST_STATE_PATH))
            
            # Define training and validating hyperparams
            criterion = CrossEntropyLoss(weight=None)
            optimizer = Adam(model.parameters(), lr = clf_cfg['mlp_lr'], weight_decay = clf_cfg['mlp_weight_decay'])
            
            # Run the training - validating pipeline
            fold_results = train_MLP.run(group, fold, train_loader, val_loader, model, criterion, optimizer, clf_cfg, save)
        

        