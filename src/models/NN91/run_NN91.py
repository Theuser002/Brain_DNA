import os
import pandas as pd
import numpy as np
import torch
import pyreadr
import config_91
import argparse
import utils_91
import train_NN91
import json
import pickle

from Dataset import CNS
from torch.utils.data import DataLoader
from Model import NN91
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from utils_91 import make_ndarray_from_csv, get_int_label
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from sklearn import preprocessing
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-scale_data', action='store_true')
    parser.add_argument('--save', type= str, default='no_save')
    parser.add_argument('--single_fold', default='disabled')
    parser.add_argument('-use_smote', action='store_true')
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    save = args.save
    scale_data = args.scale_data
    single_fold = args.single_fold
    use_SMOTE = args.use_smote
    clf_cfg = config_91.classifier_config
    device = clf_cfg['device']
    
    eval_file = open(clf_cfg['NN91_EVALUATION_RESULTS'], 'w')
    eval_file.write('EVALUATION RESULTS:\n')
    eval_file.write(f'SMOTE: {use_SMOTE} \n')
    eval_file.close()
    
    print('Runnig nn91 classifier')
    print(f'root: {config_91.root_dir}')
    print(f"device: {device}")
    print(f'save mode: {save}')
    print(f'SMOTE: {use_SMOTE}')
    print(f'single fold: {single_fold}')
    print('\n')
    
    folds = utils_91.folds
    outer_folds = utils_91.outer_folds
    inner_folds = utils_91.inner_folds
    
    if single_fold.lower() == 'disabled':
        trained_folds = outer_folds
    else:
        trained_folds = [single_fold]

    folds_results = {}
    for fold in tqdm(trained_folds, desc='Folds: ', position=0):
        folds_results[f'{fold}'] = {}
        train_features, train_labels = make_ndarray_from_csv(fold, mode='train')
        val_features, val_labels = make_ndarray_from_csv(fold, mode = 'val')
        
        if scale_data == True:
            scaler = preprocessing.StandardScaler().fit(train_features)
            train_features = scaler.transform(train_features)
            val_features = scaler.transform(val_features)
        value_counts = pd.Series(train_labels).value_counts()
        
        if use_SMOTE == True:
            smote = SMOTE(sampling_strategy = "auto", random_state = 42, k_neighbors = max(1, min(value_counts) - 1))
            train_features, train_labels = smote.fit_resample(train_features, train_labels)
        
        # Label ecoding
        train_labels_int = np.array([get_int_label(label) for label in train_labels])
        val_labels_int = np.array([get_int_label(label) for label in val_labels])
        
        # Dataset creation
        train_dataset = CNS(train_features, train_labels_int, mode='train')
        val_dataset = CNS(val_features, val_labels_int, mode='val')
        train_loader = DataLoader(train_dataset, batch_size=clf_cfg['nn91_train_batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=clf_cfg['nn91_val_batch_size'], shuffle=True)
        
        # Init model object
        in_features = clf_cfg['n_features']
        n_classes = clf_cfg['n_classes']
        model = NN91(in_features, n_classes)
        if clf_cfg['NN91_FIRST_TIME'] == False:
            BEST_STATE_PATH = os.path.join(clf_cfg['NN91_BEST_STATES_DIR'], f'{fold}.pth')
            model.load_state_dict(torch.load(BEST_STATE_PATH))
        
        # Define training parameters
        criterion = CrossEntropyLoss(weight=None)
        optimizer = Adam(model.parameters(), lr = clf_cfg['nn91_lr'], weight_decay=clf_cfg['nn91_weight_decay'])
        
        # Run the training pipeline
        fold_results, fold_history = train_NN91.run(fold, train_loader, val_loader, model, criterion, optimizer, clf_cfg, save)
        # fold_results['train_cfs'] = fold_results['train_cfs'].tolist()
        # fold_results['val_cfs'] = fold_results['val_cfs'].tolist()
        fold_results['train_cfs'] = []
        fold_results['val_cfs'] = []
        folds_results[f'{fold}'] = fold_results
        
        with open(os.path.join(clf_cfg['NN91_HISTORY_DIR'], f'{fold}.pkl'), 'wb') as handle:
            pickle.dump(fold_history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close() 
        
    with open(clf_cfg['NN91_EVALUATION_JSON'], 'w') as fp:
        json.dump(folds_results, fp)
    fp.close()