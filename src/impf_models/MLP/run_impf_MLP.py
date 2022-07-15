import sys
sys.path.append('../../../src/')
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import config
import utils
import pickle
import argparse
import train_impf_MLP

from utils import impf_make_ndarray_from_csv, get_int_label
from Model import Impf_DNAMLP
from Dataset import CNS
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type = str, default = 'no_save')
    parser.add_argument('--alg', type = str, default = 'mlp')
    parser.add_argument('-use_smote', type = str, action = 'store_true')
    parser.add_argument('-use_weight', type = bool, action = 'store_true')
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    print('Running impf_MLP classifier')
    args = parse_args()
    save = args.save
    alg = args.alg.upper()
    clf_cfg = config.classifier_config
    impf_cfg = config.impf_config
    
    groups = utils.positive_groups
    trained_folds = utils.inner_folds
    
    eval_file = open(impf_cfg[f'MLP_{alg}_EVALUATION_RESULTS'], 'w')
    eval_file.write('EVALUATION RESULTS:\n')
    eval_file.close()
    MLP_BEST_STATES_DIR = os.path.join(impf_cfg['MLP_BEST_STATES_DIR'], alg)
    
    for group in tqdm(groups, desc = 'Groups: ', position = 0):
        for fold in tqdm(trained_folds, desc = 'Fols: ', position = 1):
            outer_fold = f'{fold.split(".")[0]}.0'
            with open(os.path.join(impf_cfg['IMPORTANT_FEATURES_DIR'], alg, group, f'{outer_fold}_combined.pkl'), 'rb') as file:
                impf = pickle.load(file)    
            tqdm.write(f'[MLP] | {group} - {fold} - {alg} ({len(impf)})')                        
            train_features, train_labels, val_features, val_labels = impf_make_ndarray_from_csv(group, fold, impf)
            test_features, test_labels = impf_make_ndarray_from_csv(group, outer_fold, impf, mode = 'test')
            
            if use_weights == True:
                torch.Tensor(compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y = train_labels )).to(device)
            else:
                class_weights = None
            if use_SMOTE == True:
                smote = SMOTE(sampling_strategy = "auto", random_state = 42, k_neighbors = max(1, min(value_counts) - 1)))
                train_features, train_labels = smote.fit_resample(train_features, train_labels)
            
            # Encode the labels
            train_labels_int = np.array([get_int_label(label, group) for label in train_labels])
            val_labels_int = np.array([get_int_label(label, group) for label in val_labels])
            test_labels_int = np.array([get_int_label(label, group) for label in test_labels])
            
            # Create datasets and Dataloaders
            train_dataset = CNS(train_features, train_labels_int, mode = 'train')
            val_dataset = CNS(val_features, val_labels_int, mode = 'val')
            test_dataset = CNS(test_features, test_labels_int, mode = 'val')
            train_loader = DataLoader(train_dataset, batch_size = impf_cfg['mlp_train_batch_size'], shuffle = True)
            val_loader = DataLoader(val_dataset, batch_size = impf_cfg['mlp_val_batch_size'], shuffle = False)
            test_loader = DataLoader(test_dataset, batch_size = impf_cfg['mlp_val_batch_size'], shuffle = False)

            # Init model object
            in_features = len(impf)
            model = Impf_DNAMLP(in_features, impf_cfg['n_classes'])
            if impf_cfg['MLP_FIRST_TIME'] == False:
                # Load model based on fold
                BEST_STATE_PATH = os.path.join(impf_cfg['MLP_BEST_STATES_DIR'], alg, group, f'{fold}.pth')
                model.load_state_dict(torch.load(BEST_STATE_PATH))
                
            # Define training and validating hyperparams
            criterion = CrossEntropyLoss(weight=class_weights)
            optimizer = Adam(model.parameters(), lr = impf_cfg['mlp_lr'], weight_decay = impf_cfg['mlp_weight_decay'])
            
            tqdm.write(f'Running in {save} mode')
            best_accs = train_impf_MLP.impf_run(group, alg, fold, train_loader, val_loader, test_loader, model, criterion, optimizer, impf_cfg, save)
            # break
        # break            
            
            
            
            
            
    
    
    

    
    
