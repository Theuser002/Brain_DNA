import sys
sys.path.append('../../src/')

import os
import numpy as np
import utils
import config
import joblib
import pickle
import torch

from models.MLP.Model import DNAMLP
from tqdm import tqdm
from tqdm.auto import trange
from xgboost import XGBClassifier

dat_cfg = config.data_config
clf_cfg = config.classifier_config
impf_cfg = config.impf_config

def fold_mlp_impf (group, fold):
    # Load column names corresponding to fold from pickle file:
    FEATURE_NAMES_FILE = os.path.join(impf_cfg['FEATURE_NAMES_DIR'], f'{fold}.pkl')
    with open(FEATURE_NAMES_FILE, 'rb') as handle:
        features = pickle.load(handle)
        
    # Load model then calculate features importance
    ## Load model
    MLP_BEST_STATES_FILE = os.path.join(clf_cfg['MLP_BEST_STATES_DIR'], group, f'{fold}.pth')
    mlp = DNAMLP(clf_cfg['n_features'], clf_cfg['n_classes'])
    mlp.load_state_dict(torch.load(MLP_BEST_STATES_FILE))
    mlp_weights = mlp.densenet[0].weight.detach().cpu().numpy()
    ## Define the way we calculate feature importance for mlp model: Take the first row (corresponding to the predicted class) elements
    mlp_feature_importance = mlp_weights[0]
    ## Take absolute value
    mlp_feature_importance = list(map(abs, mlp_feature_importance))
    
    # Get important features
    sorted_indexes = np.argsort(mlp_feature_importance)
    imp_indexes = sorted_indexes[-(impf_cfg['N_IMPF']):]
    important_features = [features[i] for i in imp_indexes] 
       
    return list(important_features)

def fold_rf_impf (group, fold):
    # Load column names corresponding to fold from pickle file:
    FEATURE_NAMES_FILE = os.path.join(impf_cfg['FEATURE_NAMES_DIR'], f'{fold}.pkl')
    with open(FEATURE_NAMES_FILE, 'rb') as handle:
        features = pickle.load(handle)
    
    # Load model and RF model then calculate features importance
    RF_MODEL_FILE = os.path.join(clf_cfg['RF_MODELS_DIR'], group, f'{fold}.joblib')
    rf = joblib.load(RF_MODEL_FILE)
    rf_feature_importance = rf.feature_importances_
    sorted_indexes = np.argsort(rf_feature_importance)
    imp_indexes = sorted_indexes[-(impf_cfg['N_IMPF']):]
    important_features = [features[i] for i in imp_indexes]
           
    return list(important_features)

def fold_lr_impf (group, fold):
    # Load column names corresponding to fold from pickle file: 
    FEATURE_NAMES_FILE = os.path.join(impf_cfg['FEATURE_NAMES_DIR'], f'{fold}.pkl')
    with open(FEATURE_NAMES_FILE, 'rb') as handle:
        features = pickle.load(handle)
    
    # Load model and RF model then calculate features importance
    LR_MODEL_FILE = os.path.join(clf_cfg['LR_MODELS_DIR'], group, f'{fold}.joblib')
    lr = joblib.load(LR_MODEL_FILE)
    lr_feature_importance = lr.coef_[0]
    sorted_indexes = np.argsort(lr_feature_importance)
    imp_indexes = sorted_indexes[-(impf_cfg['N_IMPF']):]
    important_features = [features[i] for i in imp_indexes]
           
    return list(important_features)

def fold_xgb_impf (group, fold):
    # Load column names corresponding to fold from pickle file:
    FEATURE_NAMES_FILE = os.path.join(impf_cfg['FEATURE_NAMES_DIR'], f'{fold}.pkl')
    with open(FEATURE_NAMES_FILE, 'rb') as handle:
        features = pickle.load(handle)

    XGB_MODEL_FILE = os.path.join(clf_cfg['XGB_MODELS_DIR'], group, f'{fold}.json')
    xgb = XGBClassifier()
    xgb.load_model(XGB_MODEL_FILE)
    xgb_feature_importance = xgb.feature_importances_
    sorted_indexes = np.argsort(xgb_feature_importance)
    imp_indexes = sorted_indexes[-(impf_cfg['N_IMPF']):]
    important_features = [features[i] for i in imp_indexes]
           
    return list(important_features)

def choose_impf_alg(alg, group, fold):
    switcher = {
        'mlp': fold_mlp_impf(group, fold),
        'rf': fold_rf_impf(group, fold),
        'lr': fold_lr_impf(group, fold),
        'xgb': fold_xgb_impf(group, fold)  
    }
    
    return switcher.get(alg.lower(), None)

def dump_impf (algs, groups):
    for alg in tqdm(algs, desc= 'Algorithms',  position=0):
        IMPF_DIR =  os.path.join(impf_cfg['IMPORTANT_FEATURES_DIR'], alg)
        for group in tqdm(groups, desc='Class groups', position=1):
            for i in tqdm(range(1, 6), desc='Outer folds', position=2):
                out_fold = f'{i}.0'
                out_impf = choose_impf_alg(alg, group, out_fold)
                for j in tqdm(range(0, 6), desc='Inner folds', position=3):
                    in_fold = f'{i}.{j}'
                    in_impf = choose_impf_alg(alg, group, in_fold)
                    impf = set(in_impf).intersection(set(out_impf))
                    with open (os.path.join(IMPF_DIR, group, f'{in_fold}.pkl'), 'wb') as handle:
                        pickle.dump(in_impf, handle, protocol = pickle.HIGHEST_PROTOCOL)
                    with open (os.path.join(IMPF_DIR, group, f'{out_fold}-{in_fold}.pkl'), 'wb') as handle:
                        pickle.dump(impf, handle, protocol = pickle.HIGHEST_PROTOCOL)

def outer_fold_impf(alg, group, outer_fold, save = False):
    fold_index = outer_fold.split('.')[0]
    IMPF_DIR =  os.path.join(impf_cfg['IMPORTANT_FEATURES_DIR'], alg)
    IMPF_FILE = os.path.join(IMPF_DIR, group, f'{outer_fold}_combined.pkl')
    combined_impf = []
    for i in tqdm(range(0,6)):
        fold = f'{fold_index}.{i}'
        impf = choose_impf_alg(alg, group, fold)
        combined_impf.append(set(impf))
    impf_set = set.intersection(*combined_impf)
    if save == True:
        impf_file = open(IMPF_FILE, 'wb');
        pickle.dump(impf_set, impf_file, protocol = pickle.HIGHEST_PROTOCOL)
    
    return impf_set

if __name__ == "__main__": 
    print(f'Getting {impf_cfg["N_IMPF"]} important features')
    algs = ['MLP', 'RF', 'LR', 'XGB']
    groups = utils.positive_groups
    outer_folds = utils.outer_folds
    group = groups[np.random.randint(len(groups))]
    outer_fold = outer_folds[np.random.randint(len(utils.outer_folds))]    
    
    # for alg in tqdm(algs, desc= 'Algorithms',  position=0):
    #     IMPF_DIR =  os.path.join(impf_cfg['IMPORTANT_FEATURES_DIR'], alg)
    #     for group in tqdm(groups, desc='Class groups', position=1):
    #         for i in tqdm(range(1, 6), desc='Outer folds', position=2):
    #             out_fold = f'{i}.0'
    #             out_impf = set(choose_impf_alg(alg, group, out_fold))
    #             for j in tqdm(range(0, 6), desc='Inner folds', position=3):
    #                 in_fold = f'{i}.{j}'
    #                 in_impf = set(choose_impf_alg(alg, group, in_fold))
    #                 impf = sedt(in_impf).intersection(set(out_impf))
    #                 with open (os.path.join(IMPF_DIR, group, f'{in_fold}.pkl'), 'wb') as handle:
    #                     pickle.dump(in_impf, handle, protocol = pickle.HIGHEST_PROTOCOL)
    #                 with open (os.path.join(IMPF_DIR, group, f'{out_fold}-{in_fold}.pkl'), 'wb') as handle:
    #                     pickle.dump(impf, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    # dump_impf(algs, groups)
    # for alg in tqdm(algs, desc = 'Algs: ', position = 0):
    #     for group in tqdm(groups, desc = 'Groups', position = 1):
    #         for outer_fold in tqdm(outer_folds, desc = 'Outer folds', position = 2):
    #             tqdm.write(f'{alg}-{group}-{outer_fold}:')
    #             tqdm.write(f'{len(outer_fold_impf(alg, group, outer_fold, True))}')
    
    group = 'Pineal'
    fold = '1.1'
    print(choose_impf_alg(algs[0], group, fold))
    print(choose_impf_alg(algs[1], group, fold))
    print(choose_impf_alg(algs[2], group, fold))
    print(choose_impf_alg(algs[3], group, fold))
    
    fold = '1.1'
    group = 'Pineal'
    i = 0
    j = 0
    for i in range(len(algs)):
        for j in range(i+1, len(algs)):
            print(f'{algs[i]}-{algs[j]}: ', len(set(choose_impf_alg(algs[i], group, fold)).intersection(set(choose_impf_alg(algs[j], group, fold)))))
    
    print('(XGB-RF)-MLP: ', len(set(choose_impf_alg(algs[3], group, fold)).intersection(set(choose_impf_alg(algs[1], group, fold)).intersection(set(choose_impf_alg(algs[0], group, fold))))))