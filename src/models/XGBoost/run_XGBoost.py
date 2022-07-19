import sys
sys.path.append('../../../src/')

import os
import config
import utils
import numpy as np
import pandas as pd
import argparse
import utils

from xgboost import XGBClassifier
from utils import make_ndarray_from_csv, get_int_label
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score 
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_sample_weight
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

def eval(clf, features, labels_int):
    all_probs = clf.predict_proba(features)
    all_preds = [np.argmax(prob) for prob in all_probs]
    acc = clf.score(features, labels_int)
    precision = precision_score(labels_int, all_preds)
    recall = recall_score(labels_int, all_preds)
    auc = roc_auc_score(labels_int, all_preds)
    f1 = f1_score(labels_int, all_preds)
    cfs = confusion_matrix(labels_int, all_preds)
    return acc, precision, recall, auc, f1, cfs

if __name__ == "__main__":
    print("Running XGBoost classifiers")
    args = parse_args()
    save = args.save
    use_SMOTE = args.use_smote
    use_weights = args.use_weights
    scale_data = args.scale_data
    single_fold = args.single_fold
    clf_cfg = config.classifier_config
    device = clf_cfg['device']
    
    eval_file = open(clf_cfg['XGB_EVALUATION_RESULTS'], 'w')
    eval_file.write('EVALUATION RESULTS:\n')
    eval_file.write(f'SMOTE: {use_SMOTE} | weights: {use_weights}\n')
    eval_file.close()
    
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

    groups = utils.positive_groups
    
    threshold = clf_cfg['xgb_threshold']
    
    selected_metrics = ['train_acc', 'train_precision', 'train_recall', 'train_auc', 'train_f1','train_cfs', 'val_acc', 'val_precision', 'val_recall', 'val_auc', 'val_f1', 'val_cfs']

    for group in tqdm(groups, desc = 'Groups: ', position = 0):
        for fold in tqdm(trained_folds, desc = 'Folds: ', position = 1):
            history = {'train_accs': [], 'train_precisions': [], 'train_recalls': [], 'train_aucs': [], 'train_f1s':[], 'val_accs': [], 'val_precisions': [], 'val_recalls': [], 'val_aucs': [], 'val_f1s':[]}

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
                sample_weights = compute_sample_weight(class_weight='balanced', y = train_labels)
            else:
                sample_weights = None

            train_labels_int = np.array([get_int_label(label, group) for label in train_labels])
            val_labels_int = np.array([get_int_label(label, group) for label in val_labels])
            
            clf = XGBClassifier()
            clf.fit(train_features, train_labels_int, sample_weight = sample_weights)
            train_acc, train_precision, train_recall, train_auc, train_f1, train_cfs = eval(clf, train_features, train_labels_int)
            val_acc, val_precision, val_recall, val_auc, val_f1, val_cfs = eval(clf, val_features, val_labels_int)

            fold_results = {'train_accs': train_acc, 'train_precision': train_precision, 'train_recall': train_recall, 'train_auc': train_auc, 'train_f1': train_f1, 'train_cfs': train_cfs, 'val_accs': val_acc, 'val_precision': val_precision,  'val_recall': val_recall, 'val_auc': val_auc, 'val_f1': val_f1, 'val_cfs': val_cfs}
            
            tqdm.write(f'[{group.upper()}] - {fold}')
            tqdm.write('train_acc: %.3f | train_precision: %.3f | train_recall: %.3f | train_auc: %.3f | train_f1: %.3f' % (train_acc, train_precision, train_recall, train_auc, train_f1))
            tqdm.write('val_acc: %.3f | val_precision: %.3f | val_recall: %.3f | val_auc: %.3f | val_f1: %.3f' % (val_acc, val_precision, val_recall, val_auc, val_f1))
            
            eval_file = open(clf_cfg['XGB_EVALUATION_RESULTS'], 'a+')
            eval_file.write(f'\n-------------------------------\n[{group.upper()} - {fold}]:\n')
            for key, value in fold_results.items():
                if key in selected_metrics:
                    if 'cfs' in key:
                        eval_file.write(f'{key}:\n{value}\n')
                    else:
                        eval_file.write(f'{key}: {value} | ')
            eval_file.close()
            if save == 'save':
                tqdm.write('=> Saving model...')
                # dumn model parameters to json file
                MODEL_PATH = os.path.join(clf_cfg['XGB_MODELS_DIR'], group, f'{fold}.json')
                clf.save_model(MODEL_PATH)