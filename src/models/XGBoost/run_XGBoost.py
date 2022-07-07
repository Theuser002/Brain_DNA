import sys
sys.path.append('../../../src/')

import os
import config
import utils
import numpy as np
import argparse

from xgboost import XGBClassifier
from utils import make_ndarray_from_csv, get_int_label
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type = str, default='no_save')
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    print("Running XGBoost classifiers")
    args = parse_args()
    save = args.save
    clf_cfg = config.classifier_config
    
    eval_file = open(clf_cfg['XGB_EVALUATION_RESULTS'], 'w')
    eval_file.write('EVALUATION RESULTS:\n')
    eval_file.close()
    
    print(f'root: {config.root_dir}')
    print(f"device: {clf_cfg['device']}")
    print(f'save mode: {save}')
    
    folds = ['1.0', '1.1', '1.2', '1.3', '1.4', '1.5', 
            '2.0', '2.1', '2.2', '2.3', '2.4', '2.5', 
            '3.0', '3.1', '3.2', '3.3', '3.4', '3.5',
            '4.0', '4.1', '4.2', '4.3', '4.4', '4.5',
            '5.0', '5.1', '5.2', '5.3', '5.4', '5.5']
    outer_folds = ['1.0', '2.0', '3.0', '4.0', '5.0']
    inner_folds = [x for x in folds if x not in outer_folds]
    
    trained_folds = folds
    groups = utils.positive_groups
    
    for group in tqdm(groups, desc = 'Groups: ', position = 0):
        for fold in tqdm(trained_folds, desc = 'Folds: ', position = 1):
            train_features, train_labels, val_features, val_labels = make_ndarray_from_csv(group, fold)
            
            train_labels_int = np.array([get_int_label(label, group) for label in train_labels])
            val_labels_int = np.array([get_int_label(label, group) for label in val_labels])
            
            clf = XGBClassifier()
            clf.fit(train_features, train_labels_int)
            val_acc = clf.score(val_features, val_labels_int)
            eval_str = f'\n>>>>>>[{group.upper()}] - {fold} - val acc: {val_acc}'
            tqdm.write(eval_str)
            if save == 'save':
                tqdm.write('=> Saving model...')
                MODEL_PATH = os.path.join(clf_cfg['XGB_MODELS_DIR'], group, f'{fold}.json')
                clf.save_model(MODEL_PATH)