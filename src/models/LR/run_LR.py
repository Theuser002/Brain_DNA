import sys
sys.path.append('../../../src/')

import os
import config
import utils
import sklearn
import argparse
import numpy as np
import joblib
import utils

from utils import make_ndarray_from_csv, get_int_label
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type = str, default='no_save')
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    print("Running linear regression classifiers")
    args = parse_args()
    save = args.save
    clf_cfg = config.classifier_config
    
    eval_file = open(clf_cfg['LR_EVALUATION_RESULTS'], 'w')
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
    threshold = clf_cfg['lr_threshold']

    for group in tqdm(groups, desc = 'Groups: ', position = 0):
        for fold in tqdm(trained_folds, desc = 'Folds: ', position = 1):
            train_features, train_labels = make_ndarray_from_csv(group, fold, mode = 'train')
            val_features, val_labels = make_ndarray_from_csv(group, fold, mode = 'test')
            train_labels_int = np.array([get_int_label(label, group) for label in train_labels])
            val_labels_int = np.array([get_int_label(label, group) for label in val_labels])
            
            clf = LogisticRegression(n_jobs=2)
            clf.fit(train_features, train_labels_int)
            val_acc = clf.score(val_features, val_labels_int)
            
            eval_str = f'\n>>>>>>[{group.upper()}] - {fold} - val_acc: {val_acc}\n'
            tqdm.write(eval_str)
            eval_file = open(clf_cfg['LR_EVALUATION_RESULTS'], 'a+')
            eval_file.write(eval_str)
            eval_file.close()
            
            if save == 'save':
                tqdm.write('=> Saving model...')
                # dump model to pickle/cpickle/joblib(?) file
                MODEL_PATH = os.path.join(clf_cfg['LR_MODELS_DIR'], group, f'{fold}.joblib')
                joblib.dump(clf, MODEL_PATH)
          
        
    
    
    