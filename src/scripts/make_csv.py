import sys
sys.path.append('/media/data/hungnt/work/SourceCode/Brain_DNA/src/')

import os
import config
import utils
import pandas as pd

from tqdm import tqdm
from tqdm.auto import trange

def convert_to_binary(target):
    return  value if value in target else None

if __name__ == "__main__":
    groups = utils.groups
    dat_cfg = config.data_config
    clf_cfg = config.classifier_config
    
    # All folds
    folds = ['1.0', '1.1', '1.2', '1.3', '1.4', '1.5', 
             '2.0', '2.1', '2.2', '2.3', '2.4', '2.5', 
             '3.0', '3.1', '3.2', '3.3', '3.4', '3.5',
             '4.0', '4.1', '4.2', '4.3', '4.4', '4.5',
             '5.0', '5.1', '5.2', '5.3', '5.4', '5.5']
    
    raw_train_csv_dir = dat_cfg['RAW_TRAIN_CSV_DIR']
    raw_test_csv_dir = dat_cfg['RAW_TEST_CSV_DIR']
    
    for group in tqdm(groups, desc = 'groups', position = 0):
        train_csv_dir = os.path.join(dat_cfg['TRAIN_CSV_DIR'], group)
        test_csv_dir = os.path.join(dat_cfg['TEST_CSV_DIR'], group)
        for fold in tqdm(folds, desc='folds', position=1):
            raw_train_csv_file = os.path.join(raw_train_csv_dir, fold + '_train.csv')
            raw_test_csv_file = os.path.join(raw_test_csv_dir, fold + '_test.csv')
            train_csv_file = os.path.join(train_csv_dir, fold + '_train.csv')
            test_csv_file = os.path.join(test_csv_dir, fold + '_test.csv')
            
            raw_df_train = pd.read_csv(raw_train_csv_file, index_col = 0)
            df_train = raw_df_train.loc[(raw_df_train['class_group'] == group) | (raw_df_train['class_group'] == 'Control')]
            # Save the csv file
            df_train.to_csv(train_csv_file, index = True)
            # Re-read from csv file to return
            df_train = pd.read_csv(train_csv_file, index_col = 0)
            
            raw_df_test = pd.read_csv(raw_test_csv_file, index_col = 0)
            df_test = raw_df_test.loc[(raw_df_test['class_group'] == group) | (raw_df_test['class_group'] == 'Control')]
            # Save the csv file
            df_test.to_csv(test_csv_file, index = True)
            # Re-read from csv file to return
            df_test = pd.read_csv(test_csv_file, index_col = 0)
            
            
