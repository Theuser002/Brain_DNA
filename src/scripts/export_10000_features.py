import sys
sys.path.append('../../src/')

import os
import config
import utils
import pandas as pd
import pickle

from os import path
from tqdm import tqdm

if __name__ == "__main__":
    dat_cfg = config.data_config
    impf_cfg = config.impf_config
    
    folds = ['1.0', '1.1', '1.2', '1.3', '1.4', '1.5', 
             '2.0', '2.1', '2.2', '2.3', '2.4', '2.5', 
             '3.0', '3.1', '3.2', '3.3', '3.4', '3.5',
             '4.0', '4.1', '4.2', '4.3', '4.4', '4.5',
             '5.0', '5.1', '5.2', '5.3', '5.4', '5.5']
    
    FEATURE_NAMES_DIR = impf_cfg['FEATURE_NAMES_DIR']
    
    for fold in tqdm(folds):
        RAW_TEST_CSV_DIR = dat_cfg['RAW_TEST_CSV_DIR']
        RAW_TEST_CSV_FILE = os.path.join(RAW_TEST_CSV_DIR, f'{fold}_test.csv')
        df = pd.read_csv(RAW_TEST_CSV_FILE, index_col=0)
        features = list(df.columns)[:10000]
        FEATURE_NAMES_FILE = os.path.join(FEATURE_NAMES_DIR, f'{fold}.pkl')
        
        with open(FEATURE_NAMES_FILE, 'wb') as handle:
            pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)