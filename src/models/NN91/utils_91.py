import os
import numpy as np
import pandas as pd
import config_91
import pyreadr

from torch.nn.functional import softmax, one_hot

folds = ['1.0', '1.1', '1.2', '1.3', '1.4', '1.5', 
        '2.0', '2.1', '2.2', '2.3', '2.4', '2.5', 
        '3.0', '3.1', '3.2', '3.3', '3.4', '3.5',
        '4.0', '4.1', '4.2', '4.3', '4.4', '4.5',
        '5.0', '5.1', '5.2', '5.3', '5.4', '5.5',]

outer_folds = ['1.0', '2.0', '3.0', '4.0', '5.0']
inner_folds = [x for x in folds if x not in outer_folds]

label_dict = {'A IDH': 0, 'A IDH, HG': 1, 'ANA PA': 2, 'ATRT, MYC': 3, 'ATRT, SHH': 4, 'ATRT, TYR': 5, 'CHGL': 6, 'CHORDM': 7, 'CN': 8, 'CNS NB, FOXR2': 9, 'CONTR, ADENOPIT': 10, 'CONTR, CEBM': 11, 'CONTR, HEMI': 12, 'CONTR, HYPTHAL': 13, 'CONTR, INFLAM': 14, 'CONTR, PINEAL': 15, 'CONTR, PONS': 16, 'CONTR, REACT': 17, 'CONTR, WM': 18, 'CPH, ADM': 19, 'CPH, PAP': 20, 'DLGNT': 21, 'DMG, K27': 22, 'EFT, CIC': 23, 'ENB, A': 24, 'ENB, B': 25, 'EPN, MPE': 26, 'EPN, PF A': 27, 'EPN, PF B': 28, 'EPN, RELA': 29, 'EPN, SPINE': 30, 'EPN, YAP': 31, 'ETMR': 32, 'EWS': 33, 'GBM, G34': 34, 'GBM, MES': 35, 'GBM, MID': 36, 'GBM, MYCN': 37, 'GBM, RTK I': 38, 'GBM, RTK II': 39, 'GBM, RTK III': 40, 'HGNET, BCOR': 41, 'HGNET, MN1': 42, 'HMB': 43, 'IHG': 44, 'LGG, DIG/DIA': 45, 'LGG, DNT': 46, 'LGG, GG': 47, 'LGG, MYB': 48, 'LGG, PA MID': 49, 'LGG, PA PF': 50, 'LGG, PA/GG ST': 51, 'LGG, RGNT': 52, 'LGG, SEGA': 53, 'LIPN': 54, 'LYMPHO': 55, 'MB, G3': 56, 'MB, G4': 57, 'MB, SHH CHL AD': 58, 'MB, SHH INF': 59, 'MB, WNT': 60, 'MELAN': 61, 'MELCYT': 62, 'MNG': 63, 'O IDH': 64, 'PGG, nC': 65, 'PIN T,  PB A': 66, 'PIN T,  PB B': 67, 'PIN T, PPT': 68, 'PITAD, ACTH': 69, 'PITAD, FSH LH': 70, 'PITAD, PRL': 71, 'PITAD, STH DNS A': 72, 'PITAD, STH DNS B': 73, 'PITAD, STH SPA': 74, 'PITAD, TSH': 75, 'PITUI': 76, 'PLASMA': 77, 'PLEX, AD': 78, 'PLEX, PED A': 79, 'PLEX, PED B': 80, 'PTPR, A': 81, 'PTPR, B': 82, 'PXA': 83, 'RETB': 84, 'SCHW': 85, 'SCHW, MEL': 86, 'SFT HMPC': 87, 'SUBEPN, PF': 88, 'SUBEPN, SPINE': 89, 'SUBEPN, ST': 90}
reverse_label_dict = {0: 'A IDH', 1: 'A IDH, HG', 2: 'ANA PA', 3: 'ATRT, MYC', 4: 'ATRT, SHH', 5: 'ATRT, TYR', 6: 'CHGL', 7: 'CHORDM', 8: 'CN', 9: 'CNS NB, FOXR2', 10: 'CONTR, ADENOPIT', 11: 'CONTR, CEBM', 12: 'CONTR, HEMI', 13: 'CONTR, HYPTHAL', 14: 'CONTR, INFLAM', 15: 'CONTR, PINEAL', 16: 'CONTR, PONS', 17: 'CONTR, REACT', 18: 'CONTR, WM', 19: 'CPH, ADM', 20: 'CPH, PAP', 21: 'DLGNT', 22: 'DMG, K27', 23: 'EFT, CIC', 24: 'ENB, A', 25: 'ENB, B', 26: 'EPN, MPE', 27: 'EPN, PF A', 28: 'EPN, PF B', 29: 'EPN, RELA', 30: 'EPN, SPINE', 31: 'EPN, YAP', 32: 'ETMR', 33: 'EWS', 34: 'GBM, G34', 35: 'GBM, MES', 36: 'GBM, MID', 37: 'GBM, MYCN', 38: 'GBM, RTK I', 39: 'GBM, RTK II', 40: 'GBM, RTK III', 41: 'HGNET, BCOR', 42: 'HGNET, MN1', 43: 'HMB', 44: 'IHG', 45: 'LGG, DIG/DIA', 46: 'LGG, DNT', 47: 'LGG, GG', 48: 'LGG, MYB', 49: 'LGG, PA MID', 50: 'LGG, PA PF', 51: 'LGG, PA/GG ST', 52: 'LGG, RGNT', 53: 'LGG, SEGA', 54: 'LIPN', 55: 'LYMPHO', 56: 'MB, G3', 57: 'MB, G4', 58: 'MB, SHH CHL AD', 59: 'MB, SHH INF', 60: 'MB, WNT', 61: 'MELAN', 62: 'MELCYT', 63: 'MNG', 64: 'O IDH', 65: 'PGG, nC', 66: 'PIN T,  PB A', 67: 'PIN T,  PB B', 68: 'PIN T, PPT', 69: 'PITAD, ACTH', 70: 'PITAD, FSH LH', 71: 'PITAD, PRL', 72: 'PITAD, STH DNS A', 73: 'PITAD, STH DNS B', 74: 'PITAD, STH SPA', 75: 'PITAD, TSH', 76: 'PITUI', 77: 'PLASMA', 78: 'PLEX, AD', 79: 'PLEX, PED A', 80: 'PLEX, PED B', 81: 'PTPR, A', 82: 'PTPR, B', 83: 'PXA', 84: 'RETB', 85: 'SCHW', 86: 'SCHW, MEL', 87: 'SFT HMPC', 88: 'SUBEPN, PF', 89: 'SUBEPN, SPINE', 90: 'SUBEPN, ST'}

def brier_score_tensor(logits, categorical_labels):
    class_probs = softmax(logits, dim = 1)
    one_hot_labels =  one_hot(categorical_labels.long(), num_classes = class_probs.shape[1])
    class_probs = class_probs.detach().cpu().numpy()
    one_hot_labels = one_hot_labels.detach().cpu().numpy()
    return np.mean(np.sum((class_probs - one_hot_labels)**2, axis=1))

def get_int_label(label):
    return label_dict[f'{label}']

def get_str_label(label_int):
    return reverse_label_dict[f'{label_int}']

def get_one():
    return 1

def make_ndarray_from_csv(fold, mode = 'None'):
    dat_cfg = config_91.data_config
    
    if mode.lower() == 'train':
        train_csv_path = os.path.join(dat_cfg['RAW_TRAIN_CSV_DIR'],  f'{fold}_train.csv')
        df_train = pd.read_csv(train_csv_path, index_col = 0).fillna(0)
        
        train_features = np.array(df_train.iloc[:,:10000])
        train_labels = np.array(df_train.iloc[:,-2])
        return train_features, train_labels    
    
    elif mode.lower() == 'test' or mode.lower() == 'val':
        test_csv_path = os.path.join(dat_cfg['RAW_TEST_CSV_DIR'],  f'{fold}_test.csv')
        df_test = pd.read_csv(test_csv_path, index_col = 0).fillna(0)
        
        test_features = np.array(df_test.iloc[:,:10000])
        test_labels = np.array(df_test.iloc[:,-2])
        return test_features, test_labels
    
    elif mode.lower() == 'all':
        train_csv_path = os.path.join(dat_cfg['RAW_TRAIN_CSV_DIR'],  f'{fold}_train.csv')
        df_train = pd.read_csv(train_csv_path, index_col = 0).fillna(0)    
        
        train_features = np.array(df_train.iloc[:,:10000])
        train_labels = np.array(df_train.iloc[:,-2])
        
        test_csv_path = os.path.join(dat_cfg['RAW_TEST_CSV_DIR'],  f'{fold}_test.csv')
        df_test = pd.read_csv(test_csv_path, index_col = 0).fillna(0)
        
        test_features = np.array(df_test.iloc[:,:10000])
        test_labels = np.array(df_test.iloc[:,-2])
        
        features = np.append(train_features, test_features, axis = 0)
        labels = np.append(train_labels, test_labels, axis = 0)
        
        return features, labels
        
    else:        
        train_csv_path = os.path.join(dat_cfg['RAW_TRAIN_CSV_DIR'],  f'{fold}_train.csv')
        test_csv_path = os.path.join(dat_cfg['RAW_TEST_CSV_DIR'],  f'{fold}_test.csv')
        df_train = pd.read_csv(train_csv_path, index_col = 0).fillna(0)
        df_test = pd.read_csv(test_csv_path, index_col = 0).fillna(0)
        
        train_features = np.array(df_train.iloc[:,:10000])
        train_labels = np.array(df_train.iloc[:,-2])
        
        test_features = np.array(df_test.iloc[:,:10000])
        test_labels = np.array(df_test.iloc[:,-2])
        
        return train_features, train_labels, test_features, test_labels

if __name__ == "__main__":
    print('utils_91 running...')