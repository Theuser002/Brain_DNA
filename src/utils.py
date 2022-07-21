import os
import numpy as np
import pandas as pd
import config
import pyreadr

from torch.nn.functional import softmax, one_hot
'''
class classGroupUtils ():
    def __init__ (self):
        self.classes_group_map = {
            'Embryonal': ['MB, G4', 'MB, G3', 'MB, WNT', 'ETMR', 'HGNET, BCOR', 'CNS NB, FOXR2', 'ATRT, TYR', 'ATRT, SHH', 'ATRT, MYC', 'MB, SHH INF', 'MB, SHH CHL AD'],
            'Glioblastoma': ['DMG, K27', 'GBM, G34', 'GBM, MES', 'GBM, RTK I', 'GBM, RTK II','GBM, RTK III', 'GBM, MID', 'GBM, MYCN'],
            'Glio-neuronal': ['CN', 'ENB, B', 'ENB, A', 'RETB', 'LGG, GG', 'DLGNT', 'LGG, RGNT', 'LGG, DIG/DIA', 'LGG, DNT', 'LIPN', 'PGG, nC'],
            'Sella': ['PITAD, PRL', 'PITAD, FSH LH', 'PITAD, ACH', 'CPH, PAP', 'CPH, ADM', 'PITAD, STH SPA', 'PITAD, STH DNS B', 'PITAD, STH DNS A', 'PITAD, TSH', 'PITUI, SCO, GCT'],
            'Ependymal': ['EPN, RELA', 'EPN, MPE', 'SUBEPN, ST', 'SUBEPN, SPINE', 'SUBEPN, PF', 'EPN, SPINE', 'EPN, PF B', 'EPN, PF A', 'EPN, YAP'],
            'Other glioma': ['IHG', 'LGG, SEGA', 'CHGL', 'LGG, PA MID', 'LGG, PA PF', 'LGG, MYB', 'ANA PA', 'HGNET, MN1', 'LGG, PA/GG ST', 'PXA'],
            'Nerve': ['SCHW, MEL', 'SCHW'],
            'Pineal': ['PTPR, B', 'PTPR, A', 'PIN T, PB B', 'PIN T, PB A', 'PIN T, PPT'],
            'Mesenchymal': ['SFT HMPC', 'MNG', 'HMB', 'EWS', 'CHORDM', 'EFT, CIC'],
            'Melanocytic': ['MELCYT', 'MELAN'],
            'Plexus': ['PLEX, PED B', 'PLEX, PED A', 'PLEX, AD'],
            'Glioma IDH': ['A IDH, HG', 'A IDH', 'O IDH'],
            'Haematopoietic': ['PLASMA', 'LYMPHO'],
            'Control': ['REACT', 'PONS', 'PINEAL', 'INFLAM', 'HYPTHAL', 'HEMI', 'CEBM', 'WM', 'ADENOPIT']
        }
        
        self.methyl_class_list =  [
            'MB, G4', 'MB, G3', 'MB, WNT', 'ETMR', 'HGNET, BCOR', 'CNS NB, FOXR2', 'ATRT, TYR', 'ATRT, SHH', 'ATRT, MYC', 'MB, SHH INF', 'MB, SHH CHL AD', 'DMG, K27', 'GBM, G34', 'GBM, MES', 'GBM, RTK I', 'GBM, RTK II','GBM, RTK III', 'GBM, MID', 'GBM, MYCN', 'CN', 'ENB, B', 'ENB, A', 'RETB', 'LGG, GG', 'DLGNT', 'LGG, RGNT', 'LGG, DIG/DIA', 'LGG, DNT', 'LIPN', 'PGG, nC', 'PITAD, PRL', 'PITAD, FSH LH', 'PITAD, ACH', 'CPH, PAP', 'CPH, ADM', 'PITAD, STH SPA', 'PITAD, STH DNS B', 'PITAD, STH DNS A', 'PITAD, TSH', 'PITUI, SCO, GCT', 'EPN, RELA', 'EPN, MPE', 'SUBEPN, ST', 'SUBEPN, SPINE', 'SUBEPN, PF', 'EPN, SPINE', 'EPN, PF B', 'EPN, PF A', 'EPN, YAP', 'IHG', 'LGG, SEGA', 'CHGL', 'LGG, PA MID', 'LGG, PA PF', 'LGG, MYB', 'ANA PA', 'HGNET, MN1', 'LGG, PA/GG ST', 'PXA', 'SCHW, MEL', 'SCHW', 'PTPR, B', 'PTPR, A', 'PIN T, PB B', 'PIN T, PB A', 'PIN T, PPT', 'SFT HMPC', 'MNG', 'HMB', 'EWS', 'CHORDM', 'EFT, CIC', 'MELCYT', 'MELAN', 'PLEX, PED B', 'PLEX, PED A', 'PLEX, AD', 'A IDH, HG', 'A IDH', 'O IDH', 'PLASMA', 'LYMPHO', 'REACT', 'PONS', 'PINEAL', 'INFLAM', 'HYPTHAL', 'HEMI', 'CEBM', 'WM', 'ADENOPIT'
        ]
        
        self.group_names = [
            'Embryonal', 'Glioblastoma', 'Glio-neuronal', 'Sella', 'Ependymal', 'Other glioma', 'Nerve', 'Pineal', 'Mesenchymal', 'Melanocytic', 'Plexus', 'Glioma IDH', 'Haematopoietic', 'Control'
        ]
        
        self.n_classes_per_group = [11, 8, 11, 10, 9, 10, 2, 5, 6, 2, 3, 3, 2, 9],
        self.group_index_range = [11, 19, 30, 40, 49, 59, 61, 66, 72, 74, 77, 80, 82, 91]
    
    def map_class_to_group (self, methyl_class):
        methyl_class_index = self.methyl_class_list.index(methyl_class)
        for i in range(len(self.group_index_range)):
            if self.group_index_range[i] > methyl_class_index:
                break
        return self.group_names[i]
'''

class_group_map = {
    'Embryonal': ['MB, G4', 'MB, G3', 'MB, WNT', 'ETMR', 'HGNET, BCOR', 'CNS NB, FOXR2', 'ATRT, TYR', 'ATRT, SHH', 'ATRT, MYC', 'MB, SHH INF', 'MB, SHH CHL AD'],
    'Glioblastoma': ['DMG, K27', 'GBM, G34', 'GBM, MES', 'GBM, RTK I', 'GBM, RTK II','GBM, RTK III', 'GBM, MID', 'GBM, MYCN'],
    'Glio-neuronal': ['CN', 'ENB, B', 'ENB, A', 'RETB', 'LGG, GG', 'DLGNT', 'LGG, RGNT', 'LGG, DIG/DIA', 'LGG, DNT', 'LIPN', 'PGG, nC'],
    'Sella': ['PITAD, PRL', 'PITAD, FSH LH', 'PITAD, ACTH', 'CPH, PAP', 'CPH, ADM', 'PITAD, STH SPA', 'PITAD, STH DNS B', 'PITAD, STH DNS A', 'PITAD, TSH', 'PITUI'],
    'Ependymal': ['EPN, RELA', 'EPN, MPE', 'SUBEPN, ST', 'SUBEPN, SPINE', 'SUBEPN, PF', 'EPN, SPINE', 'EPN, PF B', 'EPN, PF A', 'EPN, YAP'],
    'Other glioma': ['IHG', 'LGG, SEGA', 'CHGL', 'LGG, PA MID', 'LGG, PA PF', 'LGG, MYB', 'ANA PA', 'HGNET, MN1', 'LGG, PA/GG ST', 'PXA'],
    'Nerve': ['SCHW, MEL', 'SCHW'],
    'Pineal': ['PTPR, B', 'PTPR, A', 'PIN T,  PB B', 'PIN T,  PB A', 'PIN T, PPT'],
    'Mesenchymal': ['SFT HMPC', 'MNG', 'HMB', 'EWS', 'CHORDM', 'EFT, CIC'],
    'Melanocytic': ['MELCYT', 'MELAN'],
    'Plexus': ['PLEX, PED B', 'PLEX, PED A', 'PLEX, AD'],
    'Glioma IDH': ['A IDH, HG', 'A IDH', 'O IDH'],
    'Haematopoietic': ['PLASMA', 'LYMPHO'],
    'Control': ['CONTR, REACT', 'CONTR, PONS', 'CONTR, PINEAL', 'CONTR, INFLAM', 'CONTR, HYPTHAL', 'CONTR, HEMI', 'CONTR, CEBM', 'CONTR, WM', 'CONTR, ADENOPIT']
}

methyl_class_list = ['MB, G4', 'MB, G3', 'MB, WNT', 'ETMR', 'HGNET, BCOR', 'CNS NB, FOXR2', 'ATRT, TYR', 'ATRT, SHH', 'ATRT, MYC', 'MB, SHH INF', 'MB, SHH CHL AD', 'DMG, K27', 'GBM, G34', 'GBM, MES', 'GBM, RTK I', 'GBM, RTK II','GBM, RTK III', 'GBM, MID', 'GBM, MYCN', 'CN', 'ENB, B', 'ENB, A', 'RETB', 'LGG, GG', 'DLGNT', 'LGG, RGNT', 'LGG, DIG/DIA', 'LGG, DNT', 'LIPN', 'PGG, nC', 'PITAD, PRL', 'PITAD, FSH LH', 'PITAD, ACTH', 'CPH, PAP', 'CPH, ADM', 'PITAD, STH SPA', 'PITAD, STH DNS B', 'PITAD, STH DNS A', 'PITAD, TSH', 'PITUI', 'EPN, RELA', 'EPN, MPE', 'SUBEPN, ST', 'SUBEPN, SPINE', 'SUBEPN, PF', 'EPN, SPINE', 'EPN, PF B', 'EPN, PF A', 'EPN, YAP', 'IHG', 'LGG, SEGA', 'CHGL', 'LGG, PA MID', 'LGG, PA PF', 'LGG, MYB', 'ANA PA', 'HGNET, MN1', 'LGG, PA/GG ST', 'PXA', 'SCHW, MEL', 'SCHW', 'PTPR, B', 'PTPR, A', 'PIN T,  PB B', 'PIN T,  PB A', 'PIN T, PPT', 'SFT HMPC', 'MNG', 'HMB', 'EWS', 'CHORDM', 'EFT, CIC', 'MELCYT', 'MELAN', 'PLEX, PED B', 'PLEX, PED A', 'PLEX, AD', 'A IDH, HG', 'A IDH', 'O IDH', 'PLASMA', 'LYMPHO', 'CONTR, REACT', 'CONTR, PONS', 'CONTR, PINEAL', 'CONTR, INFLAM', 'CONTR, HYPTHAL', 'CONTR, HEMI', 'CONTR, CEBM', 'CONTR, WM', 'CONTR, ADENOPIT']

groups = [
    'Embryonal', 'Glioblastoma', 'Glio-neuronal', 'Sella', 'Ependymal', 'Other glioma', 'Nerve', 'Pineal', 'Mesenchymal', 'Melanocytic', 'Plexus', 'Glioma IDH', 'Haematopoietic', 'Control'
]

positive_groups = [
    'Embryonal', 'Glioblastoma', 'Glio-neuronal', 'Sella', 'Ependymal', 'Other glioma', 'Nerve', 'Pineal', 'Mesenchymal', 'Melanocytic', 'Plexus', 'Glioma IDH', 'Haematopoietic'
]

low_perf_groups = [
    'Glio-neuronal', 'Other glioma'
]

n_classes_per_group = [11, 8, 11, 10, 9, 10, 2, 5, 6, 2, 3, 3, 2, 9],
group_index_range = [11, 19, 30, 40, 49, 59, 61, 66, 72, 74, 77, 80, 82, 91]

folds = ['1.0', '1.1', '1.2', '1.3', '1.4', '1.5', 
        '2.0', '2.1', '2.2', '2.3', '2.4', '2.5', 
        '3.0', '3.1', '3.2', '3.3', '3.4', '3.5',
        '4.0', '4.1', '4.2', '4.3', '4.4', '4.5',
        '5.0', '5.1', '5.2', '5.3', '5.4', '5.5',]

outer_folds = ['1.0', '2.0', '3.0', '4.0', '5.0']
inner_folds = [x for x in folds if x not in outer_folds]

algs = ['MLP', 'RF', 'LR', 'XGB']

def map_class_to_group (methyl_class):
    methyl_class_index = methyl_class_list.index(methyl_class)
    for i in range(len(group_index_range)):
        if group_index_range[i] > methyl_class_index:
            break
    return group_names[i]

def get_int_label (label, target):
    if label == target:
        return 0
    else:
        return 1

def brier_score_tensor(logits, categorical_labels):
    class_probs = softmax(logits, dim = 1)
    one_hot_labels =  one_hot(categorical_labels.long(), num_classes = class_probs.shape[1])
    class_probs = class_probs.detach().cpu().numpy()
    one_hot_labels = one_hot_labels.detach().cpu().numpy()
    return np.mean(np.sum((class_probs - one_hot_labels)**2, axis=1))

def make_ndarray_from_csv(group, fold, mode = 'None'):
    dat_cfg = config.data_config
    
    if mode.lower() == 'train':
        train_csv_path = os.path.join(dat_cfg['TRAIN_CSV_DIR'], group, f'{fold}_train.csv')
        df_train = pd.read_csv(train_csv_path, index_col = 0).fillna(0)
        
        train_features = np.array(df_train.iloc[:,:10000])
        train_labels = np.array(df_train.iloc[:,-1])
        return train_features, train_labels    
    
    elif mode.lower() == 'test' or mode.lower() == 'val':
        test_csv_path = os.path.join(dat_cfg['TEST_CSV_DIR'], group, f'{fold}_test.csv')
        df_test = pd.read_csv(test_csv_path, index_col = 0).fillna(0)
        
        test_features = np.array(df_test.iloc[:,:10000])
        test_labels = np.array(df_test.iloc[:,-1])
        return test_features, test_labels
    
    elif mode.lower() == 'all':
        train_csv_path = os.path.join(dat_cfg['TRAIN_CSV_DIR'], group, f'{fold}_train.csv')
        df_train = pd.read_csv(train_csv_path, index_col = 0).fillna(0)    
        
        train_features = np.array(df_train.iloc[:,:10000])
        train_labels = np.array(df_train.iloc[:,-1])
        
        test_csv_path = os.path.join(dat_cfg['TEST_CSV_DIR'], group, f'{fold}_test.csv')
        df_test = pd.read_csv(test_csv_path, index_col = 0).fillna(0)
        
        test_features = np.array(df_test.iloc[:,:10000])
        test_labels = np.array(df_test.iloc[:,-1])
        
        features = np.append(train_features, test_features, axis = 0)
        labels = np.append(train_labels, test_labels, axis = 0)
        
        return features, labels
        
    else:        
        train_csv_path = os.path.join(dat_cfg['TRAIN_CSV_DIR'], group, f'{fold}_train.csv')
        test_csv_path = os.path.join(dat_cfg['TEST_CSV_DIR'], group, f'{fold}_test.csv')
        df_train = pd.read_csv(train_csv_path, index_col = 0).fillna(0)
        df_test = pd.read_csv(test_csv_path, index_col = 0).fillna(0)
        
        train_features = np.array(df_train.iloc[:,:10000])
        train_labels = np.array(df_train.iloc[:,-1])
        
        test_features = np.array(df_test.iloc[:,:10000])
        test_labels = np.array(df_test.iloc[:,-1])
        
        return train_features, train_labels, test_features, test_labels

# def impf_make_ndarray_from_csv(group, fold, impf, mode = 'None'):
    # dat_cfg = config.data_config
    
    # if mode.lower() == 'train':
    #     train_csv_path = os.path.join(dat_cfg['TRAIN_CSV_DIR'], group, f'{fold}_train.csv')
    #     df_train = pd.read_csv(train_csv_path, index_col = 0).fillna(0)    
        
    #     train_features = np.array(df_train.loc[:, df_train.columns.isin(impf)])
    #     train_labels = np.array(df_train.iloc[:,-1])
    #     return train_features, train_labels    
    
    # elif mode.lower() == 'test' or mode.lower() == 'val':
    #     test_csv_path = os.path.join(dat_cfg['TEST_CSV_DIR'], group, f'{fold}_test.csv')
    #     df_test = pd.read_csv(test_csv_path, index_col = 0).fillna(0)
        
    #     test_features = np.array(df_test.loc[:, df_test.columns.isin(impf)])
    #     test_labels = np.array(df_test.iloc[:,-1])
    #     return test_features, test_labels
    
    # elif mode.lower() == 'all':
    #     train_csv_path = os.path.join(dat_cfg['TRAIN_CSV_DIR'], group, f'{fold}_train.csv')
    #     df_train = pd.read_csv(train_csv_path, index_col = 0).fillna(0)    
        
    #     train_features = np.array(df_train.loc[:, df_train.columns.isin(impf)])
    #     train_labels = np.array(df_train.iloc[:,-1])
        
    #     test_csv_path = os.path.join(dat_cfg['TEST_CSV_DIR'], group, f'{fold}_test.csv')
    #     df_test = pd.read_csv(test_csv_path, index_col = 0).fillna(0)
        
    #     test_features = np.array(df_test.loc[:, df_test.columns.isin(impf)])
    #     test_labels = np.array(df_test.iloc[:,-1])
        
    #     features = np.append(train_features, test_features, axis = 0)
    #     labels = np.append(train_labels, test_labels, axis = 0)
        
    #     return features, labels
        
    # else:        
    #     train_csv_path = os.path.join(dat_cfg['TRAIN_CSV_DIR'], group, f'{fold}_train.csv')
    #     test_csv_path = os.path.join(dat_cfg['TEST_CSV_DIR'], group, f'{fold}_test.csv')
    #     df_train = pd.read_csv(train_csv_path, index_col = 0).fillna(0)
    #     df_test = pd.read_csv(test_csv_path, index_col = 0).fillna(0)
        
    #     train_features = np.array(df_train.loc[:, df_train.columns.isin(impf)])
    #     train_labels = np.array(df_train.iloc[:,-1])
        
    #     test_features = np.array(df_test.loc[:, df_test.columns.isin(impf)])
    #     test_labels = np.array(df_test.iloc[:,-1])
        
    #     return train_features, train_labels, test_features, test_labels
# ==> Some of the folds have the same features alike but they might not be arranged similarly between these folds. E.g: fold 1.1 might have features a, b, c, d, e, f and fold 1.0 have them too but is more like {c, b, a, d, e, f}. impf = {c, d, f} -> we should not keep the feature order in the original dataframe but instead arange them alphabetically. This is automatically done if we use df.loc[:, impf] or df.loc[: list(impf)] pd will raise a warning if we use set as indexer for loc

def impf_make_ndarray_from_csv(group, fold, impf, mode = 'None'):
    dat_cfg = config.data_config
    
    if mode.lower() == 'train':
        train_csv_path = os.path.join(dat_cfg['TRAIN_CSV_DIR'], group, f'{fold}_train.csv')
        df_train = pd.read_csv(train_csv_path, index_col = 0).fillna(0)    
        
        train_features = np.array(df_train.loc[:,list(impf)])
        train_labels = np.array(df_train.iloc[:,-1])
        return train_features, train_labels    
    
    elif mode.lower() == 'test' or mode.lower() == 'val':
        test_csv_path = os.path.join(dat_cfg['TEST_CSV_DIR'], group, f'{fold}_test.csv')
        df_test = pd.read_csv(test_csv_path, index_col = 0).fillna(0)
        
        test_features = np.array(df_test.loc[:,list(impf)])
        test_labels = np.array(df_test.iloc[:,-1])
        return test_features, test_labels
    
    elif mode.lower() == 'all':
        train_csv_path = os.path.join(dat_cfg['TRAIN_CSV_DIR'], group, f'{fold}_train.csv')
        df_train = pd.read_csv(train_csv_path, index_col = 0).fillna(0)    
        
        train_features = np.array(df_train.loc[:,list(impf)])
        train_labels = np.array(df_train.iloc[:,-1])
        
        test_csv_path = os.path.join(dat_cfg['TEST_CSV_DIR'], group, f'{fold}_test.csv')
        df_test = pd.read_csv(test_csv_path, index_col = 0).fillna(0)
        
        test_features = np.array(df_test.loc[:,list(impf)])
        test_labels = np.array(df_test.iloc[:,-1])
        
        features = np.append(train_features, test_features, axis = 0)
        labels = np.append(train_labels, test_labels, axis = 0)
        
        return features, labels
        
    else:        
        train_csv_path = os.path.join(dat_cfg['TRAIN_CSV_DIR'], group, f'{fold}_train.csv')
        test_csv_path = os.path.join(dat_cfg['TEST_CSV_DIR'], group, f'{fold}_test.csv')
        df_train = pd.read_csv(train_csv_path, index_col = 0).fillna(0)
        df_test = pd.read_csv(test_csv_path, index_col = 0).fillna(0)
        
        train_features = np.array(df_train.loc[:,list(impf)])
        train_labels = np.array(df_train.iloc[:,-1])
        
        test_features = np.array(df_test.loc[:,list(impf)])
        test_labels = np.array(df_test.iloc[:,-1])
        
        return train_features, train_labels, test_features, test_label

if __name__ == "__main__":
    for name in sorted(group_names):
        print(f'{name}: {get_int_label(name)}')