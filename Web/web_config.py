import sys
import os

folds = ['1.0', '1.1', '1.2', '1.3', '1.4', '1.5', 
        '2.0', '2.1', '2.2', '2.3', '2.4', '2.5', 
        '3.0', '3.1', '3.2', '3.3', '3.4', '3.5',
        '4.0', '4.1', '4.2', '4.3', '4.4', '4.5',
        '5.0', '5.1', '5.2', '5.3', '5.4', '5.5',]

outer_folds = ['1.0', '2.0', '3.0', '4.0', '5.0']

inner_folds = [fold for fold in folds if fold not in outer_folds]

inner_fold_indexes = ['.1', '.2', '.3', '.4', '.5']

algs = ['MLP', 'RF', 'LR', 'XGB']

tissue_groups = [
    'Embryonal', 'Glioblastoma', 'Glio-neuronal', 'Sella', 'Ependymal', 'Other glioma', 'Nerve', 'Pineal', 'Mesenchymal', 'Melanocytic', 'Plexus', 'Glioma IDH', 'Haematopoietic'
]

cpg_info_path = '../Supplementary/humanmethylation450.csv'