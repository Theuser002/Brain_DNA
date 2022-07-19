import sys
import os

outer_folds = ['1.0', '2.0', '3.0', '4.0', '5.0']
inner_fold_indexes = ['.1', '.2', '.3', '.4', '.5']
algs = ['MLP', 'RF', 'LR', 'XGB']
tissue_groups = [
    'Embryonal', 'Glioblastoma', 'Glio-neuronal', 'Sella', 'Ependymal', 'Other glioma', 'Nerve', 'Pineal', 'Mesenchymal', 'Melanocytic', 'Plexus', 'Glioma IDH', 'Haematopoietic', 'Unknown'
]
cpg_info_path = '../Supplementary/humanmethylation450.csv'