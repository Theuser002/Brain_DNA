import sys
import os

outer_folds = ['1.0', '2.0', '3.0', '4.0', '5.0']
temp_fold = '1.1'
algs = ['MLP', 'RF', 'LR', 'XGB']
tissue_groups = [
    'Embryonal', 'Glioblastoma', 'Glio-neuronal', 'Sella', 'Ependymal', 'Other glioma', 'Nerve', 'Pineal', 'Mesenchymal', 'Melanocytic', 'Plexus', 'Glioma IDH', 'Haematopoietic', 'Unknown'
]
cpg_info_path = '../Brain_DNA/Supplementary/humanmethylation450.csv'