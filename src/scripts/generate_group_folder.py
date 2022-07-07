import sys
sys.path.append('../../src/')

import os
import config
import utils

from os import path
from tqdm import tqdm

if __name__ == "__main__":
    groups = utils.groups
    parent_folders = ['/media/data/hungnt/work/SourceCode/Brain_DNA/models/binary_classification_with_impf/mlp_best_states/LR',
'/media/data/hungnt/work/SourceCode/Brain_DNA/models/binary_classification_with_impf/mlp_best_states/MLP',
'/media/data/hungnt/work/SourceCode/Brain_DNA/models/binary_classification_with_impf/mlp_best_states/RF',
'/media/data/hungnt/work/SourceCode/Brain_DNA/models/binary_classification_with_impf/mlp_best_states/XGB']
    
    for folder in parent_folders:
        if not os.path.isdir(folder):
            print(f'{folder} is not a folder')
            continue
        for group in tqdm(groups):
            dir = os.path.join(folder, group)
            gitkeep = os.path.join(dir, '.gitkeep')
            if not os.path.isdir(dir):
                os.mkdir(dir)
            if not os.path.isfile(gitkeep):
                with open(gitkeep, 'w') as fp:
                    fp.close()
                    pass
                