import sys
sys.path.append('../../src/')

import os
import config
import utils

from os import path
from tqdm import tqdm

if __name__ == "__main__":
    groups = utils.groups
    parent_folders = ['/media/data/hungnt/work/SourceCode/Brain_DNA/data/processed/important_features/LR',
'/media/data/hungnt/work/SourceCode/Brain_DNA/data/processed/important_features/MLP',
'/media/data/hungnt/work/SourceCode/Brain_DNA/data/processed/important_features/RF',
'/media/data/hungnt/work/SourceCode/Brain_DNA/data/processed/important_features/XGB']
    
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
                