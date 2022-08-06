import os
import torch

from pathlib import Path
root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, os.pardir)

data_config = {
    'RAW_TRAIN_CSV_DIR': os.path.join(root_dir, 'data/raw/csv/train'),
    'RAW_TEST_CSV_DIR': os.path.join(root_dir, 'data/raw/csv/test')
}

classifier_config = {
    'device': ('cuda' if torch.cuda.is_available() else 'cpu'),
    
    'n_features': 10000,
    'n_classes': 91,

    'nn91_lr': 1e-4,
    'nn91_n_epochs': 30,
    'nn91_train_batch_size': 8,
    'nn91_val_batch_size': 16,
    'nn91_max_patience': 5,
    'nn91_diff_threshold': 1e-3,
    'nn91_weight_decay': 0,
    'nn91_threshold': 0.5,
    'nn91_dropout_rate': 0,
    
    'NN91_BEST_STATES_DIR': os.path.join(root_dir, 'src/models/NN91/saved_best_states'),
    'NN91_FIRST_TIME': True,
    'NN91_EVALUATION_RESULTS': os.path.join(root_dir, 'src/models/NN91/NN91_evaluation.txt'),
    'TRAIN_PROBS_DIR': os.path.join(root_dir, 'src/models/NN91/probs/train'),
    'VAL_PROBS_DIR': os.path.join(root_dir, 'src/models/NN91/probs/val'),
    'TRAIN_LABELS_DIR': os.path.join(root_dir, 'src/models/NN91/labels/train'),
    'VAL_LABELS_DIR': os.path.join(root_dir, 'src/models/NN91/labels/val'),
    'TRAIN_LOGITS_DIR': os.path.join(root_dir, 'src/models/NN91/logits/train'),
    'VAL_LOGITS_DIR': os.path.join(root_dir, 'src/models/NN91/logits/val'),
    'NN91_EVALUATION_JSON': os.path.join(root_dir, 'src/models/NN91/NN91_evaluation.json'),
    'NN91_HISTORY_DIR': os.path.join(root_dir, 'src/models/NN91/history'),
}

if __name__ == "__main__":
    print(os.system(f'ls {root_dir}'))
    print(classifier_config['NN91_EVALUATION_JSON'])