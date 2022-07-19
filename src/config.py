import os
import torch

from pathlib import Path
root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)

data_config = {
    'RAW_TRAIN_CSV_DIR': os.path.join(root_dir, 'data/raw/csv/train'),
    'RAW_TEST_CSV_DIR': os.path.join(root_dir, 'data/raw/csv/test'),
    'TRAIN_CSV_DIR': os.path.join(root_dir, 'data/processed/binary_classification_dataset/csv/train'),
    'TEST_CSV_DIR': os.path.join(root_dir, 'data/processed/binary_classification_dataset/csv/test'),
    
}
classifier_config = {
    'device': ('cuda' if torch.cuda.is_available() else 'cpu'),
    'n_features': 10000,
    'n_classes': 2,
    
    'MLP_BEST_STATES_DIR': os.path.join(root_dir, 'models/binary_classification/mlp_best_states'),
    'mlp_diff_threshold': 1e-3,
    'mlp_max_patience': 5,
    'mlp_train_batch_size': 8,
    'mlp_val_batch_size': 16,
    'MLP_FIRST_TIME': True,
    'mlp_lr': 1e-4,
    'mlp_weight_decay': 0,
    'mlp_n_epochs': 30,
    'MLP_EVALUATION_RESULTS': os.path.join(root_dir, 'src/models/MLP/evaluation.txt'),
    'mlp_threshold': 0.5,
    'mlp_dropout_rate': 0,
    
    'RF_MODELS_DIR': os.path.join(root_dir, 'models/binary_classification/rf_models'),
    'RF_EVALUATION_RESULTS': os.path.join(root_dir, 'src/models/RF/evaluation.txt'),
    'rf_threshold': 0.5,
    
    'LR_MODELS_DIR': os.path.join(root_dir, 'models/binary_classification/lr_models'),
    'LR_EVALUATION_RESULTS': os.path.join(root_dir, 'src/models/LR/evaluation.txt'),
    'lr_threshold': 0.5,
    
    'XGB_MODELS_DIR': os.path.join(root_dir, 'models/binary_classification/xgb_models'),
    'XGB_EVALUATION_RESULTS': os.path.join(root_dir, 'src/models/XGBoost/evaluation.txt'),
    'xgb_threshold': 0.5
}

impf_config = {
    'FEATURE_NAMES_DIR': os.path.join(root_dir, 'data/processed/feature_names'),
    'IMPORTANT_FEATURES_DIR': os.path.join(root_dir, 'data/processed/important_features'), 
    'N_IMPF': 3000,
    'n_classes': 2,
    'device': ('cuda' if torch.cuda.is_available() else 'cpu'),
    
    'MLP_BEST_STATES_DIR': os.path.join(root_dir, 'models/binary_classification_with_impf/mlp_best_states'),
    'MLP_FIRST_TIME': True,
    'mlp_diff_threshold': 1e-3,
    'mlp_max_patience': 5,
    'mlp_train_batch_size': 8,
    'mlp_val_batch_size': 16,
    'MLP_FIRST_TIME': True,
    'mlp_lr': 1e-4,
    'mlp_weight_decay': 0,
    'mlp_n_epochs': 30,
    'MLP_MLP_EVALUATION_RESULTS': os.path.join(root_dir, 'src/impf_models/MLP/MLP_MLP_evaluation.txt'),
    'MLP_LR_EVALUATION_RESULTS': os.path.join(root_dir, 'src/impf_models/MLP/MLP_LR_evaluation.txt'),
    'MLP_RF_EVALUATION_RESULTS': os.path.join(root_dir, 'src/impf_models/MLP/MLP_RF_evaluation.txt'),
    'MLP_XGB_EVALUATION_RESULTS': os.path.join(root_dir, 'src/impf_models/MLP/MLP_XGB_evaluation.txt')
}
