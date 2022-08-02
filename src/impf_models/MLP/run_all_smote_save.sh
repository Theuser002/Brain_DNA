#!/bin/sh
nohup python run_impf_MLP.py --alg MLP --save save -smote
nohup python run_impf_MLP.py --alg RF --save save -smote
nohup python run_impf_MLP.py --alg LR --save save -smote
nohup python run_impf_MLP.py --alg XGB --save save -smote