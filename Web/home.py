import sys
sys.path.append('../')

import streamlit as st
import os
import pandas as pd
import numpy as np
import pickle
import src.config as config
import src.impf_models.MLP as impf_MLP
import torch

from src.impf_models.MLP.Model import Impf_DNAMLP
from torch.nn.functional import softmax

# Globals
outer_folds = ['1.0', '2.0', '3.0', '4.0', '5.0']
temp_fold = '1.1'
algs = ['MLP', 'RF', 'LR', 'XGB']
tissue_groups = [
    'Embryonal', 'Glioblastoma', 'Glio-neuronal', 'Sella', 'Ependymal', 'Other glioma', 'Nerve', 'Pineal', 'Mesenchymal', 'Melanocytic', 'Plexus', 'Glioma IDH', 'Haematopoietic', 'Unknown'
]
impf_cfg = config.impf_config
device = impf_cfg['device']

@st.cache
def load_data(file_name, nrows):
    data = pd.read_csv(os.path.join(UPLOADED_CSV_PATH, file_name), nrows = nrows)

@st.cache
def get_impf_dataframe(df, outer_fold = '1.0', alg = 'MLP', group = 'Unknown'):
    with open(os.path.join(impf_cfg['IMPORTANT_FEATURES_DIR'], alg, group, f'{outer_fold}_combined.pkl'), 'rb') as file:
        impf = pickle.load(file)
    df = df.loc[:, list(impf)]
    return df, len(impf)

# MAIN
## Main page
title = 'DNA methylation-based classification of central nervous system tumours'
st.title(title, 'homepage')
st.write('---')
uploaded_file = st.file_uploader('UPLOAD A CSV FILE')
try:
    st.write(f'File ***{uploaded_file.name}*** received.')
    try:
        st.write('---')
        df = pd.read_csv(uploaded_file, nrows=1)
        st.write(df)
        alg = st.selectbox('Select alg: ', algs)
        group = st.selectbox('Select tissue origin: ', tissue_groups)
        df, n_impf = get_impf_dataframe(df, '1.0', alg, group)
        st.write(df)

        sample = np.expand_dims(np.array(df.iloc[0,:]).astype(float), axis = 0)
        sample = torch.Tensor(sample).to(device)

        print(sample.shape)
        model = Impf_DNAMLP(n_impf, impf_cfg['n_classes'])
        BEST_STATE_PATH = os.path.join(impf_cfg['MLP_BEST_STATES_DIR'], alg, group, f'{temp_fold}.pth')
        
        if torch.cuda.is_available() is False:
            model.load_state_dict(torch.load(BEST_STATE_PATH, map_location = torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(BEST_STATE_PATH))
        model.to(device)

        res = softmax(model(sample), dim = 1)
        
        st.write('---')
        st.write('**DIAGNOSIS RESULTS:**')
        st.write(res.detach().cpu().numpy())
    except Exception as e:
        st.write("Caught an error:", e)
except:
    st.write("*You haven't uploaded a csv file of correct format*")