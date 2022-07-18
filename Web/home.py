import sys
sys.path.append('../src')

import streamlit as st
import os
import pandas as pd
import numpy as np
import pickle
import config
import impf_models.MLP as impf_MLP
import scripts.get_impf as get_impf
import torch

from impf_models.MLP.Model import Impf_DNAMLP
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

def get_ranked_impf(df, outer_fold = '1.0', alg = 'MLP', group = 'Unknown'):
    return get_impf.ranked_impf_features(alg, group, outer_fold)    

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
        # Get user's input
        alg = st.selectbox('Select important features algorithm: ', algs)
        group = st.selectbox('Select tissue origin: ', tissue_groups)
        outer_fold = st.selectbox('Select fold ***(development only)***: ', outer_folds)

        df, n_impf = get_impf_dataframe(df, outer_fold, alg, group)
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

        st.write('---')

        # Frontpage column layout
        col1, col2, col3 = st.columns(3)

        # Calculate diagnosis results and display
        res = softmax(model(sample), dim = 1)
        columns = [group, 'Control']
        row = res.detach().cpu().numpy()
        res_df = pd.DataFrame(row)
        res_df.columns = columns
        col1.write('**DIAGNOSIS RESULTS**')
        col1.write(res_df)

        # Get ranked important features
        ranked_impf = get_ranked_impf(df, outer_fold, alg, group)
        ranked_impf = dict(sorted(ranked_impf.items(), key = lambda item:item[1]))
        ranked_impf = pd.DataFrame(list(ranked_impf.keys()))
        col2.write('*Important CpGs*')
        col2.write(ranked_impf)

        # Get user's input
        selected_cpg = st.selectbox
    except Exception as e:
        st.write("Caught an error:", e)
except:
    st.write("*You haven't uploaded a csv file of correct format*")