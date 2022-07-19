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
import web_config

from impf_models.MLP.Model import Impf_DNAMLP
from torch.nn.functional import softmax
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

# Globals
outer_folds = web_config.outer_folds
temp_fold = web_config.temp_fold
algs = web_config.algs
tissue_groups = web_config.tissue_groups
impf_cfg = config.impf_config
device = impf_cfg['device']

@st.cache
def get_impf_dataframe(df, outer_fold = '1.0', alg = 'MLP', group = 'Unknown'):
    with open(os.path.join(impf_cfg['IMPORTANT_FEATURES_DIR'], alg, group, f'{outer_fold}_combined.pkl'), 'rb') as file:
        impf = pickle.load(file)
    df = df.loc[:, list(impf)]
    return df, len(impf)

@st.cache
def get_ranked_impf(df, outer_fold = '1.0', alg = 'MLP', group = 'Unknown'):
    return get_impf.ranked_impf_features(alg, group, outer_fold)    

@st.cache
def get_cpg_info_file(cpg_info_path):
    return pd.read_csv(cpg_info_path, index_col = 0, low_memory=False)

# MAIN
## Main page
st.sidebar.markdown('# Home')

title = 'DNA methylation-based classification of central nervous system tumours'
st.title(title, 'homepage')
st.write('---')
st.write('### File upload')
uploaded_file = st.file_uploader('UPLOAD A CSV FILE')
try:
    st.write(f'File ***{uploaded_file.name}*** received.')
    try:
        st.write('---')
        df = pd.read_csv(uploaded_file, nrows=1)
        st.write('### CpG filtering')
        st.write('**Sample:**')
        st.write(df)
        # Get user's input
        with st.form(key = 'form_0'):
            alg = st.selectbox('Select important features algorithm: ', algs)
            group = st.selectbox('Select tissue origin: ', tissue_groups)
            outer_fold = st.selectbox('Select fold (development only): ', outer_folds)
            submit_btn = st.form_submit_button(label='Finish')

        df, n_impf = get_impf_dataframe(df, outer_fold, alg, group)
        st.write('**Sample after CpG filtering:**')
        st.write(df)
        sample = np.expand_dims(np.array(df.iloc[0,:]).astype(float), axis = 0)
        sample = torch.Tensor(sample).to(device)

        model = Impf_DNAMLP(n_impf, impf_cfg['n_classes'])
        BEST_STATE_PATH = os.path.join(impf_cfg['MLP_BEST_STATES_DIR'], alg, group, f'{temp_fold}.pth')
        
        if torch.cuda.is_available() is False:
            model.load_state_dict(torch.load(BEST_STATE_PATH, map_location = torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(BEST_STATE_PATH))
        model.to(device)

        # Calculate diagnosis results and display
        st.write('---')
        res = softmax(model(sample), dim = 1)
        columns = [group, 'Control']
        row = res.detach().cpu().numpy()
        res_df = pd.DataFrame(row)
        res_df.columns = columns
        st.write('### Diagnostic results')
        col1, col2 = st.columns(2)
        col1.write(f'|| **{group}**:  \n~{round(row[0][0]*100, 2)}%')
        col2.write(f'|| **Non-tumour**:  \n~{round(row[0][1]*100, 2)}%')

        # Get ranked important features
        st.write('---')
        st.write('### CpG information')
        ranked_impf = get_ranked_impf(df, outer_fold, alg, group)
        ranked_impf = dict(sorted(ranked_impf.items(), key = lambda item:item[1]))
        ranked_impf = list(ranked_impf.keys())
        ranked_impf_df = pd.DataFrame()
        ranked_impf_df['CpG'] = ranked_impf
        ranked_impf_df['Priority'] = range(1, len(ranked_impf)+1)
        st.write(f'**Important CpGs - {len(ranked_impf)} CpGs (the smaller the priority the more important)**')

        gb = GridOptionsBuilder.from_dataframe(ranked_impf_df)
        gb.configure_selection('single', use_checkbox=True) 
        gridOptions = gb.build()
        grid_response = AgGrid(
            ranked_impf_df,
            gridOptions=gridOptions,
            data_return_mode='AS_INPUT', 
            update_mode='SELECTION_CHANGED', 
            fit_columns_on_grid_load=True,
            theme='material', #Add theme color to the table
            height=350, 
        )
        try:
            selected_cpg = grid_response['selected_rows'][0]['CpG']
            st.write(selected_cpg)
            # Get cpg information
            cpg_info_path = web_config.cpg_info_path
            cpg_df = get_cpg_info_file(cpg_info_path)
            cpg_df = cpg_df
            cpg_info = cpg_df.loc[cpg_df.Name == selected_cpg]
            cpg_info = cpg_info.to_dict()
            st.write(cpg_info)
        except Exception as e:
            st.write('*\*Select a CpG to know more information*')

        # Display cpg information
        st.write('---')
        st.write('*Hung Nguyen, 2022*')
    except Exception as e:
        st.write("Caught an error:", e)
except:
    st.write("*You haven't uploaded a csv file of correct format*")