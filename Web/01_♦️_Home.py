import sys
sys.path.append('../src')

import streamlit as st
import os
import pandas as pd
import numpy as np
import pickle
import config
import utils
import impf_models.MLP as impf_MLP
import scripts.get_impf as get_impf
import torch
import web_config

from impf_models.MLP.Model import Impf_DNAMLP, Impf_GlioMLP
from torch.nn.functional import softmax
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

# Globals
outer_folds = web_config.outer_folds
algs = web_config.algs
inner_fold_indexes = web_config.inner_fold_indexes
tissue_groups = web_config.tissue_groups
impf_cfg = config.impf_config
device = impf_cfg['device']

@st.cache
def get_impf_dataframe(df, outer_fold = '1.0', alg = 'RF', group = 'Embryonal'):
    with open(os.path.join(impf_cfg['IMPORTANT_FEATURES_DIR'], alg, group, f'{outer_fold}_combined.pkl'), 'rb') as file:
        impf = pickle.load(file)
    df = df.loc[:, sorted(list(impf))]
    return df, len(impf)

@st.cache
def load_impf(outer_fold = '1.0', alg = 'RF', group = 'Embryonal'):
    with open(os.path.join(impf_cfg['IMPORTANT_FEATURES_DIR'], alg, group, f'{outer_fold}_combined.pkl'), 'rb') as file:
        impf = pickle.load(file)
    return impf

@st.cache
# def check_contain_features(df, impf):
    # return set(df.columns).issuperset(impf)

@st.cache
def get_ranked_impf(df, outer_fold = '1.0', alg = 'RF', group = 'Embryonal'):
    return get_impf.ranked_impf_features(alg, group, outer_fold)    

@st.cache
def get_cpg_info_file(cpg_info_path):
    return pd.read_csv(cpg_info_path, index_col = 0, low_memory=False)

# MAIN
## Main page
st.sidebar.markdown('# Home')

title = 'DNA methylation-based classification of central nervous system tumours'
st.title(title, 'homepage')
st.write('---  \n---')
st.write('### File upload')
uploaded_file = st.file_uploader('UPLOAD A CSV FILE')
try:
    st.write(f'File ***{uploaded_file.name}*** received.')
    try:
        st.write('---  \n---')
        st.write('### CpG filtering')
        # df = pd.read_csv(uploaded_file, nrows=1)
        df = pd.read_csv(uploaded_file, index_col=0)
        n_samples = df.shape[0]

        # Get user's input
        form_0_submitted = False
        with st.form(key = 'form_0'):
            sample_index = st.selectbox('Choose ur sample: ', df.index)
            alg = st.selectbox('Select important features algorithm: ', algs)
            group = st.selectbox('Select tissue origin: ', tissue_groups)
            selected_fold = st.selectbox('Select a fold model: ', web_config.inner_folds)
            submitted = st.form_submit_button(label = 'Submit')
            if submitted:
                form_0_submitted = True
        if form_0_submitted:
            st.write(f'*{sample_index} - {alg} - {group} - {selected_fold}*')
            df = pd.DataFrame(df.loc[df.index == sample_index,:])
            st.write('**Sample:**')
            st.write(df)

            outer_fold = f"{selected_fold.split('.')[0]}.0"
            df, n_impf = get_impf_dataframe(df, outer_fold, alg, group)
            st.write('**Sample after CpG filtering:**')
            st.write(df)
            sample = np.expand_dims(np.array(df.iloc[0,:]).astype(float), axis = 0)
            sample = torch.Tensor(sample).to(device)
            
            if group in utils.low_perf_groups:
                model = Impf_GlioMLP(n_impf, impf_cfg['n_classes'])
            else:
                model = Impf_DNAMLP(n_impf, impf_cfg['n_classes'])
            BEST_STATE_PATH = os.path.join(impf_cfg['MLP_BEST_STATES_DIR'], alg, group, f'{selected_fold}.pth')
            
            if torch.cuda.is_available() is False:
                model.load_state_dict(torch.load(BEST_STATE_PATH, map_location = torch.device('cpu')))
            else:
                model.load_state_dict(torch.load(BEST_STATE_PATH))
            model.to(device)

            # Calculate diagnosis results and display
            st.write('---  \n---')
            res = softmax(model(sample), dim = 1)
            columns = [group, 'Control']
            probs = res.detach().cpu().numpy()
            res_df = pd.DataFrame(probs)
            res_df.columns = columns
            st.write('### Diagnostic results')
            col1, col2 = st.columns(2)
            if alg == 'LR':
                col1.write(f'|| **{group}**:  \n~{round((1 - probs[0][0])*100, 2)}%')
                col2.write(f'|| **Non-tumour**:  \n~{round((1 - probs[0][1])*100, 2)}%')
            else:
                col1.write(f'|| **{group}**:  \n~{round(probs[0][0]*100, 2)}%')
                col2.write(f'|| **Non-tumour**:  \n~{round(probs[0][1]*100, 2)}%')

            # Get ranked important features
            st.write('---  \n---')
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
    except Exception as e:
        st.write("Caught an error:", e)
except:
    st.write("*You haven't uploaded a csv file of correct format*")