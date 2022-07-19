import streamlit as st
import web_config

st.sidebar.markdown('# [Dev] Alg Model Settings')
st.sidebar.selectbox('Choose model', web_config.algs)
n_impf = st.sidebar.slider('# n_features', 100, 10000)
st.title('Alg model tweaker')
st.write('---')
st.write(f'Select {str(n_impf)} most important features for each model.')