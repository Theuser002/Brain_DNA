import streamlit as st
import web_config

st.sidebar.markdown('# [Dev] Alg Model Settings')
st.sidebar.selectbox('Choose model', web_config.algs)