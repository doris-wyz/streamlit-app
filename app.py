import streamlit as st
import pandas as pd
import numpy as np

from navs import home, prepare_data, train, iso_forest, one_svm, seq_rnn, elliptical_env

PAGES = {
    "Home": home,
    "Prepare Data": prepare_data,
    "Train Config" : train,
    "Isolation Forest" : iso_forest,
    "Elliptical Envelope":elliptical_env,
    "SVM":one_svm,
    "RNN": seq_rnn
}

page_list = list(PAGES.keys())
st.sidebar.title('Navigation')
selection = st.sidebar.empty()
d=selection.radio("Go to", list(PAGES.keys()),0)
page = PAGES[d]
page.app()



# if ('next' in st.session_state) and (st.session_state.next == True):
#     print("next")
#     st.session_state.page += 1
#     st.session_state.next = False
#     d=selection.radio("Go to", list(PAGES.keys()),st.session_state.page)
#     page = PAGES[d]
#     page.app()
    

# if 'task_type' in st.session_state:
#     PAGES['New Page'] = prepare_data
#     selection.radio("Go to", list(PAGES.keys()))

