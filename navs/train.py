import streamlit as st
import matplotlib.pyplot as plt
from streamlit.proto.Button_pb2 import Button

def app():

    st.title("Anomaly Detection - Train Config")

    # Retrive from session
    ad = st.session_state.ad
    taks = st.session_state.task

    st.markdown("### Sample Processed Dataset")
    st.dataframe(ad.data.head(10))

    st.markdown("### Isolation Forest Configuration")
    iso_contamination = st.slider("Contamination Factor ",1,100,10,5,key='iso')

    st.markdown("### Elliptical Envelope Configuration")
    eenv_contamination = st.slider("Contamination Factor ",1,100,10,5,key='ell_env')

    st.markdown("### Upper/Lower Bound of Training erros/Support Vectors")
    nu = st.slider("Contamination Factor ",1,100,10,5,key='svm')

    st.markdown("### RNN window size")
    win_size = st.slider("Contamination Factor ",4,128,8,4,key='rnn_ws')
    
    done = st.button("Confirm Configuration")

    if done:
        st.session_state.iso_forest = {"contamination":iso_contamination/100.}
        st.session_state.eenv = {"contamination":eenv_contamination/100.}
        st.session_state.one_svm = {"contamination":nu/100.}
        st.session_state.rnn = {"win_size":win_size}
        st.session_state.retrain = True

        st.success("Model Training Configuraiton Stored")
