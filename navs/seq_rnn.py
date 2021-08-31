import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import shap
import utils
import time
st.set_option('deprecation.showPyplotGlobalUse', False)

def app():

    st.title("Anomaly Detection - Isolation Forest")

   
    task = st.session_state.task
    params = st.session_state.rnn
    if task == 'classify':
        st.write("RNN doesn't apply for Supervised Learning")
    
    
    elif task =='timeseries':
            ad = st.session_state.ad
            target = st.session_state.target

            fig = ad.process_rnn(params['win_size'])

            d = ad.data_org[ad.data_org.lstm_predict==-1]
            st.markdown("## Model Summary ")
            st.write("No of Anomalies Detected  : ",d.shape[0])
            st.write("No of Anomalies Detected percent : ",d.shape[0]/ad.data_org.shape[0])

            st.markdown("Anomalies (Sample)")

            if len(d) >1 :
                st.dataframe(d)
            else:
                st.write(" ### No anomalies predicted")

            st.markdown("## RNN prediction")
            st.pyplot(fig)

            

    else:

        st.write("Currently RNN supports single variable sequential data")
        

        
