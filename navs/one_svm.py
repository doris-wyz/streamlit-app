import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import shap
import utils
import time
st.set_option('deprecation.showPyplotGlobalUse', False)

def app():

    

   
    task = st.session_state.task
    params = st.session_state.one_svm
    if task == 'classify':
        st.title("Anomaly Detection - SVM")
        st.write("Isolation Forest doesn't apply for Supervised Learning")
    
    
    elif task =='timeseries':
        st.title("Anomaly Detection - One ClassSVM")
    
        ad = st.session_state.ad
        fig = ad.process_svm(params['contamination'])

        d = ad.data_org[ad.data_org.svm_prediction==-1]
        st.markdown("## Model Summary ")
        st.write("No of Anomalies Detected  : ",d.shape[0])
        st.write("No of Anomalies Detected percent : ",d.shape[0]/ad.data_org.shape[0])

        st.markdown("Anomalies (Sample)")

        if len(d) >1 :
            st.dataframe(d)
        else:
            st.write(" ### No anomalies predicted")

        st.markdown("## Anomaly Plot")
        st.pyplot(fig)




    else:
        st.title("Anomaly Detection - One ClassSVM")

        ad = st.session_state.ad
        p, sum_plot = ad.process_svm(params['contamination'])

        d = ad.data_org[ad.data_org.svm_prediction==-1]
        st.markdown("## Model Summary ")
        st.write("No of Anomalies Detected  : ",d.shape[0])
        st.write("No of Anomalies Detected percent : ",d.shape[0]/ad.data_org.shape[0])

        st.markdown("Anomalies (Sample)")

        if len(d) >1 :
            st.dataframe(d)
        else:
            st.write(" ### No anomalies predicted")
        
        st.markdown("## Single Row - Feature Importance")
        utils.st_shap(p)

        st.markdown("## Feature Importance Summary plot (Sampled")
        st.pyplot(sum_plot)


        
