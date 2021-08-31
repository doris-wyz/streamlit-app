import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import shap
import utils
import time
st.set_option('deprecation.showPyplotGlobalUse', False)

def app():

    st.title("Anomaly Detection - Isolation Forest")

    try:
        task = st.session_state.task
        params = st.session_state.iso_forest
        retrain = st.session_state.retrain
    except:
        st.warning("Please configure the Training Params")

    if task == 'classify':
        st.write("Isolation Forest doesn't apply for Supervised Learning")
    
    
    elif task =='timeseries':
            ad = st.session_state.ad

            fig = ad.process_isolation_forest(params['contamination'])
            
            d = ad.data_org[ad.data_org.iso_prediction==-1]
            st.markdown("## Model Summary ")
            st.write("No of Anomalies Detected  : ",d.shape[0])
            st.write("No of Anomalies Detected percent : ",d.shape[0]/ad.data_org.shape[0])

            st.markdown("Anomalies (Sample)")

            if len(d) >1 :
                st.dataframe(d)
            else:
                st.write(" ### No anomalies predicted")
                
            st.markdown("## Ground Truth vs Predicted")
            st.pyplot(fig)

            st.markdown("Anomalies (Sample)")
            d = ad.data_org[ad.data_org.iso_prediction==-1]

            if len(d) >1 :
                st.dataframe(d)
            else:
                st.write(" ### No anomalies predicted")

    else:
        

        ad = st.session_state.ad
        p, sum_plot, sum_plot_box = ad.process_isolation_forest(params['contamination'])
        d = ad.data_org[ad.data_org.iso_prediction==-1]
        st.markdown("## Model Summary ")
        st.write("No of Anomalies Detected  : ",d.shape[0])
        st.write("No of Anomalies Detected percent : ",d.shape[0]/ad.data_org.shape[0])

        st.markdown("Anomalies (Sample)")

        if len(d) >1 :
            st.dataframe(d)
        else:
            st.write(" ### No anomalies predicted")
        
        st.markdown("## Single Row - Feature Importance")
 
        st.markdown("### Importance Plot")
        utils.st_shap(p)

        st.markdown("## Feature Importance Summary plot")
        st.pyplot(sum_plot)

        

        # time.sleep(1)
        # st.markdown("## Feature Importance Box Plot")
        # st.pyplot(sum_plot_box)
        

        
