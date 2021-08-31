import streamlit as st
from collections import Counter
import matplotlib.pyplot as plt
from streamlit.proto.Selectbox_pb2 import Selectbox
from anomaly_detection import Anomaly_Detect

def app():

    st.title("Anomaly Detection - Prepare Data")

    data = st.session_state.data
    columns = data.columns.values

    cat_cols_auto = data.select_dtypes(include=['object']).columns.to_list()
    num_cols_auto = data.select_dtypes(include=['int64','float64']).columns.to_list()

    task_dict = {"Unsupervised - Sequential":'timeseries',"Unsupervised":"cluster","Supervised":"classify"}
    task_type = st.radio("Select the Task Type",
                ("Unsupervised - Sequential","Unsupervised") )#,"Supervised"))


    st.markdown("### Select Column Data Types")

    num_cols = st.multiselect("Select Numeric Columns",
                columns,num_cols_auto)

    cat_cols = st.multiselect("Select Categorical Columns",
                columns,cat_cols_auto )

    if task_type == "Unsupervised - Sequential":
        dt_col = st.selectbox("Select Date Time Column",
                    columns)
    else:
        dt_col = st.multiselect("Select Date Time Columns",
                    columns,[])


    if task_type == "Unsupervised":
        target = None
    else:
        target = st.selectbox('Select Target ',columns)

    
    done = st.button("Prepare Data")

    if done:
        if isinstance(dt_col,str):
            dt_col = [dt_col]

        com = num_cols + dt_col + cat_cols
        com_max = Counter(com).most_common(1)[0][1]

        if com_max >1:
            st.write("Some columns are selected in multiple selection. Please correct it")

        else:
            st.success("Data Preparation Succesful!")

            ad = Anomaly_Detect(data,.03)
            selected_cols, dropped_cols = ad.prepare_data(target,dt_col,num_cols,cat_cols,task_dict[task_type])

            ## 
            st.markdown("#### Selected Columns")
            st.dataframe(selected_cols)

            st.markdown("#### Dropped Columns")
            st.dataframe(dropped_cols)

            ## Add Session Data
            st.session_state.ad = ad
            st.session_state.target = target
            st.session_state.task = task_dict[task_type]
            st.success("Go to Train model")

        
