from os import name
from pandas.io.parsers import read_csv
import streamlit as st
import pandas as pd
import numpy as np
from streamlit.uploaded_file_manager import UploadedFile

def app():

    st.title("Anomaly Detection - HOME")

    st.write("""
            Un-supervised Anomaly detection for both sequential and non-sequential data.    

    """)


    st.markdown("## Data Upload")

    # Upload the dataset and save as csv
    st.markdown("### Upload a csv file for analysis.") 
    st.write("\n")

    # Code to read a single file 
    uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx','tsv'])
    
    if uploaded_file:
        try: 
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)

            elif uploaded_file.name.endswith(".tsv"):
                data = pd.read_csv(uploaded_file,sep='\t')
            
            else:
                data = pd.read_csv(uploaded_file)
        except:
            st.warning("File format not supported")

    ''' Load the data and save the columns with categories as a dataframe. 
    This section also allows changes in the numerical and categorical columns. '''
    if st.button("Load Data"):
        
        st.success("Loading Successful.!")

        st.markdown("### Data Description")
        st.write("Number of Rows     : ", data.shape[0])
        st.write("Number of Columns  : ",data.shape[1])
        # Raw data 
        st.markdown("### Data sample view")
        st.dataframe(data.head(25))

        st.session_state.data = data

        st.session_state.next = True
        st.session_state.page = 0
