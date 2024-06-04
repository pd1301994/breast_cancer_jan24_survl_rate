import streamlit as st
import matplotlib.pyplot as plt
from Programs import data_preprocessing_programs
import pandas as pd


title = "Data Preprocessing"
sidebar_name = "Data Preprocessing"


path_1 = '/content/drive/MyDrive/Datasets/df1.csv'
path_2 = "/content/drive/MyDrive/Datasets/dfB_complete.csv"

try:
    df1 = pd.read_csv(path_1, index_col='_INTEGRATION')
except FileNotFoundError:
    alternative_path_df1  = '../data/processed_data/df1.csv'
    alternative_path_dfB  = '../data/processed_data/dfB_complete.csv'
    df1 = pd.read_csv(alternative_path_df1, index_col='_INTEGRATION')
    dfB = pd.read_csv(alternative_path_dfB, index_col='_INTEGRATION')


target1 = df1['vital_status']
target1.replace(['DECEASED', 'LIVING'], [0, 1], inplace=True)
targetB = dfB['vital_status']
targetB.replace(['DECEASED', 'LIVING'], [0, 1], inplace=True)


# Display the plot
def run():
    st.title(title)
    st.markdown("---")
    st.subheader("Preprocessing Steps")
    st.markdown("This flowchart illusrtates our preprocessing steps")
    st.image("preprocessing_steps.jpg")
    st.subheader("Random Forest Imputer example result: ")
    st.image("imputing_graph.png")
    st.subheader("Visualizing Target Variable: Vital status")
    col1, col2 = st.columns(2)
    with col1:
        data_preprocessing_programs.value_counts(target1, "DF1")
    with col2:
        data_preprocessing_programs.value_counts(targetB, "DFB")
