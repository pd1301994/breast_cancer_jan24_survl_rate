import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from Programs import preprocessing_programs

title = "Data Presentation"
sidebar_name = "Data Presentation"
path = '/content/drive/MyDrive/Datasets/final_Data_merged.csv'


try:
 clinical_df =  pd.read_csv(path,  sep=',', index_col='_INTEGRATION')
 
except FileNotFoundError:
    alternative_path_unprocessed_data  = '../data/no_processed_data/final_Data_merged.csv'
    alternative_path_d1  = '../data/processed_data/df1.csv' #After imputation
    alternative_path_dfB  = '../data/processed_data/dfb_completed.csv' #After imputation
    try:
        clinical_df =  pd.read_csv(alternative_path_unprocessed_data,  sep=',', index_col='_INTEGRATION')
        df1 =  pd.read_csv(alternative_path_d1,  sep=',', index_col='_INTEGRATION')
        dfB =  pd.read_csv(alternative_path_d1,  sep=',', index_col='_INTEGRATION')

    except FileNotFoundError:
        print("Not found data")

def run():

    st.title(title)

    st.markdown(
        """
Here is where all the preprocessing work we have done will be displayed.
         """
    )
    st.subheader("Dataset head: ")
    st.dataframe(clinical_df.head())
    
    st.subheader("NaN Counts and Percentages per Column")
    st.write("To make a first approximation of the quality of the data set we had, it is first important to visualize the NaN values we had at the beginning. ")
    preprocessing_programs.calculate_nan_statistics(clinical_df)
    st.subheader("Number of categorical variables")
    st.write("This graph aims to visualize the amount of categorical values we have to deal with and what is the number of categories that they are. If a column has two unique values, would be \
             considered as 'Binary' and then we have divided the amount of unuique values in diferent sections to have an idea of which encoding we will need to apply in order to be able to analyze our data ")
    preprocessing_programs.categorize_unique_values(clinical_df)
   

   


