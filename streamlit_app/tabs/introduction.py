import streamlit as st

title = "Predicting Survival in Breast Cancer Patients: Data Science and Machine Learning Approaches"
sidebar_name = "Introduction"

def introduction():
    st.title("Introduction")


    st.header("Context")
    st.markdown("- Breast cancer has 2.3 million annual diagnoses and 670,000 global deaths.")
    st.markdown("- Factors such as demographics, clinical pathological factors etc. affect patient survival.")
    st.markdown("- Conventional manual methods have limitations identifying patterns.")

    st.markdown("- Machine learning and deep learning models have the capability to handle a wide amount of data.")

    st.header("Objectives")
    st.markdown("1. Build a model to predict survival in breast cancer patients.")
    st.markdown("2. Identify factors influencing breast cancer patient survival.")

    st.header("Datasets")
    st.markdown("Our analysis utilizes two publicly available datasets. Both datasets share the same patient ID.")
    st.markdown("1. Dataset 1: This dataset contains demographic, clinical, pathological, genetic, therapeutic information of breast cancer patients .")
    st.markdown("- The shape of the dataset is (1247, 194).")
    st.markdown("2. Dataset 2: This dataset provides survival information about the same set of patients .")
    st.markdown("- The shape of the dataset is (1236, 11).")
    st.markdown("In our analysis, we combine the two datasets which gives a final dataset with shape of (1236, 203).")
    st.write("")
    st.write("")
    st.write("")

def run():
    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    st.title(title)

    introduction()

if __name__ == "__main__":
    run()
