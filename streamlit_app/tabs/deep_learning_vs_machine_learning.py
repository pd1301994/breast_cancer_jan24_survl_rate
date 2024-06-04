import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
import xgboost as xgb
import keras
from Programs import modeling_programs

title = "Deep learning vs Machine learning"
sidebar_name = "Deep learning vs Machine learning"

def run():
    st.title(title)
    
    st.write('We have analyzed multiple models and tested different parameters, and these are the four models that have performed the best for our dataset. For each model, you will be able to see if its working better with the two coders we choose for our categorical variables: Label encoder and get dummies.')
    st.subheader('Choose a model:')
    option = st.radio('', ['xgboost_label_encoded', 'xgboost_get_dummies_encoded', 'deep_learning_get_dummies_encoded', 'deep_learning_label_encoded', 'plot_importance_features', 'model_demonstration'])
    if option == 'xgboost_label_encoded':
        modeling_programs.xgboost_label_encoded()
    elif option == 'xgboost_get_dummies_encoded':
        modeling_programs.xgboost_one_hot_encoded()
    elif option == 'deep_learning_label_encoded':
        modeling_programs.deep_learning_label_encoded()
    elif option == 'deep_learning_get_dummies_encoded':
        modeling_programs.deep_learning_one_hot_encoded()
    elif option == 'plot_importance_features':
        modeling_programs.importance_features()
    elif option == 'model_demonstration':
        modeling_programs.Model_Demonstration()

        

if __name__ == "__main__":
    run()