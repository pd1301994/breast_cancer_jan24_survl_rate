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
import h5py

def xgboost_label_encoded():
    st.write("### XGBoost Modeling with Label-encoded Data")
    # path to correct directory
    data_dir_le = '../data/model_label_data'
    ## load the trained model with XGBoost using pickle
    with open(os.path.join(data_dir_le, "best_xgb_model_label"), "rb") as xgb_le:
        loaded_ml_model_xgb_le = pickle.load(xgb_le)
    ## Load test dataset with label encoder
    ## load *.npy files
    X_test_le = np.load(os.path.join(data_dir_le, "X_test_scaled_label.npy"))
    y_test_le = np.load(os.path.join(data_dir_le, "y_test_label.npy"))
    X_train_le = np.load(os.path.join(data_dir_le, "X_train_scaled_adasyn_label.npy"))
    # use the model to make predictions
    prediction_le = loaded_ml_model_xgb_le.predict(X_test_le)
    ## Display Classiffication Report in a tabular format
    report_dict = classification_report(y_test_le, prediction_le, output_dict=True)
    # Convert classification report to DataFrame
    report_df = pd.DataFrame(report_dict).transpose()
    # Drop "macro avg" and "weighted avg" rows and the accuracy column
    report_df.drop(["macro avg", "weighted avg", "accuracy"], axis=0, inplace=True)
    # Round the values in the DataFrame to two decimal places
    report_df = report_df.round(2)
    # Create a title for the table with accuracy
    accuracy = report_dict["accuracy"]
    st.write(f'#### Classification report (Accuracy: {accuracy:.2f})')
    # Transpose the DataFrame to correctly align row labels
    report_df = report_df.T
    # Create a Plotly table from the DataFrame
    fig = go.Figure(data=[go.Table(
    header=dict(values=["Class"] + list(report_df.columns),
                fill_color='lightgray',
                align='left', height=35, 
                font=dict(color='black', size=22, family="Arial")), # Adjust font size and style
                cells=dict(values=[report_df.index] + [report_df[col] for col in report_df.columns],
                           fill_color='lightblue', align='left', height=35,
                           font=dict(color='black', size=20, family="Arial")) # Adjust font size and style
)])
    # Display the Plotly table with the title

    st.plotly_chart(fig)
    ## Display confusion matrix as a heatmap
    cm = confusion_matrix(y_test_le, prediction_le)
    plt.figure(figsize = (3, 2))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt = "g", cbar=False)
    plt.title("Confusion Matrix: XGBoost label-encoded data df1:", fontsize = 6)
    plt.xlabel("Predicted Class", fontsize = 6)
    plt.ylabel("Actual Class", fontsize = 6)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    #showPyplotGlobalUse = false
    # pass the figure object directly
    st.pyplot(plt)
    st.set_option('deprecation.showPyplotGlobalUse', False)

def xgboost_one_hot_encoded():

    st.write("### XGBoost Modelling with get_dummies encoded Data df1")
    # path to correct directory
    data_dir_gd = '../data/model_get_dummies_data'
    ## load the trained model with XGBoost using pickle
    with open(os.path.join(data_dir_gd, "best_xgb_model_get"), "rb") as xgb_gd:
        loaded_ml_model_xgb_gd = pickle.load(xgb_gd)
    ## Load test dataset with label encoder
    ## load *.npy files
    X_test_gd = np.load(os.path.join(data_dir_gd, "X_test_scaled_get.npy"))
    y_test_gd = np.load(os.path.join(data_dir_gd, "y_test_get.npy"))
    # use the model to make predictions
    prediction_gd = loaded_ml_model_xgb_gd.predict(X_test_gd)
    ## Display Classiffication Report in a tabular format
    report_dict = classification_report(y_test_gd, prediction_gd, output_dict=True)
    # Convert classification report to DataFrame
    report_df = pd.DataFrame(report_dict).transpose()
    # Drop "macro avg" and "weighted avg" rows and the accuracy column
    report_df.drop(["macro avg", "weighted avg", "accuracy"], axis=0, inplace=True)
    # Round the values in the DataFrame to two decimal places
    report_df = report_df.round(2)
    # Create a title for the table with accuracy
    accuracy = report_dict["accuracy"]
    st.write(f'#### Classification report (Accuracy: {accuracy:.2f})')
    # Transpose the DataFrame to correctly align row labels
    report_df = report_df.T
    # Create a Plotly table from the DataFrame
    fig = go.Figure(data=[go.Table(
    header=dict(values=["Class"] + list(report_df.columns),
                fill_color='lightgray',
                align='left', height=35, 
                font=dict(color='black', size=22, family="Arial")), # Adjust font size and style
                cells=dict(values=[report_df.index] + [report_df[col] for col in report_df.columns],
                           fill_color='lightblue', align='left', height=35,
                           font=dict(color='black', size=20, family="Arial")) # Adjust font size and style
)])
    # Display the Plotly table with the title
    st.plotly_chart(fig)
    ## Display confusion matrix as a heatmap
    cm = confusion_matrix(y_test_gd, prediction_gd)
    plt.figure(figsize = (3, 2))
    sns.heatmap(cm, annot=True, cmap="Reds", fmt = "g", cbar=False)
    plt.title("Confusion Matrix: XGBoost one hot-encoded data df1:", fontsize = 6)
    plt.xlabel("Predicted Class", fontsize = 6)
    plt.ylabel("Actual Class", fontsize = 6)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    #showPyplotGlobalUse = false
    # pass the figure object directly
    st.pyplot(plt)

def deep_learning_label_encoded():
    st.write("### Deep Learning Modeling with Label-encoded Data")
    
    # path to correct directory
    model_dl_le = "../data/model_label_data/deeplearning_label.h5"
    data_dir_le = "../data/model_label_data"
    ## load the trained model with XGBoost using pickle
    #loaded_model_dl_le = load.model(model_dl_le)
    loaded_model_dl_le = keras.models.load_model(model_dl_le)
    
    ## Load test dataset with label encoder
    ## load *.npy files
    X_test_le = np.load(os.path.join(data_dir_le, "X_test_scaled_label.npy"))
    y_test_le = np.load(os.path.join(data_dir_le, "y_test_label.npy"))
    X_train_le = np.load(os.path.join(data_dir_le, "X_train_scaled_adasyn_label.npy"))

    # use the model to make predictions
    prediction_dl_le_prob = loaded_model_dl_le.predict(X_test_le)
    prediction_dl_le = (prediction_dl_le_prob > 0.51).astype(int)
    ## Display Classiffication Report in a tabular format
    report_dict = classification_report(y_test_le, prediction_dl_le, output_dict=True)
    # Convert classification report to DataFrame
    report_df = pd.DataFrame(report_dict).transpose()
    # Drop "macro avg" and "weighted avg" rows and the accuracy column
    report_df.drop(["macro avg", "weighted avg", "accuracy"], axis=0, inplace=True)
    # Round the values in the DataFrame to two decimal places
    report_df = report_df.round(2)
    # Create a title for the table with accuracy
    accuracy = report_dict["accuracy"]
    st.write(f'#### Classification report (Accuracy: {accuracy:.2f})')
    # Transpose the DataFrame to correctly align row labels
    report_df = report_df.T
    # Create a Plotly table from the DataFrame
    fig = go.Figure(data=[go.Table(
    header=dict(values=["Class"] + list(report_df.columns),
                fill_color='lightgray',
                align='left', height=35, 
                font=dict(color='black', size=22, family="Arial")), # Adjust font size and style
                cells=dict(values=[report_df.index] + [report_df[col] for col in report_df.columns],
                           fill_color='lightblue', align='left', height=35,
                           font=dict(color='black', size=20, family="Arial")) # Adjust font size and style
)])
    # Display the Plotly table with the title
    st.plotly_chart(fig)
    ## Display confusion matrix as a heatmap
    cm = confusion_matrix(y_test_le, prediction_dl_le)
    plt.figure(figsize = (3, 2))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt = "g", cbar=False)
    plt.title("Confusion Matrix: DL label-encoded data df1:", fontsize = 6)
    plt.xlabel("Predicted Class", fontsize = 6)
    plt.ylabel("Actual Class", fontsize = 6)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    st.pyplot(plt)

def deep_learning_one_hot_encoded():
    st.write("### Deep Learning Modeling with get_dummies Encoded Data")
    # path to correct directory 
    model_dl_gd = "../data/model_get_dummies_data/deeplearning_get.h5"
    data_dir_gd = "../data/model_get_dummies_data"
    ## load the trained model with XGBoost using pickle
    #loaded_model_dl_le = load.model(model_dl_le)
    loaded_model_dl_gd = keras.models.load_model(model_dl_gd)
    ## Load test dataset with label encoder
    ## load *.npy files
    X_test_gd = np.load(os.path.join(data_dir_gd, "X_test_scaled_get.npy"))
    y_test_gd = np.load(os.path.join(data_dir_gd, "y_test_get.npy"))
    # use the model to make predictions
    prediction_dl_gd_prob = loaded_model_dl_gd.predict(X_test_gd)
    prediction_dl_gd = (prediction_dl_gd_prob > 0.5).astype(int)
    ## Display Classiffication Report in a tabular format
    report_dict = classification_report(y_test_gd, prediction_dl_gd, output_dict=True)
    # Convert classification report to DataFrame
    report_df = pd.DataFrame(report_dict).transpose()
    # Drop "macro avg" and "weighted avg" rows and the accuracy column
    report_df.drop(["macro avg", "weighted avg", "accuracy"], axis=0, inplace=True)
    # Round the values in the DataFrame to two decimal places
    report_df = report_df.round(2)
    # Create a title for the table with accuracy
    accuracy = report_dict["accuracy"]
    st.write(f'#### Classification report (Accuracy: {accuracy:.2f})')
    # Transpose the DataFrame to correctly align row labels
    report_df = report_df.T
    # Create a Plotly table from the DataFrame
    fig = go.Figure(data=[go.Table(
    header=dict(values=["Class"] + list(report_df.columns),
                fill_color='lightgray',
                align='left', height=35, 
                font=dict(color='black', size=22, family="Arial")), # Adjust font size and style
                cells=dict(values=[report_df.index] + [report_df[col] for col in report_df.columns],
                           fill_color='lightblue', align='left', height=35,
                           font=dict(color='black', size=20, family="Arial")) # Adjust font size and style
)])
    # Display the Plotly table with the title
    st.plotly_chart(fig)
    ## Display confusion matrix as a heatmap
    cm = confusion_matrix(y_test_gd, prediction_dl_gd)
    plt.figure(figsize = (3, 2))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt = "g", cbar=False)
    plt.title("Confusion Matrix: DL get_dummies data df1:", fontsize = 6)
    plt.xlabel("Predicted Class", fontsize = 6)
    plt.ylabel("Actual Class", fontsize = 6)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    st.pyplot(plt)

def importance_features():
    # Plot SHAP summary plot
    st.write("#### SHAP Summary Plot: Deep Learning with Labelencoded Data")
    st.image('shap_summary_plot.png')
    data_dir_le = '../data/model_label_data'

    ## plot plot_importance
    with open(os.path.join(data_dir_le, "best_xgb_model_label"), "rb") as xgb_le:
        loaded_ml_model_xgb_le = pickle.load(xgb_le)
    types = ['weight', 'gain']
    num_features = 10
    fig, axes = plt.subplots(2, 1, figsize=(12, 20))
    for i, f in enumerate(types):
        xgb.plot_importance(loaded_ml_model_xgb_le, max_num_features=num_features,
                            importance_type=f, ax=axes[i])
        axes[i].set_title('importance: ' + f, fontsize=16)
        axes[i].set_xlabel('Feature Importance', fontsize=16)  # Adjust size of x-axis label
        axes[i].set_ylabel('Feature', fontsize=16) 
        axes[i].tick_params(axis='both', which='major', labelsize=20)

    # Display feature importance plot in Streamlit
    st.write("####  Feature Importance: XGBoost Labelencoded Data")
    st.pyplot()
    # Your code for XGBoost with label-encoded data goes here
    st.write("#### The most imoprtant features for gain (accuracy) are: ")
    st.write("1. new_tumor_event_after_initial_treatment")
    st.write("2. sample_type")

    st.write("#### The most imoprtant features for weight (number of times a feature appears in a tree) are: ")
    st.write("1. days_to_last_followup")
    st.write("2. days_to_collection")
    st.write("")
    st.write("")

def Model_Demonstration():
    model_dl_le = "../data/model_label_data/deeplearning_label.h5"
    data_dir_le = "../data/model_label_data"
    ## load the trained model with XGBoost using pickle
    #loaded_model_dl_le = load.model(model_dl_le)
    loaded_model_dl_le = keras.models.load_model(model_dl_le)
    
    ## Load test dataset with label encoder
    ## load *.npy files
    X_test_le = np.load(os.path.join(data_dir_le, "X_test_scaled_label.npy"))
    y_test_le = np.load(os.path.join(data_dir_le, "y_test_label.npy"))
    X_train_le = np.load(os.path.join(data_dir_le, "X_train_scaled_adasyn_label.npy"))
    # use the model to make predictions
    prediction_dl_le = loaded_model_dl_le.predict(X_test_le)
    #prediction_dl_le = (prediction_dl_le_prob > 0.51).astype(int)
    # Identify indices for positive and negative classes
    positive_indices = np.where(y_test_le == 1)[0]
    negative_indices = np.where(y_test_le == 0)[0]
    # Select random samples from each class
    num_samples_per_class = 2
    random_positive_indices = np.random.choice(positive_indices, num_samples_per_class, replace=False)
    random_negative_indices = np.random.choice(negative_indices, num_samples_per_class, replace=False)
    # Combine the indices
    selected_indices = np.concatenate((random_positive_indices, random_negative_indices))
    # Select the corresponding samples and labels
    X_samples = X_test_le[selected_indices]
    y_samples = y_test_le[selected_indices]
    # Make predictions on the selected samples
    predictions = loaded_model_dl_le.predict(X_samples)
    # Plotting
    plt.figure(figsize=(10, 5))
    #plt.scatter(np.arange(len(selected_indices)), y_samples, color='blue', label='True Label', s=150)
    plt.figure(figsize=(10, 5))
    plt.scatter(np.arange(len(selected_indices)), predictions, color='red', label='Predicted Label')
    plt.title("Demonstration of model performance")
    plt.xlabel("Random set of four samples")
    plt.ylabel("Probability of Class Assignment")
    plt.axhline(y=0.5, color='blue', linestyle='--', label='Threshold')
    plt.legend()
    st.pyplot(plt)
    #formatted_prediction = "{:.2f}".format(prediction)
    for i, (true_label, predicted_label) in enumerate(zip(y_samples, predictions)):
        if predicted_label >.51:
            formatted_prediction = "{:.1f}".format(predicted_label[0]*100)
            formatted_prediction = formatted_prediction.strip('[]') 
            st.write(f"Sample {i+1}: Patient is predicted as diseased with a probability of {formatted_prediction}% .")
        elif predicted_label <=.51 :
            formatted_prediction = "{:.1f}".format((1-predicted_label[0])*100)
            formatted_prediction = float(formatted_prediction.strip('[]'))
            st.write(f"Sample {i+1}: Patient is predicted as living with a probability of {formatted_prediction} %.")
