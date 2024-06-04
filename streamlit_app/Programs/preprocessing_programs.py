#Importing libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


# There are 116480 total missing values
# We want to look at the distribution of missing values in each column of the dataset

def calculate_nan_statistics(df):
    # Calculate the percentage of NaN values per column
    nan_counts = df.isna().sum()
    nan_percentages = (nan_counts / len(df)) * 100

    # Create a new DataFrame with NaN counts and percentages
    nan_info = pd.DataFrame({
        'Column Name': nan_counts.index,
        'NaN Count': nan_counts,
        'NaN Percentage': nan_percentages
    })

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    nan_info['Rounded Percentage'] = nan_info['NaN Percentage'].round(1)

    # Bar plot for NaN counts
    axes[0].bar(nan_info.index, nan_info['NaN Count'], color='blue', alpha=0.7)
    axes[0].set_title('NaN Counts per Column')
    axes[0].set_ylabel('Count')
    axes[0].set_xticks([])  # Hide x-axis labels

    # Bar plot for NaN percentages
    axes[1].bar(nan_info.index, nan_info['NaN Percentage'], color='orange', alpha=0.7)
    axes[1].set_title('NaN Percentages per Column')
    axes[1].set_ylabel('Percentage')
    axes[1].set_xticks([])  # Hide x-axis labels

    # Adjust layout
    plt.tight_layout()

    # Show the plots using streamlit.pyplot
    st.pyplot(fig)

def categorize_unique_values(data):
    # Create a dictionary to store category counts
    category_count = {
        'Binary': 0,
        '2-5 categories': 0,
        '6-10 categories': 0,
        '10-20 categories': 0,
        'No categories ': 0
    }
    # Iterate on every column
    for column in data.columns:
        # Count unique values in every column
        unique_values_count = data[column].nunique()

        if unique_values_count == 2:
            category_count['Binary'] += 1
        elif 2 < unique_values_count <= 5:
            category_count['2-5 categories'] += 1
        elif 6 <= unique_values_count <= 10:
            category_count['6-10 categories'] += 1
        elif 11 <= unique_values_count <= 20:
            category_count['10-20 categories'] += 1
        elif unique_values_count > 20:
            category_count['No categories '] += 1
    # Create bargraph
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(category_count.keys(), category_count.values())

    # Add labels and title
    ax.set_xlabel('Categories')
    ax.set_ylabel('Number of columns')

    # Show the graph
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)