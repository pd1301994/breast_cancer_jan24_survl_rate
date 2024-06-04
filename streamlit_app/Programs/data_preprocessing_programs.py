import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def value_counts (df, dfname):

    deceased_count_target= df.value_counts()[0]
    living_count_target = df.value_counts()[1]
    living_count_target= df.value_counts()[1]
    deceased_count_target = df.value_counts()[0]
    st.write(f'#### {dfname}')
    fig, ax = plt.subplots()
    ax.bar(['Deceased', 'Living'], [deceased_count_target, living_count_target])
    ax.set_ylabel('count')
    st.pyplot(fig)

