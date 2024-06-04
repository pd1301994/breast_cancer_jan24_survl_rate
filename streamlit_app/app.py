from collections import OrderedDict

import streamlit as st

# TODO : change TITLE, TEAM_MEMBERS and PROMOTION values in config.py.
import config
from tabs import data_presentation, introduction, deep_learning_vs_machine_learning, data_preprocessing, data_modelling, conclusion_further_perspective



st.set_page_config(
    page_title=config.TITLE,
    page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png",
)

with open("style.css", "r") as f: 
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


# TODO: add new and/or renamed tab in this ordered dict by
# passing the name in the sidebar as key and the imported tab
# as value as follow :
TABS = OrderedDict(
    [
        (introduction.sidebar_name, introduction),
        (data_presentation.sidebar_name, data_presentation),
        (data_preprocessing.sidebar_name, data_preprocessing),
        (data_modelling.sidebar_name, data_modelling),
        (deep_learning_vs_machine_learning.sidebar_name, deep_learning_vs_machine_learning),
        (conclusion_further_perspective.sidebar_name, conclusion_further_perspective)
        
    ]
)


def run():
    st.sidebar.image(
        "https://dst-studio-template.s3.eu-west-3.amazonaws.com/logo-datascientest.png",
        width=200,
    )
    tab_name = st.sidebar.radio("", list(TABS.keys()), 0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")

    st.sidebar.markdown("### Team members:")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    tab = TABS[tab_name]

    tab.run()

if __name__ == "__main__":
    run()
