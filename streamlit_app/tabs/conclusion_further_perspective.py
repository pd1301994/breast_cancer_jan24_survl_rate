import streamlit as st


title = "Conclusions and further perspective"
sidebar_name = "Conclusions and further perspective"


def run():
    st.title(title)

    st.markdown("---")

    st.markdown("## Challenges Encountered and Conclusions")
    st.markdown("- Lack of information about the dataframe.")
    st.markdown("- High volume of missing values and redundancy.")
    st.markdown("- Data imbalance.")
    st.markdown("- Computational constraints during data imputation and model training.")
    st.markdown("- Deep learning models outperformed XGBoost on label encoded data.")

    st.markdown("## Future Perspectives")
    st.markdown("- Exploring feature reduction methods may improve model effectiveness.")
    st.markdown("- Collaboration with domain experts.")
    st.markdown("- Continued optimization and tuning of model.")

if __name__ == "__main__":
    run()
