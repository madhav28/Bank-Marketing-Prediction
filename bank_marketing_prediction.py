import streamlit as st
import pandas as pd
import numpy as np

st.title("Bank Marketing Prediction")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ['About Data', 'Objectives', 'Predictive Modelling', 'Conclusion', 'About Me'])

bank_marketing_df = pd.read_csv(
    "https://raw.githubusercontent.com/madhav28/Bank-Marketing-Prediction/master/bank-full.csv", sep=';')
bank_marketing_df.rename(columns={'y': 'outcome'},  inplace=True)

with tab1:

    st.image("overview.jpg")

    st.markdown("**Data Overview:**")

    st.markdown("The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. \
                 The classification goal is to predict if the client will subscribe a term deposit (variable outcome).")

    st.markdown("**Data Description:**")

    customer_description_df = pd.read_csv(
        "column_description.txt", names=['Type', 'Description'])

    feature_description = {}

    column_name = bank_marketing_df.columns

    feature_description['Column Name'] = column_name
    feature_description['Type'] = customer_description_df['Type']
    feature_description['Description'] = customer_description_df['Description']

    feature_description_df = pd.DataFrame(feature_description)

    idx = np.linspace(1, 17, 17)
    idx = idx.astype(int)

    feature_description_df.index = idx

    st.table(feature_description_df)

with tab2:
    st.markdown("**Objectives of the application:**")

    st.markdown("In this application, the following two questions are answered:")
    st.markdown(
        "   **💡 What kind of customers are likely to subscribe to a term deposit?**")
    st.markdown(
        "   **💡 What marketing strategies are to be employed by the bank for a successful and an economical campaign?**")
