import streamlit as st
import pandas as pd

st.title("Bank Marketing Analysis")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ['Introduction', 'Objectives', 'Data Visualisation / Analysis', 'Predictive Modelling', 'Conclusion'])

bank_marketing_df = pd.read_csv(
    "https://raw.githubusercontent.com/madhav28/Bank-Marketing-Prediction/master/bank-full.csv", sep=';')
bank_marketing_df.rename(columns={'y': 'outcome'},  inplace=True)
