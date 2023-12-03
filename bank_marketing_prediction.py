import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import altair as alt
import hiplot as hip
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

st.title("Bank Marketing Analysis")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ['Introduction', 'Objectives', 'Data Visualisation / Analysis', 'Predictive Modelling', 'Conclusion'])

bank_marketing_df = pd.read_csv(
    "https://raw.githubusercontent.com/madhav28/Bank-Marketing-Analysis/main/bank-full.csv", sep=';')
bank_marketing_df.rename(columns={'y': 'outcome'},  inplace=True)
