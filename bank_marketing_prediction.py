import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

st.title("Bank Marketing Prediction")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ['About Data', 'Objectives', 'Data Pre-processing', 'Predictive Modelling', 'Results and Discussion', 'About Me'])

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

    st.markdown(
        "💡 Build multiple classification models and predict the subscription outcome of a client.")
    st.markdown(
        "💡 Compare all the built classifcation models and identify the best classification model.")

with tab3:
    st.markdown("**Following are the pre-processing steps used in this study:**")
    st.markdown("1. Drop duplicate rows.")
    st.markdown("2. Drop the rows with NaN values.")
    st.markdown(
        "3. Identify columns with 'unknown' values and impute the values.")
    st.markdown("4. Identify and remove the outliers.")
    st.markdown("---")
    st.markdown("#### Pre-processing Steps:")
    st.markdown("---")
    st.markdown("**1. Drop duplicate rows:**")

    st.markdown(
        "No duplicate rows were found in the data. So, no rows were dropped in this step.")
    st.markdown("**2. Drop the rows with NaN values:**")
    st.markdown(
        "There are no NaN values in the data. So, no rows were dropped in this step.")
    st.markdown(
        "**3. Identify columns with 'unknown' values and impute the values:**")
    unknown_df = {'Feature name': [], 'Percentage of unknown values': []}
    for column in bank_marketing_df.columns:
        unknown_df['Feature name'].append(column)
        if column in bank_marketing_df.select_dtypes(include='O').columns:
            val = (len(bank_marketing_df.loc[bank_marketing_df[column]
                                             == 'unknown'])/len(bank_marketing_df[column]))*100
            val = round(val, 2)
            unknown_df['Percentage of unknown values'].append(val)
        else:
            unknown_df['Percentage of unknown values'].append(0)
    unknown_df = pd.DataFrame(unknown_df)
    unknown_df.index = idx

    st.dataframe(unknown_df)

    st.markdown(
        "We have 4 features which have 'unknown' values in their values. How 'unknown' values are handled for each feature is shown below:")
    st.markdown(
        "* **job:** Since there are very few rows which have 'unknown' values we can remove the rows.")
    bank_marketing_df = bank_marketing_df.loc[bank_marketing_df['job'] != 'unknown']
    st.markdown("* **education:** We know that education and job are strongly correlated quantities. So, \
                we will use job to replace 'unknown' education values. We find the mode of the education \
                values for each job type and replace 'unknown' values with it.")
    job_to_edu_df = {'Job': [], 'Education': []}
    for job in bank_marketing_df['job'].unique():
        mode = bank_marketing_df.loc[bank_marketing_df['job']
                                     == job]['education'].mode()
        job_to_edu_df['Job'].append(job)
        job_to_edu_df['Education'].append(mode.values[0])
    job_to_edu_df = pd.DataFrame(job_to_edu_df)
    st.markdown("**Job to Education Mapping:**")
    st.table(job_to_edu_df)
    education_list = []
    for index, row in bank_marketing_df.iterrows():
        job = row['job']
        education = row['education']
        if education == 'unknown':
            education = job_to_edu_df[job_to_edu_df['Job']
                                      == job]['Education'].values
            education = education[0]
        education_list.append(education)

    bank_marketing_df['education'] = education_list

    st.markdown("* **contact:** Here, we are replacing 'unknown' values with 'other' to indicate that \
                the clients were not contacted by phone but were contact by other means.")

    contact = bank_marketing_df.loc[bank_marketing_df['contact'] == 'unknown']
    contact['contact'] = 'other'
    bank_marketing_df.loc[bank_marketing_df['contact'] == 'unknown'] = contact

    st.markdown("* **poutcome:** Here, we are replacing 'unknown' values with 'first campaign' to indicate \
                 that we are dealing with newest potential clients.")

    poutcome = bank_marketing_df.loc[bank_marketing_df['poutcome'] == 'unknown']
    poutcome['poutcome'] = 'first campaign'
    bank_marketing_df.loc[bank_marketing_df['poutcome']
                          == 'unknown'] = poutcome

    st.markdown(
        "**4. Identify and remove the outliers.:**")

    figure, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    sns.boxplot(x=bank_marketing_df['age'], ax=axes[0, 0])
    sns.boxplot(x=bank_marketing_df['balance'], ax=axes[0, 1])
    sns.boxplot(x=bank_marketing_df['duration'], ax=axes[0, 2])
    sns.boxplot(x=bank_marketing_df['campaign'], ax=axes[1, 0])
    sns.boxplot(x=bank_marketing_df['pdays'], ax=axes[1, 1])
    sns.boxplot(x=bank_marketing_df['previous'], ax=axes[1, 2])

    plt.tight_layout()

    st.pyplot(figure)

    st.markdown("Based on the above plots, the following outliers are removed:")
    st.markdown("* duration > 3000")
    st.markdown("* campaign > 40")
    st.markdown("* previous > 40")

    bank_marketing_df = bank_marketing_df.loc[bank_marketing_df['campaign'] < 40]
    bank_marketing_df = bank_marketing_df.loc[bank_marketing_df['previous'] < 40]
    bank_marketing_df = bank_marketing_df.loc[bank_marketing_df['duration'] < 3000]

with tab4:

    st.markdown("""To predict the subscription outcome of a client, the following 
                classification models are developed:""")
    st.markdown("1. Logistic Regression")
    st.markdown("2. Support Vector Machine")
    st.markdown("3. Decision Tree")
    st.markdown("4. K-Nearest Neighbours")
    st.markdown("5. Naive Bayes")

    X = bank_marketing_df.drop('outcome', axis=1)
    y = bank_marketing_df['outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    numeric_features = ['age', 'balance', 'day',
                        'duration', 'campaign', 'pdays', 'previous']
    categorical_features = ['job', 'marital', 'education',
                            'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))
