import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix


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

    st.markdown("""Our primary objective in this study is to predict whether a client 
                subscribes to a term deposit or not. Given that our target variable 
                is a binary variable, the classification models will best suit for 
                prediction. To predict the subscription outcome, the following 
                classification models are developed:""")
    st.markdown("1. Logistic Regression")
    st.markdown("2. Support Vector Machine")
    st.markdown("3. Decision Tree")
    st.markdown("4. K-Nearest Neighbours")
    st.markdown("5. Naive Bayes")
    st.markdown("""Each of the above model is developed in each tab. In each of 
                the following tabs, an option is provided for users of this application 
                to develop different models with different hyperparameters for predicting 
                the subscription outcome.""")

    pm_tab1, pm_tab2, pm_tab3, pm_tab4, pm_tab5 = st.tabs(
        ['Logistic Regression', 'Support Vector Machine', 'Decision Tree', 'K-Nearest Neighbours', 'Naive Bayes'])

    with pm_tab1:

        st.markdown("""Please enter the model details and prediction details in order to build the model and 
                    predict the subscription outcome using Logistic Regression. The data from the **Model Details** 
                    section will be used for building the model and the data from the **Prediction Details** section 
                    will be used for predicting the subscription outcome.""")
        st.markdown("#### Model Details")
        st.markdown("---")

        features = st.multiselect(
            "Select features", bank_marketing_df.columns[0:16],
            default=np.array(bank_marketing_df.columns[0:16]),
            key="features_pm_tab1")
        test_size = st.slider("Select test size", 0.1,
                              0.9, 0.2, 0.1, key="test_size_pm_tab1")

        st.markdown("#### Prediction Details")
        st.markdown("---")
        X_pred = {}

        for feature in features:
            if feature == "age":
                age = st.slider(
                    "Select age:", min_value=18, max_value=100, value=35, step=1, key="age_pm_tab1")
                X_pred["age"] = [age]
            if feature == "job":
                options = ["admin.", "blue-collar", "technician", "services", "management",
                           "retired", "self-employed", "entrepreneur", "housemaid", "student"]
                job = st.selectbox(
                    "Select job:", options, key="job_pm_tab1")
                X_pred["job"] = [job]
            if feature == "marital":
                options = ["married", "single", "divorced"]
                marital = st.selectbox(
                    "Select marital:", options, key="marital_pm_tab1")
                X_pred["marital"] = [marital]
            if feature == "education":
                options = ["primary", "secondary", "tertiary"]
                education = st.selectbox(
                    "Select education:", options, key="education_pm_tab1")
                X_pred["education"] = [education]
            if feature == "default":
                options = ["no", "yes"]
                default = st.selectbox(
                    "Select default:", options, key="default_pm_tab1")
                X_pred["default"] = [default]
            if feature == "balance":
                balance = st.slider(
                    "Select balance:", min_value=-20000, max_value=100000, value=5000, step=100, key="balance_pm_tab1")
                X_pred["balance"] = [balance]
            if feature == "housing":
                options = ["no", "yes"]
                housing = st.selectbox(
                    "Select housing:", options, key="housing_pm_tab1")
                X_pred["housing"] = [housing]
            if feature == "loan":
                options = ["no", "yes"]
                loan = st.selectbox(
                    "Select loan:", options, key="loan_pm_tab1")
                X_pred["loan"] = [loan]
            if feature == "contact":
                options = ["cellular", "telephone", "other"]
                contact = st.selectbox(
                    "Enter contact:", options, key="contact_pm_tab1")
                X_pred["contact"] = [contact]
            if feature == "day":
                day = st.slider(
                    "Select day:", min_value=1, max_value=31, value=15, step=1, key="day_pm_tab1")
                X_pred["day"] = [day]
            if feature == "month":
                options = ["jan", "feb", "mar", "apr", "may", "jun",
                           "jul", "aug", "sep", "oct", "nov", "dec"]
                month = st.selectbox(
                    "Select month:", options, key="month_pm_tab1")
                X_pred["month"] = [month]
            if feature == "duration":
                duration = st.slider(
                    "Enter duration:", min_value=0, max_value=3000, value=1000, step=1, key="duration_pm_tab1")
                X_pred["duration"] = [duration]
            if feature == "campaign":
                campaign = st.slider(
                    "Select campaign:", min_value=0, max_value=40, value=10, step=1, key="campaign_pm_tab1")
                X_pred["campaign"] = [campaign]
            if feature == "pdays":
                pdays = st.slider(
                    "Select pdays:", min_value=0, max_value=1000, value=100, step=1, key="pdays_pm_tab1")
                X_pred["pdays"] = [pdays]
            if feature == "previous":
                previous = st.slider(
                    "Select previous:", min_value=0, max_value=40, value=10, step=1, key="previous_pm_tab1")
                X_pred["previous"] = [previous]
            if feature == "poutcome":
                options = ["success", "failure", "first campaign", "other"]
                poutcome = st.selectbox(
                    "Select poutcome:", options, key="poutcome_pm_tab1")
                X_pred["poutcome"] = [poutcome]

        build_model_predict_subscription_outcome = st.button(
            "Build Model and Predict Subscription Outcome", type="primary", key="build_model_predict_subscription_outcome_pm_tab1")

        if build_model_predict_subscription_outcome:

            X = bank_marketing_df[features]
            y = bank_marketing_df['outcome']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size)

            numeric_features = ['age', 'balance', 'day',
                                'duration', 'campaign', 'pdays', 'previous']
            numeric_features_temp = []
            for feature in numeric_features:
                if feature in features:
                    numeric_features_temp.append(feature)
            numeric_features = numeric_features_temp
            categorical_features = ['job', 'marital', 'education',
                                    'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
            categorical_features_temp = []
            for feature in categorical_features:
                if feature in features:
                    categorical_features_temp.append(feature)
            categorical_features = categorical_features_temp

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
            report = classification_report(y_test, y_pred, output_dict=True)
            report = pd.DataFrame(report)
            st.markdown("#### Classification Report:")
            st.table(report)

            st.markdown("#### Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            cm = pd.DataFrame(cm, index=["No", "Yes"], columns=["No", "Yes"])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d",
                        cmap="Blues", linewidths=.5, ax=ax)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)

            X_pred = pd.DataFrame(X_pred)
            y_pred = model.predict(X_pred)
            if y_pred == "yes":
                st.markdown(
                    "#### Subscription Outcome: Client will subscribe!!")
            else:
                st.markdown(
                    "#### Subscription Outcome: Client will not subscribe!!")

    with pm_tab2:

        st.markdown("""Please enter the model details and prediction details in order to build the model and 
                    predict the subscription outcome using Support Vector Machine. The data from the **Model Details** 
                    section will be used for building the model and the data from the **Prediction Details** section 
                    will be used for predicting the subscription outcome.""")
        st.markdown("#### Model Details")
        st.markdown("---")

        features = st.multiselect(
            "Select features", bank_marketing_df.columns[0:16],
            default=np.array(bank_marketing_df.columns[0:16]),
            key="features_pm_tab2")
        test_size = st.slider("Select test size", 0.1,
                              0.9, 0.2, 0.1, key="test_size_pm_tab2")

        st.markdown("#### Prediction Details")
        st.markdown("---")
        X_pred = {}

        for feature in features:
            if feature == "age":
                age = st.slider(
                    "Select age:", min_value=18, max_value=100, value=35, step=1, key="age_pm_tab2")
                X_pred["age"] = [age]
            if feature == "job":
                options = ["admin.", "blue-collar", "technician", "services", "management",
                           "retired", "self-employed", "entrepreneur", "housemaid", "student"]
                job = st.selectbox(
                    "Select job:", options, key="job_pm_tab2")
                X_pred["job"] = [job]
            if feature == "marital":
                options = ["married", "single", "divorced"]
                marital = st.selectbox(
                    "Select marital:", options, key="marital_pm_tab2")
                X_pred["marital"] = [marital]
            if feature == "education":
                options = ["primary", "secondary", "tertiary"]
                education = st.selectbox(
                    "Select education:", options, key="education_pm_tab2")
                X_pred["education"] = [education]
            if feature == "default":
                options = ["no", "yes"]
                default = st.selectbox(
                    "Select default:", options, key="default_pm_tab2")
                X_pred["default"] = [default]
            if feature == "balance":
                balance = st.slider(
                    "Select balance:", min_value=-20000, max_value=100000, value=5000, step=100, key="balance_pm_tab2")
                X_pred["balance"] = [balance]
            if feature == "housing":
                options = ["no", "yes"]
                housing = st.selectbox(
                    "Select housing:", options, key="housing_pm_tab2")
                X_pred["housing"] = [housing]
            if feature == "loan":
                options = ["no", "yes"]
                loan = st.selectbox(
                    "Select loan:", options, key="loan_pm_tab2")
                X_pred["loan"] = [loan]
            if feature == "contact":
                options = ["cellular", "telephone", "other"]
                contact = st.selectbox(
                    "Enter contact:", options, key="contact_pm_tab2")
                X_pred["contact"] = [contact]
            if feature == "day":
                day = st.slider(
                    "Select day:", min_value=1, max_value=31, value=15, step=1, key="day_pm_tab2")
                X_pred["day"] = [day]
            if feature == "month":
                options = ["jan", "feb", "mar", "apr", "may", "jun",
                           "jul", "aug", "sep", "oct", "nov", "dec"]
                month = st.selectbox(
                    "Select month:", options, key="month_pm_tab2")
                X_pred["month"] = [month]
            if feature == "duration":
                duration = st.slider(
                    "Enter duration:", min_value=0, max_value=3000, value=1000, step=1, key="duration_pm_tab2")
                X_pred["duration"] = [duration]
            if feature == "campaign":
                campaign = st.slider(
                    "Select campaign:", min_value=0, max_value=40, value=10, step=1, key="campaign_pm_tab2")
                X_pred["campaign"] = [campaign]
            if feature == "pdays":
                pdays = st.slider(
                    "Select pdays:", min_value=0, max_value=1000, value=100, step=1, key="pdays_pm_tab2")
                X_pred["pdays"] = [pdays]
            if feature == "previous":
                previous = st.slider(
                    "Select previous:", min_value=0, max_value=40, value=10, step=1, key="previous_pm_tab2")
                X_pred["previous"] = [previous]
            if feature == "poutcome":
                options = ["success", "failure", "first campaign", "other"]
                poutcome = st.selectbox(
                    "Select poutcome:", options, key="poutcome_pm_tab2")
                X_pred["poutcome"] = [poutcome]

        build_model_predict_subscription_outcome = st.button(
            "Build Model and Predict Subscription Outcome", type="primary", key="build_model_predict_subscription_outcome_pm_tab2")

        if build_model_predict_subscription_outcome:

            X = bank_marketing_df[features]
            y = bank_marketing_df['outcome']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size)

            numeric_features = ['age', 'balance', 'day',
                                'duration', 'campaign', 'pdays', 'previous']
            numeric_features_temp = []
            for feature in numeric_features:
                if feature in features:
                    numeric_features_temp.append(feature)
            numeric_features = numeric_features_temp
            categorical_features = ['job', 'marital', 'education',
                                    'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
            categorical_features_temp = []
            for feature in categorical_features:
                if feature in features:
                    categorical_features_temp.append(feature)
            categorical_features = categorical_features_temp

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
                ('classifier', SVC())
            ])

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            report = pd.DataFrame(report)
            st.markdown("#### Classification Report:")
            st.table(report)

            st.markdown("#### Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            cm = pd.DataFrame(cm, index=["No", "Yes"], columns=["No", "Yes"])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d",
                        cmap="Blues", linewidths=.5, ax=ax)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)

            X_pred = pd.DataFrame(X_pred)
            y_pred = model.predict(X_pred)
            if y_pred == "yes":
                st.markdown(
                    "#### Subscription Outcome: Client will subscribe!!")
            else:
                st.markdown(
                    "#### Subscription Outcome: Client will not subscribe!!")

    with pm_tab3:

        st.markdown("""Please enter the model details and prediction details in order to build the model and 
                    predict the subscription outcome using Decision Tree. The data from the **Model Details** 
                    section will be used for building the model and the data from the **Prediction Details** section 
                    will be used for predicting the subscription outcome.""")
        st.markdown("#### Model Details")
        st.markdown("---")

        features = st.multiselect(
            "Select features", bank_marketing_df.columns[0:16],
            default=np.array(bank_marketing_df.columns[0:16]),
            key="features_pm_tab3")
        test_size = st.slider("Select test size", 0.1,
                              0.9, 0.2, 0.1, key="test_size_pm_tab3")

        st.markdown("#### Prediction Details")
        st.markdown("---")
        X_pred = {}

        for feature in features:
            if feature == "age":
                age = st.slider(
                    "Select age:", min_value=18, max_value=100, value=35, step=1, key="age_pm_tab3")
                X_pred["age"] = [age]
            if feature == "job":
                options = ["admin.", "blue-collar", "technician", "services", "management",
                           "retired", "self-employed", "entrepreneur", "housemaid", "student"]
                job = st.selectbox(
                    "Select job:", options, key="job_pm_tab3")
                X_pred["job"] = [job]
            if feature == "marital":
                options = ["married", "single", "divorced"]
                marital = st.selectbox(
                    "Select marital:", options, key="marital_pm_tab3")
                X_pred["marital"] = [marital]
            if feature == "education":
                options = ["primary", "secondary", "tertiary"]
                education = st.selectbox(
                    "Select education:", options, key="education_pm_tab3")
                X_pred["education"] = [education]
            if feature == "default":
                options = ["no", "yes"]
                default = st.selectbox(
                    "Select default:", options, key="default_pm_tab3")
                X_pred["default"] = [default]
            if feature == "balance":
                balance = st.slider(
                    "Select balance:", min_value=-20000, max_value=100000, value=5000, step=100, key="balance_pm_tab3")
                X_pred["balance"] = [balance]
            if feature == "housing":
                options = ["no", "yes"]
                housing = st.selectbox(
                    "Select housing:", options, key="housing_pm_tab3")
                X_pred["housing"] = [housing]
            if feature == "loan":
                options = ["no", "yes"]
                loan = st.selectbox(
                    "Select loan:", options, key="loan_pm_tab3")
                X_pred["loan"] = [loan]
            if feature == "contact":
                options = ["cellular", "telephone", "other"]
                contact = st.selectbox(
                    "Enter contact:", options, key="contact_pm_tab3")
                X_pred["contact"] = [contact]
            if feature == "day":
                day = st.slider(
                    "Select day:", min_value=1, max_value=31, value=15, step=1, key="day_pm_tab3")
                X_pred["day"] = [day]
            if feature == "month":
                options = ["jan", "feb", "mar", "apr", "may", "jun",
                           "jul", "aug", "sep", "oct", "nov", "dec"]
                month = st.selectbox(
                    "Select month:", options, key="month_pm_tab3")
                X_pred["month"] = [month]
            if feature == "duration":
                duration = st.slider(
                    "Enter duration:", min_value=0, max_value=3000, value=1000, step=1, key="duration_pm_tab3")
                X_pred["duration"] = [duration]
            if feature == "campaign":
                campaign = st.slider(
                    "Select campaign:", min_value=0, max_value=40, value=10, step=1, key="campaign_pm_tab3")
                X_pred["campaign"] = [campaign]
            if feature == "pdays":
                pdays = st.slider(
                    "Select pdays:", min_value=0, max_value=1000, value=100, step=1, key="pdays_pm_tab3")
                X_pred["pdays"] = [pdays]
            if feature == "previous":
                previous = st.slider(
                    "Select previous:", min_value=0, max_value=40, value=10, step=1, key="previous_pm_tab3")
                X_pred["previous"] = [previous]
            if feature == "poutcome":
                options = ["success", "failure", "first campaign", "other"]
                poutcome = st.selectbox(
                    "Select poutcome:", options, key="poutcome_pm_tab3")
                X_pred["poutcome"] = [poutcome]

        build_model_predict_subscription_outcome = st.button(
            "Build Model and Predict Subscription Outcome", type="primary", key="build_model_predict_subscription_outcome_pm_tab3")

        if build_model_predict_subscription_outcome:

            X = bank_marketing_df[features]
            y = bank_marketing_df['outcome']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size)

            numeric_features = ['age', 'balance', 'day',
                                'duration', 'campaign', 'pdays', 'previous']
            numeric_features_temp = []
            for feature in numeric_features:
                if feature in features:
                    numeric_features_temp.append(feature)
            numeric_features = numeric_features_temp
            categorical_features = ['job', 'marital', 'education',
                                    'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
            categorical_features_temp = []
            for feature in categorical_features:
                if feature in features:
                    categorical_features_temp.append(feature)
            categorical_features = categorical_features_temp

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
                ('classifier', DecisionTreeClassifier())
            ])

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            report = pd.DataFrame(report)
            st.markdown("#### Classification Report:")
            st.table(report)

            st.markdown("#### Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            cm = pd.DataFrame(cm, index=["No", "Yes"], columns=["No", "Yes"])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d",
                        cmap="Blues", linewidths=.5, ax=ax)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)

            X_pred = pd.DataFrame(X_pred)
            y_pred = model.predict(X_pred)
            if y_pred == "yes":
                st.markdown(
                    "#### Subscription Outcome: Client will subscribe!!")
            else:
                st.markdown(
                    "#### Subscription Outcome: Client will not subscribe!!")

    with pm_tab4:

        st.markdown("""Please enter the model details and prediction details in order to build the model and 
                    predict the subscription outcome using Logistic Regression. The data from the **Model Details** 
                    section will be used for building the model and the data from the **Prediction Details** section 
                    will be used for predicting the subscription outcome.""")
        st.markdown("#### Model Details")
        st.markdown("---")

        features = st.multiselect(
            "Select features", bank_marketing_df.columns[0:16],
            default=np.array(bank_marketing_df.columns[0:16]),
            key="features_pm_tab4")
        test_size = st.slider("Select test size", 0.1,
                              0.9, 0.2, 0.1, key="test_size_pm_tab4")

        st.markdown("#### Prediction Details")
        st.markdown("---")
        X_pred = {}

        for feature in features:
            if feature == "age":
                age = st.slider(
                    "Select age:", min_value=18, max_value=100, value=35, step=1, key="age_pm_tab4")
                X_pred["age"] = [age]
            if feature == "job":
                options = ["admin.", "blue-collar", "technician", "services", "management",
                           "retired", "self-employed", "entrepreneur", "housemaid", "student"]
                job = st.selectbox(
                    "Select job:", options, key="job_pm_tab4")
                X_pred["job"] = [job]
            if feature == "marital":
                options = ["married", "single", "divorced"]
                marital = st.selectbox(
                    "Select marital:", options, key="marital_pm_tab4")
                X_pred["marital"] = [marital]
            if feature == "education":
                options = ["primary", "secondary", "tertiary"]
                education = st.selectbox(
                    "Select education:", options, key="education_pm_tab4")
                X_pred["education"] = [education]
            if feature == "default":
                options = ["no", "yes"]
                default = st.selectbox(
                    "Select default:", options, key="default_pm_tab4")
                X_pred["default"] = [default]
            if feature == "balance":
                balance = st.slider(
                    "Select balance:", min_value=-20000, max_value=100000, value=5000, step=100, key="balance_pm_tab4")
                X_pred["balance"] = [balance]
            if feature == "housing":
                options = ["no", "yes"]
                housing = st.selectbox(
                    "Select housing:", options, key="housing_pm_tab4")
                X_pred["housing"] = [housing]
            if feature == "loan":
                options = ["no", "yes"]
                loan = st.selectbox(
                    "Select loan:", options, key="loan_pm_tab4")
                X_pred["loan"] = [loan]
            if feature == "contact":
                options = ["cellular", "telephone", "other"]
                contact = st.selectbox(
                    "Enter contact:", options, key="contact_pm_tab4")
                X_pred["contact"] = [contact]
            if feature == "day":
                day = st.slider(
                    "Select day:", min_value=1, max_value=31, value=15, step=1, key="day_pm_tab4")
                X_pred["day"] = [day]
            if feature == "month":
                options = ["jan", "feb", "mar", "apr", "may", "jun",
                           "jul", "aug", "sep", "oct", "nov", "dec"]
                month = st.selectbox(
                    "Select month:", options, key="month_pm_tab4")
                X_pred["month"] = [month]
            if feature == "duration":
                duration = st.slider(
                    "Enter duration:", min_value=0, max_value=3000, value=1000, step=1, key="duration_pm_tab4")
                X_pred["duration"] = [duration]
            if feature == "campaign":
                campaign = st.slider(
                    "Select campaign:", min_value=0, max_value=40, value=10, step=1, key="campaign_pm_tab4")
                X_pred["campaign"] = [campaign]
            if feature == "pdays":
                pdays = st.slider(
                    "Select pdays:", min_value=0, max_value=1000, value=100, step=1, key="pdays_pm_tab4")
                X_pred["pdays"] = [pdays]
            if feature == "previous":
                previous = st.slider(
                    "Select previous:", min_value=0, max_value=40, value=10, step=1, key="previous_pm_tab4")
                X_pred["previous"] = [previous]
            if feature == "poutcome":
                options = ["success", "failure", "first campaign", "other"]
                poutcome = st.selectbox(
                    "Select poutcome:", options, key="poutcome_pm_tab4")
                X_pred["poutcome"] = [poutcome]

        build_model_predict_subscription_outcome = st.button(
            "Build Model and Predict Subscription Outcome", type="primary", key="build_model_predict_subscription_outcome_pm_tab4")

        if build_model_predict_subscription_outcome:

            X = bank_marketing_df[features]
            y = bank_marketing_df['outcome']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size)

            numeric_features = ['age', 'balance', 'day',
                                'duration', 'campaign', 'pdays', 'previous']
            numeric_features_temp = []
            for feature in numeric_features:
                if feature in features:
                    numeric_features_temp.append(feature)
            numeric_features = numeric_features_temp
            categorical_features = ['job', 'marital', 'education',
                                    'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
            categorical_features_temp = []
            for feature in categorical_features:
                if feature in features:
                    categorical_features_temp.append(feature)
            categorical_features = categorical_features_temp

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
                ('classifier', KNeighborsClassifier())
            ])

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            report = pd.DataFrame(report)
            st.markdown("#### Classification Report:")
            st.table(report)

            st.markdown("#### Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            cm = pd.DataFrame(cm, index=["No", "Yes"], columns=["No", "Yes"])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d",
                        cmap="Blues", linewidths=.5, ax=ax)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)

            X_pred = pd.DataFrame(X_pred)
            y_pred = model.predict(X_pred)
            if y_pred == "yes":
                st.markdown(
                    "#### Subscription Outcome: Client will subscribe!!")
            else:
                st.markdown(
                    "#### Subscription Outcome: Client will not subscribe!!")

    with pm_tab5:

        st.markdown("""Please enter the model details and prediction details in order to build the model and 
                    predict the subscription outcome using Logistic Regression. The data from the **Model Details** 
                    section will be used for building the model and the data from the **Prediction Details** section 
                    will be used for predicting the subscription outcome.""")
        st.markdown("#### Model Details")
        st.markdown("---")

        features = st.multiselect(
            "Select features", bank_marketing_df.columns[0:16],
            default=np.array(bank_marketing_df.columns[0:16]),
            key="features_pm_tab5")
        test_size = st.slider("Select test size", 0.1,
                              0.9, 0.2, 0.1, key="test_size_pm_tab5")

        st.markdown("#### Prediction Details")
        st.markdown("---")
        X_pred = {}

        for feature in features:
            if feature == "age":
                age = st.slider(
                    "Select age:", min_value=18, max_value=100, value=35, step=1, key="age_pm_tab5")
                X_pred["age"] = [age]
            if feature == "job":
                options = ["admin.", "blue-collar", "technician", "services", "management",
                           "retired", "self-employed", "entrepreneur", "housemaid", "student"]
                job = st.selectbox(
                    "Select job:", options, key="job_pm_tab5")
                X_pred["job"] = [job]
            if feature == "marital":
                options = ["married", "single", "divorced"]
                marital = st.selectbox(
                    "Select marital:", options, key="marital_pm_tab5")
                X_pred["marital"] = [marital]
            if feature == "education":
                options = ["primary", "secondary", "tertiary"]
                education = st.selectbox(
                    "Select education:", options, key="education_pm_tab5")
                X_pred["education"] = [education]
            if feature == "default":
                options = ["no", "yes"]
                default = st.selectbox(
                    "Select default:", options, key="default_pm_tab5")
                X_pred["default"] = [default]
            if feature == "balance":
                balance = st.slider(
                    "Select balance:", min_value=-20000, max_value=100000, value=5000, step=100, key="balance_pm_tab5")
                X_pred["balance"] = [balance]
            if feature == "housing":
                options = ["no", "yes"]
                housing = st.selectbox(
                    "Select housing:", options, key="housing_pm_tab5")
                X_pred["housing"] = [housing]
            if feature == "loan":
                options = ["no", "yes"]
                loan = st.selectbox(
                    "Select loan:", options, key="loan_pm_tab5")
                X_pred["loan"] = [loan]
            if feature == "contact":
                options = ["cellular", "telephone", "other"]
                contact = st.selectbox(
                    "Enter contact:", options, key="contact_pm_tab5")
                X_pred["contact"] = [contact]
            if feature == "day":
                day = st.slider(
                    "Select day:", min_value=1, max_value=31, value=15, step=1, key="day_pm_tab5")
                X_pred["day"] = [day]
            if feature == "month":
                options = ["jan", "feb", "mar", "apr", "may", "jun",
                           "jul", "aug", "sep", "oct", "nov", "dec"]
                month = st.selectbox(
                    "Select month:", options, key="month_pm_tab5")
                X_pred["month"] = [month]
            if feature == "duration":
                duration = st.slider(
                    "Enter duration:", min_value=0, max_value=3000, value=1000, step=1, key="duration_pm_tab5")
                X_pred["duration"] = [duration]
            if feature == "campaign":
                campaign = st.slider(
                    "Select campaign:", min_value=0, max_value=40, value=10, step=1, key="campaign_pm_tab5")
                X_pred["campaign"] = [campaign]
            if feature == "pdays":
                pdays = st.slider(
                    "Select pdays:", min_value=0, max_value=1000, value=100, step=1, key="pdays_pm_tab5")
                X_pred["pdays"] = [pdays]
            if feature == "previous":
                previous = st.slider(
                    "Select previous:", min_value=0, max_value=40, value=10, step=1, key="previous_pm_tab5")
                X_pred["previous"] = [previous]
            if feature == "poutcome":
                options = ["success", "failure", "first campaign", "other"]
                poutcome = st.selectbox(
                    "Select poutcome:", options, key="poutcome_pm_tab5")
                X_pred["poutcome"] = [poutcome]

        build_model_predict_subscription_outcome = st.button(
            "Build Model and Predict Subscription Outcome", type="primary", key="build_model_predict_subscription_outcome_pm_tab5")

        if build_model_predict_subscription_outcome:

            X = bank_marketing_df[features]
            y = bank_marketing_df['outcome']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size)

            numeric_features = ['age', 'balance', 'day',
                                'duration', 'campaign', 'pdays', 'previous']
            numeric_features_temp = []
            for feature in numeric_features:
                if feature in features:
                    numeric_features_temp.append(feature)
            numeric_features = numeric_features_temp
            categorical_features = ['job', 'marital', 'education',
                                    'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
            categorical_features_temp = []
            for feature in categorical_features:
                if feature in features:
                    categorical_features_temp.append(feature)
            categorical_features = categorical_features_temp

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
                ('classifier', GaussianNB())
            ])

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            report = pd.DataFrame(report)
            st.markdown("#### Classification Report:")
            st.table(report)

            st.markdown("#### Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            cm = pd.DataFrame(cm, index=["No", "Yes"], columns=["No", "Yes"])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d",
                        cmap="Blues", linewidths=.5, ax=ax)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)

            X_pred = pd.DataFrame(X_pred)
            y_pred = model.predict(X_pred)
            if y_pred == "yes":
                st.markdown(
                    "#### Subscription Outcome: Client will subscribe!!")
            else:
                st.markdown(
                    "#### Subscription Outcome: Client will not subscribe!!")

    with tab5:
        st.markdown("""Our primary goal in this study is to predict whether a client 
                    will subscribe to a term deposit or not. We developed five kinds of models 
                    to achieve this. To determine which is the best kind of model, we have 
                    a lot of performance metrics like accuracy, precision, f1-score, etc. for 
                    comparion. In our case, accuracy is a bad metric for comparison because 
                    of the disproportionate number of negative classes against positive classes. 
                    Rather, precision of positive class prediction is the best metric for 
                    model comparison. This is because, with higher precision, bank would be 
                    able to optimally allocate its resources on clients who are most likely 
                    to subscribe to a term deposit.""")

        st.markdown("##### ❌ metric for model comparison")
        data = {"Model": ["Logistic Regression", "Support Vector Machine",
                          "Decision Tree", "K-Nearest Neighbours", "Naive Bayes"],
                "Accuracy": [0.90, 0.90, 0.88, 0.89, 0.85]}
        df = pd.DataFrame(data)
        fig = px.bar(df, x="Accuracy", y="Model",
                     title="Accuracy for various models", labels={"y": "Values"})
        st.plotly_chart(fig)

        st.markdown("##### ✅ metric for model comparison")
        data = {"Model": ["Logistic Regression", "Support Vector Machine",
                          "Decision Tree", "K-Nearest Neighbours", "Naive Bayes"],
                "Precision of Yes": [0.65, 0.66, 0.46, 0.59, 0.39]}
        df = pd.DataFrame(data)
        fig = px.bar(df, x="Precision of Yes", y="Model",
                     title="Precision of Yes for various models", labels={"y": "Values"})
        st.plotly_chart(fig)

        st.markdown("""We can see that the accuracy of most of the models is very high and close 
                    to 90%. This is because only a small fraction of all the clients actually 
                    subscribe to a term deposit. Thereby, even with all No's as our prediction, 
                    we get high accuracy scores for our models. Whereas, if we see the precision 
                    scores for the models they are different for different models and highest 
                    precision is around 0.66. This high precision is achieved by Support Vector 
                    Classifier. Hence, out of all the classifiers, Support Vector Classifier is 
                    the best estimator to solve this problem.""")

    with tab6:
        st.markdown(
            "Currently pursuing Master's in Data Science at Michigan State University.")
        st.markdown("**Coursework:**")
        st.markdown("CSE482, STT810, CMSE830")
        st.markdown(
            "**Previous work experience:**")
        st.markdown(
            "* Technology Analyst at Citi for 2.5 years.")
        st.markdown("* Data Scientist at Gyan Data for 1.5 years.")
        st.markdown(
            "**Previous education:**")
        st.markdown(
            "Bachelor of Technology in Chemical Engineering, IIT Madras.")
        st.markdown("**Research Interests:**")
        st.markdown("Data-driven identificaton of ODE's and PDE's.")
        st.markdown("**Hobbies:**")
        st.markdown("Chess, Cricket and Football.")
