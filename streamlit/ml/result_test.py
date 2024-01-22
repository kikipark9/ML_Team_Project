# -*- coding:utf-8 -*-

import streamlit as st
import pandas as pd
from utils import load_test_data, load_models
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


@st.cache_data
def process_test_data(df):
    df = df.drop(columns=['RowNumber', 'Surname'])
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    customers = df['CustomerId']
    df_test = df.drop(columns=['CustomerId'])
    ndf = pd.get_dummies(df_test, columns=['Geography','Gender'])

    X = ndf.drop(['Exited'], axis=1)
    y = ndf['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
    return X_test, y_test, customers


def run_test():
    df_test = load_test_data()
    models, names = load_models()
    X_test, y_test, customers = process_test_data(df_test)

    for model in models:        
        if isinstance(model.estimator, CatBoostClassifier):
            cat = model.best_estimator_
            predictions = cat.predict(X_test)
            result_cat = pd.DataFrame({'CustomerId': customers, 'Real': y_test, 'Prediction': predictions})
            st.write(f"Predictions by Catboost:")            
            
        elif isinstance(model.estimator, XGBClassifier):
            xgb = model.best_estimator_
            predictions = xgb.predict(X_test)
            st.write(f"Predictions by XGBoost:")
            result_xgb = pd.DataFrame({'CustomerId': customers, 'Real': y_test, 'Prediction': predictions})
            st.write(f"Predictions by Catboost:")
        else:
            pass

        
