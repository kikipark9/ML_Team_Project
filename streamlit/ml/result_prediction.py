# -*- coding:utf-8 -*-

import streamlit as st
import pandas as pd
from utils import load_test_data, load_models, process_test_data
from catboost import CatBoostClassifier


def run_prediction():
    df_test = load_test_data()
    models, names = load_models()
    X, y, customers = process_test_data(df_test)

    for model in models:        
        if isinstance(model.estimator, CatBoostClassifier):
            cat = model.best_estimator_
            cat_pred = cat.predict(X)
            cat_proba = cat.predict_proba(X)[:,1]
            break
        
    ndf = df_test.copy()
    ndf['CustomerId'] = ndf['CustomerId'].astype(str)
    ndf['IsActiveMember'] = ndf['IsActiveMember'].astype(int)
    ndf['HasCrCard'] = ndf['HasCrCard'].astype(int)
    ndf = ndf.drop(['Exited', 'RowNumber'], axis=1)
    
    ndf = ndf[['CustomerId', 'Surname', 'Geography', 'Gender', 'Age']]

    st.markdown('### 고객 명단')
    st.dataframe(ndf)
    st.markdown('<hr>', unsafe_allow_html=True)
    
    with st.sidebar:
        find_customer = st.text_input('CustomerId', '15565701')
    
    if cat_proba is not None:
        ndf['Exit_Probability'] = cat_proba
        
    result = ndf[ndf['CustomerId'] == find_customer]
    
    st.markdown('### 조회 고객의 이탈 확률')    
    st.dataframe(result)
    
