# -*- coding:utf-8 -*-

import streamlit as st
import pandas as pd
from utils import load_test_data, load_models, process_test_data
from catboost import CatBoostClassifier


def highlight_exit_prob(value):
    """Exit_Probability 값에 따라 색상을 지정하는 함수"""
    if value > 0.75:
        color = 'rgba(255, 0, 0, 0.5)'  # 빨간색, 반투명
    elif value > 0.5:
        color = 'rgba(255, 165, 0, 0.5)'  # 주황색, 반투명
    elif value > 0.25:
        color = 'rgba(255, 255, 0, 0.5)'  # 노란색, 반투명
    else:
        color = 'rgba(0, 128, 0, 0.5)'  # 녹색, 반투명
    return f'background-color: {color}'


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
    st.dataframe(result.style.applymap(highlight_exit_prob, subset=['Exit_Probability']), use_container_width=True)
    
