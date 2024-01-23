# -*- coding:utf-8 -*-

import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from ml.comparing import run_comparing
from ml.result_test import run_test
from ml.result_prediction import run_prediction


def home():
    st.markdown('''
    ### 머신러닝 개요
    - 사용한 모델
        - *Logistic Regression*
        - *Decision Tree*
        - *Random Forest*
        - *XGBoost*
        - *LGBM*
        - *CatBoost* : [공식 레퍼런스](https://catboost.ai/)
    - 주요 평가 지표
        - **AUC Score** : 종합 성능 평가지표. 모델이 얼마나 다양한 임계값에서 이탈 고객과 비이탈 고객을 구별하는지를 평가.
        - **Recall** : 실제 이탈 고객 중 모델이 이탈로 예측한 비율. 고객 유지가 중요한 비즈니스에서 이탈 고객 식별을 위해 높은 재현율의 모델을 채택
    - 모델별 성능 비교 및 중요 변수 파악
    - [Bank Customer Churn Prediction](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction) 데이터를 사용한 테스트
    - 고객별 이탈 확률 제공
    ''')


def run_ml(df):
    st.markdown("""
    ## 머신러닝을 활용한 고객 이탈 예측 모델
    """)
    
    selected = option_menu(None, ['Home', '성능비교', '테스트 결과', '고객별 예측'],
                           icons=['house', 'bar-chart', 'card-checklist', 'person-bounding-box'],
                           menu_icon='cast', default_index=0, orientation='horizontal',
                           styles={
                               'container': {'padding': '0!important', 'background-color': '#fafafa'},
                               'icon': {'color': 'orange', 'font-size': '25px'},
                               'nav-link': {'font-size': 'px', 'text-align': 'left', 'margin': '0px', '--hover-color': '#eee'},
                               'nav-link-selected': {'background-color': '#D2E3FC'}
                                   }
                           )
    
    if selected == 'Home':
        home()
    elif selected == '성능비교':
        run_comparing(df)
    elif selected == '테스트 결과':
        run_test()
    elif selected == '고객별 예측':
        run_prediction()
    else:
        st.warning('Wrong')
