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
    - 모델별 성능 비교
        - Logistic Regression, Decision Tree, Random Forest, XGBoost, LGBM, CatBoost
    - 
    ''')


def run_ml(df):
    st.markdown("""
    ## 머신러닝을 활용한 고객별 이탈 예측
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