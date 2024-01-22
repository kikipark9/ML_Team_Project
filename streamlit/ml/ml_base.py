# -*- coding:utf-8 -*-

import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu


def home():
    st.markdown('''
    ### 머신러닝 예측 개요
    - 모델별 성능 비교
        - Logistic Regression, Decision Tree, Random Forest, XGBoost, LGBM, CatBoost
    - 
    ''')


def run_ml(df):
    customers = df['CustomerId']
    df = df.drop(columns=['id', 'CustomerId', 'Surname'])
    X = df.drop(['Exited'], axis=1)
    y = df['Exited']

    st.markdown("""
    ## 머신러닝 예측 개요
    머신러닝 예측 페이지입니다.
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
    elif selected == '주거형태별':
        pass
    elif selected == '자치구역별':
        pass
    elif selected == '보고서':
        pass
    else:
        st.warning('Wrong')