# -*- coding:utf-8 -*-

import streamlit as st
from streamlit_option_menu import option_menu
from home import run_home
from utils import load_data
from eda.eda_home import run_eda
from ml.ml_base import run_ml
from ml.making_prediction import predict_page

def main():
    df = load_data()
    
    with st.sidebar:
        selected = option_menu('메뉴', ['홈', '탐색적 자료 분석', '고객 이탈 예측 모델', '가상 데이터 예측'], 
                       icons=['house', 'file-bar-graph', 'graph-up-arrow', 'chat-left-text'], menu_icon='clipboard-pulse', default_index=0)
    if selected =='홈':
        run_home()
    elif selected =='탐색적 자료 분석':
        run_eda(df)
    elif selected =="고객 이탈 예측 모델":
        run_ml(df)
    elif selected == '가상 데이터 예측':
        predict_page()
    else:
        st.warning('Wrong')


if __name__ == '__main__':
    main()
