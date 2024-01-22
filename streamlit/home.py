import pandas as pd
from utils import load_data
import streamlit as st
from millify import prettify

def run_home():
    total_df = load_data()
    st.markdown('''
    ## 대시보드 개요 
    본 프로젝트는 케글 데이터를 활용해 은행 고객의 이탈 예측을 모델링합니다.\n
    여기에 내용을 추가할 수 있습니다.
    ''')

    st.dataframe(total_df)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('출처 : [Binary Classification with a Bank Churn Dataset](https://www.kaggle.com/competitions/playground-series-s4e1)')
