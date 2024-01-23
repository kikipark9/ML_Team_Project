# -*- coding:utf-8 -*-

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from utils import load_data
from eda.viz import plot_target
from eda.viz import cat_cols_graph
from eda.viz import cat_exited_graph
from eda.viz import num_exited_graph
from eda.viz import age_to_cat_graph
from eda.viz import cat_heatmap
from eda.viz import num_heatmap

def img():
    image_object = st.image(
        "https://emyrael.github.io/assets/img/churn.png"
    )
    return image_object

def home():
    st.markdown('''
    ## EDA 목표
    - 변수 간 트렌드, 패턴, 관계 파악
    - 데이터의 이상치 및 결측치 파악
    - 프로젝트 초기 가설 수립
    - 다양한 모델 비교를 통한 최적의 모델 선정
    ''')
    st.markdown('''
    ## Visualization
    - 타겟 데이터
    - 범주형 데이터
    - 수치형 데이터
    ''')

def explore():
    selection = st.sidebar.selectbox(
        ":Yellow[Option]",
        ("전체 데이터", "타겟 데이터", "범주형 데이터", "수치형 데이터"),
    )
    return selection



def run_eda(df):
    st.markdown('''
    # 탐색적 자료 분석(EDA)
    ''')
    
    selected = option_menu(None, ['Home', 'Visualization'],
                           icons=['house', 'bar-chart'],
                           menu_icon='cast', default_index=0, orientation='horizontal',
                           styles={
                               'container': {'padding': '0!important', 'background-color': '#fafafa'},
                               'icon': {'color': 'orange', 'font-size': '25px'},
                               'nav-link': {'font-size': 'px', 'text-align': 'left', 'margin': '0px', '--hover-color': '#eee'},
                               'nav-link-selected': {'background-color': '#D2E3FC'}
                                   }
                           )
    
    
    if selected == 'Home':
        img()
        home()
    elif selected == 'Visualization':
        ok = explore()
        if ok == "전체 데이터":
            tab1, tab2, tab3 = st.tabs(["데이터 분포", "이탈에 따른 데이터 분포", "상관계수"])
            with tab1:                
                plot_target()
                cat_cols_graph()
            with tab2:                
                cat_exited_graph()
                num_exited_graph()
                age_to_cat_graph()
            with tab3:                
                cat_heatmap()
                num_heatmap()
        elif ok == "타겟 데이터":
            plot_target()
        elif ok == "범주형 데이터":
            tab1, tab2, tab3 = st.tabs(["데이터 분포", "이탈에 따른 데이터 분포", "상관계수"])
            with tab1:                
                cat_cols_graph()
            with tab2:                
                cat_exited_graph()
            with tab3:                
                cat_heatmap()
        elif ok == "수치형 데이터":
            tab1, tab2 = st.tabs(["이탈에 따른 데이터 분포", "상관계수"])
            with tab1:
                num_exited_graph()
                age_to_cat_graph()
            with tab2:
                num_heatmap()

