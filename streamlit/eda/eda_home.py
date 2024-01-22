# -*- coding:utf-8 -*-

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd


def home():
    st.markdown('''
    # EDA_HOME
    - 변수 별 분석
    ''')
    st.markdown('### Visualization')
    st.markdown('### Statistics')


def run_eda(df):
    st.markdown('''
    ## 탐색적 자료 분석 개요
    탐색적 자료 분석 페이지입니다.\n
    내용을 추가할 수 있습니다.
                ''')
    
    selected = option_menu(None, ['Home', 'Visualization', 'Statistics'],
                           icons=['house', 'bar-chart', 'file-spreadsheet'],
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
    elif selected == 'Visualization':
        pass
    elif selected == 'Statistics':
        pass
