# -*- coding:utf-8 -*-

import streamlit as st
import pandas as pd
from utils import load_test_data


def run_prediction():
    df_test = load_test_data()
    
    st.markdown('### 고객별 이탈 예측')
    st.dataframe(df_test)
    
    with st.sidebar:
        st.number_input('CustomerId', min_value=15565701, max_value=15815690)
    
    
        
    

