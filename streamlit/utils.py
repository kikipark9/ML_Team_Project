# -*- coding:utf-8 -*-

import pandas as pd
import streamlit as st


@st.cache_data
def load_data():
    data = pd.read_csv('../data/train.csv/train.csv')
    return data
