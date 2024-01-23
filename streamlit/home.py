import pandas as pd
from utils import load_data
import streamlit as st
from millify import prettify
from io import StringIO

def run_home():
    df = load_data()
    st.markdown('''
    ## 고객 이탈 예측 
    - 본 프로젝트는 케글 데이터를 활용해 은행 고객의 이탈 예측을 모델링합니다.
    - 은행 고객 데이터를 활용하여 고객의 이탈 여부를 예측하는 모델을 개발하고, 서비스 개선 및 고객 유지 전략 수립을 지원하는 것을 목표로 합니다.
    - 데이터 : [Binary Classification with a Bank Churn Dataset](https://www.kaggle.com/competitions/playground-series-s4e1)
    ''')
    buffer = StringIO()
    df.info(buf=buffer)
    info = buffer.getvalue()

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('### 데이터 현황')
    
    tab1, tab2 = st.tabs(['데이터 요약', '변수 분류'])
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.text(info)
        with col2:
            st.markdown('''
            - 데이터 포인트 : 165,034 
            - 컬럼 수 : 14개
            - 결측치 : 0개
            ''')
                        
    with tab2:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('''
            ##### 범주형 변수
            - HasCrCard : 신용카드 보유 여부
            - IsActiveMember : 활성 멤버 여부
            - Geography : 거주 국가
            - Gender : 성별
            ##### 타겟 변수
            - Exited : 고객 이탈 여부(0, 1)
            ''')
        with col2:
            st.markdown('''
            ##### 수치형 변수
            - CreditScore : 신용점수
            - Age : 나이
            - Tenure : 거래 기간
            - Balance : 잔고
            - NumOfProducts : 가입 상품
            - EstimatedSalary : 추정 연봉
            ''')        

    # col1, col2 = st.columns([2, 1])
    # with col1:
    #     st.text(info)
    # with col2:
    #     st.markdown('''
    #     ##### 범주형 변수
    #     - HasCrCard : 신용카드 보유
    #     - IsActiveMember : 활성멤버
    #     - Geography : 거주 국가
    #     - Gender : 성별
    #     ##### 수치형 변수
    #     - CreditScore : 신용점수
    #     - Age : 나이
    #     - Tenure : 거래 기간
    #     - Balance : 잔고
    #     - NumOfProducts : 가입 상품
    #     - EstimatedSalary : 추정 연봉
    #     ##### 타겟 변수
    #     - Exited : 고객 이탈 여부(0, 1)
    #     ''')
        
    st.markdown('<hr>', unsafe_allow_html=True)                
    df['CustomerId'] = df['CustomerId'].astype(str)
    df['IsActiveMember'] = df['IsActiveMember'].astype(int)
    df['HasCrCard'] = df['HasCrCard'].astype(int)

    column_configuration = {
        'id': st.column_config.NumberColumn(
            'id', help='The id of the user'
        ),
        'CustomerId': st.column_config.TextColumn(
            'CustomerId', help='The unique id of the customer'
        ),
        'HasCrCard': st.column_config.CheckboxColumn(
            'HasCrCard', help='HasCrCard'
        ),
        'IsActiveMember': st.column_config.CheckboxColumn(
            'IsActiveMember', help='IsActiveMember'
        )
    }
    st.markdown('### 데이터 프레임')
    st.data_editor(
        data=df,
        use_container_width=True,
        column_config=column_configuration,
        num_rows="fixed",
        hide_index=True,
    )

    st.markdown('<hr>', unsafe_allow_html=True)
