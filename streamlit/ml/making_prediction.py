# -*- coding: utf-8 -*-

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
from utils import load_models, load_data
from catboost import CatBoostClassifier

Geography = ("France", "Spain", "Germany")
Gender = ("Male", "Female")
HasCrCard = ("Yes", "No")
IsActiveMember = ("Yes", "No")


def predict_page():
    st.title("은행의 이탈 고객 예측 모델 :bank:")
    st.write("""#### 이탈 여부를 예측하려고 하는 고객의 데이터를 입력하세요""")
    
    creditscore = st.number_input("신용점수", 350, 850, 500)
    age = st.number_input("나이", 18, 100)
    tenure = int(st.slider("거래 기간", 0, 10, 1))
    balance = st.number_input("잔고", 0, 250000, 55000)
    nop = int(
        st.slider("가입 상품 수", 0, 4, 1)
    )
    hasCreditCard = st.radio("당행의 신용카드를 보유하고 있습니까?", HasCrCard)
    isActiveMember = st.radio("현재 활성멤버 입니까?", IsActiveMember)
    estimatedSalary = st.number_input("추정 연봉을 입력하세요", 0, 200000, 100000)    
    geography_any = st.selectbox("거주 국가", Geography)
    gender_any = st.radio("성별", Gender)    

    # converting input to desired format
    # Has Credit Card
    for item in HasCrCard:
        if item == hasCreditCard:
            hasCreditCard = 1
        elif item == hasCreditCard:
            hasCreditCard = 0

    # Is active member
    for item in IsActiveMember:
        if item == isActiveMember:
            isActiveMember = 1
        elif item == isActiveMember:
            isActiveMember = 0

    # Geographical of the user
    Geography_France, Geography_Germany, Geography_Spain = None, None, None
    if geography_any == "France":
        Geography_France = 1
        Geography_Germany = 0
        Geography_Spain = 0
    elif geography_any == "Germany":
        Geography_France = 0
        Geography_Germany = 1
        Geography_Spain = 0
    else:
        Geography_France = 0
        Geography_Germany = 0
        Geography_Spain = 1

    # Gender of the user
    Gender_Female, Gender_Male = None, None
    if gender_any == "Female":
        Gender_Female = 1
        Gender_Male = 0
    else:
        Gender_Female = 0
        Gender_Male = 1

    st.divider()
    button_pressed = st.button("예측하시겠습니까?", help="Click to make prediction")

    if button_pressed:
        predictor = np.array(
            [
                [
                    creditscore,
                    age,
                    tenure,
                    balance,
                    nop,
                    hasCreditCard,
                    isActiveMember,
                    estimatedSalary,
                    Geography_France,
                    Geography_Germany,
                    Geography_Spain,
                    Gender_Female,
                    Gender_Male,
                ]
            ]
        )
        columns = [
            "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", 
            "IsActiveMember", "EstimatedSalary", "Geography_France", "Geography_Germany", 
            "Geography_Spain", "Gender_Female", "Gender_Male"
        ]

        X = pd.DataFrame(predictor, columns=columns)
        models, names = load_models()
        
        for model in models:        
            if isinstance(model.estimator, CatBoostClassifier):
                cat = model.best_estimator_
                result = cat.predict(X)
                break

        if result[0] == 1:
            message = "입력된 고객은 이탈할 것으로 예측됩니다. :thumbsdown:"
            st.subheader(message)
        else:
            message = "입력된 고객은 이탈 위험이 낮습니다.  :thumbsup:"
            st.subheader(message)
