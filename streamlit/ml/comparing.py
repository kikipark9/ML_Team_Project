# -*- coding:utf-8 -*-

import streamlit as st
import pandas as pd
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go
from ml.clf_evaluation import *
from utils import do_preprocess, load_models
from xgboost import XGBClassifier


def get_result(model, X_test, y_test):
    model_proba = model.predict_proba(X_test)
    model_pred = model.predict(X_test)
    return get_clf_eval(y_test, model_pred, model_proba[:, 1])


def get_result_pd(models, model_names, X_test, y_test):
    col_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    result_pd = pd.DataFrame(columns=col_names, index=model_names)
    for model, model_name in zip(models, model_names):
        accuracy, precision, recall, f1, confusion, roc_score = get_result(model, X_test, y_test)
        result_pd.loc[model_name] = [accuracy, precision, recall, f1, roc_score]
    return result_pd


def plot_roc_curve(models, names, X_test, y_test):
    traces = []
    for model, name in zip(models, names):
        pred_proba = model.predict_proba(X_test)[:, 1]
        fprs, tprs, thresholds = roc_curve(y_test, pred_proba)
        roc_auc = auc(fprs, tprs)
        traces.append(go.Scatter(x=fprs, y=tprs, name=f'{name} AUC = {round(roc_auc, 4)}', mode='lines'))
    
    traces.append(go.Scatter(x=[0, 1], y=[0, 1], name='Random', mode='lines'))
    layout = go.Layout(xaxis=dict(title='FPR'), yaxis=dict(title='TPR'), margin=dict(l=50, r=50, t=50, b=50))
    fig = go.Figure(data=traces, layout=layout)
    st.plotly_chart(fig, use_container_width=True)


def get_feature_importance(importance, X_train):
    fi = pd.DataFrame(list(zip(X_train.columns, importance)), 
                        columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)
    return fi


def run_comparing(df):
    X_test, y_test, customers = do_preprocess(df)

    models, names = load_models()
    result = get_result_pd(models, names, X_test, y_test)

    st.markdown('''
    #### 모델의 성능비교
    ''')
    
    st.data_editor(data=result, use_container_width=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('''
    #### ROC Curve 비교
    ''')
    plot_roc_curve(models, names, X_test, y_test)
    
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('''
    #### 특성 중요도(XGBoost)
    ''')
    for model in models:        
        if isinstance(model.estimator, XGBClassifier):
            xgb = model.best_estimator_
        else:
            pass
    
    importance = xgb.feature_importances_
    fi = get_feature_importance(importance, X_test)
    fi_sorted = fi.sort_values(by='Importance', ascending=False).iloc[::-1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=fi_sorted['Importance'],
        y=fi_sorted['Feature'],
        orientation='h'
    ))
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(fi)
    with col2:
        st.plotly_chart(fig, use_container_width=True)
    
    