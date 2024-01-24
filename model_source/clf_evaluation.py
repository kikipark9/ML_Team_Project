import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import Binarizer
import pandas as pd

def get_clf_eval(y_test, pred, pred_proba=None):
    """
    분류 모델의 평가 지표를 계산합니다.

    매개변수:
    y_test (array-like): 테스트 세트의 실제 레이블.
    pred (array-like): 테스트 세트의 예측 레이블.
    pred_proba (array-like, optional): 테스트 세트의 예측 확률. 기본값은 None입니다.

    반환값:
    accuracy, precision, recall, f1, confusion, roc_score: 각 평가 지표의 값들.
    """

    # 이진 분류와 다중 클래스 분류를 구분합니다.
    class_count = len(np.unique(y_test))

    # 이진 분류인 경우와 다중 클래스 분류인 경우를 나누어 평가 지표를 계산합니다.
    if class_count == 2:
        average_type = 'binary'
    else:
        average_type = 'macro'

    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average=average_type)
    recall = recall_score(y_test, pred, average=average_type)
    f1 = f1_score(y_test, pred, average=average_type)
    
    roc_score = None
    if pred_proba is not None:
        if class_count == 2:
            roc_score = roc_auc_score(y_test, pred_proba)
        else:
            roc_score = roc_auc_score(y_test, pred_proba, average=average_type, multi_class='ovr')
    
    return accuracy, precision, recall, f1, confusion, roc_score


def print_eval(y_test, pred, pred_proba=None):
    """
    분류 모델의 평가 지표를 출력합니다.

    매개변수:
    y_test (array-like): 테스트 세트의 실제 레이블.
    pred (array-like): 테스트 세트의 예측 레이블.
    pred_proba (array-like, optional): 테스트 세트의 예측 확률. 기본값은 None입니다.
    """
    accuracy, precision, recall, f1, confusion, roc_score = get_clf_eval(y_test, pred, pred_proba)

    print('Confusion Matrix')
    print(confusion)
    print('Accuracy:', round(accuracy, 4), 'Precision:', round(precision, 4), 
          'Recall:', round(recall, 4))
    
    if roc_score is not None:
        print('F1:', round(f1, 4), 'ROC AUC:', round(roc_score, 4))
    else:
        print('F1:', round(f1, 4))


def get_eval_by_threshold(y_test, pred_proba, thresholds):
    """
    서로 다른 임계값에 따른 평가 지표를 계산합니다.
    
    매개변수:
    y_test (array-like): 테스트 세트의 실제 레이블.
    pred_proba (array-like): 긍정 클래스에 대한 예측 확률.
    thresholds (array-like): 평가할 임계값들.

    반환값:
    custom_results_df (DataFrame): 각 임계값별 평가 지표 (정확도, 정밀도, 재현율, F1, ROC AUC)를 포함하는 DataFrame.
    """

    thresholds_lst = []
    results = []
    
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba)
        custom_predict = binarizer.transform(pred_proba)
        accuracy, precision, recall, f1, confusion, roc_score = get_clf_eval(y_test, custom_predict, pred_proba)
        results.append([accuracy, precision, recall, f1, roc_score])
        thresholds_lst.append(custom_threshold)
    
    custom_results_df = pd.DataFrame(results, columns=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC'], index=thresholds_lst)
    return custom_results_df


def roc_curve_plot(y_test, pred_proba_c1):
    """
    예측 확률과 실제 레이블을 바탕으로 ROC 곡선을 그립니다.

    매개변수:
    y_test (array-like): 이진 분류 문제의 실제 레이블.
    pred_proba_c1 (array-like): 양성 클래스의 예측 확률.

    반환값:
    없음

    이 함수는 roc_curve 함수를 사용하여 거짓 양성률(FPRs), 진짜 양성률(TPRs) 및 임계값을 계산하고,
    roc_auc_score 함수를 사용하여 ROC 점수를 계산합니다. 그런 다음 Matplotlib 라이브러리의 plot 함수를 사용하여
    그래프에 FPRs, TPRs 및 임계값을 표시합니다. 무작위 분류자를 나타내는 선도 추가됩니다.
    x축은 FPR(1 - 민감도)을 나타내고 y축은 TPR(재현율)을 나타냅니다.
    그래프는 Matplotlib 라이브러리의 show 함수를 사용하여 표시됩니다.
    ROC 점수는 소수점 네 자리로 반올림하여 출력됩니다.
    """
    fprs, tprs, thresholds = roc_curve(y_test, pred_proba_c1)
    roc_score = roc_auc_score(y_test, pred_proba_c1)
    plt.plot(fprs, tprs, label='ROC')
    plt.plot([0, 1], [0, 1], 'k-', label='Random')

    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xticks(np.round(np.arange(0.1, 1.0, 0.1), 2))
    plt.xlabel('FPR(1 - Sensitivity)')
    plt.ylabel('TPR(Recall)')
    plt.legend()
    plt.show()

    print('ROC SCORE : ', round(roc_score, 4))


def precision_recall_curve_plot(y_test, pred_proba_c1):
    """
    클래스 1의 예측 확률과 실제 레이블을 기반으로 정밀도-재현율 곡선을 그립니다.

    매개변수:
    y_test (array-like): 실제 레이블.
    pred_proba_c1 (array-like): 클래스 1의 예측 확률.

    반환값:
    없음
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)
    
    plt.figure(figsize=(8, 6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[:threshold_boundary], ls='--', label='precision')
    plt.plot(thresholds, recalls[:threshold_boundary], ls='-', label='recall')
    
    plt.xlim((0.1, 0.9))
    plt.xticks(np.round(np.arange(0.1, 0.9, 0.1), 2))

    plt.xlabel('Threshold value')
    plt.ylabel('Precision and Recall value')
    plt.legend()
    plt.show()
