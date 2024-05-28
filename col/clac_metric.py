import numpy as np
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def handle_invalid_values(data):
    mask = np.isnan(data) | np.isinf(data)
    data[mask] = 0.0
    return data

def get_metrics_1(real_score, predict_score):

    fpr, tpr, thersholds = roc_curve(real_score.reshape(-1), predict_score.reshape(-1), pos_label=1)
    precision, recall, thresholds = precision_recall_curve(real_score.reshape(-1), predict_score.reshape(-1), pos_label=1)

    roc_auc = auc(fpr, tpr)
    print('auc',roc_auc)
    aupr = auc(recall, precision)
    print('aupr',aupr)
    

    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_score = handle_invalid_values(f1_scores)

    max_f1_index = np.argmax(f1_score)
    max_f1 = f1_score[max_f1_index]
    optimal_precision = precision[max_f1_index]
    optimal_recall = recall[max_f1_index]
    
    print('precision',optimal_precision)
    print('recall',optimal_recall)
    print('f1_score',max_f1)
    
    return [roc_auc, aupr, optimal_precision, optimal_recall, max_f1]


def cv_model_evaluate(interaction_matrix, predict_matrix, testset,k):

    predict_score = predict_matrix[:, testset]
    real_score = interaction_matrix[:, testset]


    return get_metrics_1(real_score, predict_score)

