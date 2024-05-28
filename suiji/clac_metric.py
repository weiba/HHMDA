import numpy as np
import os
# -*- coding: gbk -*-
# -*- coding:utf-8 -*-

from sklearn.metrics import roc_curve, auc, precision_recall_curve


def get_metrics(real_score, predict_score):
    


    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score) #sorted_predict_score_num 3072

 
    
    thresholds = sorted_predict_score[np.int32(sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]
    

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]

    recall = recall_list[max_index]
    precision = precision_list[max_index]

    return [auc[0, 0], aupr[0, 0], precision, recall, f1_score]

def cv_model_evaluate(interaction_matrix, predict_matrix, neg_index, pos_index,k):

    test_pos_index = pos_index
    test_neg_index = neg_index

    test_index = test_pos_index + test_neg_index

    
    
    predict_score = predict_matrix[tuple(np.array(test_index).T)] #3076
    real_score = interaction_matrix[tuple(np.array(test_index).T)] #3076

    return get_metrics(real_score, predict_score)
