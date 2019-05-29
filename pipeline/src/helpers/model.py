import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import warnings

def print_model_eval(y_true, y_pred, model_name, overall_cost):
    unique_values = np.unique(y_true)
    print("\n" + model_name + ":")
    confus = confusion_matrix(y_true, y_pred, labels=unique_values)
    #Correct headers
    confus_ex = pd.DataFrame(confus,
                   index=['true:'+x for x in unique_values],
                   columns=['pred:'+x for x in unique_values])
    print('CONFUSION MATRIX:')
    print(confus_ex)
    print('CLASSIFICATION REPORT:')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        print(classification_report(y_true, y_pred, target_names=unique_values))
    print("Sum of explosive misclassifications: %.3f kg/m3" % overall_cost)

def calc_overall_cost(y_true, y_pred, cost_dict):
    if len(y_pred) != len(y_true):
        raise ValueError("Predicted vector and true vector must have the same length (and order!)")
    #Get predicted and actual costs
    pred_cost = np.vectorize(cost_dict.get)(y_pred)
    true_cost = np.vectorize(cost_dict.get)(y_true)

    diff_vector = np.abs(pred_cost - true_cost)

    return diff_vector.sum()
