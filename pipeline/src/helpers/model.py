import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_predict
import warnings
from collections import Counter


def evaluate_model(y_true, y_pred, model_name, eval_time, cost_dict):
    unique_values = np.unique(y_true)
    print("\n" + model_name.upper() + ":")
    print("Evaluated in %.2f s" % eval_time)
    confus = confusion_matrix(y_true, y_pred, labels=unique_values)
    #Correct headers
    confus_ex = pd.DataFrame(confus,
                   index=['true:'+x for x in unique_values],
                   columns=['pred:'+x for x in unique_values])
    print('Confusion Matrix:')
    print(confus_ex)
    print('Classification Report:')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        f1 = f1_score(y_true, y_pred, average="macro")
        print(classification_report(y_true, y_pred, labels=unique_values))
    overall_cost = calc_overall_cost(y_true, y_pred, cost_dict)
    print("Sum of explosive misclassifications: %.3f kg/m3" % overall_cost)

    acc = accuracy_score(y_true, y_pred)


    return model_name, acc, f1, eval_time, overall_cost

def calc_overall_cost(y_true, y_pred, cost_dict):
    if len(y_pred) != len(y_true):
        raise ValueError("Predicted vector and true vector must have the same length (and order!)")
    #Get predicted and actual costs
    pred_cost = np.vectorize(cost_dict.get)(y_pred)
    true_cost = np.vectorize(cost_dict.get)(y_true)

    diff_vector = np.abs(pred_cost - true_cost)

    return diff_vector.sum()

def cros_val_predict_oversample(estimator, X, y, oversampler, cv):
    pred_array = np.ndarray(len(y), dtype=object)
    for train_index, test_index in cv.split(X, y):
        X_train = X.loc[train_index]
        X_test = X.loc[test_index]
        y_train = y.loc[train_index]
        y_test = y.loc[test_index]
        X_train_res, y_train_res = oversampler.fit_resample(X_train, y_train)

        #Train model in oversampled training set
        estimator.fit(X_train_res, y_train_res)
        pred_values = estimator.predict(X_test)
        pred_array[test_index] = pred_values

    return pred_array

def custom_oversample(X, y, class_list, num_samples, random_state=None):
    index_vec = np.arange(len(y))

    for c in class_list:
        resample_indeces = np.random.RandomState(seed=random_state).choice(index_vec[y == c],
                                            size=num_samples,
                                            replace=True)
        index_vec = np.append(index_vec, resample_indeces)

    return X[index_vec], y[index_vec]
