#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
import sys
import pickle
from helpers.model import calc_overall_cost, evaluate_model, cros_val_predict_oversample, custom_oversample
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from joblib import dump, load
from time import time
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn import FunctionSampler

# makefile command
# python build_model.py non_telem_features.csv rock_class model_results.txt

train_path = sys.argv[1]
powder_path = sys.argv[2]
results_path = sys.argv[3]

# Read train dataset from files
df = pd.read_csv(train_path, low_memory=False)
print('Training data dimensions:\n', df.shape)

# Filter out holes/rows which have no target label
df = df.dropna(subset = ["litho_rock_class"])

# Exclude columns from features
cols_to_exclude = ["hole_id", "exp_rock_type", "exp_rock_class", "litho_rock_type", "litho_rock_class", 'ActualX_mean', 'ActualY_mean']

# Separate target and features
X = df.drop(columns=cols_to_exclude)
y = df.litho_rock_class # Target column

# K-fold strategy
kfold = StratifiedKFold(10, shuffle=True, random_state=123)

# Load explosive density data
df_exp = pd.read_csv(powder_path)
cost_dict = dict(zip(df_exp.rock_class, df_exp["kg/m3"]))

# Exploration model
y_pred_m0 = df.exp_rock_class
name_m0 = "Exploration model"
acc_m0, f1_m0, time_m0, cost_m0 = evaluate_model(y,
                                                 y_pred_m0,
                                                 name_m0,
                                                 0, cost_dict)

# Simple random forest model to test evaluate function
clf = RandomForestClassifier(n_estimators=430, random_state=0)
tic = time()
y_pred_m1 = cross_val_predict(clf, X, y, cv=kfold)
toc = time()
name_m1 = "Basic model"
acc_m1, f1_m1, time_m1, cost_m1 = evaluate_model(y,
                                                 y_pred_m1,
                                                 name_m1,
                                                 (toc-tic), cost_dict)

# Simple random forest model after grouping QZ & LIM
y_grouped = pd.Series(np.where(np.logical_or(y == "LIM", y == "QZ"), "LIM", y), name="litho_rock_class")
clf = RandomForestClassifier(n_estimators=430, random_state=0)
tic = time()
y_pred_m2 = cross_val_predict(clf, X, y_grouped, cv=kfold)
toc = time()
name_m2 = "Basic model after grouping QZ & LIM"
acc_m2, f1_m2, time_m2, cost_m2 = evaluate_model(y_grouped,
                                                 y_pred_m2,
                                                 name_m2,
                                                 (toc-tic), cost_dict)

# Naive oversampled on simple random forest
ros = RandomOverSampler(sampling_strategy='minority', random_state=123)
clf = RandomForestClassifier(n_estimators=430, random_state=0)
tic = time()
y_pred_m3 = cros_val_predict_oversample(clf, X, y, ros, cv=kfold)
toc = time()
name_m3 = "Naive oversampled on simple random forest"
acc_m3, f1_m3, time_m3, cost_m3 = evaluate_model(y,
                                                 y_pred_m3,
                                                 name_m3,
                                                 (toc-tic), cost_dict)

# # SMOTE oversampling on simple random forest
# ros = SMOTE(sampling_strategy='minority', random_state=123)
# clf = RandomForestClassifier(n_estimators=430, random_state=0)
# tic = time()
# y_pred_m4 = cros_val_predict_oversample(clf, X, y, ros, cv=kfold)
# toc = time()
# name_m4 = "SMOTE oversampling on simple random forest"
# acc_m4, f1_m4, time_m4, cost_m4 = evaluate_model(y,
#                                                  y_pred_m4,
#                                                  name_m4,
#                                                  (toc-tic), cost_dict)
#
# # Custom oversampling with 25 extra random samples on QZ
# ros = FunctionSampler(func=custom_oversample,
#                       kw_args={"random_state": 123,
#                                "class_list": ["QZ"],
#                                "num_samples": 25})
# clf = RandomForestClassifier(n_estimators=430, random_state=0)
# tic = time()
# y_pred_m4 = cros_val_predict_oversample(clf, X, y, ros, cv=kfold)
# toc = time()
# name_m5 = "Custom oversampling with 25 extra random samples on QZ"
# acc_m5, f1_m5, time_m5, cost_m5 = evaluate_model(y,
#                                                  y_pred_m5,
#                                                  name_m5,
#                                                  (toc-tic), cost_dict)

# Simple random forest model after removing QZ
y_no_qz = y.loc[y != "QZ"]
X_no_qz = X.loc[y != "QZ"]
clf = RandomForestClassifier(n_estimators=430, random_state=0)
tic = time()
y_pred_m6 = cross_val_predict(clf, X_no_qz, y_no_qz, cv=kfold)
toc = time()
name_m6 = "Basic model after removing QZ"
acc_m6, f1_m6, time_m6, cost_m6 = evaluate_model(y_no_qz,
                                                 y_pred_m6,
                                                 name_m6,
                                                 (toc-tic), cost_dict)

df_summary = pd.DataFrame(data={
    "Model Description": [name_m0, name_m1, name_m2, name_m3, name_m6],
    "Accuracy": [acc_m0, acc_m1, acc_m2, acc_m3, acc_m6],
    "Macro F1": [f1_m0, f1_m1, f1_m2, f1_m3, f1_m6],
    "Evaluation Time": [time_m0, time_m1, time_m2, time_m3, time_m6],
    "Absolute Explosive Diff": [cost_m0, cost_m1, cost_m2, cost_m3, cost_m6]
})
df_summary.to_csv("doc/model_eval.csv")
