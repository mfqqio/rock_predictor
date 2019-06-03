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

# Lists of summary metrics
names = []
accs = []
f1s = []
times = []
costs = []
lists = [names, accs, f1s, times, costs]

# Exploration model
y_pred = df.exp_rock_class
appenders = evaluate_model(y, y_pred,"Exploration model", 0, cost_dict)
[lst.append(x) for lst, x in zip(lists, appenders)]

# Simple random forest model to test evaluate function
clf = RandomForestClassifier(n_estimators=430, random_state=0)
tic = time()
y_pred = cross_val_predict(clf, X, y, cv=kfold)
toc = time()
appenders = evaluate_model(y, y_pred, "Basic model", (toc-tic), cost_dict)
[lst.append(x) for lst, x in zip(lists, appenders)]

# Simple random forest model after grouping QZ & LIM
y_grouped = pd.Series(np.where(np.logical_or(y == "LIM", y == "QZ"), "LIM", y), name="litho_rock_class")
clf = RandomForestClassifier(n_estimators=430, random_state=0)
tic = time()
y_pred = cross_val_predict(clf, X, y_grouped, cv=kfold)
toc = time()
appenders = evaluate_model(y_grouped, y_pred,
    "Basic model after grouping QZ & LIM", (toc-tic), cost_dict)
[lst.append(x) for lst, x in zip(lists, appenders)]

# Naive oversampled on simple random forest
ros = RandomOverSampler(random_state=123)
clf = RandomForestClassifier(n_estimators=430, random_state=0)
tic = time()
y_pred = cros_val_predict_oversample(clf, X, y, ros, cv=kfold)
toc = time()
appenders = evaluate_model(y, y_pred,
    "Naive oversampled on simple random forest", (toc-tic), cost_dict)
[lst.append(x) for lst, x in zip(lists, appenders)]

# SMOTE oversampling on simple random forest
ros = SMOTE(sampling_strategy='minority', random_state=123)
clf = RandomForestClassifier(n_estimators=430, random_state=0)
tic = time()
y_pred = cros_val_predict_oversample(clf, X, y, ros, cv=kfold)
toc = time()
appenders = evaluate_model(y, y_pred, "SMOTE oversampling on simple random forest", (toc-tic), cost_dict)
[lst.append(x) for lst, x in zip(lists, appenders)]

# Custom oversampling with 25 extra random samples on QZ
ros = FunctionSampler(func=custom_oversample,
                      kw_args={"random_state": 123,
                               "class_list": ["QZ"],
                               "num_samples": 25})
clf = RandomForestClassifier(n_estimators=430, random_state=0)
tic = time()
y_pred = cros_val_predict_oversample(clf, X, y, ros, cv=kfold)
toc = time()
appenders = evaluate_model(y, y_pred,
    "Custom oversampling with 25 extra random samples on QZ", (toc-tic), cost_dict)
[lst.append(x) for lst, x in zip(lists, appenders)]

# Custom oversampling with 50 extra random samples on QZ
ros = FunctionSampler(func=custom_oversample,
                      kw_args={"random_state": 123,
                               "class_list": ["QZ"],
                               "num_samples": 50})
clf = RandomForestClassifier(n_estimators=430, random_state=0)
tic = time()
y_pred = cros_val_predict_oversample(clf, X, y, ros, cv=kfold)
toc = time()
appenders = evaluate_model(y, y_pred,
    "Custom oversampling with 50 extra random samples on QZ", (toc-tic), cost_dict)
[lst.append(x) for lst, x in zip(lists, appenders)]

# Custom oversampling with 75 extra random samples on QZ
ros = FunctionSampler(func=custom_oversample,
                      kw_args={"random_state": 123,
                               "class_list": ["QZ"],
                               "num_samples": 75})
clf = RandomForestClassifier(n_estimators=430, random_state=0)
tic = time()
y_pred = cros_val_predict_oversample(clf, X, y, ros, cv=kfold)
toc = time()
appenders =  evaluate_model(y, y_pred,
    "Custom oversampling with 75 extra random samples on QZ", (toc-tic), cost_dict)
[lst.append(x) for lst, x in zip(lists, appenders)]

# Custom oversampling with 1 extra random samples on QZ
ros = FunctionSampler(func=custom_oversample,
                      kw_args={"random_state": 123,
                               "class_list": ["QZ"],
                               "num_samples": 1 })
clf = RandomForestClassifier(n_estimators=430, random_state=0)
tic = time()
y_pred = cros_val_predict_oversample(clf, X, y, ros, cv=kfold)
toc = time()
appenders =  evaluate_model(y, y_pred,
    "Custom oversampling with 1 extra random samples on QZ", (toc-tic), cost_dict)
[lst.append(x) for lst, x in zip(lists, appenders)]

# Simple random forest model after removing QZ
y_no_qz = y.loc[y != "QZ"]
X_no_qz = X.loc[y != "QZ"]
clf = RandomForestClassifier(n_estimators=430, random_state=0)
tic = time()
y_pred = cross_val_predict(clf, X_no_qz, y_no_qz, cv=kfold)
toc = time()
appenders = evaluate_model(y_no_qz, y_pred,
    "Basic model after removing QZ", (toc-tic), cost_dict)

df_summary = pd.DataFrame(data={
    "Model Description": names,
    "Accuracy": accs,
    "Macro F1": f1s,
    "Evaluation Time": times,
    "Absolute Explosive Diff": costs})

df_summary.to_csv("doc/model_eval.csv")
