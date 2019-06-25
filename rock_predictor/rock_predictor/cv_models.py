#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import re
import sys
import pickle
import glob
from helpers.model import calc_overall_cost, evaluate_model, ColumnSelector, cros_val_predict_oversample, custom_oversample
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from joblib import dump, load
from time import time
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn import FunctionSampler

# makefile command
# python build_model.py non_telem_features.csv rock_class model_results.txt

train_path = sys.argv[1]
powder_path = sys.argv[2]
results_path = sys.argv[3]

# Read train dataset from files
df = pd.read_csv(train_path, low_memory=False)
df.drop(columns=["ActualX_mean", "ActualY_mean"], inplace=True)
print('Training data dimensions:\n', df.shape)

# Assert data integrity
assert df.litho_rock_class.isna().any() == False

# Separate target and features
X = df.select_dtypes(include=[np.number])
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

# All pipelines
models_path = os.path.join(os.getcwd(), "models/unfitted/*.joblib")

# Regular models
for f in glob.glob(models_path):
    pipe = load(f)
    tic = time()
    y_pred = cross_val_predict(pipe, X, y, cv=kfold)
    toc = time()
    appenders = evaluate_model(y, y_pred,
        pipe.description + " - Regular model", (toc-tic), cost_dict)#, export_dir="data/pipeline/cv_cross_val_pred")
    [lst.append(x) for lst, x in zip(lists, appenders)]

# Grouping QZ & LIM
# y_grouped = pd.Series(np.where(np.logical_or(y == "LIM", y == "QZ"), "LIM", y), name="litho_rock_class")
# for f in glob.glob(models_path):
#     pipe = load(f)
#     tic = time()
#     y_pred = cross_val_predict(pipe, X, y_grouped, cv=kfold)
#     toc = time()
#     appenders = evaluate_model(y_grouped, y_pred,
#         pipe.description + " - Grouping QZ & LIM", (toc-tic), cost_dict, export_dir="data/pipeline/cv_cross_val_pred")
#     [lst.append(x) for lst, x in zip(lists, appenders)]

# Naive Oversampling models
ros = RandomOverSampler(random_state=123)
for f in glob.glob(models_path):
    pipe = load(f)
    tic = time()
    y_pred = cros_val_predict_oversample(pipe, X, y, ros, cv=kfold)
    toc = time()
    appenders = evaluate_model(y, y_pred,
        pipe.description + " - Naive Oversampling", (toc-tic), cost_dict)#, export_dir="data/pipeline/cv_cross_val_pred")
    [lst.append(x) for lst, x in zip(lists, appenders)]

# SMOTE Oversampling models
ros = SMOTE(random_state=123)
for f in glob.glob(models_path):
    pipe = load(f)
    tic = time()
    y_pred = cros_val_predict_oversample(pipe, X, y, ros, cv=kfold)
    toc = time()
    appenders = evaluate_model(y, y_pred,
        pipe.description + " - SMOTE Oversampling", (toc-tic), cost_dict)#, export_dir="data/pipeline/cv_cross_val_pred")
    [lst.append(x) for lst, x in zip(lists, appenders)]

# # Custom oversampling with 25 extra random samples on QZ
# ros = FunctionSampler(func=custom_oversample,
#                       kw_args={"random_state": 123,
#                                "class_list": ["QZ"],
#                                "num_samples": 25})
# for f in glob.glob(models_path):
#     pipe = load(f)
#     tic = time()
#     y_pred = cros_val_predict_oversample(pipe, X, y, ros, cv=kfold)
#     toc = time()
#     appenders = evaluate_model(y, y_pred,
#         pipe.description + " - Custom Oversampling - 25", (toc-tic), cost_dict)#, export_dir="data/pipeline/cv_cross_val_pred")
#     [lst.append(x) for lst, x in zip(lists, appenders)]
#
# # Custom oversampling with 50 extra random samples on QZ
# ros = FunctionSampler(func=custom_oversample,
#                       kw_args={"random_state": 123,
#                                "class_list": ["QZ"],
#                                "num_samples": 50})
# for f in glob.glob(models_path):
#     pipe = load(f)
#     tic = time()
#     y_pred = cros_val_predict_oversample(pipe, X, y, ros, cv=kfold)
#     toc = time()
#     appenders = evaluate_model(y, y_pred,
#         pipe.description + " - Custom Oversampling - 50", (toc-tic), cost_dict)#, export_dir="data/pipeline/cv_cross_val_pred")
#     [lst.append(x) for lst, x in zip(lists, appenders)]
#
# # Custom oversampling with 75 extra random samples on QZ
# ros = FunctionSampler(func=custom_oversample,
#                       kw_args={"random_state": 123,
#                                "class_list": ["QZ"],
#                                "num_samples": 75})
# for f in glob.glob(models_path):
#     pipe = load(f)
#     tic = time()
#     y_pred = cros_val_predict_oversample(pipe, X, y, ros, cv=kfold)
#     toc = time()
#     appenders = evaluate_model(y, y_pred,
#         pipe.description + " - Custom Oversampling - 75", (toc-tic), cost_dict)#, export_dir="data/pipeline/cv_cross_val_pred")
#     [lst.append(x) for lst, x in zip(lists, appenders)]

df_summary = pd.DataFrame(data={
    "Model Description": names,
    "Accuracy": accs,
    "Macro F1": f1s,
    "Evaluation Time": times,
    "Absolute Explosive Diff": costs})

df_summary.to_csv("doc/eval_models.csv")
