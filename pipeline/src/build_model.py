#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
import sys
import pickle
from helpers.model import calc_overall_cost, evaluate_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from joblib import dump, load
from time import time

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
kfold = StratifiedKFold(10, random_state=123)

# Load explosive density data
df_exp = pd.read_csv(powder_path)
cost_dict = dict(zip(df_exp.rock_class, df_exp["kg/m3"]))

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

df_summary = pd.DataFrame(data={
    "Model Description": [name_m1, name_m2],
    "Accuracy": [acc_m1, acc_m2],
    "Macro F1": [f1_m1, f1_m2],
    "Evaluation Time": [time_m1, time_m2],
    "Absoltute Explosive Diff": [cost_m1, cost_m2]
})
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_summary)
