#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:55:39 2019

@author: Carrie Cheung

This step trains an initial model using the provided training dataset.
"""

import pandas as pd
import numpy as np
import re
import sys
import pickle
from helpers.model import calc_overall_cost, print_model_eval
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_predict
from joblib import dump, load

# makefile command
# python build_model.py non_telem_features.csv rock_class model_results.txt

#### MAIN
# First check if command line arguments are provided before launching main script
if len(sys.argv) == 4:
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

    # Load explosive density data
    df_exp = pd.read_csv(powder_path)
    cost_dict = dict(zip(df_exp.rock_class, df_exp["kg/m3"]))

    # Simple random forest model to test evaluate function
    clf = RandomForestClassifier(n_estimators=430, random_state=0)
    y_pred = cross_val_predict(clf, X, y, cv=8)
    overall_cost = calc_overall_cost(y, y_pred, cost_dict)
    model_name = "Basic model"
    print_model_eval(y, y_pred, model_name, overall_cost)

    # Simple random forest model after grouping QZ & LIM
    y_grouped = pd.Series(np.where(np.logical_or(y == "LIM", y == "QZ"), "LIM", y), name="litho_rock_class")
    clf = RandomForestClassifier(n_estimators=430, random_state=0)
    y_pred = cross_val_predict(clf, X, y_grouped, cv=8)
    model_name = "Basic model after grouping QZ & LIM"
    overall_cost = calc_overall_cost(y_grouped, y_pred, cost_dict)
    print_model_eval(y_grouped, y_pred, model_name, overall_cost)
