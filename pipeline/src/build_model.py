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
from helpers.model import evaluate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
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
    print('Training data dimensions:', df.shape)

    # Filter out holes/rows which have no target label
    df = df.dropna(subset = ["litho_rock_class"])

    # Exclude columns from features
    cols_to_exclude = ["hole_id", "exp_rock_type", "exp_rock_class", "litho_rock_type", "litho_rock_class", 'ActualX_mean', 'ActualY_mean']

    # Separate target and features
    X = df.drop(columns=cols_to_exclude)
    y = df.litho_rock_class # Target column

    ###### PLACE MODEL(S) HERE
    # Simple random forest model to test evaluate function
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf.fit(X, y)
    # Saves model evaluation results to a text file
    with open(results_path, "w") as outfile:
        evaluate(clf, 'random forest', X, y, powder_path, outfile, cv_folds=8)

    # Simple random forest model after grouping QZ & LIM
    y_grouped = pd.Series(np.where(np.logical_or(y == "LIM", y == "QZ"), "LIMQZ", y), name="litho_rock_class")
    clf2 = RandomForestClassifier(n_estimators=430, random_state=0)
    clf2.fit(X, y_grouped)
    results_path2 = "doc/qzlim_grouped_model_results.txt"
    with open(results_path2, "w") as outfile:
        evaluate(clf2, 'random forest grouping qz and lim', X, y_grouped, powder_path, outfile, cv_folds=8)

    #dump(clf, 'random_forest_model.pkl')
