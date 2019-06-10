import os
import pandas as pd
import numpy as np
import re
import sys
import argparse
from time import time

from helpers.model import ColumnSelector, custom_oversample, evaluate_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from joblib import dump, load
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn import FunctionSampler

parser = argparse.ArgumentParser()
parser.add_argument("pipeline_path")
parser.add_argument("train_path")
parser.add_argument("test_path")
parser.add_argument("oversampling_strategy")
args = parser.parse_args()

pipeline_path = args.pipeline_path
train_path = args.train_path
test_path = args.test_path
oversampling = args.oversampling_strategy

df_exp = pd.read_csv("data/input_mapping/explosive_by_rock_class.csv")
cost_dict = dict(zip(df_exp.rock_class, df_exp["kg/m3"]))

pipe = load(pipeline_path)
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

if oversampling == "SMOTE":
    # Separate target and features
    X_train = df_train.select_dtypes(include=[np.number])
    y_train = df_train.litho_rock_class
    X_test = df_test.select_dtypes(include=[np.number])
    y_test = df_test.litho_rock_class
    columns = X_train.columns

    #Oversample
    ros = SMOTE()
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
    X_train_res = pd.DataFrame(data=X_train_res, columns=columns)

    #Calculate and export accuracy
    tic = time()
    pipe.fit(X_train_res, y_train_res)
    y_pred = pipe.predict(X_test)
    toc = time()
    evaluate_model(y_test, y_pred, "Final model", (toc - tic), cost_dict)

    #Train model with entire dataset
    df = pd.concat([df_train, df_test], ignore_index=True, sort=False)
    X = df.select_dtypes(include=[np.number])
    y = df.litho_rock_class
    X_res, y_res = ros.fit_resample(X, y)
    X_res = pd.DataFrame(data=X_res, columns=columns)
    pipe.fit(X_res, y_res)
    dump(pipe, "models/fitted/final_model.joblib")
