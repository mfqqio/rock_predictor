#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Allows for predictions from a trained model saved to file.
"""

import pandas as pd
import pickle
import numpy as np
import sys
from itertools import compress
from joblib import load
import argparse
import warnings

parser = argparse.ArgumentParser()
parser.add_argument("final_model_path")
args = parser.parse_args()

final_model_path = args.final_model_path

train_features_path = "data/pipeline/train_features.csv"
test_features_path = "data/pipeline/test_features.csv"
predict_features_path = "data/pipeline/predict_features.csv"
output_file_path = "data/output/predictions.csv"

print("Loading data...")
pred_feats = pd.read_csv(predict_features_path)
X = (pred_feats
    .drop(columns=["ActualX_mean", "ActualY_mean"])
    .select_dtypes(include=[np.number]))

print('Loading model...')

# Mute warning in terminal temporarily for demo
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    pipe = load(final_model_path)

print('\nPredicting rock classes...')
probs = pipe.predict_proba(X)

#Find the model within the pipeline. A model is the only object that has the method 'predict'.
for est in pipe.named_steps.values():
    if hasattr(est, "predict"):
        model = est

# Format output and get most likely label
y_pred = pd.DataFrame(probs, columns=[x for x in model.classes_])
y_pred['pred'] = y_pred.idxmax(axis=1)

print('Done! Saving output for visualization tool...')

# Attach features and useful information back to predictions
feat_pred = pd.concat([pred_feats, y_pred], sort=False, axis=1)

# Combine together train features and predict/test features as
# input for web app visualization
train_feats = pd.read_csv(train_features_path)
test_feats = pd.read_csv(test_features_path)
train_feats = pd.concat([train_feats, test_feats], axis=0, sort=False)
train_feats['data_type'] = 'train'
feat_pred['data_type'] = 'predict'
all_data = pd.concat([train_feats, feat_pred], axis=0, sort=False)
all_data.to_csv(output_file_path)
