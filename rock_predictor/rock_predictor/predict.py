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

parser = argparse.ArgumentParser()
parser.add_argument("final_model_path")
args = parser.parse_args()

final_model_path = args.final_model_path

features_path = "data/pipeline/predict_features.csv"
output_file_path = "data/output/predictions.csv"

X = pd.read_csv(features_path)
print('Loading model...')
pipe = load(final_pipe_path)

print('\nPredicting rock classes...')
probs = pipe.predict_proba(X)

#Find the model within the pipeline. A model is the only object that has the method 'predict'.
for est in pipe.named_steps.values():
    if hasattr(est, "predict"):
        model = est

# Format output and get most likely label
y_pred = pd.DataFrame(probs, columns=[x for x in model.classes_])
y_pred['pred'] = y_pred.idxmax(axis=1)

print('Done!')

# Attach features and useful information back to predictions
feat_pred = pd.concat([feat, y_pred], sort=False, axis=1)

feat_pred.to_csv(output_file_path)
