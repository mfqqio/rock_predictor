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

# Reads a trained model from a pickle file and returns the model object. 
# Also prints out to screen the model retuned.
def load_model(path):
    model = ''
    print('Loading model...')
    try:
        model = pickle.load(open(path, 'rb'))
        print(model)
    except:
        print('Model could not be read. Check if filepath is correct.')
    return model

    
# Returns an array of values predicted by model for each example
# from features data located at a specified filepath.
def predict(model, features_path):
    # Read in features
    feat = pd.read_csv(features_path)

    # Remove non-predictive variables and columns not one-hot encoded
    to_exclude = ['hole_id',
                  'exp_rock_type',
                  'exp_rock_class']
    
    # If dataset was upsampled, remove oversampled flag from predictors
    if 'oversample' in feat.columns:
        to_exclude.append('oversample')

    X = feat[feat.columns.difference(to_exclude)] # Feature columns
    
    # Check if all features as named from model fit are present
    diff_features = set(X) - set(model.feature_names)
    feat_bool = [feature in list(X) for feature in model.feature_names] 
    missing = list(compress(model.feature_names, ~np.array(feat_bool)))
    
    # Check for same number of features in model fit vs predict
    if model.n_features_ == X.shape[1]:        
        if not all(feat_bool): # We are missing features
            print('Feature columns are missing: {} \nAborted!'.format(missing))
            return
            
    # Mismatch in number of features 
    elif model.n_features_ < X.shape[1]: 
        if all(feat_bool):
            # We have all features plus extra
            print('Warning: There are more prediction features than the model was trained on!')
            new_features = set(X) - set(model.feature_names)
            print('New features:', new_features)
            print('Note that new features will not be used for prediction. To use them, re-train model including new features.')
        
            # Discard the extra features
            X = X[X.columns.difference(list(new_features))] 
        else: 
            # We are missing features
            print('Feature columns are missing: {} \nAborted!'.format(missing))
            return
            
    else: # We are missing features
        print('Feature columns are missing: {} \nAborted!'.format(missing))
        return
        
    # Continue with predictions
    # Sort features in same order as used when model was fit
    X = X.reindex(columns=model.feature_names)
    print('\nPredicting rock classes...')
    probs = model.predict_proba(X)
    
    # Format output and get most likely label
    y_pred = pd.DataFrame(probs, columns=[x for x in model.classes_])
    y_pred['pred'] = y_pred.idxmax(axis=1)
    
    print('Done!')
    
    # Attach features and useful information back to predictions
    joined = pd.concat([feat, y_pred], sort=False, axis=1)
    return joined


if len(sys.argv) == 4:
    model_path = sys.argv[1]
    features_path = sys.argv[2]
    output_file_path = sys.argv[3]
    
    model = load_model(model_path)
    preds = predict(model, features_path)
    
    # Write out predictions df to file
    preds.to_csv(output_file_path) # need to check if this object is empty or not before writing to file 