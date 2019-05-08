#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:55:39 2019

@author: Carrie Cheung

This step trains an initial model using the provided training dataset. 
"""

import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# makefile command
# python build_model.py train_features.csv target_col
# python build_model.py non_telem_features.csv rock_class

#### MAIN 
# First check if command line arguments are provided before launching main script
if len(sys.argv) == 3: 
    train_path = sys.argv[1]
    target_col = sys.argv[2]

    # Read train dataset from files
    df = pd.read_csv(train_path, low_memory=False)
    print('Training data dimensions:', df.shape)
    
    # Filter out holes/rows which have no target label
    df = df[pd.notnull(df['rock_class'])]

    # Filter out holes/rows where drilling time is less than 1 minute
    df = df[df['total_drill_time'] > 60]
    
    # Separate target and features
    exclude_cols = [target_col, 'hole_id'] # Non-feature columns to exclude
    
    X = df.loc[:, ~df.columns.isin(exclude_cols)]
    y = df[[target_col]].astype(str) # Enforce target label column as a consistent type (string)   
    
    # Find the max number of folds we can do for cross-validation
    #class_sizes = y.groupby('rock_class').size()
    #max_k = max(class_sizes)
    
    #### Try random forest
    print("Training random forest model...")
    rf = RandomForestClassifier(n_estimators = 100, random_state=2019)

    # Use cross validation to score model
    print("Assessing random forest model...")
    rf_scores = cross_val_score(rf, X, y.values.ravel(), cv=10) # For now, use k=10
    print("RANDOM FOREST Accuracy: %0.3f (+/- %0.2f)" % (rf_scores.mean(), rf_scores.std() * 2))
    
    #### Try multiclass logistic regression
    # TO DO
    
    
    
    
    
    

    

