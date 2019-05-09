#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:55:39 2019

@author: Carrie Cheung

This step trains an initial model using the provided training dataset. 
"""

import pandas as pd
import numpy as np
import sys
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# makefile command
# python build_model.py non_telem_features.csv rock_class model_results.txt

# Saves a trained model as a pickle file
def save_model(model, output_filename):
    with open(output_filename, 'wb') as file:
        pickle.dump(model, file)

# Saves modelling rersults to a text file
def save_model_results(results, output_path):
    with open(output_path, "w") as text_file:
        print(results, file=text_file)
        
#### MAIN 
# First check if command line arguments are provided before launching main script
if len(sys.argv) == 4: 
    train_path = sys.argv[1]
    target_col = sys.argv[2]
    results_path = sys.argv[3]

    # Read train dataset from files
    df = pd.read_csv(train_path, low_memory=False)
    print('Training data dimensions:', df.shape)
    
    # Filter out holes/rows which have no target label
    df = df[pd.notnull(df[target_col])]

    # Filter out holes/rows where drilling time is less than 1 minute
    df = df[df['total_drill_time'] > 60]
    
    target_col = 'rock_class'

    feature_cols = ['total_drill_time',
                 'redrill_flag',
                 'drill_operator',
                 'penetration_rate_mph']
    
    # Separate target and features
    X = df.loc[:, feature_cols] # Features columns
    y = df[[target_col]].astype(str) # Target column
    
    # Models to try
    models = {
        'random forest' : RandomForestClassifier(n_estimators=10),
        'logistic regression': LogisticRegression(solver='lbfgs', 
                                                  multi_class='multinomial',
                                                  max_iter=2000) # For now, this uses L2 regularization
    }

    # Save results of models
    results_dict = {
                    'Classifier':[],
                    'CV Accuracy':[]
                   }
    
    # Assess each model with cross-validation
    for model_name, model in models.items():
        print("Fitting %s model..." % model_name)
        cv_scores = cross_val_score(model, X, y.values.ravel(), cv=4)
        mean_cv_score = np.mean(cv_scores)
        print("Cross-validation accuracy: %f\n" % mean_cv_score)
        
        results_dict['Classifier'].append(model_name)
        results_dict['CV Accuracy'].append(mean_cv_score)
    
    # Save results in a table to save as text file
    results_table = pd.DataFrame(results_dict)
    save_model_results(results_table, results_path)
    
    #save_model(rf, 'random_forest_model.pkl')
    
    
    
    
    
    

    

