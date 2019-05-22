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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

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
        
# For a given model, shows avg cross-validation score,
# confusion matrix and classification report of evaluation
# metrics including precision, recall, F1 score.
# Usage: evaluate(clf, 'random forest', X, y, cv_folds=8)
def evaluate(model, model_name, X, y, cv_folds):
    predictions = model.predict(X)
    print('Evaluating %s...' % model_name)
    print('Trained on dataset of size {0} with {1} features\n'.format(X.shape, len(list(X))))
    
    # Calculate and print cross validation score
    cv_scores = cross_val_score(model, X, y.values.ravel(), cv=cv_folds)
    mean_cv_score = np.mean(cv_scores)
    print("Cross-validation accuracy (%i-folds): %f\n" % (cv_folds, mean_cv_score))
    
    # Create confusion matrix
    rock_labels = list(clf.classes_)
    confus = confusion_matrix(y, predictions, labels=rock_labels)

    # Print confusion matrix with headers
    confus_ex = pd.DataFrame(confus, 
                   index=['true:'+x for x in rock_labels], 
                   columns=['pred:'+x for x in rock_labels])
    print('CONFUSION MATRIX\n', confus_ex)
    
    # Classification report
    report = classification_report(y, predictions, target_names=rock_labels)
    print('\nCLASSIFICATION REPORT\n', report)
    return 
        
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

    # With vibration
    vib_cols = ['Horizontal Vibration',
                 'Vertical Vibration']
    
    feature_cols = ['total_drill_time',
                 'penetration_rate_mph']
    
    telem_cols = ['prop_nowater', 'prop_max_pulldown', 'prop_half_pulldown']
    
    # Exclude columns as features
    to_exclude = [target_col, 'hole_id']
                 
    # Gets one-hot encoded drill operator column names and add to list of feature columns
    drillop_cols = [col for col in list(df) if re.search(r'operator[0-9]+', col)] 
    feature_cols = feature_cols + drillop_cols + telem_cols

    # Separate target and features
    #X = df.loc[:, feature_cols] # Features columns
    X = df[df.columns.difference(to_exclude)]
    y = df[[target_col]].astype(str) # Target column
    
    # Models to try
    models = {
        'random forest' : RandomForestClassifier(n_estimators=100),
        'gradient boost': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                 max_depth=2, random_state=0),
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
    k_folds = 8
    for model_name, model in models.items():
        print("Fitting %s model..." % model_name)
        cv_scores = cross_val_score(model, X, y.values.ravel(), cv=k_folds)
        mean_cv_score = np.mean(cv_scores)
        print("Cross-validation accuracy (%i-folds): %f\n" % (k_folds,mean_cv_score))
        
        results_dict['Classifier'].append(model_name)
        results_dict['CV Accuracy'].append(mean_cv_score)
    
    # Save results in a table to save as text file
    results_table = pd.DataFrame(results_dict)
    save_model_results(results_table, results_path)
    
    #save_model(rf, 'random_forest_model.pkl')
    
    
    
    
    
    

    

