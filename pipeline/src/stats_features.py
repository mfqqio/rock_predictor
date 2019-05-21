#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 20:29 2019

Calculates the standard deviation of a telemetry feature
"""

# import the libraries
import pandas as pd
import sys

# Function to calculate standard deviation of a list of variables/features
def get_std(df, list_of_var):
    col_names = []
    final = pd.DataFrame()
    for var in list_of_var: # loop through the list of features
        groups = df.groupby('redrill_id') # group by redrill id
        std_results = pd.DataFrame(groups.agg('std')[var]) # calc 'std' on 'FieldData'
        std_results.columns = [str(var) + '_std' for var in std_results.columns]
        col_names.append(str(var)+'_std')
        final = pd.concat([final, std_results], axis=1) # concatenate list of columns
    final = final.reset_index(drop=True)
    return final

def get_median(df, list_of_var):
    final = pd.DataFrame()
    col_names = []
    for var in list_of_var:
        groups = df.groupby('redrill_id')
        std_results = pd.DataFrame(groups.agg('median')[var])
        std_results.columns=[str(var) + '_med' for var in std_results.columns]
        col_names.append(str(var)+'_med')
        final = pd.concat([final, std_results], axis=1)
    final = final.reset_index(drop=True)
    return final

def get_mean(df, list_of_var):
    final = pd.DataFrame()
    col_names = []
    for var in list_of_var:
        groups = df.groupby('redrill_id')
        std_results = pd.DataFrame(groups.agg('mean')[var])
        std_results.columns=[str(var) + '_mean' for var in std_results.columns]
        col_names.append(str(var)+'_mean')
        final = pd.concat([final, std_results], axis=1)
    final = final.reset_index(drop=True)
    return final

def get_min_max(df, list_of_var):
    final = pd.DataFrame()
    col_names = []
    minmax = ['min', 'max']
    for var in list_of_var:
        for m in minmax:
            groups = df.groupby('redrill_id')
            std_results = pd.DataFrame(groups.agg(m)[var])
            std_results.columns=[str(var) + '_' + m for var in std_results.columns]
            col_names.append(str(var)+'_' + m)
            final = pd.concat([final, std_results], axis=1)
    final = final.reset_index(drop=True)
    return final

def get_percentile(df, list_of_var, list_of_perc):
    final = pd.DataFrame()
    col_names = []
    for var in list_of_var:
        for p in list_of_perc:
            groups = df.groupby('redrill_id')[var].quantile(p)
            std_results = pd.DataFrame(groups)
            std_results.columns=[str(var) + '_' + str(p) for var in std_results.columns]
            col_names.append(str(var)+'_' + str(p))
            final = pd.concat([final, std_results], axis=1)
    final = final.reset_index(drop=True)
    return final

#### MAIN
# First check if command line arguments are provided before launching main script
if len(sys.argv) == 3:
    df_path = sys.argv[1]
    output_file_path = sys.argv[2]

    df = pd.read_csv(df_path, low_memory=False)
    print('Master joined table dimensions:', df.shape)

    telem_feats = get_std(df, ['Horizontal Vibration', 'Vertical Vibration'])

    # Output calculated features to file
    telem_feats.to_csv(output_file_path, index=False)

    print('Telemetry features calculated and written to file')
