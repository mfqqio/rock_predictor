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
    final = pd.DataFrame()
    for var in list_of_var: # loop through the list of features
        mask = (df.FieldDesc == var)
        df1 = df[mask]
        groups = df1.groupby('redrill_id') # group by redrill id
        std_results = pd.DataFrame(groups.agg('std')['FieldData']) # calc 'std' on 'FieldData'
        std_results.columns=[var]
        final = pd.concat([final, std_results], axis=1) # concatenate list of columns
    final['hole_id'] = final.index
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
