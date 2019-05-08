#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:04:42 2019

@author: Carrie Cheung

Creates "higher-level" non-telemetry features
(e.g. drill operator, total drill time)
"""

import pandas as pd
import sys

# makefile command
# python create_nontelem_features.py ../eda_viz/join_master.csv non_telem_features.csv

# Get non-telemetry features for each hole, 
# specifically total drill time, drill operator, redrill flag.
def create_nontelem_features(data, target_col, hole_id_col, drilltime_col, operator_col, redrill_col, hole_depth_col):
    # Group by hole IDs
    hole_grps = data.groupby(hole_id_col)
    
    # Rename columns to meaningful feature names
    feat_names = {drilltime_col: 'total_drill_time', 
                  operator_col:'drill_operator',
                  redrill_col: 'redrill_flag',
                  target_col: 'rock_class'} # Adds target labels
    
    # Calculate appropriate stat for each feature
    features = hole_grps.agg({drilltime_col: 'mean', 
                              redrill_col:'max', 
                              operator_col: 'first',
                              target_col: 'first'}).rename(columns=feat_names)
    
    # Add hole ID as a column in output dataframe
    features['hole_id'] = features.index
    features = features.reset_index(drop=True)
    
    # Map drill operator names to numerical values
    features['drill_operator'] = pd.factorize(features['drill_operator'])[0]
    
    # Make rock_class column a consistent type as string
    features['rock_class'] = features['rock_class'].astype(str)
    
    # Add penetration rate as a feature
    depth_telem = df[df.FieldDesc == 'Hole Depth'] # Subset to hole depth data only
    features = calc_penetration_rate(depth_telem, features, hole_id_col)
    
    return features

# Calculates penetration rate for each hole (metres per hour) 
# and returns feature dataframe with this features included
def calc_penetration_rate(depth_telem, feature_df, hole_id_col):
    # Get min & max of hole depth in time series
    df = depth_telem.groupby(hole_id_col).agg(['min', 'max'])['FieldData'] 
    
    # Calculate actual depth of hole drilled
    df['actual_hole_depth'] = df['max'] - df['min'] 
    df['hole_id'] = df.index
    df = df.reset_index(drop=True)
    
    # Join in features/hole data
    df = pd.merge(df, feature_df,  
                    how='left', 
                    left_on=['hole_id'], 
                    right_on = ['hole_id'])
    
    # Calculate penetration rate
    df['penetration_rate_mph'] = df['actual_hole_depth']/(df['total_drill_time']/3600)
    
    # Filter out unnecessary columns
    exclude_cols = ['min', 'max', 'actual_hole_depth']
    return df.loc[:, ~df.columns.isin(exclude_cols)]
    
#### MAIN 
# First check if command line arguments are provided before launching main script
if len(sys.argv) == 3: 
    data_path = sys.argv[1]
    output_file_path = sys.argv[2]
    
    # Read master joined data from file
    print('Reading input data...')
    df = pd.read_csv(data_path, low_memory=False)
    print('Master joined table dimensions:', df.shape)
    
    nontelem_feats = create_nontelem_features(df, 
                                              target_col='rock_type_mfq',
                                              hole_id_col='Unique_HoleID_y', 
                                              drilltime_col='DrillTime', 
                                              operator_col='FirstName', 
                                              redrill_col='Redrill', 
                                              hole_depth_col='Hole Depth')
    
    # Output calculated features to file
    nontelem_feats.to_csv(output_file_path, index=False)
    
    print('Non-telemetry features calculated and written to file:', output_file_path)
    
    

