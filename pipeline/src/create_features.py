#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main driver for creating features. 

This script imports and calls functions containing logic that engineers 
features from input data. Output is a dataframe of features for each hole.
"""

import pandas as pd
import numpy as np
import sys
from nontelem_features import create_nontelem_features

#### MAIN
# First check if command line arguments are provided before launching main script
# python src/create_features.py data/raw/sample_wide.csv data/intermediate/features.csv
if len(sys.argv) == 3:
    data_path = sys.argv[1]
    output_file_path = sys.argv[2]

    # Read master joined data from file
    print('Reading input data...')
    df = pd.read_csv(data_path, low_memory=False)
    print('Master joined table dimensions:', df.shape)
    
    # Creates non-telemetry features
    features = create_nontelem_features(df,
                                              target_col='litho_rock_class',
                                              hole_id_col='redrill_id',
                                              drilltime_col='DrillTime',
                                              operator_col='FirstName',
                                              redrill_col='redrill',
                                              hole_depth_col='Hole Depth')
    
    
    # Add feature on proportion of time where water flow was zero
    #features['prop_nowater'] = zero_water_flow(df, hole_id_col, 'Water Flow')
    
    # Drop any intermediate columns
    features = features.drop(['drill_operator'], axis=1)
    
    # Output calculated features to file
    features.to_csv(output_file_path, index=False)

    print('Features calculated and written to file:', output_file_path)