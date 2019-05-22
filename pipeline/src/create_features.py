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
from telem_features import zero_water_flow, prop_max_pulldown, prop_half_pulldown
from stats_features import get_std, get_median, get_mean, get_percentile

#### MAIN
# First check if command line arguments are provided before launching main script
# python src/create_features.py data/raw/sample_wide.csv data/intermediate/features.csv
if len(sys.argv) == 3:
    data_path = sys.argv[1]
    output_file_path = sys.argv[2]

    # Read master joined data from file
    print('Reading joined input data...')
    df = pd.read_csv(data_path, low_memory=False)
    print('Master joined input data dimensions:', df.shape)
    
    # Column name mapping
    target_col='litho_rock_class'
    hole_id_col='hole_id'
    drilltime_col='DrillTime'
    operator_col='FirstName'
    hole_depth_col='depth'
    water_col = 'water'
    pulldown_col = 'pull'
    
    telem_features = ["hvib", "vvib", "pull",
    "air", "pos", "depth", "rot", "water"]
    
    # Creates non-telemetry features
    features = create_nontelem_features(df,
                                        target_col,
                                        hole_id_col,
                                        drilltime_col,
                                        operator_col,
                                        hole_depth_col)
    
    # Add proportion of time with zero water flow
    features['prop_nowater'] = zero_water_flow(df, hole_id_col, water_col)
    
    # Add proportion of time at maximum pulldown force
    features['prop_max_pulldown'] = prop_max_pulldown(df, hole_id_col, pulldown_col)
    
    # Add proportion of time at less than half of max pulldown force
    features['prop_half_pulldown'] = prop_half_pulldown(df, hole_id_col, pulldown_col)
    #print(features.groupby('rock_class').agg(['mean'])['prop_half_pulldown'])

    # Add stat summary features
    features = pd.concat([features, get_std(df, telem_features)], axis=1)
    features = pd.concat([features, get_median(df, telem_features)], axis=1)
    features = pd.concat([features, get_mean(df, telem_features)], axis=1)
    
    percentile_list = [0.1, 0.25, 0.75, 0.95]
    features = pd.concat([features, get_percentile(df, telem_features, percentile_list)], axis=1)

    # Drop any intermediate columns
    features = features.drop(['drill_operator'], axis=1)

    # Output calculated features to file
    features.to_csv(output_file_path, index=False)

    print('Features calculated and written to file:', output_file_path)
