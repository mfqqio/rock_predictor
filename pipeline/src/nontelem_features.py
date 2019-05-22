#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates "higher-level" non-telemetry features for each hole
(e.g. drill operator, total drill time)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Get non-telemetry features for each hole,
# (e.g. total drill time, drill operator)
def create_nontelem_features(data, target_col, hole_id_col, drilltime_col, operator_col, hole_depth_col):
    # Group by hole IDs
    hole_grps = data.groupby(hole_id_col)

    # Rename columns to meaningful feature names
    feat_names = {drilltime_col: 'total_drill_time',
                  operator_col:'drill_operator',
                  target_col: 'rock_class'} # Adds target labels

    # Calculate appropriate stat for each feature
    features = hole_grps.agg({drilltime_col: 'mean',
                              operator_col: 'first',
                              target_col: 'first'}).rename(columns=feat_names)

    # Add hole ID as a column in output dataframe
    features['hole_id'] = features.index
    features = features.reset_index(drop=True)

    # One-hot encode drill operator names 
    features = encode_operator('drill_operator', data, features)

    # Make rock_class column a consistent type as string
    features['rock_class'] = features['rock_class'].astype(str)

    # Add penetration rate as a feature
    depth_telem = data[[hole_depth_col, hole_id_col, 'timestamp']] # Subset to hole depth data only
    features = calc_penetration_rate(depth_telem, features, hole_id_col, hole_depth_col)

    return features

# Performs one-hot encoding on specified drill operator column and
# returns original dataframe with one-hot encoded columns added
def encode_operator(op_col, df, features):
    # First label encode drill operators
    values = np.array(features[op_col])
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    # Then one-hot encode
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    
    # Create column headings for each operator
    num_ops = len(features[op_col].unique()) # Number of different drill operators
    col_ops = ['operator'+str(n) for n in range(1, num_ops+1)] # Columns named Operator1, Operator2, etc.
    onehot_op = pd.DataFrame(onehot_encoded, columns=col_ops)
    
    # Add one-hot encoded operator columns to features dataframe
    features = pd.concat([features.reset_index(drop=True), onehot_op.reset_index(drop=True)], axis=1)
    return features

# Calculates penetration rate for each hole (metres per hour)
# and returns feature dataframe with this features included
def calc_penetration_rate(depth_telem, feature_df, hole_id_col, hole_depth_col):
    # Get min & max of hole depth in time series
    df = depth_telem.groupby(hole_id_col).agg(['min', 'max'])[hole_depth_col]

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
