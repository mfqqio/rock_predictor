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
import argparse
from helpers.feature_eng import calc_penetration_rate, calc_prop_zero, calc_prop_max, calc_prop_half, count_oscillations, class_distance

parser = argparse.ArgumentParser()

parser.add_argument("mode") # Mode can be "for_train" or "for_predict"
args = parser.parse_args()

mode = args.mode

if mode == "for_train":
    paths = [
        ("data/pipeline/train.csv", "data/pipeline/train_features.csv"),
        ("data/pipeline/test.csv", "data/pipeline/test_features.csv")
    ]
elif mode == "for_predict":
    paths = [
        ("data/pipeline/predict_data.csv", "data/pipeline/predict_features.csv")
    ]

for input_path, output_path in paths:

    print('Reading clean input data...')
    df = pd.read_csv(input_path, low_memory=False)
    print('Clean input data dimensions:', df.shape)


    #Create feature for second derivative of hole_depth progression
    df["pos_lagOfLag"] = df.pos_lag1_diff.diff().fillna(0)

    # For cases when exp_rock_type is "AIR" or "OB"
    df.exp_rock_class.fillna(value="Unknown", inplace=True)

    # Handle grouping for creating features for training (contains litho columns)
    # and for predict (does not contain litho columns/are blank)
    groupby_cols = []
    if mode == 'for_train':
        groupby_cols = ["hole_id", "exp_rock_type", "exp_rock_class", "litho_rock_type", "litho_rock_class"]
    elif mode == 'for_predict':
        groupby_cols = ["hole_id", "exp_rock_type", "exp_rock_class"]

    # Creating major dataframe with summarizing metrics for every hole
    features = (df.groupby(groupby_cols)
    .agg({"pos_lagOfLag": "median",
        "pos_lag1_diff": "median",
        "utc_field_timestamp": "count",
        "ActualX": "mean",
        "ActualY": "mean",
        "hvib": ["std", "max", "min", "sum", "median",
                ("10th_quant", lambda x: x.quantile(0.1)),
                ("25th_quant", lambda x: x.quantile(0.25)),
                ("75th_quant", lambda x: x.quantile(0.75)),
                ("90th_quant", lambda x: x.quantile(0.9)),
                ("num_oscillations", count_oscillations),
                ],
        "vvib": ["std", "max", "min", "sum", "median",
                ("10th_quant", lambda x: x.quantile(0.1)),
                ("25th_quant", lambda x: x.quantile(0.25)),
                ("75th_quant", lambda x: x.quantile(0.75)),
                ("90th_quant", lambda x: x.quantile(0.9)),
                ("num_oscillations", count_oscillations),
                ],
        "pull": ["std", "max", "min", "sum", "median",
                ("10th_quant", lambda x: x.quantile(0.1)),
                ("25th_quant", lambda x: x.quantile(0.25)),
                ("75th_quant", lambda x: x.quantile(0.75)),
                ("90th_quant", lambda x: x.quantile(0.9)),
                ("prop_max", calc_prop_max),
                ("prop_half", calc_prop_half),
                ("num_oscillations",count_oscillations)
                ],
        "air": ["std", "max", "min", "sum", "median",
                ("10th_quant", lambda x: x.quantile(0.1)),
                ("25th_quant", lambda x: x.quantile(0.25)),
                ("75th_quant", lambda x: x.quantile(0.75)),
                ("90th_quant", lambda x: x.quantile(0.9)),
                ("num_oscillations", count_oscillations)
                ],
        "pos": ["std", "max", "min", "sum", "median",
                ("10th_quant", lambda x: x.quantile(0.1)),
                ("25th_quant", lambda x: x.quantile(0.25)),
                ("75th_quant", lambda x: x.quantile(0.75)),
                ("90th_quant", lambda x: x.quantile(0.9)),
                ],
        "depth": ["std", "max", "min", "sum", "median",
                ("10th_quant", lambda x: x.quantile(0.1)),
                ("25th_quant", lambda x: x.quantile(0.25)),
                ("75th_quant", lambda x: x.quantile(0.75)),
                ("90th_quant", lambda x: x.quantile(0.9)),
                ],
        "rot": ["std", "max", "min", "sum", "median",
                ("10th_quant", lambda x: x.quantile(0.1)),
                ("25th_quant", lambda x: x.quantile(0.25)),
                ("75th_quant", lambda x: x.quantile(0.75)),
                ("90th_quant", lambda x: x.quantile(0.9)),
                ("num_oscillations", count_oscillations)
                ],
        "water": ["std", "max", "min", "sum", "median",
                ("10th_quant", lambda x: x.quantile(0.1)),
                ("25th_quant", lambda x: x.quantile(0.25)),
                ("75th_quant", lambda x: x.quantile(0.75)),
                ("90th_quant", lambda x: x.quantile(0.9)),
                ("prop_zero", calc_prop_zero)
                ],
        })
        .reset_index()
    )
    features.columns = ['_'.join(col).strip() if col[1] != "" else col[0] for col in features.columns.values]
    features.rename(columns={"utc_field_timestamp_count": "time_count"},
                    inplace=True)
    features["penetration_rate"] = calc_penetration_rate(features.depth_max,
                                                         features.depth_min,
                                                         features.time_count)

    #Add one hot encoding for Exploration Rock Type
    features["exp_rock_type_onehot"] = features.exp_rock_type
    features = pd.get_dummies(data=features, columns=["exp_rock_type_onehot"])

    # Add dist features
    unique_labels = features.exp_rock_class.unique()
    for label in unique_labels:
        features["dist_"+ label] = class_distance(features.ActualX_mean,
                                                 features.ActualY_mean,
                                                 features.exp_rock_class,
                                                 label)

    # Output calculated features to file
    features.to_csv(output_path, index=False)

    print('Features calculated and written to file:', output_path)
