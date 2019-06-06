#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates features based on telemetry data for each hole.
"""

import pandas as pd
import numpy as np
from scipy.spatial import distance

# Calculates proportion of time in time series
# where water flow is zero. Returns a series of values.

def calc_prop_zero(num_vector):
    vec_is_zero = np.isclose(num_vector, 0)
    prop_zero = vec_is_zero.sum() / len(vec_is_zero)
    return prop_zero

def calc_prop_max(num_vector):
    max = num_vector.max()
    vec_close_max = np.isclose(num_vector, max, rtol=0.05)
    prop_max = vec_close_max.sum() / len(vec_close_max)
    return prop_max

def calc_prop_half(num_vector):
    half = num_vector.max()/2
    vec_below_half = (num_vector <= half)
    prop_below_half = vec_below_half.sum() / len(vec_below_half)
    return prop_below_half

def calc_penetration_rate(vec_max_depth, vec_min_depth, vec_count_time):
    vec_penetration_rate = (vec_max_depth - vec_min_depth) / vec_count_time
    return vec_penetration_rate

def count_oscillations(num_vector):
    """
    Input: pd.Series
    Output: int, count of number of oscillations
    """
    #to avoid zeros
    diff = num_vector.diff().fillna(0)
    diff = diff.replace(0, method="ffill")

    prev_diff = diff.shift(1).fillna(0)
    prev_diff2 = diff.shift(2).fillna(0)
    prev_diff3 = diff.shift(3).fillna(0)
    post_diff = diff.shift(-1).fillna(0)
    post_diff2 = diff.shift(-2).fillna(0)
    post_diff3 = diff.shift(-3).fillna(0)

    # Changes in signal result in diff_mult being negative
    diff_mult = diff * prev_diff

    # To keep only the ones in which the change progresses
    diff_mult = diff_mult * (np.sign(prev_diff) != np.sign(diff))
    diff_mult = diff_mult * (np.sign(prev_diff2) != np.sign(diff))
    diff_mult = diff_mult * (np.sign(prev_diff3) != np.sign(diff))
    diff_mult = diff_mult * (np.sign(post_diff) == np.sign(diff))
    diff_mult = diff_mult * (np.sign(post_diff2) == np.sign(diff))
    diff_mult = diff_mult * (np.sign(post_diff3) == np.sign(diff))

    return diff_mult[diff_mult < 0].count()

def class_distance(df):
    # Create dictionary to hold output.
    Dist_dict = {"AMP_dist":[], "IF_dist":[], "QZ_dist":[], "LIM_dist":[]}

    # name rock classes to iterate through.
    rock_class = ['AMP', 'IF', 'QZ', 'LIM']

    #df = df[pd.notnull(df[['rock_class', 'ActualX_mean', 'ActualY_mean']])]

    # grab each row to grab the closest distances of each rock type and return to dictionary.
    for row in zip(df.ActualX_mean, df.ActualY_mean):
        row = np.reshape(row,(1,2))
        for r in rock_class:
            a = df.loc[df['exp_rock_class'] == r, 'ActualX_mean':'ActualY_mean']
            n = a.shape[0]
            new_a = a.values.reshape(n,2)
            d = distance.cdist(row, new_a, 'euclidean')[0]
            m = min(x for x in d if x > 0)
            r = r + str("_dist")
            Dist_dict[r].append(m)

    # Create dataframe of minimal distances
    pd_df = pd.DataFrame(Dist_dict)
    #new_df = pd.concat([df, pd_df], axis=1)
    return pd_df
