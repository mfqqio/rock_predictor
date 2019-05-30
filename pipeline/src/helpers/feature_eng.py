#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates features based on telemetry data for each hole.
"""

import pandas as pd
import numpy as np

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
    return vec_penetration_rate.values

def count_oscillations(num_vector):
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
#
# def zero_water_flow(telem, hole_id_col, colname):
#     # Select only rows where water flow is 0
#     zero_water = telem[telem[colname] == 0]
#     has_water = telem[telem[colname] > 0]
#
#     samples_nowater = zero_water.groupby(hole_id_col).agg(['count'])[colname]
#     samples_nowater.rename(columns={'count':'samples_nowater'}, inplace=True)
#
#     samples_haswater = has_water.groupby(hole_id_col).agg(['count'])[colname]
#     samples_haswater.rename(columns={'count':'samples_haswater'}, inplace=True)
#
#     joined = pd.merge(samples_nowater, samples_haswater,
#                     how='left',
#                     left_index=True,
#                     right_index=True)
#     joined.fillna(0, inplace=True)
#     joined['total_samples'] = joined['samples_nowater'] + joined['samples_haswater']
#     joined['prop_nowater'] = joined['samples_nowater']/joined['total_samples']
#
#     return joined['prop_nowater'].values
#
# # Calculates proportion of time in time series at maximum pulldown force.
# def prop_max_pulldown(telem, hole_id_col, colname):
#     holes = telem.groupby(hole_id_col)
#     max_pd = holes.agg(['max'])[colname] # Maximum pulldown force for each hole
#
#     # Attach max pulldown force value to telem data
#     joined = pd.merge(telem, max_pd,
#                      how='left',
#                      left_on=[hole_id_col],
#                      right_index=True)
#
#     # Count samples within a margin (e.g. 5%) of max pulldown for each hole
#     telem_max = joined[(joined[colname] <= joined['max']+(joined['max']*0.05)) & (joined[colname] >= joined['max']-(joined['max']*0.05))]
#     samples_at_max = telem_max.groupby(hole_id_col).agg(['count'])[colname]
#     samples_at_max.rename(columns={'count':'samples_max_pd'}, inplace=True)
#
#     total_samples = holes.agg(['count'])[colname]
#     total_samples.rename(columns={'count':'total_samples'}, inplace=True)
#
#     pd_df = pd.merge(samples_at_max, total_samples,
#                     how='left',
#                     left_index=True,
#                     right_index=True)
#     pd_df['prop_max_pulldown'] = pd_df['samples_max_pd']/pd_df['total_samples']
#
#     return pd_df['prop_max_pulldown'].values
#
# # Calculates proportion of time in time series at less than half of maximum pulldown force.
# def prop_half_pulldown(telem, hole_id_col, colname):
#     holes = telem.groupby(hole_id_col)
#     max_pd = holes.agg(['max'])[colname] # Maximum pulldown force for each hole
#     half_max_pd = max_pd/2
#
#     # Attach max pulldown force value to telem data
#     joined = pd.merge(telem, half_max_pd,
#                      how='left',
#                      left_on=[hole_id_col],
#                      right_index=True)
#
#     # Count samples at less than half of max pulldown for each hole
#     telem_half = joined[joined[colname] < joined['max']]
#     samples_half = telem_half.groupby(hole_id_col).agg(['count'])[colname]
#     samples_half.rename(columns={'count':'samples_half_pd'}, inplace=True)
#
#     total_samples = holes.agg(['count'])[colname]
#     total_samples.rename(columns={'count':'total_samples'}, inplace=True)
#
#     pd_df = pd.merge(samples_half, total_samples,
#                     how='left',
#                     left_index=True,
#                     right_index=True)
#     pd_df['prop_half_max_pd'] = pd_df['samples_half_pd']/pd_df['total_samples']
#
#     return pd_df['prop_half_max_pd'].values
