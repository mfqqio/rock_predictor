"""
Created in May 2019
@authors: Carrie Cheung, Gabriel Bogo, Shayne Andrews, Jim Pushor

Collection of functions to clean and join raw data
"""
import pandas as pd
import numpy as np

def hello():
  print("hello")

def convert_time(list):
    return list

def reindex_duplicates(list):
    return list

def make_wide(df):
    return df # reduce from ~2m rows to 300k

def identify_telem_holes(hole_depth, max_diff):
    return list # new_hole_index vector

def group_holes(new_hole_index, timestamps, hole_depth):
    return df # 1309 rows grouped by new_hole_index

#def remove_na(df)

def remove_drill_time(times, min_time):
    return time_diffs > min_time

def remove_zero_rotation(rotations):
    return rotations > 0

def remove_head_position(head_diffs, min_head_diff):
    return head_diffs > min_head_diff

def create_hole_id(drill_pattern, hole):
    return drill_pattern + '-' + hole

# Converts UTC format to UNIX timestamp
def convert_utc2unix(utc, timezone, unit="ns"):
    utc = pd.to_datetime(utc, unit=unit)
    utc = utc.dt.tz_localize(timezone)
    utc = utc.dt.tz_convert(None)

    return (utc - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

def get_rock_class(rock_types, df_mapping):
    df_rock_types = rock_types.to_frame()
    df_rock_types.set_index(df_rock_types.columns[0], inplace=True)
    rock_classes = pd.merge(df_rock_types, df_mapping, how='left', left_index=True, right_on = ['rock_type'])
    return rock_classes["rock_class"].values

def match_telemetry_labels(telem_id, telem_start, telem_end, labels_start, labels_end, labels_id):
    labels_midpoints = (labels_start + labels_end)/2

    i, j = np.nonzero((labels_midpoints[:, None] >= drill_starts) & (labels_midpoints[:, None] <= drill_ends))
