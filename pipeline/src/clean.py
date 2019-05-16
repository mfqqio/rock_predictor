"""
Created in May 2019
@authors: Carrie Cheung, Gabriel Bogo, Shayne Andrews, Jim Pushor

Collection of functions to clean and join raw data
"""
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
