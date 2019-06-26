import pandas as pd
import numpy as np
import random, glob, os

def get_files(path, mode, cols=None, dtype=None, encoding=None):
    if cols:
        usecols = list(cols.keys())
    else:
        usecols = None

    if os.path.isdir(path):
        print("Loading all files from:", path)
        files = glob.glob(os.path.join(path, "*.csv"))
        df = (pd.read_csv(f, usecols=usecols,  encoding = encoding) for f in files)
        df = pd.concat(df, ignore_index=True, sort=False).drop_duplicates()
        print("...concatenated shape:", df.shape)
    else:
        print("Loading this file:", path)
        df = pd.read_csv(path, usecols=usecols)
        print("...shape of file:", df.shape)
    if cols:
        df.rename(columns=cols,inplace=True)
    return df

def remove_zero_rotation(rotations):
    return rotations > 0

def remove_head_position(head_diffs, min_head_diff):
    return head_diffs > min_head_diff

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

def join_prod_telemetry(prod_start, prod_end, prod_id, telem_start, telem_end, telem_id):
    prod_midpoints = (prod_start.values + prod_end.values)/2
    np_telem_start = telem_start.values
    np_telem_end = telem_end.values

    i, j = np.nonzero((prod_midpoints[:, None] >= np_telem_start) & (prod_midpoints[:, None] <= np_telem_end))

    df_prod_telem = pd.DataFrame(
        data={"hole_id": prod_id.values[i],
              "telem_id": telem_id.values[j]
        }
    # np.column_stack([prod_id.values[i], telem_id.values[j]]),
    # columns=["prod_id", "telem_id"]
    )
    # import pdb; pdb.set_trace()
    return df_prod_telem

def identify_double_joins(telem_id):
    g = telem_id.groupby(telem_id).count()
    double_joins_list = g[g > 1].index.tolist()
    double_joins_mask = ~telem_id.index.isin(double_joins_list)

    return double_joins_mask

def train_test_split(df, id_col, test_prop=0.2, stratify_by=None, seed=123):
    random.seed(seed)
    test_holes = []
    if stratify_by is None:
        strat_holes = df[id_col].unique()
        n_test = round(len(strat_holes) * test_prop)
        test_holes.extend(random.sample(list(strat_holes), n_test))
    else:
        strat_values = df[stratify_by].unique()
        for s in strat_values:
            strat_holes = df[df[stratify_by]==s][id_col].unique()
            n_test = round(len(strat_holes) * test_prop)
            test_holes.extend(random.sample(list(strat_holes), n_test))

    df_train = df[~df[id_col].isin(test_holes)]
    df_test = df[df[id_col].isin(test_holes)]

    return df_train, df_test
