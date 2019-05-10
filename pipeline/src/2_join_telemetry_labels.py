#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 17:04 2019

@author: Gabriel Bogo

Telemetry - Label join algorithm

This algorithm works in 4 steps:

1. Load telemetry data and perform preliminary cleaning
2. Identify different drills based purely on hole depth signal. Let's call them "holes from telemetry"
3. Match **holes from telemetry** with **holes from Provision**. Needs to be done in a separate step due to memory restrictions
4. Fully join Telemetry Data with Provision data and perform final cleaning.

To run the script:

python src/2_join_telemetry_labels.py data/raw/190416001_MCMcshiftparam.csv data/raw/dbo.MCCONFMcparam_rawdata.csv data/intermediate/1_labelled_holes.csv data/intermediate/2_joined_data_test.csv
"""

import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("data_path")
parser.add_argument("field_names_path")
parser.add_argument("labelled_holes_path")
parser.add_argument("output_file_path")

args = parser.parse_args()

data_path = args.data_path
field_names_path = args.field_names_path
labelled_holes_path = args.labelled_holes_path
output_file_path = args.output_file_path

# ## Step 1 - Load telemetry data and perform preliminary cleaning
# #### 1.1 Load telemetry data ignoring useless columns

usecols = ['FieldTimestamp', 'ShiftId','Id', 'FieldOperid', 'FieldStatus', 'FieldId', 'FieldData', 'FieldX', 'FieldY']
df = pd.read_csv(data_path, usecols=usecols, dtype={"FieldOperid": object})

# #### 1.2 Join FieldDesc and correct timezone

# Join FieldDesc
df_field_names = pd.read_csv(field_names_path)
df_field_names = df_field_names[["FieldId", "FieldDesc"]].set_index("FieldId")
df = df.join(df_field_names, on="FieldId", how="left")

#FieldTimestamp is not in UTC. Let's correct that
df["utc_field_timestamp"] = pd.to_datetime(df.FieldTimestamp, unit="s")
df["utc_field_timestamp"] = df.utc_field_timestamp.dt.tz_localize("Canada/Eastern")
df["utc_field_timestamp"] = df.utc_field_timestamp.dt.tz_convert(None)
df["utc_field_timestamp"] = (df.utc_field_timestamp - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
df.head(1)


# #### 1.3 Disambiguate repeated timestamps

# Disambiguation of repeated timestamps
index = ["utc_field_timestamp", "FieldDesc"]
df.set_index(index, inplace=True)
df.sort_index(inplace=True) #Making sure that the telemetry data is properly sorted
rep_id = df.reset_index().groupby(index).cumcount()
df["rep_id"] = rep_id.values
df.reset_index(inplace=True)

# Checking the de-ambiguation process
assert df.query("FieldTimestamp == 1553931647").rep_id.max() == 32
assert df.query("FieldTimestamp == 1553931646").rep_id.max() == 2


# ## Step 2 - Identify different drills based purely on hole depth signal.

# #### 2.1 - Subsetting only Hole Depth data

df_timeseries = (df.query("FieldDesc == 'Hole Depth'")
         .copy()
         .set_index(["utc_field_timestamp", "rep_id"])
         .sort_index()
        )#to make sure it's ordered in time

# #### 2.2 Identify drops in Hole Depth

df_timeseries["hd_diff"] = df_timeseries.FieldData.diff().fillna(0)
df_timeseries["hole_index"] = df_timeseries.hd_diff < 0
df_timeseries["hole_index"] = df_timeseries.hole_index * 1
df_timeseries.hole_index = df_timeseries.hole_index.cumsum()

# #### 2.3 Getting start and end of drill purely on telemetry data

# In[7]:


df_holes = (df_timeseries
            .reset_index()
            .groupby("hole_index")
            .agg({"utc_field_timestamp": ['min', 'max']})
            )
df_holes.columns = ["drill_start", "drill_end"]
df_holes.reset_index(inplace=True)
df_holes.dropna(inplace=True) #remove ALL missing values. We need complete signals.

# ## Step 3 - Match holes from telemetry with holes from Provision.

# #### 3.1 Read in labelled data from previous and perform basic cleaning

df_labels = pd.read_csv(labelled_holes_path)
df_labels = df_labels[df_labels.collar_type != "DESIGN"] # We just want ACTUAL drills, not designed ones.
df_labels["unix_midpoint"] = (df_labels.unix_start + df_labels.unix_end) / 2
df_labels.dropna(subset=["unix_midpoint"], inplace=True) # We need complete drills, just as before

# This function should be on the First Step of creating labelled_holes.csv
def clean_label_data(df_labels):
    #1 rule - If Actual Depth is zero, we can remove
    #2 rule - If drilling time is less than a minute, we can remove it
    #3 rule - Whenever there is a dark time in the signal (either remove it or separate the sinal and get both start
    #and end of the signal)
    df = df_labels[df_labels.ActualDepth != 0] #Remove drills that did not actually drilled
    df = df[(df.unix_end - df.unix_start) > 60] #Remove drills that lasted less than a minute

    return df

df_labels = clean_label_data(df_labels)


# #### 3.2 Match holes from Provision with holes from telemetry

#Preparing numpy vectors
avg_epochs = df_labels.unix_midpoint.values # PVDrill
drill_starts = df_holes.drill_start.values # Telemetry
drill_ends = df_holes.drill_end.values # Telemetry

# #Checking input shape
# display(avg_epochs.shape)
# display(drill_starts.shape)
# display(drill_ends.shape)
#
# #Checking dtype
# display(avg_epochs.dtype)
# display(drill_starts.dtype)
# display(drill_ends.dtype)

# Complicated logic! Matching happens whenever the midpoint of a hole from Provision sits within
# the drilling period from a hole from telemetry.
# Every pair a[i], b[j] satisfy the condition below.
i, j = np.nonzero((avg_epochs[:, None] >= drill_starts) & (avg_epochs[:, None] <= drill_ends))

df_holes_labels = pd.DataFrame(
    np.column_stack([df_labels.values[i], df_holes.values[j]]),
    columns=df_labels.columns.append(df_holes.columns)
)
df_holes_labels.set_index("hole_index", inplace=True)

def treat_bad_signal(df_holes_labels):
    #Removing bad signals for now. We may think of a better solution later.
    g = (pd.pivot_table(data=df_holes_labels,
               index=["hole_index"],
               aggfunc={"hole_id": lambda x: x.nunique()})
        )
    bad_signal_list = g[g.hole_id > 1].index.tolist()

    df = df_holes_labels[~df_holes_labels.index.isin(bad_signal_list)]

    return df

df_holes_labels = treat_bad_signal(df_holes_labels)

# ## Step 4 - Fully join Telemetry Data with Provision data and perform final cleaning.

# #### 4.1 - Joining original data with identified holes from telemetry. Then joining labels.

# every timestamp belongs to a hole. No change in original dataset shape
df_join = df.join(df_timeseries.hole_index, on=["utc_field_timestamp", "rep_id"], how="left")
assert len(df_join) == len(df)

# every hole has a label. No change in original dataset shape
df_join = df_join.join(other=df_holes_labels, on="hole_index", how="left")
assert len(df_join) == len(df)

# Remove signal that was not assigned to any Provision hole
df_join.dropna(subset=["hole_id"], inplace=True)

# #### 4.2 - Exporting data as .csv

df_join.to_csv(output_file_path)
