#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process raw data into a fully cleaned and joined dataset, then export train-test-split into csv.
Inputs: 5 raw csv files: COLLAR, MCM (telemetry), PVDrill, MCCONFM, rock_class_mapping
Outputs: 2 csv files: train, test
"""

from helpers import clean, feature_eng # support functions live here
import pandas as pd
import numpy as np
import argparse, sys

parser = argparse.ArgumentParser()

parser.add_argument("mode")

parser.add_argument('--max_diff', default=0,
    help='Maximum diff in hole depth that will be tolerated. More than this diff will be considered a new hole (default: 0)')

parser.add_argument('--min_time', default=60, help='Minimum drilling time for a drill to be considered valid (default: 60)')

parser.add_argument('--min_depth', default=5, help='Minimum drilling depth for a drill to be considered valid (default: 5)')

parser.add_argument('--test_prop', default=0.2, help='Proportion of data that will be held for testing (default: 0.2)')

args = parser.parse_args()

mode = args.mode # Mode can be "for_train" or "for_predict"
max_diff = args.max_diff
min_time = args.min_time
min_depth = args.min_depth
test_prop = args.test_prop

print('Processing data %s...' % mode)

input_labels = ""
input_production = ""
input_telemetry = ""

# Read in main data sources
if mode == 'for_train':
    input_labels = "data/input_train/COLLAR"
    input_production = "data/input_train/PVDrillProduction"
    input_telemetry = "data/input_train/MCMcshiftparam"

if mode == 'for_predict':
    input_labels = "data/input_predict/COLLAR"
    input_production = "data/input_predict/PVDrillProduction"
    input_telemetry = "data/input_predict/MCMcshiftparam"

# Read in tables used for mapping parameters (same for both train/predict)
input_class_mapping = "data/input_mapping/rock_class_mapping.csv"
input_telem_headers = "data/input_mapping/dbo.MCCONFMcparam_rawdata.csv"    

# Define output files
output_train = "data/pipeline/train.csv" # for_train
output_test = "data/pipeline/test.csv" # for_train
output_predict = "data/pipeline/predict.csv" # for_predict

# Specify required columns
input_labels_cols = {
    'hole_id':'hole_id',
    'x':'x_collar',
    'y':'y_collar',
    'z':'z_collar',
    'LITHO':'litho_rock_type',
    'COLLAR_TYPE':'collar_type',
    'PLANNED_RTYPE':'exp_rock_type',
    'BLAST': 'blast',
    'HOLE_NAME': 'hole_name'
}
input_telemetry_cols = {
    'FieldTimestamp': 'FieldTimestamp',
    'FieldId': 'FieldId',
    'FieldData': 'FieldData',
    'FieldX': 'FieldX',
    'FieldY': 'FieldY'
}

# Read all raw files into dataframes
df_labels = clean.get_files(input_labels, mode, cols=input_labels_cols)
df_class_mapping = clean.get_files(input_class_mapping, mode)
df_production = clean.get_files(input_production, mode)
df_telemetry = clean.get_files(input_telemetry, mode, cols=input_telemetry_cols)
df_telem_headers = clean.get_files(input_telem_headers, mode)

# Cleaning df_labels (COLLAR)
df_labels['hole_id'] = df_labels['blast'] + "-" + df_labels['hole_name']
df_labels['exp_rock_type'] = df_labels['exp_rock_type'].str.strip()

if mode == 'for_train':
    df_labels['litho_rock_type'] = df_labels['litho_rock_type'].str.strip()
    df_labels.dropna(subset=["litho_rock_type"], inplace=True) # Only drop NA litho rock type rows if processing for training
    df_labels["litho_rock_class"] = clean.get_rock_class(df_labels["litho_rock_type"], df_class_mapping)

df_labels["exp_rock_class"] = clean.get_rock_class(df_labels["exp_rock_type"], df_class_mapping)
print("Clean chemical assay data dimensions:", df_labels.shape)

# Cleaning df_production
df_production['hole_id'] = df_production['DrillPattern'] + "-" + df_production['HoleID']
df_production = df_production.drop(columns=['DrillPattern', 'HoleID'])
df_production['unix_start'] = clean.convert_utc2unix(df_production['UTCStartTime'], timezone="UTC")
df_production['unix_end'] = clean.convert_utc2unix(df_production['UTCEndTime'], timezone="UTC")
df_production = df_production.drop_duplicates(subset=["hole_id"], keep=False)
print("Clean Provision data dimensions:", df_production.shape)

# Join df_production with df_labels and more cleaning
df_prod_labels = pd.merge(df_production, df_labels, how='left', left_on=['hole_id'], right_on = ['hole_id'])
print('Joined labels + production dimensions:', df_prod_labels.shape)
df_prod_labels = df_prod_labels[df_prod_labels.collar_type != "DESIGN"] # We just want ACTUAL drills, not designed ones.
df_prod_labels = df_prod_labels[df_prod_labels.ActualDepth != 0] #Remove drills that did not actually drilled
df_prod_labels = df_prod_labels[(df_prod_labels.unix_end - df_prod_labels.unix_start) > 60] #Remove drills that lasted less than a minute

if mode == 'for_train': # Don't drop na rows when we process data for prediction
    df_prod_labels.dropna(inplace=True)

print('Clean joined Provision and Assay dataset dimension:', df_prod_labels.shape)

# Cleaning df_telemetry and making wide
df_telemetry["utc_field_timestamp"] = clean.convert_utc2unix(df_telemetry.FieldTimestamp, timezone="Canada/Eastern",unit="s")
df_telemetry = pd.pivot_table(df_telemetry,
    values='FieldData',
    columns='FieldId',
    index=['utc_field_timestamp','FieldX', 'FieldY'],
    aggfunc='mean')

cols = {"42010001": "rot", "42010005": "pull",
        "42010008": "air", "4201000B": "vvib",
        "4201000C": "hvib", "4201000E": "water",
        "4201000F": "depth", "42010010": "pos"}
df_telemetry = df_telemetry.rename(columns=cols)
print('df_telemetry dimensions in wide format:', df_telemetry.shape)
df_telemetry.dropna(inplace=True)
df_telemetry["pos_lag1_diff"] = df_telemetry.pos.diff().fillna(0)

# Identify signals purely on Hole Depth
df_telemetry.sort_index(inplace=True)
df_telemetry["depth_diff"] = df_telemetry.depth.diff().fillna(0)
df_telemetry["telem_id"] = df_telemetry.depth_diff < max_diff
df_telemetry["telem_id"] = df_telemetry.telem_id * 1
df_telemetry["telem_id"] = df_telemetry.telem_id.cumsum()

#Count the times the drill goes up and down
df_telemetry["count_change_direction"] = (df_telemetry
    .groupby("telem_id")["pos"]
    .transform(feature_eng.count_oscillations))

#Primary cleaning
df_telemetry = df_telemetry[df_telemetry.pos_lag1_diff > 0]
df_telemetry = df_telemetry[df_telemetry.rot > 0]
print('df_telemetry dimensions after initial cleaning:', df_telemetry.shape)
print("Number of telemetry holes: ", df_telemetry.telem_id.nunique())

# New df grouped by telemetry holes
df_telem_holes = (df_telemetry
    .reset_index()
    .groupby("telem_id")
    .agg({"utc_field_timestamp": ['min', 'max'],
          "depth": ['min', 'max']})
    )
df_telem_holes.columns = ["drill_start", "drill_end", "initial_depth", "final_depth"]
df_telem_holes["drilling_time"] = df_telem_holes.drill_end - df_telem_holes.drill_start
df_telem_holes["drilling_depth"] = df_telem_holes.final_depth - df_telem_holes.initial_depth
df_telem_holes.reset_index(inplace=True)

# Cleaning noisy drills
df_telem_holes.dropna(inplace=True)
df_telem_holes = df_telem_holes[df_telem_holes.drilling_time > min_time]
df_telem_holes = df_telem_holes[df_telem_holes.drilling_depth > min_depth]
print("Number of telemetry holes after cleaning: ", df_telem_holes.shape)

# Match time frames (for report purposes)
latest_start_time = max(df_telem_holes.drill_start.min(),
    df_prod_labels.unix_start.min())
earliest_end_time = min(df_telem_holes.drill_end.max(),
    df_prod_labels.unix_end.max())
df_telem_holes = df_telem_holes.query("drill_start > @latest_start_time")
df_telem_holes = df_telem_holes.query("drill_end < @earliest_end_time")
df_prod_labels = df_prod_labels.query("unix_start > @latest_start_time")
df_prod_labels = df_prod_labels.query("unix_end < @earliest_end_time")
print("Number of telemetry holes after time adjustment: ", df_telem_holes.shape)
print("Number of Assay/Provision holes after time adjustment: ",
    df_prod_labels.shape)

# Match Provision and Telemetry data
df_join_lookup = clean.join_prod_telemetry(df_prod_labels.unix_start,df_prod_labels.unix_end,df_prod_labels.hole_id,
                                           df_telem_holes.drill_start,df_telem_holes.drill_end,df_telem_holes.telem_id)
print("Number of joined holes from Assay, Provision and Telemetry: ", df_join_lookup.shape)

# Clean joining anomalies
double_joins_mask = clean.identify_double_joins(df_join_lookup.telem_id)
df_join_lookup = df_join_lookup[double_joins_mask]
print("Number of joined holes from Assay, Provision and Telemetry, after cleaning: ", df_join_lookup.shape)

# Join df_telemetry with df_join_lookup
df_output = pd.merge(df_telemetry.reset_index(), df_join_lookup, how="inner", on="telem_id")
df_output = pd.merge(df_output, df_prod_labels, how="left", on="hole_id")
print("Final data shape: ", df_output.shape)

# Split data into train/test if processing data for training.
# and save to appropriately named output files.
if mode == 'for_train':
    df_train, df_test = clean.train_test_split(df_output, id_col="hole_id", test_prop=test_prop, stratify_by="litho_rock_type")

    print("Final train shape: ", df_train.shape)
    print("Final test shape: ", df_test.shape)

    print("Saving output files...")
    df_train.to_csv(output_train, index=True)
    df_test.to_csv(output_test, index=True)

# Save everything to file if processing data for prediction.
elif mode == 'for_predict':
    df_output.to_csv(output_predict, index=True)

print("Data cleaning complete!")
