#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process raw data into a fully cleaned and joined dataset, then export train-test-split into csv.
Inputs: 5 raw csv files: COLLAR, MCM (telemetry), PVDrill, MCCONFM, rock_class_mapping
Outputs: 2 csv files: train, test
"""

from helpers import clean # support functions live here
import pandas as pd
import numpy as np
import argparse, sys


if len(sys.argv)<7: # then running in dev mode (can remove in production)
    data_root = sys.argv[1]
    mode =  sys.argv[2] # Mode can be "for_train" or "for_predict"
    print('Processing data %s...' % mode)
    
    # Read in main data sources
    input_labels = data_root + "/COLLAR"
    input_production = data_root + "/PVDrillProduction"
    input_telemetry = data_root + "/MCMcshiftparam"
    
    # Read in tables used for mapping parameters
    input_class_mapping = "../data/business/rock_class_mapping.csv"
    input_telem_headers = "../data/mapping/dbo.MCCONFMcparam_rawdata.csv"
    
    # Define output files
    output_train = "data/train.csv" # for_train
    output_test = "data/test.csv" # for_train
    output_predict = "data/predict_data.csv" # for_predict
    
    max_diff = 0
    min_time = 60
    min_depth = 5
    test_prop = 0.2

else: # parse input parameters from terminal or makefile
    parser = argparse.ArgumentParser()
    parser.add_argument("input_labels")
    parser.add_argument("input_class_mapping")
    parser.add_argument("input_production")
    parser.add_argument("input_telemetry")
    parser.add_argument("input_telem_headers")
    parser.add_argument("output_train")
    parser.add_argument("output_test")
    parser.add_argument("max_diff")
    parser.add_argument("min_time")
    parser.add_argument("min_depth")
    parser.add_argument("test_prop")
    args = parser.parse_args()

    input_labels = args.input_labels
    input_class_mapping = args.input_class_mapping
    input_production = args.input_production
    input_telemetry = args.input_telemetry
    input_telem_headers = args.input_telem_headers
    output_train = args.output_train
    output_test = args.output_test
    max_diff = args.max_diff
    min_time = args.min_time
    min_depth = args.min_depth
    test_prop = args.test_prop

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
df_labels = clean.get_files(input_labels, cols=input_labels_cols)
df_class_mapping = clean.get_files(input_class_mapping)
df_production = clean.get_files(input_production)
df_telemetry = clean.get_files(input_telemetry, cols=input_telemetry_cols)
df_telem_headers = clean.get_files(input_telem_headers)

# Cleaning df_labels (COLLAR)
df_labels['hole_id'] = df_labels['blast'] + "-" + df_labels['hole_name']
df_labels['exp_rock_type'] = df_labels['exp_rock_type'].str.strip()

if mode == 'for_train': 
    df_labels['litho_rock_type'] = df_labels['litho_rock_type'].str.strip()
    df_labels.dropna(subset=["litho_rock_type"], inplace=True) # Only drop NA litho rock type rows if processing for training
    df_labels["litho_rock_class"] = clean.get_rock_class(df_labels["litho_rock_type"], df_class_mapping)
    
df_labels["exp_rock_class"] = clean.get_rock_class(df_labels["exp_rock_type"], df_class_mapping)

# Cleaning df_production
df_production['hole_id'] = df_production['DrillPattern'] + "-" + df_production['HoleID']
df_production = df_production.drop(columns=['DrillPattern', 'HoleID'])
df_production['unix_start'] = clean.convert_utc2unix(df_production['UTCStartTime'], timezone="UTC")
df_production['unix_end'] = clean.convert_utc2unix(df_production['UTCEndTime'], timezone="UTC")
df_production = df_production.drop_duplicates(subset=["hole_id"], keep=False)

# Join df_production with df_labels and more cleaning
df_prod_labels = pd.merge(df_production, df_labels, how='left', left_on=['hole_id'], right_on = ['hole_id'])
print('Joined labels + production dimensions:', df_prod_labels.shape)
df_prod_labels = df_prod_labels[df_prod_labels.collar_type != "DESIGN"] # We just want ACTUAL drills, not designed ones.
df_prod_labels = df_prod_labels[df_prod_labels.ActualDepth != 0] #Remove drills that did not actually drilled
df_prod_labels = df_prod_labels[(df_prod_labels.unix_end - df_prod_labels.unix_start) > 60] #Remove drills that lasted less than a minute
print('df_prod_labels dimensions after cleaning:', df_prod_labels.shape)

if mode == 'for_train': # Don't drop na rows when we process data for prediction
    df_prod_labels.dropna(inplace=True)
    
print('df_prod_labels dimensions after dropna:', df_prod_labels.shape)

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
df_telemetry = df_telemetry[df_telemetry.pos_lag1_diff > 0]
df_telemetry = df_telemetry[df_telemetry.rot > 0]
print('df_telemetry dimensions after initial cleaning:', df_telemetry.shape)

# Identify signals purely on Hole Depth
df_telemetry.sort_index(inplace=True)
df_telemetry["depth_diff"] = df_telemetry.depth.diff().fillna(0)
df_telemetry["telem_id"] = df_telemetry.depth_diff < max_diff
df_telemetry["telem_id"] = df_telemetry.telem_id * 1
df_telemetry["telem_id"] = df_telemetry.telem_id.cumsum()
print("Number of holes: ", df_telemetry.telem_id.nunique())

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
print("Number of holes after cleaning: ", df_telem_holes.shape)

# Match Provision and Telemetry data
df_join_lookup = clean.join_prod_telemetry(df_prod_labels.unix_start,df_prod_labels.unix_end,df_prod_labels.hole_id,
                                           df_telem_holes.drill_start,df_telem_holes.drill_end,df_telem_holes.telem_id)
print("Number of matches between Provision and Telemetry: ", df_join_lookup.shape)

# Clean joining anomalies
double_joins_mask = clean.identify_double_joins(df_join_lookup.telem_id)
df_join_lookup = df_join_lookup[double_joins_mask]
print("Number of matches between Provision and Telemetry, after cleaning: ", df_join_lookup.shape)

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
