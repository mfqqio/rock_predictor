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

input_labels = ""
input_production = ""
input_telemetry = ""

# Read in main data sources
if mode == 'for_train':
    input_labels = "data/input_train/labels"
    input_production = "data/input_train/production"
    input_telemetry = "data/input_train/telemetry"

if mode == 'for_predict':
    input_labels = "data/input_predict/labels"
    input_production = "data/input_predict/production"
    input_telemetry = "data/input_predict/telemetry"

# Read in tables used for mapping parameters (same for both train/predict)
input_class_mapping = "data/input_mapping/rock_class_mapping.csv"
input_telem_headers = "data/input_mapping/telemetry_mapping.csv"
input_explosive_mapping = "data/input_mapping/explosive_by_rock_class.csv"

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

input_provision_cols = {
    'DrillPattern': 'DrillPattern',
    'DesignX': 'DesignX',
    'DesignY': 'DesignY',
    'DesignZ': 'DesignZ',
    'DesignDepth': 'DesignDepth',
    'ActualX': 'ActualX',
    'ActualY': 'ActualY',
    'ActualZ': 'ActualZ',
    'ActualDepth': 'ActualDepth',
    'ColletZ': 'ColletZ',
    'HoleID': 'HoleID',
    'FullName': 'FullName',
    'FirstName': 'FirstName',
    'UTCStartTime': 'UTCStartTime',
    'UTCEndTime': 'UTCEndTime',
    'StartTimeStamp': 'StartTimeStamp',
    'EndTimeStamp': 'EndTimeStamp',
    'DrillTime': 'DrillTime'
}

print('Reading all data %s...\n' % mode)

# Read all raw files into dataframes
df_labels = clean.get_files(input_labels, mode, cols=input_labels_cols)
df_class_mapping = clean.get_files(input_class_mapping, mode)
df_explosive_mapping = clean.get_files(input_explosive_mapping, mode)
df_production = clean.get_files(input_production, mode,
    cols=input_provision_cols, encoding="ISO-8859-1")
df_telemetry = clean.get_files(input_telemetry, mode, cols=input_telemetry_cols)
df_telem_headers = clean.get_files(input_telem_headers, mode)

shape_collar = df_labels.shape
shape_prov = df_production.shape
shape_telem = df_telemetry.shape

print("Number of raw datapoints: ")
print("Labels:", shape_collar)
print("Production:",shape_prov)
print("Telemetry:", shape_telem)
print("\n")

# Cleaning df_labels (COLLAR)
print("Cleaning Label data...")
df_labels['hole_id'] = df_labels['blast'] + "-" + df_labels['hole_name']
df_labels['exp_rock_type'] = df_labels['exp_rock_type'].str.strip()
df_labels["exp_rock_class"] = clean.get_rock_class(df_labels["exp_rock_type"], df_class_mapping)

if mode == 'for_train':
    df_labels['litho_rock_type'] = df_labels['litho_rock_type'].str.strip()
    df_labels.dropna(subset=["litho_rock_type"], inplace=True) # Only drop NA litho rock type rows if processing for training
    df_labels["litho_rock_class"] = clean.get_rock_class(df_labels["litho_rock_type"], df_class_mapping)
    if len(df_labels) < shape_collar[0]:
        diff = shape_collar[0] - len(df_labels)
        print("- %s rows from the Labels table do not contain labels. Removing those." % diff)
        shape_collar = df_labels.shape
    df_labels = df_labels[df_labels.collar_type != "DESIGN"] # We just want ACTUAL drills, not designed ones.
if len(df_labels) < shape_collar[0]:
    diff = shape_collar[0] - len(df_labels)
    print("- %s rows from the Label are DESIGN holes. Removing those." % diff)
    shape_collar = df_labels.shape
print("Shape of Labels table after cleaning:", shape_collar)
print("\n")

# Cleaning df_production
print("Cleaning Production data...")
df_production['hole_id'] = df_production['DrillPattern'] + "-" + df_production['HoleID']
df_production = df_production.drop(columns=['DrillPattern', 'HoleID'])
df_production['unix_start'] = clean.convert_utc2unix(df_production['UTCStartTime'], timezone="UTC")
df_production['unix_end'] = clean.convert_utc2unix(df_production['UTCEndTime'], timezone="UTC")
df_production = df_production.drop_duplicates(subset=["hole_id"], keep=False)
if len(df_production) < shape_prov[0]:
    diff = shape_prov[0] - len(df_production)
    print("- %s rows from the Production table are holes that have been redrilled. Removing those." % diff)
    shape_prov = df_production.shape
#Remove drills that did not actually drilled
df_production = df_production[df_production.ActualDepth != 0]
if len(df_production) < shape_prov[0]:
    diff = shape_prov[0] - len(df_production)
    print("- %s rows from the Production table have ActualDepth = 0. Removing those." % diff)
    shape_prov = df_production.shape
#Remove drills that lasted less than a minute
df_production = df_production[
    (df_production.unix_end - df_production.unix_start) > min_time]
if len(df_production) < shape_prov[0]:
    diff = shape_prov[0] - len(df_production)
    print("- %s rows from the Production table lasted less than %s. Removing those." % (diff, min_time))
    shape_prov = df_production.shape
print("Shape of Production table after cleaning:", shape_prov)
print("\n")

# Join df_production with df_labels and more cleaning
print("Joining Labels and Production...")
df_prod_labels = pd.merge(df_production, df_labels, how='left', left_on=['hole_id'], right_on = ['hole_id'])

# Join in df_explosive_mapping
df_prod_labels = pd.merge(df_prod_labels, df_explosive_mapping, how='left', left_on=['exp_rock_class'], right_on=['rock_class'])
df_prod_labels["prior_explosive"] = df_prod_labels["kg/m3"]
df_prod_labels = df_prod_labels.drop(columns=["kg/m3","kg/t"])
shape_prod_labels = df_prod_labels.shape
if mode == 'for_train': # Don't drop na rows when we process data for prediction
    df_prod_labels.dropna(inplace=True)
    if shape_prod_labels[0] < shape_prov[0]:
        diff = shape_prov[0] - shape_prod_labels[0]
        print("- %s rows from the Production table have no corresponding label from the Labels table. Removing those." % diff)
        shape_prod_labels = df_prod_labels.shape
print("Shape of joined Production + Labels:", shape_prod_labels)
print("\n")

# Cleaning df_telemetry and making wide
print("Processing Telemetry...")
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
shape_telem = df_telemetry.shape
print('Telemetry shape in wide format:', shape_telem)
#Dropping all missing values
df_telemetry.dropna(inplace=True)
if len(df_telemetry) < shape_telem[0]:
    diff = shape_telem[0] - len(df_telemetry)
    print("- %s rows from the Telemetry table contain missing values. Removing those." % diff)
    shape_telem = df_telemetry.shape
#Dropping timestamps where the drill head is stopped
df_telemetry["pos_lag1_diff"] = df_telemetry.pos.diff().fillna(0)
df_telemetry = df_telemetry[df_telemetry.pos_lag1_diff > 0]
if len(df_telemetry) < shape_telem[0]:
    diff = shape_telem[0] - len(df_telemetry)
    print("- %s rows from the Telemetry table correspond to a stopped drill head. Removing those." % diff)
    shape_telem = df_telemetry.shape
#Dropping timestamps where the drill is not rotating
df_telemetry = df_telemetry[df_telemetry.rot > 0]
if len(df_telemetry) < shape_telem[0]:
    diff = shape_telem[0] - len(df_telemetry)
    print("- %s rows from the Telemetry table correspond to a non-rotating drill. Removing those." % diff)
    shape_telem = df_telemetry.shape
print("Shape of Telemetry table (wide):", shape_telem)

# Identify signals purely on Hole Depth
print("Identifying holes based on Telemetry data...")
df_telemetry.sort_index(inplace=True)
df_telemetry["depth_diff"] = df_telemetry.depth.diff().fillna(0)
df_telemetry["telem_id"] = df_telemetry.depth_diff < max_diff
df_telemetry["telem_id"] = df_telemetry.telem_id * 1
df_telemetry["telem_id"] = df_telemetry.telem_id.cumsum()

#Count the times the drill goes up and down
df_telemetry["count_change_direction"] = (df_telemetry
    .groupby("telem_id")["pos"]
    .transform(feature_eng.count_oscillations))

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
assert df_telem_holes.telem_id.nunique() == len(df_telem_holes)
num_holes = len(df_telem_holes)
print("Number of telemetry holes: ", num_holes)
df_telem_holes.dropna(inplace=True)
if len(df_telem_holes) < num_holes:
    diff = num_holes - len(df_telem_holes)
    print("- %s telemetry holes contain missing values. Removing those." % diff)
    num_holes = len(df_telem_holes)
# Remove holes that last than min time
df_telem_holes = df_telem_holes[df_telem_holes.drilling_time > min_time]
if len(df_telem_holes) < num_holes:
    diff = num_holes - len(df_telem_holes)
    print("- %s telemetry holes have a duration that is lower than %ss. Removing those." % (diff, min_time))
    num_holes = len(df_telem_holes)
# Remove holes that are less deep than min_depth
df_telem_holes = df_telem_holes[df_telem_holes.drilling_depth > min_depth]
if len(df_telem_holes) < num_holes:
    diff = num_holes - len(df_telem_holes)
    print("- %s telemetry holes have depth that is lower than %sm. Removing those." % (diff, min_depth))
    num_holes = len(df_telem_holes)
print("Number of telemetry holes after cleaning: ", num_holes)
print("\n")

# Join all datasets
print("Joining Telemetry with Production/Labels...")
# Match time frames (for report purposes)
latest_start_time = max(df_telem_holes.drill_start.min(),
    df_prod_labels.unix_start.min())
print("Beginning of overlapping time window:",
    pd.to_datetime(latest_start_time, unit="s"))
earliest_end_time = min(df_telem_holes.drill_end.max(),
    df_prod_labels.unix_end.max())
print("End of overlapping time window:",
    pd.to_datetime(earliest_end_time, unit="s"))
df_telem_holes = df_telem_holes.query("drill_start > @latest_start_time")
df_telem_holes = df_telem_holes.query("drill_end < @earliest_end_time")
if len(df_telem_holes) < num_holes:
    diff = num_holes - len(df_telem_holes)
    print("- %s telemetry holes have are out of the overlapping time window. Removing those." % diff)
    num_holes = len(df_telem_holes)

df_prod_labels = df_prod_labels.query("unix_start > @latest_start_time")
df_prod_labels = df_prod_labels.query("unix_end < @earliest_end_time")
if len(df_prod_labels) < shape_prod_labels[0]:
    diff = shape_prod_labels[0] - len(df_prod_labels)
    print("- %s rows from the Production/Labels table are out of the overlapping time window. Removing those." % diff)
    shape_prod_labels = df_prod_labels.shape

print("Number of telemetry holes within overlapping time window:", df_telem_holes.shape)
print("Number of Production/Labels holes within overlapping time window:",
    df_prod_labels.shape)

df_join_lookup = clean.join_prod_telemetry(df_prod_labels.unix_start,df_prod_labels.unix_end,df_prod_labels.hole_id,
                                           df_telem_holes.drill_start,df_telem_holes.drill_end,df_telem_holes.telem_id)
assert df_join_lookup.hole_id.nunique() == len(df_join_lookup)
num_holes_join = len(df_join_lookup)
if num_holes_join < shape_prod_labels[0]:
    diff = shape_prod_labels[0] - num_holes_join
    print("- %s holes from the Production/Labels were lost on the joining process. That represents a loss of %.2f." % (diff, diff/shape_prod_labels[0]))
if num_holes_join < num_holes:
    diff = num_holes - num_holes_join
    print("- %s holes from Telemetry were lost on the joining process. That represents a loss of %.2f." % (diff, diff/num_holes))
# Clean joining anomalies
double_joins_mask = clean.identify_double_joins(df_join_lookup.telem_id)
df_join_lookup = df_join_lookup[double_joins_mask]
if len(df_join_lookup) < num_holes_join:
    diff = num_holes_join - len(df_join_lookup)
    print("- %s holes were considered joining anomalies. Removing those." % diff)
    num_holes_join = len(df_join_lookup)

print("Number of joined holes from Labels, Production and Telemetry: ", num_holes_join)


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
