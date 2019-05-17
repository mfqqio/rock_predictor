#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in May 2019

@authors: Carrie Cheung, Gabriel Bogo, Shayne Andrews, Jim Pushor

Process raw data into a fully cleaned and joined dataset.
Inputs: 5 raw csv files: COLLAR, MCM (telemetry), PVDrill, MCCONFM, rock_class_mapping
Outputs: 2 csv files: train, test
"""

import clean # support functions live here
import pandas as pd
import numpy as np
import argparse, sys

if len(sys.argv)<7: # then running in dev mode (can remove in production)
    input_labels = "data/raw/190501001_COLLAR.csv"
    input_class_mapping = "data/raw/rock_class_mapping.csv"
    input_production = "data/raw/190506001_PVDrillProduction.csv"
    input_telemetry = "data/raw/190416001_MCMcshiftparam.csv"
    input_telem_headers = "data/raw/dbo.MCCONFMcparam_rawdata.csv"
    output_train = "data/intermediate/train.csv"
    output_test = "data/intermediate/test.csv"
    max_diff = 0
    min_time = 60
    min_depth = 5

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
    args = parser.parse_args()

    input_labels = args.input_labels
    input_class_mapping = args.input_class_mapping
    input_production = args.input_production
    input_telemetry = args.input_telemetry
    input_telem_headers = args.input_telem_headers
    output_train = args.output_train
    output_test = args.output_test

# Read all raw csv files
cols = ['hole_id', 'x', 'y', 'z', 'COLLAR_TYPE', 'LITHO', 'PLANNED_RTYPE']
df_labels = pd.read_csv(input_labels, usecols=cols)
print('Labels table dimensions:', df_labels.shape)
df_class_mapping = pd.read_csv(input_class_mapping)
print('Class mapping table dimensions:', df_class_mapping.shape)
df_production = pd.read_csv(input_production)
print('Production table dimensions:', df_production.shape)
cols = ['FieldTimestamp', 'ShiftId', 'FieldStatus', 'FieldId', 'FieldData', 'FieldX', 'FieldY']
df_telemetry = pd.read_csv(input_telemetry, usecols=cols, dtype={"FieldOperid": object})
print('Telemetry table dimensions:', df_telemetry.shape)
df_telem_headers = pd.read_csv(input_telem_headers)
print('Telemetry headers table dimensions:', df_telem_headers.shape)

# Cleaning df_labels and join with df_class_mapping
df_labels.rename(columns={'x':'x_collar',
                       'y':'y_collar',
                       'z':'z_collar',
                       'LITHO':'litho_rock_type',
                       'COLLAR_TYPE':'collar_type',
                       'PLANNED_RTYPE':'exp_rock_type'},
                 inplace=True)
df_labels['exp_rock_type'] = df_labels['exp_rock_type'].str.strip()
df_labels['litho_rock_type'] = df_labels['litho_rock_type'].str.strip()
df_labels["litho_rock_class"] = clean.get_rock_class(df_labels["litho_rock_type"], df_class_mapping)
df_labels["exp_rock_class"] = clean.get_rock_class(df_labels["exp_rock_type"], df_class_mapping)

df_production['hole_id'] = clean.create_hole_id(df_production['DrillPattern'], df_production['HoleID'])
df_production = df_production.drop(columns=['DrillPattern', 'HoleID'])
df_production['unix_start'] = clean.convert_utc2unix(df_production['UTCStartTime'], timezone="UTC")
df_production['unix_end'] = clean.convert_utc2unix(df_production['UTCEndTime'], timezone="UTC")
df_production = df_production.drop_duplicates(subset=["hole_id"], keep=False)

df_prod_labels = pd.merge(df_production, df_labels, how='left', left_on=['hole_id'], right_on = ['hole_id'])
print('Joined labels + production dimensions:', df_prod_labels.shape)

#Removing problematic datapoints
df_prod_labels = df_prod_labels[df_prod_labels.collar_type != "DESIGN"] # We just want ACTUAL drills, not designed ones.
df_prod_labels = df_prod_labels[df_prod_labels.ActualDepth != 0] #Remove drills that did not actually drilled
df_prod_labels = df_prod_labels[(df_prod_labels.unix_end - df_prod_labels.unix_start) > 60] #Remove drills that lasted less than a minute
print('df_prod_labels dimensions after cleaning:', df_prod_labels.shape)

## TELEMETRY
# import pdb; pdb.set_trace()
#FieldTimestamp is not in UTC. Let's correct that
df_telemetry["utc_field_timestamp"] = clean.convert_utc2unix(
    df_telemetry.FieldTimestamp,
    timezone="Canada/Eastern",
    unit="s")

#There are repeated timestamps for the same FieldIds, which is an anomaly.
#We're taking the mean of those points
df_telemetry = pd.pivot_table(df_telemetry,
    values='FieldData',
    columns='FieldId',
    index=['utc_field_timestamp','ShiftId','FieldX', 'FieldY'],
    aggfunc='mean')

cols = {"42010001": "rot", "42010005": "pull",
        "42010008": "air", "4201000B": "vvib",
        "4201000C": "hvib", "4201000E": "water",
        "4201000F": "depth", "42010010": "head"}
df_telemetry = df_telemetry.rename(columns=cols)

print(df_telemetry.shape)

# Identify signals purely on Hole Depth
df_telemetry.sort_index(inplace=True)
df_telemetry["depth_diff"] = df_telemetry.depth.diff().fillna(0)
df_telemetry["hole_index"] = df_telemetry.depth_diff < max_diff
df_telemetry["hole_index"] = df_telemetry.hole_index * 1
df_telemetry["hole_index"] = df_telemetry.hole_index.cumsum()
print("Number of holes: ", df_telemetry.hole_index.nunique())

# Getting info from individual holes
df_telem_drills = (df_telemetry
    .reset_index()
    .groupby("hole_index")
    .agg({"utc_field_timestamp": ['min', 'max'],
          "depth": ['min', 'max']})
    )
df_telem_drills.columns = ["drill_start", "drill_end", "initial_depth", "final_depth"]
df_telem_drills["drilling_time"] = df_telem_drills.drill_end - df_telem_drills.drill_start
df_telem_drills["drilling_depth"] = df_telem_drills.final_depth - df_telem_drills.initial_depth
df_telem_drills.reset_index(inplace=True)

# Cleaning noisy drills
df_telem_drills.dropna(inplace=True)
df_telem_drills = df_telem_drills[df_telem_drills.drilling_time > min_time]
df_telem_drills = df_telem_drills[df_telem_drills.drilling_depth > min_depth]

print("Number of holes after cleaning: ", df_telem_drills.shape)







## need to add train test split
