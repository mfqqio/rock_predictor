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
    input_labels = "../data/raw/190501001_COLLAR.csv"
    input_class_mapping = "../data/raw/rock_class_mapping.csv"
    input_production = "../data/raw/190506001_PVDrillProduction.csv"
    input_telemetry = "../data/raw/190416001_MCMcshiftparam.csv"
    input_telem_headers = "../data/raw/dbo.MCCONFMcparam_rawdata.csv"
    output_train = "../data/intermediate/train.csv"
    output_test = "../data/intermediate/test.csv"
else: # parse input parameters from terminal or makefile
    parser = argparse.ArgumentParser()
    parser.add_argument("input_labels")
    parser.add_argument("input_class_mapping")
    parser.add_argument("input_production")
    parser.add_argument("input_telemetry")
    parser.add_argument("input_telem_headers")
    parser.add_argument("output_train")
    parser.add_argument("output_test")
    args = parser.parse_args()

    input_labels = args.input_labels
    input_class_mapping = args.input_class_mapping
    input_production = args.input_production
    input_telemetry = args.input_telemetry
    input_telem_headers = args.input_telem_headers
    output_train = args.output_train
    output_test = args.output_test
clean.hello()
# Read all raw csv files
df_labels = pd.read_csv(input_labels)
print('Labels table dimensions:', df_labels.shape)
df_class_mapping = pd.read_csv(input_class_mapping)
print('Class mapping table dimensions:', df_class_mapping.shape)
df_production = pd.read_csv(input_production)
print('Production table dimensions:', df_production.shape)
df_telemetry = pd.read_csv(input_telemetry)
print('Telemetry table dimensions:', df_telemetry.shape)
df_telem_headers = pd.read_csv(input_telem_headers)
print('Telemetry headers table dimensions:', df_telem_headers.shape)

# Cleaning df_labels and join with df_class_mapping
df_labels = df_labels[['hole_id', 'x', 'y', 'z', 'COLLAR_TYPE', 'LITHO', 'PLANNED_RTYPE']]
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

## need to test for duplicates

df_production['hole_id'] = clean.create_hole_id(df_production['DrillPattern'], df_production['HoleId'])
exclude_cols = ['DrillPattern', 'HoleID']
df_production = df_production.loc[:, ~df_production.columns.isin(exclude_cols)]
df_production['unix_start'] = clean.convert_utc2unix(df_production['UTCStartTime'])
df_production['unix_end'] = clean.convert_utc2unix(df_production['UTCEndTime'])

## need to drop redrills (duplicates) from df_production - let's use Gabriel's logic

df_joined_holes = pd.merge(df_production, df_labels, how='left', left_on=['hole_id'], right_on = ['hole_id'])
print('Joined labels + production dimensions:', df_joined_holes.shape)

print(df_joined_holes.head())
## Gabriel to add cleaning logic here

## to include call to clean.make_wide

## need to add train test split
