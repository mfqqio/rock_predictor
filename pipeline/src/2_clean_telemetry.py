#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in May 2019
@author: Shayne Andrews

Pre-process telemetry data to put into wide format and remove noise.
Input: raw telemetry data in long format
Output: wide format telemetry data with noise removed

Sample: python3 src/2_clean_telemetry.py data/raw/190416001_MCMcshiftparam.csv data/raw/dbo.MCCONFMcparam_rawdata.csv data/intermediate/2_telemetry_cleaned.csv
"""
import pandas as pd
import numpy as np
import argparse
import clean

parser = argparse.ArgumentParser()
parser.add_argument("input_telemetry_file")
parser.add_argument("input_mapping_file")
parser.add_argument("output_file")

args = parser.parse_args()

input_telemetry_file = args.input_telemetry_file
input_mapping_file = args.input_mapping_file
output_file = args.output_file

clean.hello()

# # ## Step 1 - Load telemetry data and perform preliminary cleaning
# # #### 1.1 Load telemetry data ignoring useless columns
#
# usecols = ['FieldTimestamp', 'ShiftId', 'FieldStatus', 'FieldId', 'FieldData', 'FieldX', 'FieldY']
# df = pd.read_csv(input_telemetry_file, usecols=usecols, dtype={"FieldOperid": object})
#
# # #### 1.2 Join FieldDesc and correct timezone
#
# # Join FieldDesc
# df_field_names = pd.read_csv(input_mapping_file)
# df_field_names = df_field_names[["FieldId", "telem_short"]].set_index("FieldId")
# df = df.join(df_field_names, on="FieldId", how="left")
#
# #FieldTimestamp is not in UTC. Let's correct that
# df["utc_field_timestamp"] = pd.to_datetime(df.FieldTimestamp, unit="s")
# df["utc_field_timestamp"] = df.utc_field_timestamp.dt.tz_localize("Canada/Eastern")
# df["utc_field_timestamp"] = df.utc_field_timestamp.dt.tz_convert(None)
# df["utc_field_timestamp"] = (df.utc_field_timestamp - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
#
# df = pd.pivot_table(df, values='FieldData', columns='telem_short', index=['FieldTimestamp','utc_field_timestamp','ShiftId','FieldX', 'FieldY'])
#
# print(df.head(5))
# print(df[pd.isnull(df).any(axis=1)])
