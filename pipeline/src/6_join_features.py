#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 13:46 2019

@author: Jim Pushor

Joins the telemetry and non-telemetry features of train df
"""

# import the libraries
import pandas as pd
import sys


def join_feat(non_telem_df, telem_df):
    final = pd.merge(non_telem_df, telem_df, how='left', right_on=['hole_id'], left_on=['hole_id'])
    return final

#### MAIN
# First check if command line arguments are provided before launching main script
if len(sys.argv) == 4:
    telem_path = sys.argv[1]
    non_telem_path = sys.argv[2]
    output_file_path = sys.argv[3]

    telem_df = pd.read_csv(telem_path, low_memory=False)
    non_telem_df = pd.read_csv(non_telem_path, low_memory=False)
    print('Telem table dimensions:', telem_df.shape)
    print('Non-telem table dimensions:', non_telem_df.shape)


    join_features = join_feat(non_telem_df, telem_df)

    # Output calculated features to file
    join_features.to_csv(output_file_path, index=False)

    print('Telemetry features and non-telemetry features joined and written to file')
