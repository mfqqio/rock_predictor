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

# Read all raw csv files
df_labels = pd.read_csv(input_labels)
df_class_mapping = pd.read_csv(input_class_mapping)
df_production = pd.read_csv(input_production)
df_telemetry = pd.read_csv(input_telemetry)
df_telem_headers = pd.read_csv(input_telem_headers)
