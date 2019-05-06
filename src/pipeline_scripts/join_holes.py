#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:54:08 2019

@author: Carrie Cheung

Joins hole-identified data, specifically collar table and PV drill data. 
"""

import pandas as pd
import sys

# Read in input parameters from command line
collar_path = sys.argv[1]
pvdrill_path = sys.argv[2]

# Read files (collar and pvdrill)
collar_df = pd.read_csv(collar_path)
print(collar_df.shape)

pvdrill_df = pd.read_csv(pvdrill_path)
print(collar_df.shape)
