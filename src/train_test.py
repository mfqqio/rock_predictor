#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in May 2019
@author: Shayne Andrews

Separates train and test data with stratification.
Input: labelled_holes.csv (n rows, m columns)
Output: labelled_holes_tts.csv (n rows, m+1 columns)

Sample terminal call: python3 train_test.py joined_data.csv litho_rock_type 0.2 train.csv test.csv
"""
import sys, random
import pandas as pd

input_file = sys.argv[1]
stratify_by = sys.argv[2]
test_prop = float(sys.argv[3])
output_file_train = sys.argv[4]
output_file_test = sys.argv[5]

df_input = pd.read_csv(input_file, low_memory=False)
print("Input file loaded...")
strat_values = df_input[stratify_by].unique()
test_holes = []

for s in strat_values:
    strat_holes = df_input[df_input[stratify_by]==s]["hole_id"].unique()
    n_test = round(len(strat_holes) * test_prop)
    test_holes.extend(random.sample(list(strat_holes), n_test))

df_train = df_input[~df_input["hole_id"].isin(test_holes)]
df_test = df_input[df_input["hole_id"].isin(test_holes)]

print("Saving output files...")
df_train.to_csv(output_file_train, index=False)
df_test.to_csv(output_file_test, index=False)

print("Train-test split complete!")
