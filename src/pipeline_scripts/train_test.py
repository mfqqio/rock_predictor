#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in May 2019
@author: Shayne Andrews

Separates train and test data with stratification.
Input: labelled_holes.csv (n rows, m columns)
Output: labelled_holes_tts.csv (n rows, m+1 columns)
"""
import sys, random
import pandas as pd

input_file = sys.argv[1]
stratify_by = sys.argv[2]
test_prop = float(sys.argv[3])
output_file = sys.argv[4]
# output? extra column on df1 or separate files?

df_input = pd.read_csv(input_file)
strat_values = df_input[stratify_by].unique()

test_holes = []

for s in strat_values:
    strat_holes = df_input[df_input[stratify_by]==s]["hole_id"].unique()
    n_test = round(len(strat_holes) * test_prop)
    print(n_test)
    test_holes.extend(random.sample(list(strat_holes), n_test))

print(test_holes[0])
print(len(test_holes))
