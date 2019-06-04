import pytest
import numpy as np
import pandas as pd
import os
import sys

# By default, python only searches for packages and modules to import
# from the directories listed in sys.path
# The idea is to add another path that inlcudes the function we want to test
# This allows us to import that function
#sys.path.append("/ubc_drilltelemetry/pipeline/src/helpers")
# os.path.dirname(os.path.dirname(os.path.realpath("./pipeline/src/helpers"))))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
# Now that we have added the path, we can import the function we want to test
from src.helpers.feature_eng import count_oscillations

# Dummy input DataFrame
sample_vector = [2.5,4.4,0,0,6.3,9.2,0.0,0,0.4,3.1]
# expected_bool = [False, False, True, True, False,False, True, True, False]
# expected_prop = 0.5
#print(sample_vector)


# Let's get the output of our function with the test DataFrame
# result = calc_prop_zero(sample_vector)


# Test to check if the output matches expectations using sample vector
# def test_calc_prop_zero_output():
#     assert result == expected_prop, "Failed to achieve expected proportion"

def count_oscillations(num_vector):
    #to avoid zeros
    diff = num_vector.diff().fillna(0)
    diff = diff.replace(0, method="ffill")

    prev_diff = diff.shift(1).fillna(0)
    prev_diff2 = diff.shift(2).fillna(0)
    prev_diff3 = diff.shift(3).fillna(0)
    post_diff = diff.shift(-1).fillna(0)
    post_diff2 = diff.shift(-2).fillna(0)
    post_diff3 = diff.shift(-3).fillna(0)

    # Changes in signal result in diff_mult being negative
    diff_mult = diff * prev_diff

    # To keep only the ones in which the change progresses
    diff_mult = diff_mult * (np.sign(prev_diff) != np.sign(diff))
    diff_mult = diff_mult * (np.sign(prev_diff2) != np.sign(diff))
    diff_mult = diff_mult * (np.sign(prev_diff3) != np.sign(diff))
    diff_mult = diff_mult * (np.sign(post_diff) == np.sign(diff))
    diff_mult = diff_mult * (np.sign(post_diff2) == np.sign(diff))
    diff_mult = diff_mult * (np.sign(post_diff3) == np.sign(diff))

    return diff_mult[diff_mult < 0].count()

count_oscillations(sample_vector)
print(sample_vector)
