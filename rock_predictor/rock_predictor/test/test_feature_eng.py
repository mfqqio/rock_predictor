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
from src.helpers.feature_eng import calc_penetration_rate, calc_prop_half, calc_prop_max, calc_prop_zero, class_distance, count_oscillations


# TEST FUNCTION: calc_prop_half test function

# Dummy input vector and set expected_prop_half given sample_vector
sample_vector = np.linspace(0, 100, num=100)
expected_prop_half = 0.5

# Test to check if the output matches expectations using sample_vector
def test_calc_prop_half_output():
    result = calc_prop_half(sample_vector)
    assert result == expected_prop_half, "Failed to achieve expected proportion"


# TEST FUNCTION: calc_prop_max

# Set expected proportion of max values based on sample_vector
expected_prop_max = 0.05

# Test to check if the output matches expectations using sample vector
def test_calc_prop_max_output():
    result = calc_prop_max(sample_vector)
    assert result == expected_prop_max, "Failed to achieve expected proportion"


# TEST FUNCTION: calc_prop_zero

# Dummy input array and expected proportion of values at zero
sample_vector_zero = np.array([2.5,4.4,0,0,6.3,9.2,0.0,0,0,3.1])
expected_prop_zero = 0.5

# Test to check if the output matches expectations using sample_vector_zero
def test_calc_prop_zero_output():
    result = calc_prop_zero(sample_vector_zero)
    assert result == expected_prop_zero, "Failed to achieve expected proportion"


# TEST FUNCTION: class_distance

# Dummy input DataFrame to calculate the distance to each nearest rock_class
frames = pd.DataFrame({'rock_class': ["LIM","QZ", "AMP","IF", "LIM","QZ", "AMP","IF"],
                       'exp_rock_class': ["LIM","QZ", "AMP","IF","LIM","QZ", "AMP","IF"],
                       'ActualX_mean': [643.94, 642.16, 654.83, 651.17, 643.99, 642.76, 624.83, 611.17],
                       'ActualY_mean': [559.98, 504.08, 513.77, 583.66, 555.98, 544.08, 511.77, 523.66]
                      })


# Test to check if the input to our function is a DataFrame


def test_class_distance_input():
    unique_labels = frames.exp_rock_class.unique()
    for label in unique_labels:
        frames["dist_"+label] = class_distance(frames.ActualX_mean, frames.ActualY_mean, frames.exp_rock_class, label)
    assert isinstance(frames, pd.DataFrame), "Input is not DataFrame"


# TEST FUNCTION: calc_penetration_rate

# Dummy input vectors and set expected_rate given the input vectors
vec_max_depth = np.array([10.0,40.0,75.0])
vec_min_depth = np.array([0,0,0])
vec_count_time = np.array([10,20,25])
expected_rate = np.array([1.0, 2.0, 3.0])

# Test to check if the output matches expectations using sample vector
def test_calc_penetration_rate_output():
    result = calc_penetration_rate(vec_max_depth, vec_min_depth, vec_count_time)
    assert np.array_equal(result, expected_rate), "Failed to achieve expected rate"


# TEST FUNCTION: count_oscillations

# Test to check if the output matches expectations using sample vector
def test_calc_prop_osc_output():
    sample_vector = pd.Series([1,2,3,4,5,4,3,2,1,3,4,5,6,7,8])
    result = count_oscillations(sample_vector)
    expected_prop = 2
    assert result == expected_prop, "Failed to achieve expected proportion"
