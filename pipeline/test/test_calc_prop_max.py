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
from src.helpers.feature_eng import calc_prop_max

# Dummy input vector
sample_vector = np.linspace(0, 100, num=100)
expected_prop = 0.05
#print(sample_vector)


# Let's get the output of our function with the test vector
result = calc_prop_max(sample_vector)
print(result)


# Test to check if the output matches expectations using sample vector
def test_calc_prop_max_output():
    assert result == expected_prop, "Failed to achieve expected proportion"
