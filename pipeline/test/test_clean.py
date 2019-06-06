import pytest
import numpy as np
import pandas as pd
import os
import sys
import random, glob

# By default, python only searches for packages and modules to import
# from the directories listed in sys.path
# The idea is to add another path that inlcudes the function we want to test
# This allows us to import that function
#sys.path.append("/ubc_drilltelemetry/pipeline/src/helpers")
# os.path.dirname(os.path.dirname(os.path.realpath("./pipeline/src/helpers"))))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
# Now that we have added the path, we can import the function we want to test
from src.helpers.clean import convert_utc2unix


# TEST FUNCTION: convert_utc2unix

# Set expected UNIX time from give utc_test
utc_test = pd.DataFrame(['2019-05-23 14:51:18.417'], columns=['UTC'])
expected_UNIX = 1558623078

# Ensure that the expected UNIX time is correct given the utc_test value
def test_convert_utc2unix_output():
    result = convert_utc2unix(utc_test['UTC'], timezone='UTC')
    assert result[0] == expected_UNIX, "Failed to achieve expected UNIX time"


# TEST FUNCTION: convert_utc2unix

# test to ensure that the output is never negative
# create a series of dates and convert to string
from datetime import datetime
date_rng = pd.date_range(start='4/1/2019', end='4/08/2019', freq='S')
#print(len(date_rng))
#print(date_rng)
#convert date to string

date_rng = pd.DataFrame(date_rng, columns=['UTC'])

# ensure the result is not negative
def test_convert_utc2unix_neg():
    result = convert_utc2unix(date_rng['UTC'], timezone='UTC')
    #result *= -1
    assert np.all(result > 0)

test_convert_utc2unix_neg()
