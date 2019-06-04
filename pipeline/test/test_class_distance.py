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
from src.helpers.feature_eng import class_distance

# Dummy input DataFrame
frames = pd.DataFrame({'rock_class': ["LIM","QZ", "AMP","IF", "LIM","QZ", "AMP","IF"],
                       'exp_rock_class': ["LIM","QZ", "AMP","IF","LIM","QZ", "AMP","IF"],
                       'ActualX_mean': [643.94, 642.16, 654.83, 651.17, 643.99, 642.76, 624.83, 611.17],
                       'ActualY_mean': [559.98, 504.08, 513.77, 583.66, 555.98, 544.08, 511.77, 523.66]
                      })

# Let's get the output of our function with the test DataFrame
result_df = class_distance(frames)


# Test to check if the input to our function is a DataFrame
def test_correct_input():
    assert isinstance(frames, pd.DataFrame), "Input is not DataFrame"

# Test to check if the data type of the first column of the output DataFrame
# is string
#def test_first_column_is_string():
    #assert pd.api.types.is_string_dtype(result_df['columns']), "1st column data type is not string"

# Test to check if the data type of the second column of the output DataFrame
# is numeric
#def test_second_column_is_numeric():
    #assert pd.api.types.is_numeric_dtype(result_df['no_of_missing_vals']), "2nd column data type is not numeric"

# Test to check if the data type of the third column of the output DataFrame
# is numeric
#def test_third_column_is_numeric():
    #assert pd.api.types.is_numeric_dtype(result_df['perecent_missing_vals']), "3rd column data type is not numeric"
