#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:22:39 2019

@author: Carrie Cheung
"""

import pytest
import pandas as pd
import numpy as np
import os, sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

# File to test
from create_nontelem_features import create_nontelem_features, calc_penetration_rate

# Below are test stubs to be filled in

def test_calc_penrate_out_df():
    """
    Test that output is a dataframe object
    """

def test_calc_penrate_out_feature():
    """
    Test that output contains a new column called
    'penetration_rate_mph' of numerics (floats)
    """

def test_create_nontelem_features_out_df():
    """
    Test that output is a dataframe object
    """

def test_create_nontelem_features_out_holeid():
    """
    Test that 'hole_id' is a column in the output df
    and that it contains unique strings (no duplicate hole ids in the table)
    """

def test_create_nontelem_features_out_drillop():
    """
    Test that 'drill_operator' is a column in the output df
    and that it contains strings
    """

def test_create_nontelem_features_out_drilltime():
    """
    Test that 'total_drill_time' is a column in the output df
    and that it contains floats
    """

def test_create_nontelem_features_out_redrill():
    """
    Test that 'redrill_flag' is a column in the output df
    and that it contains 0 or 1 (binary)
    """
