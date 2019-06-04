#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:09:28 2019

@author: Carrie Cheung

This script tests join_holes.py.

"""

import pytest
import pandas as pd
import numpy as np

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

# File to test
from join_holes import label_rock_class, convert_utc2unix

# Below are test stubs to be filled in

def test_label_rock_class_out_df():
    """ 
    Test that output is a dataframe object
    """

def test_label_rock_class_out_labels():
    """ 
    Test that dataframe output contains 2 new columns
    specifying rock class
    """
    
def test_convert_utc2unix_out():
    """ 
    Test that output is a series object of numerics (timestamps)
    """

def test_convert_utc2unix_out():
    """ 
    Test that output is a series object
    """
    
    

