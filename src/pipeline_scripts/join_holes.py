#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:54:08 2019

@author: Carrie Cheung

Joins hole-identified data, specifically collar table and PV drill data. 
"""

import pandas as pd
import sys


#### HELPER FUNCTIONS
# Based on mapping specified in rock_class_mapping.csv, 
# label records in collar dataframe with appropriate rock class
# for both litho and exploratory rock types.
def label_rock_class(map_path, df):
    
    # Read rock class mapping from file
    rclass_map = pd.read_csv(map_path)
    
    # Attach litho rock class labels by joining
    collar_class = pd.merge(df, rclass_map,  
                           how='left', 
                           left_on=['litho_rock_type'], 
                           right_on = ['rock_type'])
    
    # Clarify/clean up columns
    collar_class.rename(columns={'rock_class':'litho_rock_class'},inplace=True)
    collar_class.drop('rock_type', axis=1, inplace=True)
    
    # Attach exploration rock class labels by joining
    collar_class = pd.merge(collar_class, rclass_map,  
                           how='left', 
                           left_on=['exp_rock_type'], 
                           right_on = ['rock_type'])
    
    # Clarify/clean up columns
    collar_class.rename(columns={'rock_class':'exp_rock_class'},inplace=True)
    collar_class.drop('rock_type', axis=1, inplace=True)
    
    return collar_class


# Converts UTC format to UNIX timestamp
def convert_utc2unix(utc_col, df):
    df[utc_col] = pd.to_datetime(df[utc_col])
    return (df[utc_col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')


# Read in input parameters from command line
collar_path = sys.argv[1]
pdrill_path = sys.argv[2]
rock_map_path = sys.argv[3]
output_file_path = sys.argv[4]

#### WRANGLE COLLAR DATA
# Read collar data from file
collar = pd.read_csv(collar_path)
print('Collar table dimensions:', collar.shape)


# Filter collar table to only relevant columns
collar = collar[['hole_id', 'x', 'y', 'z', 'COLLAR_TYPE', 'LITHO', 'PLANNED_RTYPE']]

# Rename collar columns to clarify column meanings
collar.rename(columns={'x':'x_collar',
                       'y':'y_collar',
                       'z':'z_collar',
                       'LITHO':'litho_rock_type',
                       'COLLAR_TYPE':'collar_type',
                       'PLANNED_RTYPE':'exp_rock_type'}, 
                 inplace=True)

# Remove any leading/trailing spaces from rock type names
collar['exp_rock_type'] = collar['exp_rock_type'].str.strip()
collar['litho_rock_type'] = collar['litho_rock_type'].str.strip()

# Check that hole IDs are unique in collar table
if len(collar['hole_id']) > len(set(collar['hole_id'])):
    print('There are duplicate hole IDs in collar table! Pipeline stopped.')
else:
    print('No duplicate hole IDs in collar table - OK, continue')
    
# Add rock class labels to collar data
collar = label_rock_class(rock_map_path, collar)


#### WRANGLE PVDRILL DATA
# Read production drill data from file, attaching header column names
pdrill_cols = ['DrillPattern','DesignX','DesignY','DesignZ','DesignDepth','ActualX','ActualY',
                   'ActualZ','ActualDepth','ColletZ','HoleID','FullName','FirstName',
                   'UTCStartTime','UTCEndTime','StartTimeStamp','EndTimeStamp','DrillTime']

pdrill = pd.read_csv(pdrill_path, 
                     names = pdrill_cols)

print('PVdrill data dimensions:', pdrill.shape)

# Create hole ID column in pdrill data using DrillPattern + Hole number
pdrill['hole_id'] = pdrill['DrillPattern'] + '-' + pdrill['HoleID']

# Filter out no longer relevant columns from pdrill
exclude_cols = ['DrillPattern', 'HoleID']
pdrill = pdrill.loc[:, ~pdrill.columns.isin(exclude_cols)]

# Convert UTC start and end times to UTC UNIX timestamps
pdrill['unix_start'] = convert_utc2unix('UTCStartTime', pdrill)
pdrill['unix_end'] = convert_utc2unix('UTCEndTime', pdrill)


# Handle duplicate hole IDs in pdrill data by creating redrill IDs
# (since duplicate hole IDs indicate redrills).
pdrill['redrill_id'] = pdrill['hole_id']
pdrill['redrill'] = 0 # Initially set to 0
pdrill.sort_values(by=['redrill_id'], inplace=True)

dupes = pd.concat(g for _, g in pdrill.groupby('redrill_id') if len(g) > 1)

# Adds additional identifier in hole ID to account for re-drills (e.g. BP-662-008-3310-1)
current_id = dupes['redrill_id'].head(1).values[0]
count = 1

for index, row in dupes.iterrows():
    if row['redrill_id'] == current_id:
        new_id = row['redrill_id'] + '-' + str(count)
        dupes.loc[index, 'redrill_id'] = new_id # Update hole ID
        count = count + 1
    else:
        current_id = row['redrill_id']
        new_id = row['redrill_id'] + '-1'
        dupes.loc[index, 'redrill_id'] = new_id # Update hole ID
        count = 2
        
# Explicity flag redrills in these duplicate cases as 1
dupes['redrill'] = 1

# Add back non-duplicate hole records to re-create pdrill dataframe
non_dupes = pd.concat(g for _, g in pdrill.groupby('redrill_id') if len(g) == 1)
pdrill_uniqueid = pd.concat([dupes, non_dupes])
print('PVdrill with unique id added dimensions:', pdrill_uniqueid.shape)

# Finally attach rock type and class labels to pvdrill data
joined_holes = pd.merge(pdrill_uniqueid, collar,  
                        how='left', 
                        left_on=['hole_id'], 
                        right_on = ['hole_id'])

print('Joined collar + PV drill dimensions:', joined_holes.shape)

# Sort by time and write results to file
joined_holes.sort_values(by=['unix_start'], inplace=True)
joined_holes.to_csv(output_file_path, index=False)

print('Joined holes data set written to file')
