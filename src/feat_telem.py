#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 20:29 2019

@author: Jim Pushor

Performs preliminary EDA on joined and telemetry df. Useful for preliminary sanity check on data joins & manipulation
"""

# Preliminary EDA script
# import the required data sets
# import the libraries
import pandas as pd
import sys



# Read in input parameters from command line
df_path = sys.argv[1]
output_file_path = sys.argv[2]

df = pd.DataFrame(df_path)

def get_std(df, list_of_var):
    final = pd.DataFrame()
    for var in list_of_var:
        mask = (df.FieldDesc == var)
        df1 = df[mask]
        groups = df1.groupby('Unique_HoleID_y')
        std_results = pd.DataFrame(groups.agg('std')['FieldData'])
        std_results.columns=[var]
        final = pd.concat([final, std_results], axis=1)
    final['hole_id'] = final.index
    final = final.reset_index(drop=True)
    return final

std.to_csv(output_file_path, index=False)


# Look at distribution of other eight features from telemetry df
# num_bins = 50
# plt.hist(eda_data['Other 8 features'], num_bins, normed=1, facecolor='blue', alpha=0.5)
# plt.show()

































# # Add rock class labels to collar data
# collar = label_rock_class(rock_map_path, collar)
#
#
# #### WRANGLE PVDRILL DATA
# # Read production drill data from file, attaching header column names
# pdrill_cols = ['DrillPattern','DesignX','DesignY','DesignZ','DesignDepth','ActualX','ActualY',
#                    'ActualZ','ActualDepth','ColletZ','HoleID','FullName','FirstName',
#                    'UTCStartTime','UTCEndTime','StartTimeStamp','EndTimeStamp','DrillTime']
#
# pdrill = pd.read_csv(pdrill_path,
#                      names = pdrill_cols)
#
# print('PVdrill data dimensions:', pdrill.shape)
#
# # Create hole ID column in pdrill data using DrillPattern + Hole number
# pdrill['hole_id'] = pdrill['DrillPattern'] + '-' + pdrill['HoleID']
#
# # Filter out no longer relevant columns from pdrill
# exclude_cols = ['DrillPattern', 'HoleID']
# pdrill = pdrill.loc[:, ~pdrill.columns.isin(exclude_cols)]
#
# # Convert UTC start and end times to UTC UNIX timestamps
# pdrill['unix_start'] = convert_utc2unix('UTCStartTime', pdrill)
# pdrill['unix_end'] = convert_utc2unix('UTCEndTime', pdrill)
#
#
# # Handle duplicate hole IDs in pdrill data by creating redrill IDs
# # (since duplicate hole IDs indicate redrills).
# pdrill['redrill_id'] = pdrill['hole_id']
# pdrill['redrill'] = 0 # Initially set to 0
# pdrill.sort_values(by=['redrill_id'], inplace=True)
#
# dupes = pd.concat(g for _, g in pdrill.groupby('redrill_id') if len(g) > 1)
#
# # Adds additional identifier in hole ID to account for re-drills (e.g. BP-662-008-3310-1)
# current_id = dupes['redrill_id'].head(1).values[0]
# count = 1
#
# for index, row in dupes.iterrows():
#     if row['redrill_id'] == current_id:
#         new_id = row['redrill_id'] + '-' + str(count)
#         dupes.loc[index, 'redrill_id'] = new_id # Update hole ID
#         count = count + 1
#     else:
#         current_id = row['redrill_id']
#         new_id = row['redrill_id'] + '-1'
#         dupes.loc[index, 'redrill_id'] = new_id # Update hole ID
#         count = 2
#
# # Explicity flag redrills in these duplicate cases as 1
# dupes['redrill'] = 1
#
# # Add back non-duplicate hole records to re-create pdrill dataframe
# non_dupes = pd.concat(g for _, g in pdrill.groupby('redrill_id') if len(g) == 1)
# pdrill_uniqueid = pd.concat([dupes, non_dupes])
# print('PVdrill with unique id added dimensions:', pdrill_uniqueid.shape)
#
# # Finally attach rock type and class labels to pvdrill data
# joined_holes = pd.merge(pdrill_uniqueid, collar,
#                         how='left',
#                         left_on=['hole_id'],
#                         right_on = ['hole_id'])
#
# print('Joined collar + PV drill dimensions:', joined_holes.shape)
#
# # Sort by time and write results to file
# joined_holes.sort_values(by=['unix_start'], inplace=True)
# joined_holes.to_csv(output_file_path, index=False)
#
# print('Joined holes data set written to file')
