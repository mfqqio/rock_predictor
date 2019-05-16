#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates features based on telemetry data for each hole.
"""

import pandas as pd
import numpy as np

def zero_water_flow(telem, hole_id_col, colname):
    
    # Select only rows where water flow is 0
    zero_water = telem[telem[colname] == 0]
    has_water = telem[telem[colname] > 0]
    
    samples_nowater = zero_water.groupby(hole_id_col).agg(['count'])[colname]
    samples_nowater.rename(columns={'count':'samples_nowater'}, inplace=True)
    
    samples_haswater = has_water.groupby(hole_id_col).agg(['count'])[colname]   
    samples_haswater.rename(columns={'count':'samples_haswater'}, inplace=True)
    
    joined = pd.merge(samples_nowater, samples_haswater,
                    how='left',
                    left_index=True,
                    right_index=True)
    joined.fillna(0, inplace=True)
    joined['total_samples'] = joined['samples_nowater'] + joined['samples_haswater']
    joined['prop_nowater'] = joined['samples_nowater']/joined['total_samples']
    
    return joined['prop_nowater'].values
