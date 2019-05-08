# Makefile to run pipeline for
# Rock Type Prediction from Drill Telemetry Data (QIO3)

# Join hole-identified data sets (collar + PV drill), add rock class labels
labelled_holes.csv : ../../data/190501001_COLLAR.csv ../../data/190418001_PVDrillProduction.csv ../../data/rock_class_mapping.csv
	python join_holes.py ../../data/190501001_COLLAR.csv ../../data/190418001_PVDrillProduction.csv ../../data/rock_class_mapping.csv labelled_holes.csv

# Label telemetry data

# Split data into train and test sets

# EDA and sanity check report (1)
# eda_1.py /join_master.csv
# Feature engineering for non-telemetry data

# Feature engineering for telemetry Data
# feat_telem.py join_holes.py

# Join and prepare inputs for model

# EDA and sanity check report (2)

# Build model
