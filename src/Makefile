# Makefile to run pipeline for
# Rock Type Prediction from Drill Telemetry Data (QIO3)



# Join hole-identified data sets (collar + PV drill), add rock class labels
labelled_holes.csv : ../data/190501001_COLLAR.csv ../data/190506001_PVDrillProduction.csv ../data/rock_class_mapping.csv
	python join_holes.py ../data/190501001_COLLAR.csv ../data/190506001_PVDrillProduction.csv ../data/rock_class_mapping.csv labelled_holes.csv

# Label telemetry data

# Split data into train and test sets

# EDA and sanity check report (1)
# Use train data (telemetry and non-telemetry data)
eda_1.pdf : eda_1.ipynb
	jupyter nbconvert --to pdf --execute eda_1.ipynb

# Feature engineering for non-telemetry data
non_telem_features.csv : ../eda_viz/join_master.csv
	python create_nontelem_features.py ../eda_viz/join_master.csv non_telem_features.csv

# Feature engineering for telemetry Data
telem_features.csv : ../eda_viz/join_master.csv
	python telem_features.py ../eda_viz/join_master.csv telem_features.csv

# Join and prepare inputs for model

# EDA and sanity check report (2)

# Build model

# Clean up files
clean :
	rm -f labelled_holes.csv
	rm -f non_telem_features.csv
	rm -f telem_features.csv
	rm -f eda_1.pdf
