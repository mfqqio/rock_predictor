# Makefile to run pipeline for
# Rock Type Prediction from Drill Telemetry Data (QIO3)

all : doc/model_results.txt

# Join hole-identified data sets (collar + PV drill), add rock class labels
data/intermediate/1_labelled_holes.csv : data/raw/190501001_COLLAR.csv data/raw/190506001_PVDrillProduction.csv data/raw/rock_class_mapping.csv
	python src/1_join_holes.py data/raw/190501001_COLLAR.csv data/raw/190506001_PVDrillProduction.csv data/raw/rock_class_mapping.csv data/intermediate/1_labelled_holes.csv

# Label telemetry data
data/intermediate/2_joined_data.csv : data/raw/190416001_MCMcshiftparam.csv data/raw/dbo.MCCONFMcparam_rawdata.csv data/intermediate/1_labelled_holes.csv
	python src/2_join_telemetry_labels.py data/raw/190416001_MCMcshiftparam.csv data/raw/dbo.MCCONFMcparam_rawdata.csv data/intermediate/1_labelled_holes.csv data/intermediate/2_joined_data.csv

# Split data into train and test sets
data/intermediate/3-1_train.csv : data/intermediate/2_joined_data.csv
	python src/3_train_test.py data/intermediate/2_joined_data.csv litho_rock_type 0.2 data/intermediate/3-1_train.csv data/intermediate/3-2_test.csv

# EDA and sanity check report (1)
# Use train data (telemetry and non-telemetry data)
doc/eda_1.pdf : notebook/eda_1.ipynb
	jupyter nbconvert --to pdf --execute notebook/eda_1.ipynb

# Feature engineering for non-telemetry data
data/intermediate/4_non_telem_features.csv : data/intermediate/3-1_train.csv
	#python src/4_create_nontelem_features.py data/intermediate/3-1_train.csv data/intermediate/4_non_telem_features.csv
	python src/4_create_nontelem_features.py data/raw/sample_wide.csv data/intermediate/4_non_telem_features.csv # For now, this points to a sample wide dataset

# Feature engineering for telemetry data
data/intermediate/5_telem_features.csv : data/intermediate/3-1_train.csv
	python src/5_telem_features.py data/intermediate/3-1_train.csv data/intermediate/5_telem_features.csv

# Join and prepare inputs for model
data/intermediate/6_join_feat.csv : data/intermediate/4_non_telem_features.csv data/intermediate/5_telem_features.csv
	python src/6_join_features.py data/intermediate/5_telem_features.csv data/intermediate/4_non_telem_features.csv data/intermediate/6_join_feat.csv

# EDA and sanity check report (2)

# Build model
doc/model_results.txt : data/intermediate/6_join_feat.csv
	python src/7_build_model.py data/intermediate/6_join_feat.csv rock_class doc/model_results.txt

# Clean up files
clean :
	rm -f data/intermediate/*
	rm -f doc/*
