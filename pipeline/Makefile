# Makefile to run pipeline for
# Rock Type Prediction from Drill Telemetry Data (QIO3)

all : doc/model_results.txt

# Split data into train and test sets
data/train.csv : ../data/COLLAR/* ../data/PVDrillProduction/* ../data/MCMcshiftparam/* src/process_data.py
	python src/process_data.py # need to add parameters later (they are hardcoded in script for now)

# EDA and sanity check report (1)
# Use train data (telemetry and non-telemetry data)
doc/eda_1.pdf : notebook/eda_1.ipynb
	jupyter nbconvert --to pdf --execute notebook/eda_1.ipynb

# Feature engineering
data/features.csv : data/train.csv src/create_features.py
	python src/create_features.py data/train.csv data/features.csv

# EDA and sanity check report (2)

# Build model
doc/model_results.txt : data/features.csv ../data/business/explosive_by_rock_class.csv
	python src/7_build_model.py data/features.csv ../data/business/explosive_by_rock_class.csv rock_class doc/model_results.txt

# Clean up files
clean :
	rm -f data/*
	rm -f doc/*
