# Makefile to run pipeline for
# Rock Type Prediction from Drill Telemetry Data (QIO3)

all : doc/model_results.txt

# Split data into train and test sets
data/intermediate/train.csv : data/raw/*
	python src/process_data.py # need to add parameters later (they are hardcoded in script for now)

# EDA and sanity check report (1)
# Use train data (telemetry and non-telemetry data)
doc/eda_1.pdf : notebook/eda_1.ipynb
	jupyter nbconvert --to pdf --execute notebook/eda_1.ipynb

# Feature engineering
data/intermediate/features.csv : data/intermediate/train.csv
	python src/create_features.py data/intermediate/train.csv data/intermediate/features.csv

# EDA and sanity check report (2)

# Build model
doc/model_results.txt : data/intermediate/features.csv data/raw/explosive_by_rock_class.csv
	python src/7_build_model.py data/intermediate/features.csv data/raw/explosive_by_rock_class.csv rock_class doc/model_results.txt

# Clean up files
clean :
	rm -f data/intermediate/*
	rm -f doc/*
