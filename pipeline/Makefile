# Makefile to run pipeline for
# Rock Type Prediction from Drill Telemetry Data (QIO3)

all : doc/model_results.txt

#### TRAINING
# Split data into train and test sets
data/train.csv : src/process_data.py ../data/train
	python src/process_data.py ../data/train for_train

# EDA and sanity check report (1)
# Use train data (telemetry and non-telemetry data)
doc/eda_1.pdf : notebook/eda_1.ipynb
	jupyter nbconvert --to pdf --execute notebook/eda_1.ipynb

# Feature engineering
data/features.csv : data/train.csv src/create_features.py
	python src/create_features.py data/train.csv data/features.csv

# EDA and sanity check report (2)

#### PREDICTION
# Process input data for new predictions
data/predict_data.csv: src/process_data.py ../data/predict
	python src/process_data.py ../data/predict for_predict
	

# Build model
doc/model_results.txt : data/features.csv ../data/business/explosive_by_rock_class.csv src/build_model.py src/helpers/model.py
	python src/build_model.py data/features.csv ../data/business/explosive_by_rock_class.csv doc/model_results.txt

# Clean up files
clean :
	rm -f data/*
	rm -f doc/*
