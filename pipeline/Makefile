# Makefile to run pipeline for
# Rock Type Prediction from Drill Telemetry Data (QIO3)

all : doc/model_results.txt

#### TRAINING
# Split data into train and test sets
data/train.csv : ../data/train src/process_data.py src/helpers/clean.py
	python src/process_data.py ../data/train for_train

# EDA and sanity check report (1)
# Use train data (telemetry and non-telemetry data)
doc/eda_1.pdf : notebook/eda_1.ipynb
	jupyter nbconvert --to pdf --execute notebook/eda_1.ipynb

# Feature engineering
data/features.csv : data/train.csv src/create_features.py src/helpers/feature_eng.py
	python src/create_features.py data/train.csv data/features.csv for_train

# EDA and sanity check report (2)

# Build model
doc/eval_models.csv : data/features.csv ../data/business/explosive_by_rock_class.csv src/cv_models.py src/helpers/model.py
	python src/cv_models.py data/features.csv ../data/business/explosive_by_rock_class.csv doc/eval_models.csv

#### PREDICTION
# Process input data for new predictions
data/predict_data.csv : src/process_data.py ../data/predict
	python src/process_data.py ../data/predict for_predict

# Create features for new predictions
data/predict_features.csv : src/create_features.py data/predict_data.csv
	python src/create_features.py data/predict_data.csv data/predict_features.csv for_predict

# Make predictions using a saved model
data/predictions.csv : src/predict.py models/dummy_model data/predict_features.py
	python src/predict.py models/dummy_model data/predict_features.csv data/predictions.csv

# Clean up files
clean :
	rm -f data/*
	rm -f doc/*
