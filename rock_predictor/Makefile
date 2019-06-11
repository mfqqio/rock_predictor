# Makefile to run pipeline for
# Rock Type Prediction from Drill Telemetry Data (QIO3)

train : models/fitted/final_model.joblib doc/eval_models.csv
predict : data/output/predictions.csv

#### TRAINING
# Split data into train and test sets
data/pipeline/train.csv : data/input_train data/input_mapping rock_predictor/process_data.py rock_predictor/helpers/clean.py
	python rock_predictor/process_data.py for_train

# EDA and sanity check report
doc/eda_1.pdf : notebook/eda_1.ipynb
	jupyter nbconvert --to pdf --execute notebook/eda_1.ipynb

# Feature engineering
data/pipeline/train_features.csv : data/pipeline/train.csv rock_predictor/create_features.py rock_predictor/helpers/feature_eng.py
	python rock_predictor/create_features.py for_train

# Evaluate models
doc/eval_models.csv : data/pipeline/train_features.csv data/input_mapping/explosive_by_rock_class.csv rock_predictor/cv_models.py rock_predictor/helpers/model.py models/unfitted
	python rock_predictor/cv_models.py data/pipeline/train_features.csv data/input_mapping/explosive_by_rock_class.csv doc/eval_models.csv > doc/eval_models.txt

models/fitted/final_model.joblib : rock_predictor/train_final_model.py data/pipeline/train_features.csv data/pipeline/test_features.csv
	python rock_predictor/train_final_model.py models/unfitted/rfc.joblib data/pipeline/train_features.csv data/pipeline/test_features.csv SMOTE > doc/final_eval.txt

#### PREDICTION
# Process input data for new predictions
data/pipeline/predict_data.csv : data/input_predict data/input_mapping rock_predictor/process_data.py
	python rock_predictor/process_data.py for_predict

# Create features for new predictions
data/pipeline/predict_features.csv : rock_predictor/create_features.py data/input_predict data/pipeline/predict_data.csv
	python rock_predictor/create_features.py for_predict

# Make predictions using a saved model
data/output/predictions.csv : rock_predictor/predict.py models/unfitted/rfc.joblib data/pipeline/predict_features.csv
	python rock_predictor/predict.py models/unfitted/rfc.joblib

# Clean up files
clean :
	rm -f data/pipeline/*
	rm -f doc/*
