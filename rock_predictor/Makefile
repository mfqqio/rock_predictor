# Makefile to run pipeline for Rock Predictor

# Selected model that will be saved as joblib file
selected_model = models/unfitted/randomforest.joblib
oversampling = SMOTE # Oversampling strategy

#### MODEL TRAINING
# To train selected model, use `make train`
train : models/fitted/final_model.joblib doc/final_eval.txt

# To update evaluation reports for all models, use `make model_evals`
model_evals : doc/eval_models.csv doc/eval_models.txt

# Defines targets for each separate step of the pipeline
# 1) Split data into train and test sets
data/pipeline/train.csv data/pipeline/test.csv : data/input_train/* data/input_mapping/* rock_predictor/process_data.py
	python rock_predictor/process_data.py for_train

# 2) Creates features for both train and test data sets
data/pipeline/train_features.csv data/pipeline/test_features.csv : data/pipeline/train.csv data/pipeline/test.csv rock_predictor/create_features.py
	python rock_predictor/create_features.py for_train

# Optional Step: Evaluates all models and outputs performance reports + confusion matrices as files
doc/eval_models.csv doc/eval_models.txt : data/pipeline/train_features.csv data/input_mapping/explosive_by_rock_class.csv models/unfitted/* rock_predictor/cv_models.py
	python -u rock_predictor/cv_models.py data/pipeline/train_features.csv data/input_mapping/explosive_by_rock_class.csv doc/eval_models.csv | tee doc/eval_models.txt

# 3) Train on entire dataset for selected model and save final model as joblib file
models/fitted/final_model.joblib doc/final_eval.txt : data/pipeline/train_features.csv data/pipeline/test_features.csv rock_predictor/train_final_model.py
	python rock_predictor/train_final_model.py $(selected_model) data/pipeline/train_features.csv data/pipeline/test_features.csv $(oversampling) > doc/final_eval.txt

#### PREDICTION USING SELECTED MODEL
## To make predictions on new data, use `make predict`
predict : data/input_predict/* data/input_mapping/* rock_predictor/process_data.py
	python rock_predictor/process_data.py for_predict # 1) Process input data for new predictions
	python rock_predictor/create_features.py for_predict # 2) Create features for new predictions
	python rock_predictor/predict.py models/fitted/final_model.joblib # 3) Make predictions using a saved model
	# Uses train & test features for web app visualization, but does not regenerate them

all : model_evals train predict

# Clean up files
clean :
	rm -rf data/pipeline/*
	rm -f doc/*
