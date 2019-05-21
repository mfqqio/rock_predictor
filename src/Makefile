# Makefile to run pipeline for
# Rock Type Prediction from Drill Telemetry Data (QIO3)

all : model_results.txt

# Join hole-identified data sets (collar + PV drill), add rock class labels
labelled_holes.csv : ../data/190501001_COLLAR.csv ../data/190506001_PVDrillProduction.csv ../data/rock_class_mapping.csv
	python join_holes.py ../data/190501001_COLLAR.csv ../data/190506001_PVDrillProduction.csv ../data/rock_class_mapping.csv labelled_holes.csv

# Label telemetry data
# joined_data.csv : list dependencies here
# insert python script to run here

# Split data into train and test sets
train.csv : joined_data.csv
	python train_test.py joined_data.csv litho_rock_type 0.2 train.csv test.csv

# EDA and sanity check report (1)
# Use train data (telemetry and non-telemetry data)
eda_1.pdf : eda_1.ipynb
	jupyter nbconvert --to pdf --execute eda_1.ipynb

# Feature engineering for non-telemetry data
non_telem_features.csv : train.csv
	python create_nontelem_features.py train.csv non_telem_features.csv

# Feature engineering for telemetry data
telem_features.csv : train.csv
	python telem_features.py train.csv telem_features.csv

# Join and prepare inputs for model
join_feat.csv : telem_features.csv non_telem_features.csv
	python join_features.py telem_features.csv non_telem_features.csv join_feat.csv

# EDA and sanity check report (2)

# Build model
model_results.txt : join_feat.csv
	python build_model.py join_feat.csv rock_class model_results.txt

# Clean up files
clean :
	rm -f labelled_holes.csv
	rm -f train.csv
	rm -f test.csv
	rm -f non_telem_features.csv
	rm -f telem_features.csv
	rm -f eda_1.pdf
	rm -f join_feat.csv
	rm -f model_results.txt
