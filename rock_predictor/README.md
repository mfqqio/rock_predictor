# Rock Predictor: Predicting Rock Type from Drill Telemetry Data

## Summary
Rock Predictor is a Python package that leverages telemetry data collected from iron ore mining drills to predict the most likely type of rock that was drilled. It provides a pipeline that processes and integrates the telemetry data in order to create features which are used to train a machine learning model. This model can then be used to produce rock type predictions, which are visualized by a [Dash](https://plot.ly/products/dash/) web application.

This project was completed as a 2019 capstone project for the [UBC Master of Data Science](https://masterdatascience.ubc.ca/) program, in partnership with [Quebec Iron Ore](https://mineraiferquebec.com/?lang=en).

## Motivation
During the production phase of mining iron ore, explosive blasts fragment rock surfaces so that rock material can be extracted. Blast designs optimized for the rock type are used to maximize the extraction of material. However, inaccurate knowledge about the expected rock type can lead to designs resulting in sub-optimal rock fragment size after blasting. This leads to additional costs and production delays in downstream production processes.

The project's objective is to mitigate these costs by predicting the rock type as soon as possible after/while drilling blast holes. This allows for an opportunity to update blast designs before the explosive charges are loaded in the blast holes.

## Installation Instructions

*Note: The below instructions assumes you have Python 3 installed on your computer. If you need to install Python, you can refer to [Anaconda](https://www.anaconda.com/distribution/#macos), an easy-to-install distribution of Python.*

First, obtain the code by cloning the repository. Open up the command line console and enter the following commands:

```
$ git clone https://github.com/carrieklc/test-repo3
$ cd test-repo3
```

Next, create the directories where input data will be located and intermediary outputs from the pipeline will be saved (***Note: this will not be needed once we have dummy data provided in the folders***).
```
$ mkdir doc \
  data \
  data/input_train data/input_predict data/input_mapping data/output data/pipeline \
  data/input_train/COLLAR data/input_train/MCMcshiftparam data/input_train/PVDrillProduction \
  data/input_predict/COLLAR data/input_predict/MCMcshiftparam data/input_predict/PVDrillProduction \
  models/fitted
```
Add relevant .csv data files into the `input_x` folders we just created such that each file type is placed in its own folder as per the folder name.

To allow the Rock Predictor pipeline and web app to be run from a designated virtual environment, we create a virtual environment called `rock_venv` and install the necessary packages:

```
$ python -m venv rock_venv
$ source rock_venv/bin/activate
# For Windows machines, instead use: rock_venv\Scripts\activate
$ pip install -r requirements.txt
```

## Running the Pipeline

The Rock Predictor pipeline uses the .csv files placed into the `input_train` and `input_mapping` folders as input and outputs a trained predictive model. The steps of the pipeline are automated through a Makefile.

You will need “GNU Make” installed on your computer to run Make. To see if you already have it installed, type `make -v` into your terminal (Linux/Mac) or `make --version` (Windows). The version will display if you have Make installed. If you need to install it, please see the “Software” section of this [reference](https://swcarpentry.github.io/make-novice/setup).

To train the model and save it as a joblib file output to the `models/fitted` folder, run the training-specific target from the Makefile in the pipeline:

```
$ make train
```

If you instead want to run a specific step of the pipeline, you can run a part of the Makefile by typing a make command that identifies the specific target you want to create - for example:

```
$ make data/pipeline/train.csv data/pipeline/test.csv
```
To get predictions from the fitted model based on input data placed in the `input_predict` folder, run the predict-specific target:

```
$ make predict
```

When you are done, exit the virtual environment:
```
deactivate
```
In the future, when you want to run the pipeline again, you can simply activate the same virtual environment and run the Makefile or parts of the Makefile, without having to re-install the requirements again.

## Running the Web App

To visualize the rock class predictions from a model created by the pipeline, we provide a [Dash](https://plot.ly/products/dash/) application that can be run locally via localhost (instructions below). If you choose, you can also deploy the app to Heroku or another platform.

Please note that web app will only launch if the pipeline has already been run to produce predictions, outputted in the form of a .csv file in the `data/output/` folder.

To run the web app locally, activate the virtual environment first (as above, if not completed already), then enter the following commands:

```
cd web_app
python app.py
```

To close down the app, hit CTRL+C in command line. If you also wish to exit the virtual environment, remember to deactivate.
