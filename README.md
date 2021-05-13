# Disaster-Response-Pipelines

### Description
This Project is part of the Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains pre-labelled tweet and messages from real-life disaster events. The goal of the project is is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.

This project is divided in the following key sections:

Processing data: An ETL pipeline is built to extract data from source, clean and save the data in an SQLite DB

Model Build: A ML pipeline is built to train the model to classify text message.

Web app: Shows model results in real time

### Getting Started

#### Dependencies
Python 3.5+
Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
Natural Language Process Libraries: NLTK
SQLlite Database Libraqries: SQLalchemy
Model Loading and Saving Library: Pickle
Web App and Data Visualization: Flask, Plotly

### Installing
To clone the git repository:

git clone https://github.com/Pullooo/Disaster_Response_Pipeline.git

### Executing Program:
You can run the following commands in the project's directory to set up the database, train model and save the model.

To run ETL pipeline to clean data and store the processed data in the database python data/process_data.py data/disaster_messages.csv data/categories.csv data/disaster_response_db.db
To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl
Run the following command in the app's directory to run your web app. python run.py

To see this live, copy and paste this in your web browser http://0.0.0.0:3001/


#### Additional Material
In the data and models folder you can find two jupyter notebook that will help you understand how the model works step by step:

ETL Preparation Notebook: learn everything about the implemented ETL pipeline
ML Pipeline Preparation Notebook: look at the Machine Learning Pipeline developed with NLTK and Scikit-Learn
You can use ML Pipeline Preparation Notebook to re-train the model or tune it through a dedicated Grid Search section.


#### Important Files
app/templates/*: templates/html files for web app

data/process_data.py: Extract Train Load (ETL) pipeline used for data cleaning, feature extraction, and storing data in a SQLite database

models/train_classifier.py: A machine learning pipeline that loads data, trains a model, and saves the trained model as a .pkl file for later use

run.py: This file can be used to launch the Flask web app used to classify disaster messages


#### Author
Paul Fru

