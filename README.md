# Disaster Response Pipeline Project
### Project Summary:
One of the most important problems that we're trying to solve in Data Science and Machine Learning right now, is following a disaster.
Typically you will get millions and millions of communications either direct or via social media right at the time
when disaster response organizations have the least capacity to filter and then pull out the messages which are the most important.
Often it really is only one in every thousand messages that might be relevant to the disaster response professionals.

So, the way that disasters are typically responded to, is that different organizations will take care of different parts of the problem.
So one organization will care about water,another will care about blocked roads, another will care about medical supplies.
So, when you look at the data, you'll see that these are the categories that we have pulled out for each of these datasets.
So, we actually used Figure Eight for many of the disasters from which these messages are taken then combined these datasets and relabeled them so that
they're consistent labels across the different disasters,and this is to allow you to investigate the different trends that you
might be able to find and to build supervised Machine Learning models.
So you'll see for example, when you look at a keyword like water,that's actually a very small percent of the time that,that will map to someone who needs
fresh drinking water and also that will miss people who say they're thirsty,but don't use the word, "Water."
So, supervised Machine Learning-based approaches are going to be a lot more accurate than anyone could do with keyword searching,
but this is actually a big gap right now in disaster response contexts. It extremly critical to discover new trends and new ways of building
Machine Learning models that can help us respond to future disasters.
------------
### Project Description
In this project disaster data from [Appen](https://appen.com/) is being analysed to build a model for an API that classifies disaster messages.

The project consists of:
- ETL pipeline: The first part of the data pipeline is the Extract, Transform, and Load process. Here, the dataset is being read, cleaned, and then stored it in a SQLite database.
- Machine Learning Pipeline: The machine learning portion, data us being split into a training set and a test set. Then, a machine learning pipeline is being created that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification).
- Flask app: In the last step, the result will bed displayed in a Flask web app.

------------
### Project files

```

+-- app
|   +-- template
|   |    +--master.html # main page of web app                        
|   |    +--go.html # classification result page of web app
|   +-- run.py # Flask file that runs app
+-- data                      
|   +-- disaster_categories.csv # data to process
|   +-- disaster_messages.csv # data to process
|   +-- process_data.py
|   +-- YourDatabaseName.db  # database to save clean data to
+-- models
|   +-- train_classifier.py
|   +-- classifier.pkl # saved model
+-- README.md

```

------------
- The app folder contains all the files that is needed for the flask app to run.
- The data folder, contains the raw data in csv files, and the clean dataset that is being analysed and created in the ETL pipeline.
- The models folder, contains the machine learning pipeline, and the saved model after training.
- The readme file for the project details.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
