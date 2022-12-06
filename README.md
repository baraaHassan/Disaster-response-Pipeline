# Disaster Response Pipeline Project
Analyzing disaster data from [Appen](https://appen.com/) to build a model for an API that classifies disaster messages.

The project consists of:
- ETL pipeline: The first part of the data pipeline is the Extract, Transform, and Load process. Here, the dataset is being read, cleaned, and then stored it in a SQLite database.
- Machine Learning Pipeline: The machine learning portion, data us being split into a training set and a test set. Then, a machine learning pipeline is being created that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification).
- Flask app: In the last step, the result will bed displayed in a Flask web app.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
