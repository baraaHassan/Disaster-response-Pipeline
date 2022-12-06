import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import pickle
import re
import string
# import libraries
import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
    '''
    Load the data from databaset.

            Parameters:
                    database file path (str): path to the databaset

            Returns:
                    return training data (X), label (Y), and the categories names
    '''
    # load data from database
    engine = create_engine('sqlite:///data/YourDatabaseName.db')
    df = pd.read_sql_table('InsertTableName', engine)

    remove_indx = np.array([])
    
    # Removing messages that contain only spaces!
    _,indeces = np.unique(df.message.values,return_inverse=True)
    indeces_of_the_empty_messages = np.where((indeces==0)|(indeces==1)|(indeces==2))[0]
    remove_indx = np.append(remove_indx,indeces_of_the_empty_messages)

    # Removing nan catogries
    for col_ind in range(4,40):
        remove_indx = np.append(remove_indx,df[df[df.columns[col_ind]].isna()].index.values)

    df.drop(np.unique(remove_indx), inplace=True)

    X = df.message.values
    Y = df[df.columns[4:]].values
    
    return X, Y, df.columns[4:]

def tokenize(text):
    '''
    tokenize the messages.

            Parameters:
                    text (str): message

            Returns:
                    return clean tokens from the input message
    '''
    # find URLs
    detected_urls = re.findall(url_regex, text)
    
    # Replace the URLs with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Remove numbers from the text
    text = re.sub(r'\d+', '', text)
    
    # Remove panctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # clean tokens of the messages
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok.lower().strip())
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    build the model.

            Returns:
                    The model pipeline.
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return pipeline 
    
    


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    build the model.
            
            Parameters:
                    model (sklearn model): the trained model
                    X_test (numpy array): test input data
                    Y_test (numpy array): test label data
                    category_names (list of strings): list of the categories names
            
            Returns:
                    The model pipeline.
    '''
    Y_pred = pipeline.predict(X_test)
    
    for i in range(36):
        print("report of the category",category_names[i],classification_report(Y_test[:,i], Y_pred[:,i]))
    
    




def save_model(model, model_filepath):
    '''
    build the model.
            
            Parameters:
                    model (sklearn model): the trained model
                    model_filepath (str): the path where the model will be saved
            
            Returns:
                    The model pipeline.
    '''
    
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()