import sys
from sqlalchemy import create_engine
import pandas as pd

def load_data(messages_filepath, categories_filepath):
    '''
    Load the data from disaster_messages.csv and disaster_categories.csv, besides merging them.

            Parameters:
                    messages_filepath (str): path to disaster_messages.csv
                    categories_filepath (str): path to disaster_categories.csv

            Returns:
                    return the merged data
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on = 'id')
    
    # create a dataframe of the 36 individual category columns
    categories = categories["categories"].str.split(";",expand=True)
    
    # select the first row of the categories dataframe
    row = categories.loc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda entry: entry[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype("str").str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype("str").astype(int)
        
        # convert the labels into binary
        categories.loc[categories[column]>=1,column] = 1
        categories.loc[categories[column]<=0,column] = 0
    
    # drop the original categories column from `df`
    df.drop(["categories"], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    return df


def clean_data(df):
    '''
    clean the dataframe from the duplicates.

            Parameters:
                    df (Dataframe): merged data

            Returns:
                    return the clean dataframe
    '''
    # drop duplicates
    df.drop(df[df.duplicated()].index, inplace=True)
    return df
    
    

def save_data(df, database_filename):
    '''
    Save the data to database.

            Parameters:
                    df (Dataframe): The dataframe to be saved to database.
                    database_filename (str): the name of the database.
                    
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('InsertTableName', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print(database_filepath)

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print(df)
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
