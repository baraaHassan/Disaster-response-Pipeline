import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
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

# load data
engine = create_engine('sqlite:////home/workspace/data/YourDatabaseName.db')
df = pd.read_sql_table('InsertTableName', engine)

# load model
model = joblib.load("/home/workspace/models/your_model_name.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    offer_counts = df.groupby('offer').count()['message']
    offer_names = list(offer_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
        ,
        {
            'data': [
                Bar(
                    x=offer_names,
                    y=offer_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Offers',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Offer"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()