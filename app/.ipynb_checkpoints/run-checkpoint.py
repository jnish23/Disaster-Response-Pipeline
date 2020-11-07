import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from nltk.corpus import stopwords


app = Flask(__name__)

def tokenize(text):
    """converts text to tokens, lemmatizes, and removes stop words"""
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    filtered_text = [tok for tok in clean_tokens if tok not in stopwords.words('english')]

    return filtered_text

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Bar Graph 1 Data
    message_counts = df.iloc[:, 4:].sum().reset_index().sort_values(0, ascending=False)
    
    # Series of the possible message categories
    message_categories = message_counts['index']     
    
    # Number of messages in each category
    counts = message_counts[0]                       
    
    # Bar Graph 2 Data
    
    # Gets the word count for each message
    message_lengths = df['message'].str.split(' ').str.len().reset_index()     
    
    # X-value ranges
    bins = list(range(0, 101, 5))                                              
    
    
    # Groups the messages into the bins above based on word_count
    word_count_bins = pd.cut(message_lengths.loc[message_lengths['message']<=100, 'message'],bins=bins).value_counts().sort_index().reset_index()   
    
    # Reformats the bins into strings because JSON could not handle the interval objects
    word_count_bins['index'] = word_count_bins['index'].astype(str).str[1:-1].str.replace(',', ' -')

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=message_categories,
                    y=counts
                )
            ],

            'layout': {
                'title': {'text':'Message Category Distribution in Training Set', 'x':0.5, 'y':0.85, 'font':{'size':25}},
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Category"
                }
            }
        },
        
        {
            'data' : [
                Bar(
                    x = word_count_bins['index'],
                    y = word_count_bins['message']
                )
            ],
            
            'layout' : {
                'title' : {'text':'Message Length', 'x':0.5, 'y':0.85, 'font':{'size':25}},
                'xaxis' : {'tickmode':'linear', 'tick0':1, 'dtick':5, 'title':{'text':'Word Count', 'font':{'size':16}}},
                'yaxis' : {'title': {'text':'Frequency', 'font':{'size':16}}}
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()