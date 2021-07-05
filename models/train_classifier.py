import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Table, Column, Integer, MetaData
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'stopwords'])

from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
import pickle


def load_data(database_filepath):
    """Loads data from database"""
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    
    table_name = engine.table_names()[0]
    
    df = pd.read_sql_table(table_name, engine)
    
    X = df['message']
    y = df.iloc[:, 4:]
    category_names = list(y.columns)
    
    return X, y, category_names

def tokenize(text):
    """Converts text to tokens, lemmatizes, and removes stop words"""
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    
    filtered_sentence = [tok for tok in clean_tokens if tok not in stopwords.words('english')]
    
    return filtered_sentence


def build_model():
    """Creates a pipeline that converts text into word vectors, calculates Term Frequency Inverse Document Frequency, and uses Random Forest to classify message"""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {'vect__ngram_range' : [(1, 1), (1,3)],
                  'clf__estimator__min_samples_split':[2, 4]
                 }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """Prints a classification report for each target category"""
    y_preds = model.predict(X_test)
    y_preds = pd.DataFrame(y_preds, columns=category_names, dtype='int32')
    
    for c in category_names:
        print(c)
        print(classification_report(Y_test[c], y_preds[c]))


def save_model(model, model_filepath):
    """Saves model as a pkl file"""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """Combines all previous functions into one function, and checks to ensure there are three provided arguments"""
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