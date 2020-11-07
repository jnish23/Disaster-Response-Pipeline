# Disaster Response Pipeline Project

### Project Overview
When a disaster strikes, there is often a flood of messages and communications on social media. This can be overwhelming to the organizations that are responding to the disaster. For example, and organization might only want to see the messages that deal with problems they are responsible for, or messages that are urgent and require immediate response.  

In this project, I will use data provided by Figure Eight to prepare an ETL (Extract-Transform-Load) pipeline, and then train a supervised machine learning model to classify these messages according to various categories (a message can be more than one category). Using this model, the messages could then be directed to the relevant organization. 

The model will then go on a web app built with Flask that can classify disaster messages.  

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### File Desciptions

- `process_data.py`: an ETL pipeline that
    - Loads the messages and categories datasetes. 
    - Merges the two datasets. Cleans the
    - Loads the messages and categories datasets
    - Merges the two datasets
    - Cleans the data
    - Stores it in a SQLite database
   
- `train_classifier.py`: Supervised Machine Learning model that
    - Loads the data from the SQLite database
    - Splits data into training and test set
    - Builds a text processing and machine learning pipeline
    - Trains and tunes a model using GridSearchCV
    - Outputs results on the test set
    - Exports final model as a pickle file
    
 - `run.py`: a Flask Web App that visualizes the results
 
 
 ### Licensing, Authors, Acknowledgements
 
 I want to acknowledge Udacity for the project idea, guidance, and starter code. The labeled data used in this project was provided by Figure Eight.
