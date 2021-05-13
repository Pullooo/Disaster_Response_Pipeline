import sys
import nltk
import numpy as np
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sqlalchemy import create_engine
import re
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

def load_data(database_filepath):
    """
    input:
    df_filepath = (str) filepath of sql database containing data
    input_df = (str) table to read in
    
    Output:
    X = table containing input variables i.e. the messages
    Y = Target categories of the dataset
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('df_disaster', engine)

    #drop colummns with missing values
    df.dropna(subset = ['related'], inplace = True)

    #assign X and Y
    X = np.array(df['message'])
    Y = df.iloc[:,-36:]
    cat_columns = Y.columns
    return X, Y, cat_columns

    

def tokenize(text):
    """This function takes in text as input and normalises it by removing punctuation and converting to lower case, 
    tokenizing, removing stop words, lemmatizing and then stemming. 
    
    Output: list containing list of processed words"""
    normalised_text = re.sub(r"\W"," ", text.lower())
    words = word_tokenize(normalised_text)
    
    # remove stop words to reduce dimensionality
    no_stop_words = [word for word in words if word not in stopwords.words('english')]  #tokens

    # lemmatize words - first lemmatize the noun forms
    lem_n = [WordNetLemmatizer().lemmatize(word) for word in no_stop_words]

    # now lemmatize verb forms
    lem_v = [WordNetLemmatizer().lemmatize(word, pos = 'v') for word in lem_n]

    # apply stemming to reduce words to their stems
    stems = [PorterStemmer().stem(word) for word in lem_v]
    return stems

def build_pipeline():
    pipeline = Pipeline([
        
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        
    ])
    
    return pipeline
    
def classification_rep(y_true, y_pred, cat_columns):

    #convert y_true and y_pred from numpy arrays to pandas dataframes
    y_pred = pd.DataFrame(y_pred, columns = cat_columns)
#     y_true = pd.DataFrame(y_true, columns = cat_columns)
    
    for col in cat_columns:
        class_rep = classification_report(y_true = y_true[col], y_pred = y_pred[col])
        print(col)
        print(class_rep)
        


def save_model(model, model_filepath):
    """Save model as pickle file"""
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        """Load the data, run the model and save model"""

        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

        print('Building model...')
        model = build_pipeline()

        print('checking if saved model already exist') #pickle file path should be one of input variables
        if os.path.exists(model_filepath):
            print('model already exist\n-------loading model from pickle file-------')
            with open(model_filepath, 'rb') as file:
                model = pickle.load(file)

        else:
            print('Training model...')
            model.fit(X_train, Y_train)
            print('Training complete')

            # save to a file in the current working directory
            print('saving the trained model')
            with open(model_filepath, 'wb') as file:
                pickle.dump(model, file)

        print('Evaluating model...')
        y_pred = model.predict(X_test)
        classification_rep(y_true = Y_test, y_pred=y_pred,cat_columns =category_names)
        print('evaluation complete')
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db pickle_model.pkl')


if __name__ == '__main__':
    main()
