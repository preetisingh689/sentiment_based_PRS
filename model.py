from flask import Flask, render_template, request
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn import *
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cosine
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd
import re
from pandas import DataFrame
import string
import warnings
import joblib
import nltk


nltk.download('stopwords')
warnings.filterwarnings("ignore")


def SentimentAnalysis(df):
    df['reviews_title'] = df['reviews_title'].fillna('')
    df['user_reviews'] = df[['reviews_title', 'reviews_text']].agg('. '.join, axis=1).str.lstrip('. ')

    def get_wordnet_pos(pos_tag):
        if pos_tag.startswith('J'):
            return wordnet.ADJ
        elif pos_tag.startswith('V'):
            return wordnet.VERB
        elif pos_tag.startswith('N'):
            return wordnet.NOUN
        elif pos_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
        
        
    def clean_text(text):
        # lower text
        text = text.lower()

        # tokenize text and remove puncutation
        text = [word.strip(string.punctuation) for word in text.split(" ")]
        # remove words that contain numbers
        text = [word for word in text if not any(c.isdigit() for c in word)]
        # remove stop words
        stop = stopwords.words('english')
        text = [x for x in text if x not in stop]
        # remove empty tokens
        text = [t for t in text if len(t) > 0]

        # pos tag text
        pos_tags = pos_tag(text)
        # lemmatize text
        text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
        # remove words with only one letter
        text = [t for t in text if len(t) > 1]

        # join all
        text = " ".join(text)
        return(text)
    
    df["Reviews"] = df.apply(lambda x: clean_text(x['user_reviews']),axis=1)
    df['user_reviews'] = pd.DataFrame(df.Reviews.tolist(), index=df.index)

    countVector = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), token_pattern=r'\w{1,}')
    countVector.fit(df["user_reviews"])
    tfidf = countVector.transform(df["user_reviews"])

    model = joblib.load("sentiment_model.pkl")
    result = model.predict(tfidf)
    df['prediction'] = result

    return df


def RecommendationSystem(df):
    df['avg_ratings'] = df.groupby(['id', 'reviews_username'])['reviews_rating'].transform('mean')
    df['avg_ratings'] = df['avg_ratings'].round(2)
    ratings = df.drop_duplicates(subset={"reviews_username", "id"}, keep="first")
    ratings = ratings.dropna(subset=['reviews_username'])

    train, test = train_test_split(ratings, test_size=0.30, random_state=31)
    df_pivot = train.pivot(
        index='reviews_username',
        columns='name',
        values='reviews_rating'
    ).fillna(0)

    dummy_train = train.copy()
    dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x >= 1 else 1)
    dummy_train = dummy_train.pivot(
        index='reviews_username',
        columns='name',
        values='reviews_rating'
    ).fillna(1)

    # Creating the User Similarity Matrix using pairwise_distance function.
    user_correlation = 1 - pairwise_distances(df_pivot, metric='cosine')
    user_correlation[np.isnan(user_correlation)] = 0
    # Create a user-product matrix.
    df_pivot = train.pivot(
        index='reviews_username',
        columns='name',
        values='reviews_rating'
    )
    mean = np.nanmean(df_pivot, axis=1)
    df_subtracted = (df_pivot.T - mean).T
    user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
    user_correlation[np.isnan(user_correlation)] = 0

    user_correlation[user_correlation < 0] = 0
    user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
    user_final_rating = np.multiply(user_predicted_ratings, dummy_train)
    finalratingpath = "user_final_rating.csv"
    user_final_rating.to_csv(finalratingpath)


