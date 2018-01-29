import matplotlib
matplotlib.use('Agg')
import collections
import csv
import dateutil.parser
import html
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import numpy as np
import string
import os

PATH = '../data_utils/'#'/mnt/mounted_bucket/'
NUM_MONTHS = 35
food_names = ['lentils', 'oil', 'wheat', 'salt', 'wheat flour', 'milk', 'sugar', 'black tea', 'potato', 'ghee', 'rice', 'onions', 'tomato']

def get_features():
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = []
    food_cnt = []
    for i in range(len(food_names)):
        sentiment_scores.append([])
        food_cnt.append([])
        for j in range(NUM_MONTHS):
            sentiment_scores[i].append({'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0})
            food_cnt[i].append(0)
    
    f_sentiment = open("Delhi_sentiment_scores.txt", "w")
    f_cnt = open("Delhi_food_count.txt", "w")
    with open(PATH + "Delhi_tweets.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            date = dateutil.parser.parse(row["postedTime"])
            date_index = (date.year - 2014)*12 + (date.month - 1)
            
            tweet = ' '.join([word for word in html.unescape(row["tweet"]).split() if "@" not in word and "http" not in word])
            scores = sid.polarity_scores(tweet)
            tweet = tweet.lower()
            for i in range(len(food_names)):
                if food_names[i] in tweet:
                    for k, v in scores.items():
                        sentiment_scores[i][date_index][k] += v
                    food_cnt[i][date_index] += 1

    for i in range(len(food_names)):
        for j in range(NUM_MONTHS):
            for k, v in sentiment_scores[i][j].items():
                f_sentiment.write("%f " % v)
            f_sentiment.write("\n")
            f_cnt.write("%d\n" % food_cnt[i][j])
    f_sentiment.close()
    f_cnt.close()
    
get_features()
