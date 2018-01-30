import matplotlib
matplotlib.use('Agg')
from collections import defaultdict
import csv
import dateutil.parser
import html
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.linear_model import Ridge
import string
import os
import sys
sys.path.insert(0, '../data_utils')
from util import *

PATH = '../data_utils/'#'/mnt/mounted_bucket/'
NUM_MONTHS = 35
food_names = ['lentils', 'oil', 'wheat', 'salt', 'wheat flour', 'milk', 'sugar', 'black tea', 'potato', 'ghee', 'rice', 'onions', 'tomato']
food_to_predict = ['Lentils', 'Wheat', 'Salt (iodised)', 'Lentils (masur)', 'Sugar', 'Tea (black)', 'Potatoes', 'Oil (mustard)', 'Rice', 'Onions', 'Milk (pasteurized)', 'Tomatoes']
food_to_index = [0, 2, 3, 0, 6, 7, 8, 1, 10, 11, 5, 12]

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
    
def read_features():
    sentiment_feat = np.zeros((len(food_names), NUM_MONTHS, 4), dtype=np.float32)
    food_cnt = np.zeros((len(food_names), NUM_MONTHS), dtype=np.float32)
    f_sentiment = open("Delhi_sentiment_scores.txt", "r")
    f_cnt = open("Delhi_food_count.txt", "r")
    for i in range(len(food_names)):
        for j in range(NUM_MONTHS):
            line = f_sentiment.readline().split()
            sentiment_feat[i][j] = np.array([float(x) for x in line])
            line = f_cnt.readline()
            food_cnt[i][j] = float(line)
    f_sentiment.close()
    f_cnt.close()
    return sentiment_feat, food_cnt

def lin_reg(food_idx, sentiment_feat, food_cnt, prices):
    feat = np.hstack((sentiment_feat, np.expand_dims(food_cnt, axis=1)))
    train_feat, test_feat = feat[:24], feat[24:]
    train_prices, test_prices = prices[:24], prices[24:]

    reg = Ridge()
    reg.fit(train_feat, train_prices)
    print(food_to_predict[food_idx])
    #print(reg.coef_)
    print("score: %f" % reg.score(train_feat, train_prices))
    pred = reg.predict(test_feat)
    print("MAPE: %f" % mape(test_prices, pred)) 
    plot_price_trend(2*food_idx, reg.predict(train_feat), train_prices, food_to_predict[food_idx] + "_train_sentiment")
    plot_price_trend(2*food_idx + 1, pred, test_prices, food_to_predict[food_idx] + "_test_sentiment")

def main():
    #get_features()
    sentiment_feat, food_cnt = read_features()
    prices = get_prices("Delhi")
    for i in range(len(food_to_predict)):
        idx = food_to_index[i]
        lin_reg(i, sentiment_feat[idx], food_cnt[idx], prices[i].T)

if __name__ == "__main__":
    main()
