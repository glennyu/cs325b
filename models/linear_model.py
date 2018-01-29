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

def get_prices(city):
    with open(PATH + 'India_Food_Prices.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        food_to_prices = defaultdict(list)
        for row in reader:
            if (row[CITY_COL] == city):
                month, year = int(row[MONTH_COL]), int(row[YEAR_COL])
                if ((year >= START_YEAR and year < END_YEAR) or (year == END_YEAR and month <= END_MONTH)):
                    food = row[FOOD_TYPE_COL]
                    price = float(row[FOOD_PRICE_COL])
                    food_to_prices[food].append(price)

        city_prices = []
        for food in food_to_prices:	   
            city_prices.append([food] + food_to_prices[food])
        city_prices = np.array(city_prices)
        assert(city_prices.shape == (21, 36))
        return city_prices

def lin_reg(sentiment_feat, food_cnt):
    feat = np.hstack((sentiment_feat, food_cnt))
    prices = get_prices("Delhi")
    print(prices)
    Ridge(fit_intercept=False)
    reg.fit(feat, prices)
    print(reg.coef_)
    print(reg.score(feat, prices))

def main():
    #get_features()
    sentiment_feat, food_cnt = read_features()
    #for i in range(len(food_names)):
    lin_reg(sentiment_feat[11:12].reshape((NUM_MONTHS, 4)), food_cnt[11:12].reshape((NUM_MONTHS, 1)))

if __name__ == "__main__":
    main()
