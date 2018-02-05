
# coding: utf-8

# In[ ]:

# Imports
from collections import defaultdict
import csv
import dateutil.parser
import HTMLParser
import nltk
from nltk.tokenize import casual_tokenize
import numpy as np


# In[ ]:

# Constants
PATH = '../../../../'
EMBEDDINGS_FILE = 'glove.twitter.27B/glove.twitter.27B.50d.txt'
MISSING_WORDS_FILE = 'missing_words.txt'
WORDS_FILE = 'words.txt'
TWEET_CNTS_FILE = 'tweet_counts.txt'

# In[ ]:

# Global variables
word_to_idx = dict()


# In[ ]:

def create_words_file():
    print 'Begin creating words file...'
    with open(WORDS_FILE, 'w') as wf:
        with open(PATH + EMBEDDINGS_FILE, 'r') as ef:
            idx = 0
            for line in ef:
                word = line.split()[0]
                wf.write(word + '\n')
                word_to_idx[word] = idx
                idx += 1
    print 'Finished creating words file!'

def output_tweet_counts(month_to_tweets):
    with open(TWEET_CNTS_FILE, 'w') as f:
        f.write(','.join([str(len(month_to_tweets[month])) for month in month_to_tweets]))

def output_missing_words(missing_words):
    with open(MISSING_WORDS_FILE, 'w') as f:
        for word in missing_words:
            f.write(word + '\n')

# In[ ]:

def output_embeddings(month_to_tweets, title):
    print 'Begin outputting word embeddings...'
    for month in month_to_tweets:
        with open(str(month) + '_embeddings_' + title + '.csv', 'w') as output:
            for tweet in month_to_tweets[month]:
                output.write(','.join([str(num) for num in tweet]) + '\n')
    print 'Finished outputting word embeddings!'


# In[ ]:

def get_word_embeddings():
    print 'Begin getting word embeddings...'
    tweets_file = 'Delhi_tweets.csv'
    title = 'delhi'
    cnt = 0
    with open(PATH + tweets_file) as csvfile:
        month_to_tweets = defaultdict(list)
        reader = csv.DictReader(csvfile)
        missing_words = set()
        missing_words_cnt = 0
        for row in reader:
            if (cnt % 100000 == 0): print str(cnt) + ' tweets processed...'
            cnt += 1
            date = dateutil.parser.parse(row['postedTime'])
            month_idx = (date.year - 2014)*12 + (date.month - 1)
            tweet = ' '.join([word for word in casual_tokenize(row['tweet']) 
                              if '@' not in word and 'http' not in word and '#' not in word])
            tweet = tweet.lower()
            tweet_embedding = []
            for word in tweet.split():
                if (word in word_to_idx):
                    tweet_embedding.append(word_to_idx[word])
                else:
                    missing_words_cnt += 1
                    missing_words.add(word)
            if (len(tweet_embedding) > 0): month_to_tweets[month_idx].append(tweet_embedding)
        output_embeddings(month_to_tweets, title)
        output_tweet_counts(month_to_tweets)
        output_missing_words(missing_words)
    print 'Finished getting word embeddings!'
    print 'Total number of missing words: ' + str(missing_words_cnt)


# In[ ]:

def main():
    create_words_file()
    get_word_embeddings()


# In[ ]:

if (__name__ == '__main__'):
    main()


# In[ ]:



