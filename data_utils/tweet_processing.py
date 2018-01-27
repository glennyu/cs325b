import matplotlib
matplotlib.use('Agg')
import collections
import csv
import html
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import string
import os

PATH = '/mnt/mounted_bucket/'
SAMPLING_RATE = 0.01
SEED = 325

keys = ["id", "tweet", "postedTime", "actorLoc", "locName", "locGeoType", "areakm2", "lng", "lat"]
keywords = set(['lentils', 'oil', 'soybean', 'soybeans', 'moong', 'wheat', 'salt', 'masur', 'flour', 'milk', 'pasteurized', 'sugar', 'palm', 'sunflower', 'jaggery', 'gur', 'tea', 'urad','potato', 'potatoes', 'ghee', 'vanaspati', 'mustard', 'rice', 'onion', 'onions', 'tomato', 'tomatoes', 'groundnut', 'groundnuts', 'price', 'sell', 'sells', 'sold', 'buy', 'buys', 'bought'])

def create_files():
    np.random.seed(SEED)
    sampledTweetFile = open("sampled_tweets.csv", "w")
    sampledTweetWriter = csv.DictWriter(sampledTweetFile, keys)
    sampledTweetWriter.writeheader()
    geoTweetFile = open("geo_tweets.csv", "w")
    geoTweetWriter = csv.DictWriter(geoTweetFile, keys)
    geoTweetWriter.writeheader()

    for filename in os.listdir(PATH):
        if filename.endswith(".csv") and filename.startswith("tweet"):
            print("reading", filename)
            numEntries = len(open(PATH + filename).readlines()) - 1
            numSamples = int(SAMPLING_RATE*numEntries)
            print("sampling", numSamples)
            indices = set(np.random.choice(numEntries, numSamples, replace=False))
            with open(PATH + filename) as csvfile:
                reader = csv.DictReader(csvfile)
                index = 0
                for row in reader:
                    if index in indices:
                        sampledTweetWriter.writerow(row)
                    if row["lng"] != "" and row["lat"] != "":
                        geoTweetWriter.writerow(row)
                    index += 1

    sampledTweetFile.close()
    geoTweetFile.close()

def geo_sample():
    np.random.seed(SEED)
    sampledTweetFile = open("sampled_geo_tweets.csv", "w")
    sampledTweetWriter = csv.DictWriter(sampledTweetFile, keys)
    sampledTweetWriter.writeheader()

    numEntries = len(open("geo_tweets.csv").readlines()) - 1
    numSamples = int(SAMPLING_RATE*numEntries)
    print("sampling", numSamples)
    indices = set(np.random.choice(numEntries, numSamples, replace=False))
    with open("geo_tweets.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        index = 0
        for row in reader:
            if index in indices:
                sampledTweetWriter.writerow(row)
            index += 1

    sampledTweetFile.close()

def tweet_length_distribution(tweetWordLen):
    tweetWordLen = dict(tweetWordLen)
    maxLen = max(tweetWordLen.keys())
    wordLenX, wordLenY = [], []
    for i in range(1, maxLen + 1):
        wordLenX.append(i)
        wordLenY.append(tweetWordLen[i] if i in tweetWordLen else 0)
    total = sum(wordLenY)
    sumLengths = sum([wordLenX[i]*wordLenY[i] for i in range(len(wordLenX))])
    mean = 1.0*sumLengths/total
    median, cur = 0, 0
    for i in range(len(wordLenX)):
        cur += wordLenY[i]
        if 2*cur >= total:
            median = i + 1
            break
    print("mean", mean)
    print("median", median)
    for i in range(30, len(wordLenX)):
        print(i + 1, wordLenY[i])
    plt.bar(wordLenX, wordLenY, align='center', alpha=0.5)
    plt.xlabel('# of Words')
    plt.ylabel('Frequency')
    plt.title('Tweet Word Length Distribution')
    plt.savefig('tweet_length_distribution.png', format='png')

def word_distribution(wordCnt):
    f = open("word_distribution.txt", "w")
    for word in keywords:
        f.write("%s appeared %d times\n" % (word, wordCnt[word]))
    topWords = sorted(wordCnt.items(), key=lambda x: x[1],reverse=True)[:1000]
    for word, cnt in topWords:
        f.write("%s: %d\n" % (word, cnt))
    f.close()

def parse_files():
    tweetWordLen = collections.defaultdict(int)
    wordCnt = collections.defaultdict(int)
    stop_words = set(stopwords.words('english'))
    with open("sampled_tweets.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tokens = word_tokenize(html.unescape(row["tweet"]))
            tokens = [w.lower() for w in tokens]
            words = [word for word in tokens if word.isalpha()]
            tweetWordLen[len(words)] += 1
            for word in words:
                if word not in stop_words:
                    wordCnt[word] += 1
    tweet_length_distribution(tweetWordLen)
    word_distribution(wordCnt)
    
def main():
    #create_files()
    #geo_sample()
    parse_files()
    
if __name__ == "__main__":
    main()
