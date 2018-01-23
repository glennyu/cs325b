import collections
import csv
import html
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import string
import os

PATH = '/mnt/mounted_bucket/'
SAMPLING_RATE = 0.01
SEED = 325

keys = ["id", "tweet", "postedTime", "actorLoc", "locName", "locGeoType", "areakm2", "lng", "lat"]
keywords = set(['lentils', 'oil', 'soybean', 'moong', 'wheat', 'salt', 'masur', 'flour', 'milk', 'pasteurized', 'sugar', 'palm', 'sunflower', 'jaggery', 'gur', 'tea', 'urad','potatoes', 'ghee', 'vanaspati', 'mustard', 'rice', 'onions', 'tomatoe', 'groundnut'])

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

def parse_files():
    tweetWordLen = collections.defaultdict(int)
    for filename in os.listdir(PATH):
        if filename.endswith(".csv") and filename == 'tweets_201401.csv':
            print("reading", filename)
            numEntries = len(open(PATH + filename).readlines()) - 1
            numSamples = int(SAMPLING_RATE*numEntries)
            print("sampling", numSamples)
            indices = set(np.random.choice(numEntries, numSamples, replace=False))
            with open(PATH + filename) as csvfile:
                reader = csv.DictReader(csvfile)
                numEntriesRead, index, cnt = 0, 0, 0
                for row in reader:
                    if index in indices:
                        tokens = word_tokenize(html.unescape(row["tweet"]))
                        tokens = [w.lower() for w in tokens]
                        words = [word for word in tokens if word.isalpha()]
                        #filter out stop words
                        #from nltk.corpus import stopwords
                        #stop_words = set(stopwords.words('english'))
                        #words = [w for w in words if not w in stop_words]
                        tweetWordLen[len(words)] += 1
                        numEntriesRead += 1

                        good = False
                        for word in words:
                            if word in keywords:
                                good = True
                                break
                        if good:
                            if 'price' in words or 'sell' in words:
                                cnt += 1

                        if numEntriesRead % 100000 == 0:
                            print(numEntriesRead)
                            #print(row["tweet"])
                            #print(words)
                            print("good tweets", cnt)
                    index += 1

    print(tweetWordLen)

def main():
    #create_files()
    geo_sample()
    #parse_files()
    
if __name__ == "__main__":
    main()
