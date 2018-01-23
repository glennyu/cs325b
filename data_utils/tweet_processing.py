import collections
import csv
import html
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import string
import os

PATH = './'#'/mnt/mounted_bucket/'
SAMPLING_RATE = 0.01
SEED = 325

keys = ["id", "tweet", "postedTime", "actorLoc", "locName", "locGeoType", "areakm2", "lng", "lat"]
keywords = set(['lentils', 'oil', 'soybean', 'soybeans', 'moong', 'wheat', 'salt', 'masur', 'flour', 'milk', 'pasteurized', 'sugar', 'palm', 'sunflower', 'jaggery', 'gur', 'tea', 'urad','potato', 'potatoes', 'ghee', 'vanaspati', 'mustard', 'rice', 'onion', 'onions', 'tomato', 'tomatoes', 'groundnut', 'groundnuts'])

def create_files():
    np.random.seed(SEED)
    sampledTweetFile = open("../sampled_tweets.csv", "w")
    sampledTweetWriter = csv.DictWriter(sampledTweetFile, keys)
    sampledTweetWriter.writeheader()
    geoTweetFile = open("../geo_tweets.csv", "w")
    geoTweetWriter = csv.DictWriter(geoTweetFile, keys)
    geoTweetWriter.writeheader()

    for filename in os.listdir(PATH):
        if filename.endswith(".csv"):
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

def tweet_length_distribution(tweetWordLen):
    tweetWordLen = dict(tweetWordLen)
    maxLen = max(tweetWordLen.keys())
    wordLenX, wordLenY = [], []
    for i in range(1, maxLen + 1):
        wordLenX.append(i)
        wordLenY.append(tweetWordLen[i] if i in tweetWordLen else 0)
    plt.bar(wordLenX, wordLenY, align='center', alpha=0.5)
    plt.xticks(wordLenX, wordLenX)
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
    for filename in os.listdir(PATH):
        if filename.endswith(".csv") and filename == 'sampled_tweets.csv':
            with open(PATH + filename) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    tokens = word_tokenize(html.unescape(row["tweet"]))
                    tokens = [w.lower() for w in tokens]
                    words = [word for word in tokens if word.isalpha()]
                    #from nltk.corpus import stopwords
                    #stop_words = set(stopwords.words('english'))
                    #words = [w for w in words if not w in stop_words]
                    tweetWordLen[len(words)] += 1
                    for word in words:
                        wordCnt[word] += 1

    tweet_length_distribution(tweetWordLen)
    word_distribution(wordCnt)
    
def main():
    #create_files()
    parse_files()
    
if __name__ == "__main__":
    main()
