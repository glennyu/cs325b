import collections
import csv
import html
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import string
import os

PATH = './'#'/mnt/mounted_bucket/'
SAMPLING_RATE = 0.1

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
                numEntriesRead, index = 0, 0
                for row in reader:
                    if index in indices:
                        tokens = word_tokenize(html.unescape(row["tweet"]))
                        tokens = [w.lower() for w in tokens]
                        words = [word for word in tokens if word.isalpha()]
                        # filter out stop words
                        # from nltk.corpus import stopwords
                        # stop_words = set(stopwords.words('english'))
                        # words = [w for w in words if not w in stop_words]
                        tweetWordLen[len(words)] += 1
                        numEntriesRead += 1
                        if numEntriesRead % 100000 == 0:
                            print(numEntriesRead)
                            print(row["tweet"])
                            print(words)
                    index += 1

    print(tweetWordLen)

def main():
    parse_files()
    
if __name__ == "__main__":
    main()
