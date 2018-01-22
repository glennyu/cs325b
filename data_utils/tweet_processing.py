import collections
import csv
import nltk
from nltk.tokenize import word_tokenize
import string
import os

PATH = '/mnt/mounted_bucket/'

def parse_files():
    tweetWordLen = collections.defaultdict(int)
    for filename in os.listdir(PATH):
        if filename.endswith(".csv") and filename == 'tweets_201401.csv':
            with open(PATH + filename) as csvfile:
                reader = csv.DictReader(csvfile)
                numRowsRead = 0
                for row in reader:
                    tokens = word_tokenize(row["tweet"])
                    tokens = [w.lower() for w in tokens]
                    table = str.maketrans('', '', string.punctuation)
                    stripped = [w.translate(table) for w in tokens]
                    words = [word for word in stripped if word.isalpha()]
                    # filter out stop words
                    #from nltk.corpus import stopwords
                    #stop_words = set(stopwords.words('english'))
                    #words = [w for w in words if not w in stop_words]
                    tweetWordLen[len(words)] += 1
                    numRowsRead += 1
                    if numRowsRead % 1000000 == 0:
                        print(numRowsRead)
    print(tweetWordLen)

def main():
    parse_files()
    
if __name__ == "__main__":
    main()
