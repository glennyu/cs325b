from collections import defaultdict
import csv
import HTMLParser
import numpy as np
import os

PATH = '../data/'
K = 4.5 # threshold distance for nearest neighbors search
DIR = '../../../data_utils/'
NUM_MONTHS = 35

word_to_embedding = dict()

def read_file(filename):
    print 'Reading file...'
    with open(filename, 'r') as f:
        cnt = 0
        for line in f:
            if (cnt % 1e5 == 0): print cnt
            line = line.strip().split()
            word = line[0]
            embedding = [float(num) for num in line[1:]]
            word_to_embedding[word] = embedding
            cnt += 1
    print ' -done'

# Outputs nearest words to word using L2 distance < K (threshold value)
def find_nearest_words(word):
    print 'Finding nearest words...'
    if (word not in word_to_embedding):
        print ' -done'
        print '%s not found in dictionary' % word
    else:
        cnt = 0
        for neighbor in word_to_embedding:
            dist = np.linalg.norm(np.array(word_to_embedding[neighbor]) - np.array(word_to_embedding[word]))
            if (dist <= K):
                cnt += 1
                print 'word: %s, dist: %f' % (neighbor, dist)
        print '%d neighbors found' % cnt
        print ' -done'

# Outputs file with related tweet counts by city and month
def get_tweet_cnts():
    onion_embedding = np.array(word_to_embedding['onion'])
    city_to_cnt = defaultdict(lambda: [0 for i in range(NUM_MONTHS)])
    for filename in os.listdir(DIR):
        city = filename.split('_')[0]
        with open(DIR + filename) as csvfile:
            if '.csv' not in filename:
                continue
            if 'India_tweets' in filename:
                continue
            if 'India_Food_Prices' in filename:
                continue
            print 'Reading file %s...' % filename
            reader = csv.DictReader(csvfile)
            for row in reader:
                tweet = [word for word in HTMLParser.HTMLParser().unescape(row['tweet']).lower().split() if '@' not in word and 'http' not in word]
                time = row['postedTime']
                month = int(time[5:7])
                year = int(time[:4])
                idx = (year - 2014)*12 + (month - 1)
                for word in tweet:
                    if (word in word_to_embedding):
                        embedding = np.array(word_to_embedding[word])
                        dist = np.linalg.norm(embedding - onion_embedding)
                        if (dist <= K):
                            city_to_cnt[city][idx] += 1
            print ' -done'
    
    with open('related_tweet_counts.txt', 'w') as output:
        for city in city_to_cnt:
            cnts = [str(cnt) for cnt in city_to_cnt[city]]
            line = '\t'.join([city] + cnts) + '\n'
            output.write(line)

def main():
    read_file(PATH + 'glove.twitter.27B.50d.txt')
    find_nearest_words('onion')
    #get_tweet_cnts()

if __name__ == '__main__':
    main()

