import numpy as np

PATH = '../data/'
K = 3.0 # threshold distance for nearest neighbors search

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
        for neighbor in word_to_embedding:
            dist = np.linalg.norm(np.array(word_to_embedding[neighbor]) - np.array(word_to_embedding[word]))
            if (dist <= K):
                print 'word: %s, dist: %f' % (neighbor, dist)
        print ' -done'

def main():
    read_file(PATH + 'glove.twitter.27B.50d.txt')
    find_nearest_words('onion')

if __name__ == '__main__':
    main()

