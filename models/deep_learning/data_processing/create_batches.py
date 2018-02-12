
# coding: utf-8

# In[2]:

# Imports
import numpy as np


# In[3]:

# Constants
TWEET_CNTS_FILE = 'tweet_counts.txt'
NUM_MONTHS = 35
K = 50 # number of tweets per batch
NUM_RESAMPLES = 5 # number of times to resample from city-month tweets to generate batches
MIN_TWEETS = K * 1000 # minimum number of tweets per month


# In[4]:

# Get tweet counts
tweet_cnts = []
with open(TWEET_CNTS_FILE, 'r') as f:
    line = f.readline()
    tweet_cnts = [int(num) for num in line.strip().split(',')]
print tweet_cnts


# In[4]:

# Create batches for a city given a tweet count
# Inputs: tweet count (int), city name (str), month index (int), trend (int = 0, 1, 2), spike (int = 0 ,1)
# Outputs: [city name]_[month index]_batch.txt, a file containing the indexes of each tweet batch input
# Returns: number of batches created for that month
def create_batches(tweet_count, city_name, month_idx, trend, spike):
    np.random.seed(10)
    
    n = tweet_count
    rand_seq = []
    for i in range(NUM_RESAMPLES):
        cur = [str(num) for num in np.random.choice(n, n, replace=False)]
        remaining = K - (n % K) # used to make the length of rand_seq a multiple of K
        cur += [str(num) for num in np.random.choice(n, remaining, replace=False)]
        rand_seq += cur

    num_batches = len(rand_seq) / K
    for i in range(num_batches):
        folder = 'batches_train'
        if (i <= 0.7 * num_batches):
            folder = 'batches_train'
        elif (i <= 0.9 * num_batches):
            folder = 'batches_val'
        else:
            folder = 'batches_test'
        output_file = '%s%s/%s_%s_batch.txt' % (PATH, folder, city_name, str(month_idx))
        with open(output_file, 'w') as output:
                if (i == 0):
                    output.write('%d,%d\n' % (trend, spike))
                suffix = '\n'
                if (i == num_batches - 1): suffix = ''
                output.write('\t'.join(rand_seq[i * K : (i + 1) * K]) + suffix)
    
    return num_batches


# In[ ]:



