"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf
import numpy as np
import os
from random import shuffle

NUM_MONTHS = 35

def pad_tweets(tweets, max_tweet_len):
    np_tweets = np.zeros((len(tweets), max_tweet_len), dtype=np.int32)
    for i, tweet in enumerate(tweets):
        if (len(tweet) > max_tweet_len):
            print("found tweet of length", len(tweet))
        idx = 0
        while idx < len(tweet):
            np_tweets[i][idx] = tweet[idx]
            idx += 1
    return np_tweets

def get_tweet_len(tweets):
    tweet_len = [len(tweet) for tweet in tweets]
    return tweet_len

MIN_DIST = 3
ONION = 20115
TOMATO = 20279

def is_relevant(tweet, word_embeddings):
    for word in tweet:
        embedding = np.array(word_embeddings[word])
        if (np.linalg.norm(embedding - word_embeddings[ONION]) <= MIN_DIST):
            return True
    return False

def load_tweets_and_prices(path_embeddings, path_batches, word_embeddings, params):
    """Create tf.data Instance from txt file

    Args:
        path_embeddings: (string) path to embeddings
        path_batches: (string) path to batches
        params: data parameters

    Returns:
        tweets, prices: (tf.Dataset) yielding list of tweet tokens, lengths, and price deviations
    """
    tweets, prices = [], []
    tweet_month_idx, monthly_price = [], []
    label_distribution = np.zeros((3,), dtype=np.int32)
    for filename in os.listdir(path_batches):
        city_month = filename[:filename.find("batch")]
        with open(path_batches + filename, "r") as batchf:
            priceLine = True
            yVal = -1
            for batch in batchf:
                if priceLine:
                    yVal = int(batch.split(',')[3 - params.class_size]) #0 = predict price change, 1 = predict price spike
                    monthly_price.append(yVal)
                    priceLine = False
                else:
                    break
        with open(path_embeddings + city_month + "embeddings.csv", "r") as embf:
            for tweet in embf:
                cur_tweet = [int(num) for num in tweet.split(',')]
                if (is_relevant(cur_tweet, word_embeddings)):
                    tweets.append(cur_tweet)
                    tweet_month_idx.append(len(monthly_price) - 1)
                    prices.append(yVal)
                    label_distribution[yVal] += 1

    tweet_month_idx, monthly_price = np.array(tweet_month_idx), np.array(monthly_price)
    tweet_lens = np.array(get_tweet_len(tweets))
    tweets = pad_tweets(tweets, params.tweet_max_len)
    #np.random.seed(325)
    #idx = np.random.choice(5000, params.train_size)
    #tweets = tweets[idx]
    #prices = np.array(prices)[idx]
    #tweet_lens = tweet_lens[idx]
    print(tweets.shape)
    print(monthly_price.shape)
    print(label_distribution)

    tweet_len = tf.data.Dataset.from_tensor_slices(tf.constant(tweet_lens, dtype=tf.int32))
    tweets = tf.data.Dataset.from_tensor_slices(tf.constant(tweets, dtype=tf.int32))
    tweets = tf.data.Dataset.zip((tweets, tweet_len))
    prices = tf.data.Dataset.from_tensor_slices(tf.constant(prices, dtype=tf.int32))
    return tweets, prices, tweet_month_idx, monthly_price

def load_word_embeddings(path_word_embeddings, params):
    """Create np array of word embeddings

    Args:
        path_word_embeddings: (string) path to word embeddings

    Returns:
        tweets: (np.array) yielding np array of word embeddings
    """
    word_embeddings = []
    with open (path_word_embeddings, "r") as f:
        for line in f:
            if (line.split()[0] == 'onion'):
                print(len(word_embeddings))
            word_embeddings.append([float(x) for x in line.split()[1:]])
            while (len(word_embeddings[-1]) < params.embedding_size):
                word_embeddings[-1].append(0.0)
            word_embeddings[-1] = word_embeddings[-1][:params.embedding_size]
    return np.array(word_embeddings)

def input_fn(mode, tweets, prices, tweet_month_idx, monthly_price, params):
    """Input function

    Args:
        mode: (string) 'train', 'eval'
                     At training, we shuffle the data and have multiple epochs
        tweets: (tf.Dataset) yielding list of tweets
        prices: (tf.Dataset) yielding list of prices
        tweet_month_idx: maps tweet index to the city-month index
        monthly_price: price for the city-month index
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)

    """
    # Load all the dataset in memory for shuffling is training
    is_training = (mode == 'train')
    buffer_size = tf.placeholder(tf.int64, shape=())

    # Zip the tweets and the prices together
    dataset = tf.data.Dataset.zip((tweets, prices))

    seed = tf.placeholder(tf.int64, shape=())
    dataset = (dataset
        .shuffle(buffer_size=buffer_size, seed=seed)
        .batch(params.batch_size)
        .prefetch(1)  # make sure you always have one batch ready to serve
    )

    # Create initializable iterator from this dataset so that we can reset at each epoch
    iterator = dataset.make_initializable_iterator()

    # Query the output of the iterator for input to the model
    ((tweets, tweet_len), prices) = iterator.get_next()
    init_op = iterator.initializer

    # Build and return a dictionary containing the nodes / ops
    inputs = {
        'tweets': tweets,
        'prices': prices,
        'tweet_lengths': tweet_len,
        'tweet_month_idx': tweet_month_idx,
        'monthly_price': monthly_price,
        'iterator_init_op': init_op,
        'buffer_size': buffer_size,
        'seed': seed
    }
    
    return inputs
