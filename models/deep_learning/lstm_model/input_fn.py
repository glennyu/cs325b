"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf
import numpy as np
import os

def pad_tweets(tweets, time_step, tweet_batch_size, max_tweet_len):
    np_tweets = np.zeros((len(tweets), time_step, tweet_batch_size, max_tweet_len), dtype=np.int32)
    for i in range(len(tweets)):
        for t, batch in enumerate(tweets[i]):
            for j, tweet in enumerate(batch):
                if (len(tweet) > max_tweet_len):
                    print("found tweet of length", len(tweet))
                idx = 0
                while idx < len(tweet) and idx < max_tweet_len:
                    np_tweets[i][t][j][idx] = tweet[idx]
                    idx += 1
    return np_tweets

def get_tweet_len(tweets):
    tweet_len = [[[len(tweet) for tweet in batch] for batch in x] for x in tweets]
    return np.array(tweet_len)

def load_tweets_and_prices(path_embeddings, path_batches, params, cap):
    """Create tf.data Instance from txt file

    Args:
        path_embeddings: (string) path to embeddings
        path_batches: (string) path to batches
        params: data parameters

    Returns:
        tweets, prices: (tf.Dataset) yielding list of tweet tokens, lengths, and price deviations
    """
    tweets, prv_price_dir, price_dir = [], [], []
    for filename in os.listdir(path_batches):
        city = filename[:filename.find("_weekly_batch")]
        print("processing", city)
        city_tweets = []
        with open(path_embeddings + city + "_embeddings.csv", "r") as embf:
            for tweet in embf:
                city_tweets.append([int(num) for num in tweet.split(',')])
        with open(path_batches + filename, "r") as batchf:
            total = 0
            for batch in batchf:
                nums = np.array([int(x) for x in batch.split('\t')])
                tweet_idx = np.reshape(nums[:params.time_step*params.tweet_batch_size], (params.time_step, params.tweet_batch_size))
                cur_price_dir = nums[params.time_step*params.tweet_batch_size:]
                tweets.append([[city_tweets[idx] for idx in tweet] for tweet in tweet_idx])
                prv_price_dir.append(cur_price_dir[:params.time_step])
                price_dir.append(cur_price_dir[-1])
                total += 1
                if total == cap:
                    break
    prv_price_dir = np.array(prv_price_dir)
    price_dir = np.array(price_dir)
    print(len(tweets))

    tweet_len = tf.data.Dataset.from_tensor_slices(get_tweet_len(tweets))
    prv_price_dir_tf = tf.data.Dataset.from_tensor_slices(prv_price_dir)
    tweets = tf.data.Dataset.from_tensor_slices(pad_tweets(tweets, params.time_step, params.tweet_batch_size, params.tweet_max_len))
    tweets = tf.data.Dataset.zip((tweets, tweet_len, prv_price_dir_tf))
    price_dir = tf.data.Dataset.from_tensor_slices(np.array(price_dir))
    return tweets, price_dir

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
            word_embeddings.append([float(x) for x in line.split()[1:]])
            while (len(word_embeddings[-1]) < params.embedding_size):
                word_embeddings[-1].append(0.0)
            word_embeddings[-1] = word_embeddings[-1][:params.embedding_size]
    return np.array(word_embeddings)

def input_fn(mode, tweets, prices, params):
    """Input function

    Args:
        mode: (string) 'train', 'eval'
                     At training, we shuffle the data and have multiple epochs
        tweets: (tf.Dataset) yielding list of tweets
        prices: (tf.Dataset) yielding list of prices
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)

    """
    # Load all the dataset in memory for shuffling is training
    is_training = (mode == 'train')
    buffer_size = params.train_size if is_training else 1

    # Zip the tweets and the prices together
    dataset = tf.data.Dataset.zip((tweets, prices))

    dataset = (dataset
        .shuffle(buffer_size=buffer_size)
        .batch(params.batch_size)
        .prefetch(1)  # make sure you always have one batch ready to serve
    )

    # Create initializable iterator from this dataset so that we can reset at each epoch
    iterator = dataset.make_initializable_iterator()

    # Query the output of the iterator for input to the model
    ((tweets, tweet_len, prv_prices), prices) = iterator.get_next()
    init_op = iterator.initializer

    # Build and return a dictionary containing the nodes / ops
    inputs = {
        'tweets': tweets,
        'tweet_lengths': tweet_len,
        'prv_prices': prv_prices,
        'prices': prices,
        'iterator_init_op': init_op
    }
    
    return inputs
