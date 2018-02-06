"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf
import numpy as np

NUM_MONTHS = 5

def pad_tweets(tweets):
    np_tweets = np.zeros((len(tweets), len(tweets[0]), 40), dtype=np.int32)
    for i, batch in enumerate(tweets):
        for j, tweet in enumerate(batch):
            idx = 0
            while idx < len(tweet) and idx < 40:
                np_tweets[i][j][idx] = tweet[idx]
                idx += 1
    return np_tweets

def get_tweet_len(tweets):
    tweet_len = [[len(tweet) for tweet in batch] for batch in tweets]
    return tweet_len

def load_tweets_and_prices(path_embeddings, path_batches, path_prices):
    """Create tf.data Instance from txt file

    Args:
        path_embeddings: (string) path to embeddings
        path_batches: (string) path to batches
        path_prices: (string) path to prices

    Returns:
        tweets, prices: (tf.Dataset) yielding list of tweet tokens, lengths, and price deviations
    """
    tweets, prices = [], []
    with open(path_prices, "r") as pricef:
        p_dev = [float(p) for p in pricef.readline().strip('\n').split('\t')]
        for i in range(NUM_MONTHS):
            month_tweets = []
            with open(path_embeddings + str(i) + "_embeddings_delhi.csv", "r") as embf:
                for tweet in embf:
                    month_tweets.append([int(num) for num in tweet.split(',')])
            with open(path_batches + str(i) + "_batches_delhi.txt", "r") as batchf:
                for batch in batchf:
                    tweets.append([month_tweets[int(idx)] for idx in batch.split('\t')])
                    prices.append(p_dev[i])

    tweet_len = tf.data.Dataset.from_tensor_slices(tf.constant(get_tweet_len(tweets), dtype=tf.int32))
    tweets = tf.data.Dataset.from_tensor_slices(tf.constant(pad_tweets(tweets), dtype=tf.int32))
    tweets = tf.data.Dataset.zip((tweets, tweet_len))
    prices = tf.data.Dataset.from_tensor_slices(tf.constant(prices, dtype=tf.float32))
    return tweets, prices

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
    ((tweets, tweet_len), prices) = iterator.get_next()
    init_op = iterator.initializer

    # Build and return a dictionary containing the nodes / ops
    inputs = {
        'tweets': tweets,
        'prices': prices,
        'tweet_lengths': tweet_len,
        'iterator_init_op': init_op
    }
    
    return inputs
