"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf

def pad_tweets(tweets):
    for batch in tweets:
        for tweet in batch:
            while (len(tweet) < 30):
                tweet.append(0)
    return tweets

def get_tweet_len(tweets):
    tweet_len = [[len(tweet) for tweet in batch] for batch in tweets]
    return tweet_len

def load_tweets(path):
    """Create tf.data Instance from txt file

    Args:
        path: (string) path containing one example per line

    Returns:
        tweets: (tf.Dataset) yielding list of ids of tokens and lengths for each example
    """
    tweets = [[[3, 10, 2, 10], [20, 25, 1]], [[6, 11, 4, 20, 1], [2, 5, 12]], [[7, 1], [2, 8, 9, 1, 4]]]
    tweet_len = tf.data.Dataset.from_tensor_slices(get_tweet_len(tweets))
    tweets = tf.data.Dataset.from_tensor_slices(tf.constant(pad_tweets(tweets), dtype=tf.float32))
    tweets = tf.data.Dataset.zip((tweets, tweet_len))
    return tweets

def load_prices(path):
    """Create tf.data Instance from txt file

    Args:
        path: (string) path containing one price per line

    Returns:
        prices: (tf.Dataset) yielding list of prices
    """
    prices = tf.data.Dataset.from_tensor_slices(tf.constant([3.28, 9.4, 8.2, 2.3], dtype=tf.float32))
    return prices

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
