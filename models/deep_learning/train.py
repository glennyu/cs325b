"""Train the model"""

import argparse
import logging
import os

import tensorflow as tf

from model.utils import Params
from model.utils import set_logger
from model.training import train_and_evaluate
from model.input_fn import input_fn
from model.input_fn import load_tweets_and_prices, load_word_embeddings
from model.model_fn import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--results_dir', default='experiments/base_model/results',
                    help="Directory containing results")
parser.add_argument('--data_dir', default='data/', help="Directory containing the dataset")
parser.add_argument('--restore_dir', default=None,
                    help="Optional, directory containing weights to reload before training")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(325)

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    model_dir_has_best_weights = os.path.isdir(os.path.join(args.results_dir, "best_weights"))
    overwritting = model_dir_has_best_weights and args.restore_dir is None
    # assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    set_logger(os.path.join(args.results_dir, 'train.log'))

    # Get paths for dataset
    path_train_embeddings = os.path.join(args.data_dir, 'embeddings/')
    path_train_batches = os.path.join(args.data_dir, 'batches_train/')
    path_eval_embeddings = os.path.join(args.data_dir, 'embeddings/')
    path_eval_batches = os.path.join(args.data_dir, 'batches_val/')
    path_word_embeddings = os.path.join(args.data_dir, 'glove.twitter.27B.50d.txt')
    
    logging.info("Creating the datasets...")
    train_tweets, train_prices, train_tweet_month_idx, train_monthly_price = load_tweets_and_prices(
        path_train_embeddings, path_train_batches, params)
    eval_tweets, eval_prices, eval_tweet_month_idx, eval_monthly_price = load_tweets_and_prices(
        path_eval_embeddings, path_eval_batches, params)

    # Create the two iterators over the two datasets
    train_inputs = input_fn('train', train_tweets, train_prices, train_tweet_month_idx, train_monthly_price, params)
    eval_inputs = input_fn('eval', eval_tweets, eval_prices, eval_tweet_month_idx, eval_monthly_price, params)
    logging.info("- done.")

    # Load word embeddings
    logging.info("Loading word embeddings...")
    word_embeddings = load_word_embeddings(path_word_embeddings, params)
    logging.info("- done.")

    # Define the models (2 different set of nodes that share weights for train and eval)
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', word_embeddings, train_inputs, params)
    eval_model_spec = model_fn('eval', word_embeddings, eval_inputs, params, reuse=True)
    logging.info("- done.")

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.results_dir, params, args.restore_dir)
