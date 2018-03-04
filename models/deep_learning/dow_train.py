"""Train the model"""

import argparse
import logging
import os

import tensorflow as tf

from dow_model.utils import Params
from dow_model.utils import set_logger
from dow_model.training import train_and_evaluate
from dow_model.input_fn import input_fn
from dow_model.input_fn import load_features_and_labels
from dow_model.model_fn import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/dow_model',
                    help="Directory containing params.json")
parser.add_argument('--results_dir', default='experiments/dow_model/results',
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
    path_train_features = os.path.join(args.data_dir, 'dow_delhi_feat.txt')
    path_eval_features = os.path.join(args.data_dir, 'dow_delhi_feat.txt')
    
    # Create the input data pipeline
    logging.info("Creating the datasets...")
    train_features, train_labels = load_features_and_labels(path_train_features, params, 'train')
    eval_features, eval_labels = load_features_and_labels(path_eval_features, params, 'eval')

    # Create the two iterators over the two datasets
    train_inputs = input_fn('train', train_features, train_labels, params)
    eval_inputs = input_fn('eval', eval_features, eval_labels, params)
    logging.info("- done.")

    # Define the models (2 different set of nodes that share weights for train and eval)
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)
    logging.info("- done.")

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.results_dir, params, args.restore_dir)
