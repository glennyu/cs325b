#! /bin/sh
cd models/deep_learning/

mkdir experiments/single_tweet_model/price_dir_results
python3 single_tweet_train.py --json_file=experiments/single_tweet_model/price_dir.json --results_dir=experiments/single_tweet_model/price_dir_results/
mkdir experiments/single_tweet_model/price_spike_results
python3 single_tweet_train.py --json_file=experiments/single_tweet_model/price_spike.json --results_dir=experiments/single_tweet_model/price_spike_results/

mkdir experiments/batch_model/price_dir_results
python3 batch_model_train.py --json_file=experiments/batch_model/price_dir.json --results_dir=experiments/batch_model/price_dir_results/
mkdir experiments/batch_model/price_spike_results
python3 batch_model_train.py --json_file=experiments/batch_model/price_spike.json --results_dir=experiments/batch_model/price_spike_results/
