"""Tensorflow utility functions for evaluation"""

import logging
import os

from tqdm import trange
import tensorflow as tf
import numpy as np

from model.utils import save_dict_to_json


def evaluate_sess(sess, model_spec, num_steps, epoch, writer=None, params=None):
    """Train the model on `num_steps` batches.

    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        epoch: (int) epoch number
        writer: (tf.summary.FileWriter) writer for summaries. Is None if we don't log anything
        params: (Params) hyperparameters
    """
    prices = model_spec['prices']
    predictions = model_spec['predictions']
    update_metrics = model_spec['update_metrics']
    eval_metrics = model_spec['metrics']
    tweet_dist = model_spec['tweet_dist']
    tweet_month_idx = model_spec['tweet_month_idx']
    global_step = tf.train.get_global_step()

    # Load the evaluation dataset into the pipeline and initialize the metrics init op
    sess.run(model_spec['iterator_init_op'], feed_dict={model_spec['seed']: epoch, model_spec['buffer_size'] : 1})
    sess.run(model_spec['metrics_init_op'])

    cur_conf_matrix = np.zeros((params.class_size, params.class_size), dtype=np.int32)
    agg_conf_matrix = np.zeros((params.class_size, params.class_size), dtype=np.int32)
    cur_votes = np.zeros((model_spec['monthly_price'].shape[0], params.class_size), dtype=np.int32)
    
    # compute metrics over the dataset
    for i in range(num_steps):
        _, pred, pri = sess.run([update_metrics, predictions, prices], feed_dict={model_spec['is_training'] : False})
        #print(pred)
        #print(pri)
        for j in range(len(pred)):
            cur_conf_matrix[pred[j]][pri[j]] += 1
            index = i*params.batch_size + j
            cur_month_idx = tweet_month_idx[index]
            cur_dist = tweet_dist[index]
            cur_votes[cur_month_idx][pred[j]] += (1.0 / (cur_dist + 1))
    
    month_pred = np.argmax(cur_votes, axis=1)
    aggregate_acc = 1.0*np.sum(month_pred == model_spec['monthly_price'])/cur_votes.shape[0]
    for i in range(month_pred.shape[0]):
        agg_conf_matrix[month_pred[i]][model_spec['monthly_price'][i]] += 1

    # Get the values of the metrics
    metrics_values = {k: v[0] for k, v in eval_metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Eval metrics: " + metrics_string)

    # Add summaries manually to writer at global_step_val
    if writer is not None:
        global_step_val = sess.run(global_step)
        for tag, val in metrics_val.items():
            summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
            writer.add_summary(summ, global_step_val)

    return metrics_val, cur_conf_matrix, aggregate_acc, agg_conf_matrix


def evaluate(model_spec, model_dir, params, restore_from):
    """Evaluate the model

    Args:
        model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
    """
    # Initialize tf.Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize the lookup table
        sess.run(model_spec['variable_init_op'])

        # Reload weights from the weights subdirectory
        save_path = os.path.join(model_dir, restore_from)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)

        # Evaluate
        num_steps = (params.eval_size + params.batch_size - 1) // params.batch_size
        metrics, _ = evaluate_sess(sess, model_spec, num_steps)
        metrics_name = '_'.join(restore_from.split('/'))
        save_path = os.path.join(model_dir, "metrics_test_{}.json".format(metrics_name))
        save_dict_to_json(metrics, save_path)
