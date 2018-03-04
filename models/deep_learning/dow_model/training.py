"""Tensorflow utility functions for training"""

import logging
import os

from tqdm import trange
import tensorflow as tf
import numpy as np

from dow_model.utils import save_dict_to_json
from dow_model.evaluation import evaluate_sess


def train_sess(sess, model_spec, num_steps, epoch, writer, params):
    """Train the model on `num_steps` batches

    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        epoch: (int) epoch number
        writer: (tf.summary.FileWriter) writer for summaries
        params: (Params) hyperparameters
    """
    # Get relevant graph operations or nodes needed for training
    loss = model_spec['loss']
    labels = model_spec['labels']
    logits = model_spec['logits']
    predictions = model_spec['predictions']
    train_op = model_spec['train_op']
    update_metrics = model_spec['update_metrics']
    metrics = model_spec['metrics']
    summary_op = model_spec['summary_op']
    global_step = tf.train.get_global_step()

    # Load the training dataset into the pipeline and initialize the metrics local variables
    sess.run(model_spec['iterator_init_op'], feed_dict={model_spec['seed']: epoch})
    sess.run(model_spec['metrics_init_op'])

    cur_conf_matrix = np.zeros((params.class_size, params.class_size), dtype=np.int32)

    # Use tqdm for progress bar
    t = trange(num_steps)
    for i in t:
        # Evaluate summaries for tensorboard only once in a while
        if i % params.save_summary_steps == 0:
            # Perform a mini-batch update
            _, _, loss_val, summ, global_step_val, pred, pri, lg = sess.run([train_op, update_metrics, loss,
                                                              summary_op, global_step, predictions, labels, logits], 
                                                              feed_dict={model_spec['is_training'] : True})
            #print(pred)
            #print(pri)
            #print(lg)
            # Write summaries for tensorboard
            writer.add_summary(summ, global_step_val)
        else:
            _, _, loss_val, pred, pri, lg = sess.run([train_op, update_metrics, loss, predictions, labels, logits],
                                        feed_dict={model_spec['is_training'] : True})
            #print(pred)
            #print(pri)
            #print(lg)
        # Log the loss in the tqdm progress bar
        t.set_postfix(loss='{:05.3f}'.format(loss_val))
        for j in range(len(pred)):
            cur_conf_matrix[pri[j]][pred[j]] += 1

    metrics_values = {k: v[0] for k, v in metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Train metrics: " + metrics_string)
    return cur_conf_matrix


def train_and_evaluate(train_model_spec, eval_model_spec, results_dir, params, restore_from=None):
    """Train the model and evaluate every epoch.

    Args:
        train_model_spec: (dict) contains the graph operations or nodes needed for training
        eval_model_spec: (dict) contains the graph operations or nodes needed for evaluation
        results_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
    """
    # Initialize tf.Saver instances to save weights during training
    last_saver = tf.train.Saver() # will keep last 5 epochs
    best_saver = tf.train.Saver(max_to_keep=1)  # only keep 1 best checkpoint (best on eval)
    begin_at_epoch = 0

    with tf.Session() as sess:
        # Initialize model variables
        sess.run(train_model_spec['variable_init_op'])

        # Reload weights from directory if specified
        if restore_from is not None:
            logging.info("Restoring parameters from {}".format(restore_from))
            if os.path.isdir(restore_from):
                restore_from = tf.train.latest_checkpoint(restore_from)
                begin_at_epoch = int(restore_from.split('-')[-1])
            last_saver.restore(sess, restore_from)

        # For tensorboard (takes care of writing summaries to files)
        train_writer = tf.summary.FileWriter(os.path.join(results_dir, 'train_summaries'), None)
        eval_writer = tf.summary.FileWriter(os.path.join(results_dir, 'eval_summaries'), None)

        train_conf_matrix = np.zeros((params.class_size, params.class_size), dtype=np.int32)
        eval_conf_matrix = np.zeros((params.class_size, params.class_size), dtype=np.int32)

        best_eval_acc = 0.0
        for epoch in range(begin_at_epoch, begin_at_epoch + params.num_epochs):
            # Run one epoch
            logging.info("Epoch {}/{}".format(epoch + 1, begin_at_epoch + params.num_epochs))
            # Compute number of batches in one epoch (one full pass over the training set)
            num_steps = (params.train_size + params.batch_size - 1) // params.batch_size
            train_conf_matrix = train_sess(sess, train_model_spec, num_steps, epoch, train_writer, params)

            # Save weights
            # last_save_path = os.path.join(results_dir, 'last_weights', 'after-epoch')
            # last_saver.save(sess, last_save_path, global_step=epoch + 1)

            # Evaluate for one epoch on validation set
            num_steps = (params.eval_size + params.batch_size - 1) // params.batch_size
            metrics, m = evaluate_sess(sess, eval_model_spec, num_steps, epoch, eval_writer, params)

            #logging.info("- Current Training Confusion Matrix:\n {}".format(train_conf_matrix))
            #logging.info("- Current Evaluation Confusion Matrix:\n {}".format(m))

            # If best_eval, best_save_path
            eval_acc = metrics['accuracy']
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                # Save weights
                best_save_path = os.path.join(results_dir, 'best_weights', 'after-epoch')
                best_save_path = best_saver.save(sess, best_save_path, global_step=epoch + 1)
                logging.info("- Found new best accuracy, saving in {}".format(best_save_path))
                # Save best eval metrics in a json file in the model directory
                best_json_path = os.path.join(results_dir, "metrics_eval_best_weights.json")
                save_dict_to_json(metrics, best_json_path)
                eval_conf_matrix = m

            # Save latest eval metrics in a json file in the model directory
            last_json_path = os.path.join(results_dir, "metrics_eval_last_weights.json")
            save_dict_to_json(metrics, last_json_path)


        logging.info("- Training Confusion Matrix:\n {}".format(train_conf_matrix))
        logging.info("- Evaluation Confusion Matrix:\n {}".format(eval_conf_matrix))
