"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf
import numpy as np
import os
from random import shuffle

def load_features_and_labels(file_path, params, mode):
    features, labels = [], []
    label_distribution = np.zeros(params.class_size, dtype=np.int32)
    with open(file_path, "r") as f:
        idx = 0
        for line in f:
            if (mode == 'train' and idx < 700) or (mode != 'train' and idx >= 700):
                features.append([float(x) for x in line.split()[:-1]])
                labels.append(int(line.split()[-1]))
                label_distribution[labels[-1]] += 1
            idx += 1
    features = np.array(features)
    labels = np.array(labels)
    print(features.shape)
    print(labels.shape)
    print(label_distribution)
    features = tf.data.Dataset.from_tensor_slices(tf.constant(features, dtype=tf.float32))
    labels = tf.data.Dataset.from_tensor_slices(tf.constant(labels, dtype=tf.int32))
    return features, labels

def input_fn(mode, features, labels, params):
    # Load all the dataset in memory for shuffling is training
    is_training = (mode == 'train')
    buffer_size = params.train_size if is_training else 1

    dataset = tf.data.Dataset.zip((features, labels))

    seed = tf.placeholder(tf.int64, shape=())
    dataset = (dataset
        .shuffle(buffer_size=buffer_size, seed=seed)
        .batch(params.batch_size)
        .prefetch(1)  # make sure you always have one batch ready to serve
    )

    # Create initializable iterator from this dataset so that we can reset at each epoch
    iterator = dataset.make_initializable_iterator()

    # Query the output of the iterator for input to the model
    (features, labels) = iterator.get_next()
    init_op = iterator.initializer

    # Build and return a dictionary containing the nodes / ops
    inputs = {
        'features': features,
        'labels': labels,
        'iterator_init_op': init_op,
        'seed': seed
    }
    
    return inputs
