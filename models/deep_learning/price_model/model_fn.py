"""Define the model."""

import tensorflow as tf

def build_model(mode, inputs, is_training, params):
    """Compute logits of the model (output distribution)

    Args:
        mode: (string) 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (tf.Tensor) output of the model
    """

    prices = inputs['prices']
    #print('prices shape: ', prices.get_shape())

    if params.model_version == 'lstm1':
        # Apply LSTM over the prices
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=1.0 - params.dropout_rate)
        _, output = tf.nn.dynamic_rnn(lstm_cell, prices, dtype=tf.float32)


        hidden_layer1 = tf.layers.dense(output[1], 50, activation=tf.nn.tanh, kernel_regularizer=tf.contrib.layers.l2_regularizer(params.reg_term))
        hidden_layer2 = tf.layers.dense(hidden_layer1, 30, activation=tf.nn.tanh, kernel_regularizer=tf.contrib.layers.l2_regularizer(params.reg_term))
        hidden_layer3 = tf.layers.dense(hidden_layer2, 20, activation=tf.nn.tanh, kernel_regularizer=tf.contrib.layers.l2_regularizer(params.reg_term))
        dropout = tf.layers.dropout(hidden_layer3, rate=params.dropout_rate, training=is_training)
        predictions = tf.layers.dense(dropout, 1, kernel_regularizer=tf.contrib.layers.l2_regularizer(params.reg_term))
        return predictions

    else:
        raise NotImplementedError("Unknown model version: {}".format(params.model_version))

    return predictions

def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.
    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights
    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    train_placeholder = tf.placeholder(tf.bool)
    deltas = inputs['deltas']
    deltas = tf.cast(deltas, tf.float32)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        predictions = build_model(is_training, inputs, train_placeholder, params)

    # Define loss and profit
    loss = tf.reduce_mean(tf.nn.l2_loss((predictions - deltas)))

    # Define training step that minimizes the loss with the Adam optimizer
    lr_placeholder = tf.placeholder(tf.float32)
    if is_training:
        optimizer = tf.train.AdamOptimizer(lr_placeholder)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)


    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'loss': tf.metrics.mean(loss),
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec['is_training'] = train_placeholder
    model_spec['learning_rate'] = lr_placeholder
    model_spec['predictions'] = predictions
    model_spec['deltas'] = deltas
    model_spec['loss'] = loss
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
