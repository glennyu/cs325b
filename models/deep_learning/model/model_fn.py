"""Define the model."""

import tensorflow as tf

def build_model(mode, word_embeddings, inputs, is_training, params):
    """Compute logits of the model (output distribution)

    Args:
        mode: (string) 'train', 'eval', etc.
        word_embeddings: GloVe word embeddings
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (tf.Tensor) output of the model
    """
    tweets = inputs['tweets'] # (batch_size, tweet_batch, tweet_length)
    tweet_feat = inputs['tweet_features']

    embeddings = tf.constant(word_embeddings, dtype=tf.float32)
    tweets = tf.nn.embedding_lookup(embeddings, tweets)
    tweet_len = inputs['tweet_lengths']
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
    
    for lstm_variable in lstm_cell.variables:
        weight_decay = tf.multiply(tf.nn.l2_loss(lstm_variable), params.reg_term, name='weight_decay')
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_decay)

    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=1.0 - params.dropout_rate)
    _, output = tf.nn.dynamic_rnn(lstm_cell, tweets, sequence_length=tweet_len, dtype=tf.float32)

    output = tf.layers.dense(output[1], 20, activation=tf.nn.tanh, kernel_regularizer=tf.contrib.layers.l2_regularizer(params.reg_term))
    dropout = tf.layers.dropout(output, rate=params.dropout_rate, training=is_training)
    output2 = tf.layers.dense(dropout, 5, activation=tf.nn.tanh, kernel_regularizer=tf.contrib.layers.l2_regularizer(params.reg_term))
    all_feat = tf.concat([output2, tweet_feat], axis=1)
    output3 = tf.layers.dense(all_feat, 10, activation=tf.nn.tanh, kernel_regularizer=tf.contrib.layers.l2_regularizer(params.reg_term))
    predictions = tf.layers.dense(output3, params.class_size, kernel_regularizer=tf.contrib.layers.l2_regularizer(params.reg_term))
    return predictions

def model_fn(mode, word_embeddings, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) 'train', 'eval', etc.
        word_embeddings: GloVe word embeddings
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    train_placeholder = tf.placeholder(tf.bool)
    prices = inputs['prices']

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(mode, word_embeddings, inputs, train_placeholder, params)
        predictions = tf.argmax(logits, -1, output_type=tf.int32)

    # Define loss and accuracy (we need to apply a mask to account for padding)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(prices, params.class_size), logits=logits))
    reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
#     total_reg_losses = tf.Print(total_reg_losses, total_reg_losses, "Total regularization loss:")
    total_loss = loss + reg_loss    
 
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prices, predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(total_loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'loss': tf.metrics.mean(total_loss),
            'accuracy': tf.metrics.accuracy(labels=prices, predictions=predictions),
            'auc': tf.metrics.auc(labels=tf.one_hot(prices, params.class_size), predictions=tf.nn.softmax(logits))
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('reg_loss', reg_loss)
    tf.summary.scalar('accuracy', accuracy)
    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec['is_training'] = train_placeholder
    model_spec['logits'] = logits
    model_spec['predictions'] = predictions
    model_spec['prices'] = prices
    model_spec['loss'] = total_loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
