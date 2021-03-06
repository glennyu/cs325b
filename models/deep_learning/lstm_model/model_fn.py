"""Define the model."""

import tensorflow as tf

def build_model(mode, word_embeddings, inputs, params):
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
    tweets = inputs['tweets'] #(batch_size, time, tweet_batch, tweet_length)
    prv_prices = inputs['prv_prices'] #(batch_size, time, time_step)

    embeddings = tf.constant(word_embeddings, dtype=tf.float32)
    tweets = tf.nn.embedding_lookup(embeddings, tweets)
    #print("after embedding shape:", tweets.get_shape())
    
    reshaped_tweets = tf.reshape(tweets, (-1, params.tweet_max_len, params.embedding_size))
    #print("after reshaping tweets shape:", reshaped_tweets.get_shape())
    tweet_len = inputs['tweet_lengths']
    reshaped_tweet_len = tf.reshape(tweet_len, (-1,))

    with tf.variable_scope("LSTM1", reuse=False):
        lstm_tweet_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
        _, output = tf.nn.dynamic_rnn(lstm_tweet_cell, reshaped_tweets, sequence_length=reshaped_tweet_len, dtype=tf.float32)
        #print("after lstm shape", output[1].get_shape()) 
        output = tf.reshape(output[1], (-1, params.time_step, params.tweet_batch_size, params.lstm_num_units))
        #print("after reshape:", output.get_shape())
        averaged_output = tf.reduce_mean(output, axis=2)
        #print("after average:", averaged_output.get_shape())


    with tf.variable_scope("LSTM2", reuse=False):
        tweet_price = tf.concat([averaged_output, tf.cast(tf.expand_dims(prv_prices, axis=-1), tf.float32)], axis=-1)
        #print("after concat:", tweet_price.get_shape())
        lstm_time_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
        _, time_output = tf.nn.dynamic_rnn(lstm_time_cell, tweet_price, dtype=tf.float32)
        time_output = time_output[1]
        #print("after time lstm:", time_output.get_shape())

    hidden_layer = tf.layers.dense(time_output, 30, activation=tf.nn.tanh)
    predictions = tf.layers.dense(hidden_layer, params.class_size)
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
    prices = inputs['prices']

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(mode, word_embeddings, inputs, params)
        predictions = tf.argmax(logits, -1, output_type=tf.int32)

    # Define loss and accuracy (we need to apply a mask to account for padding)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(prices, params.class_size), logits=logits))
    #trainable_vars = tf.trainable_variables()
    #reg_losses = tf.reduce_sum([tf.nn.l2_loss(v) for v in trainable_vars])
    #reg_term = 0.05
    #print("reg_losses:", reg_losses) 
    #loss = loss + reg_term * reg_losses    
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(prices, tf.int32), predictions), tf.float32))
    # mape = tf.reduce_mean(tf.abs((prices - predictions)/prices))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        #print(prices)
        #print(predictions)
        metrics = {
            'loss': tf.metrics.mean(loss),
            'accuracy': tf.metrics.accuracy(labels=prices, predictions=predictions),
            'auc': tf.metrics.auc(labels=tf.one_hot(prices, params.class_size), predictions=tf.nn.softmax(logits))
            #'mean_per_class_accuracy': tf.metrics.mean_per_class_accuracy(labels=tf.one_hot(prices, params.class_size), predictions=tf.nn.softmax(logits), num_classes=params.class_size)
            #'true_positives': tf.metrics.true_positives(prices, predictions),
            #'false_positives': tf.metrics.false_positives(prices, predictions),
            #'true_negatives': tf.metrics.true_negatives(prices, predictions),
            #'false_negatives': tf.metrics.false_negatives(prices, predictions)
            #'mape': tf.metrics.mean(mape)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    #tf.summary.scalar('mape', mape)
    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec['predictions'] = predictions
    model_spec['prices'] = prices
    model_spec['loss'] = loss
    #model_spec['mape'] = mape
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
