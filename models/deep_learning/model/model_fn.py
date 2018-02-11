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
    tweets = inputs['tweets'] # (batch_size, tweet_batch, tweet_length)

    if params.model_version == "lstm1":
        embeddings = tf.constant(word_embeddings, dtype=tf.float32)
        tweets = tf.nn.embedding_lookup(embeddings, tweets)
        #print("after embedding shape:", tweets.get_shape())
        
        reshaped_tweets = tf.reshape(tweets, (-1, 140, params.embedding_size))
        #print("after reshaping tweets shape:", reshaped_tweets.get_shape())
        tweet_len = inputs['tweet_lengths']
        reshaped_tweet_len = tf.reshape(tweet_len, (-1,))
        
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
        _, output = tf.nn.dynamic_rnn(lstm_cell, reshaped_tweets, sequence_length=reshaped_tweet_len, dtype=tf.float32)
        #print("after lstm shape", output[1].get_shape()) 
        output = tf.reshape(output[1], (-1, params.tweet_batch_size, params.lstm_num_units))
        #print("after reshape:", output.get_shape())
        averaged_output = tf.reduce_mean(output, axis=1)
        #print("after average:", averaged_output.get_shape())
        hidden_layer = tf.layers.dense(averaged_output, 20, activation=tf.nn.tanh)
        predictions = tf.layers.dense(hidden_layer, 2)
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
        predictions = tf.argmax(logits, -1)

    # Define loss and accuracy (we need to apply a mask to account for padding)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf.one_hot(prices)))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prices, predictions), tf.float32))
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
        metrics = {
            'loss': tf.metrics.mean(loss),
            'accuracy': tf.metrics.accuracy(labels=prices, predictions=predictions)
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
