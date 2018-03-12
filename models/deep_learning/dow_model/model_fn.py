"""Define the model."""

import tensorflow as tf

def build_model(mode, inputs, is_training, params):
    """Compute logits of the model (output distribution)

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (tf.Tensor) output of the model
    """
    features = inputs['features']
    hidden_layer1 = tf.layers.dense(features, 200, activation=tf.nn.tanh)
    hidden_layer2 = tf.layers.dense(hidden_layer1, 200, activation=tf.nn.tanh)
    hidden_layer3 = tf.layers.dense(hidden_layer2, 100, activation=tf.nn.tanh)
    hidden_layer4 = tf.layers.dense(hidden_layer3, 100, activation=tf.nn.tanh)
    hidden_layer5 = tf.layers.dense(hidden_layer4, 50, activation=tf.nn.tanh)
    hidden_layer6 = tf.layers.dense(hidden_layer5, 50, activation=tf.nn.tanh)
    hidden_layer7 = tf.layers.dense(hidden_layer6, 20, activation=tf.nn.tanh)
    hidden_layer8 = tf.layers.dense(hidden_layer7, 20, activation=tf.nn.tanh)
    predictions = tf.layers.dense(hidden_layer8, params.class_size)
    return predictions

def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    train_placeholder = tf.placeholder(tf.bool)
    labels = inputs['labels']

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(mode, inputs, train_placeholder, params)
        predictions = tf.argmax(logits, -1, output_type=tf.int32)

    # Define loss and accuracy (we need to apply a mask to account for padding)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(labels, params.class_size), logits=logits))
    reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    total_loss = loss + reg_loss    
 
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

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
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions)
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
    model_spec['labels'] = labels
    model_spec['loss'] = total_loss
    model_spec['reg_loss'] = reg_loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
