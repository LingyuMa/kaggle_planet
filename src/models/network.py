import tensorflow as tf

import src.settings as settings


def inference(images, training=True):
    #regularizer = tf.contrib.layers.l2_regularizer(scale=settings.REGULARIZER)
    # Feature level
    conv1 = tf.layers.conv2d(
        inputs=images,
        filters=32,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=32,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )

    pool1 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )

    dropout1 = tf.layers.dropout(
        inputs=pool1,
        rate=0.25,
        training=training
    )

    conv3 = tf.layers.conv2d(
        inputs=dropout1,
        filters=64,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )

    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=64,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )

    pool2 = tf.layers.max_pooling2d(
        inputs=conv4,
        pool_size=[2, 2],
        strides=2
    )

    dropout2 = tf.layers.dropout(
        inputs=pool2,
        rate=0.25,
        training=training
    )

    conv5 = tf.layers.conv2d(
        inputs=dropout2,
        filters=128,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )

    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=128,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )

    pool3 = tf.layers.max_pooling2d(
        inputs=conv6,
        pool_size=[2, 2],
        strides=2
    )

    dropout3 = tf.layers.dropout(
        inputs=pool3,
        rate=0.25,
        training=training
    )

    conv7 = tf.layers.conv2d(
        inputs=dropout3,
        filters=256,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )

    conv8 = tf.layers.conv2d(
        inputs=conv7,
        filters=256,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )

    pool4 = tf.layers.max_pooling2d(
        inputs=conv8,
        pool_size=[2, 2],
        strides=2
    )

    dropout4 = tf.layers.dropout(
        inputs=pool4,
        rate=0.25,
        training=training
    )

    # Classifier level
    dropout4_flat = tf.contrib.layers.flatten(dropout4)

    dense1 = tf.layers.dense(
        inputs=dropout4_flat,
        units=512,
        activation=tf.nn.relu
    )

    norm1 = tf.layers.batch_normalization(
        inputs=dense1,
        training=training
    )

    dropout5 = tf.layers.dropout(
        inputs=norm1,
        rate=0.5,
        training=training
    )

    logits = tf.layers.dense(
        inputs=dropout5,
        units=settings.LABELS_SIZE
        #activation=tf.nn.sigmoid
    )

    return logits


def loss(logits, labels):
    total_loss = tf.losses.sigmoid_cross_entropy(labels, logits)
    return total_loss


def train(total_loss, global_step):
    # Variables that affect learning rate
    num_batches_per_epoch = settings.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / settings.BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * settings.NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(settings.INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    settings.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    tf.summary.scalar('learning_rate', lr)

    # Compute gradients
    #opt = tf.train.GradientDescentOptimizer(lr)
    opt = tf.train.AdamOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

    # Apply gradients
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        settings.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op




