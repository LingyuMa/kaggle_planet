import tensorflow as tf
import src.settings as settings
import src.models.architecture as architecture


def inference(images, n, reuse=False):
    logits = architecture.resnet(images, n, reuse)
    return logits


def loss(logits, labels):
    logits = tf.cast(logits, tf.float32)
    labels = tf.cast(labels, tf.float32)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels,
                                                            name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([cross_entropy_mean] + reg_loss)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('fn_loss', cross_entropy_mean)
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




