import tensorflow as tf
import src.settings as settings

def _activation_summary(x):
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def network8(images, training):
    # regularizer = tf.contrib.layers.l2_regularizer(scale=settings.REGULARIZER)
    # Feature level
    pre_bn = tf.layers.batch_normalization(
        inputs=images,
        training=training,
        name='bn_pre'
    )

    conv1 = tf.layers.conv2d(
        inputs=pre_bn,
        filters=32,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu,
        name='conv1'
    )
    _activation_summary(conv1)

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=32,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu,
        name='conv2'
    )
    _activation_summary(conv2)

    pool1 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2,
        name='pool1'
    )

    dropout1 = tf.layers.dropout(
        inputs=pool1,
        rate=0.25,
        training=training,
        name='dropout1'
    )

    conv3 = tf.layers.conv2d(
        inputs=dropout1,
        filters=64,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu,
        name='conv3'
    )
    _activation_summary(conv3)

    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=64,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu,
        name='conv4'
    )
    _activation_summary(conv4)

    pool2 = tf.layers.max_pooling2d(
        inputs=conv4,
        pool_size=[2, 2],
        strides=2,
        name='pool2'
    )

    dropout2 = tf.layers.dropout(
        inputs=pool2,
        rate=0.25,
        training=training,
        name='dropout2'
    )

    conv5 = tf.layers.conv2d(
        inputs=dropout2,
        filters=128,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu,
        name='conv5'
    )
    _activation_summary(conv5)

    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=128,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu,
        name='conv6'
    )
    _activation_summary(conv6)

    pool3 = tf.layers.max_pooling2d(
        inputs=conv6,
        pool_size=[2, 2],
        strides=2,
        name='pool3'
    )

    dropout3 = tf.layers.dropout(
        inputs=pool3,
        rate=0.25,
        training=training,
        name='dropout3'
    )

    conv7 = tf.layers.conv2d(
        inputs=dropout3,
        filters=256,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu,
        name='conv7'
    )
    _activation_summary(conv7)

    conv8 = tf.layers.conv2d(
        inputs=conv7,
        filters=256,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu,
        name='conv8'
    )
    _activation_summary(conv8)

    pool4 = tf.layers.max_pooling2d(
        inputs=conv8,
        pool_size=[2, 2],
        strides=2,
        name='pool4'
    )

    dropout4 = tf.layers.dropout(
        inputs=pool4,
        rate=0.25,
        training=training,
        name='dropout4'
    )

    # Classifier level
    dropout4_flat = tf.contrib.layers.flatten(dropout4)

    dense1 = tf.layers.dense(
        inputs=dropout4_flat,
        units=512,
        activation=tf.nn.relu,
        name='dense1'
    )

    norm1 = tf.layers.batch_normalization(
        inputs=dense1,
        training=training,
        name='norm1'
    )

    dropout5 = tf.layers.dropout(
        inputs=norm1,
        rate=0.5,
        training=training,
        name='dropout5'
    )

    logits = tf.layers.dense(
        inputs=dropout5,
        units=settings.LABELS_SIZE,
        # activation=tf.nn.sigmoid
        name='logits'
    )
    _activation_summary(logits)
    return logits


def network2(images, training):
    pre_bn = tf.layers.batch_normalization(
        inputs=images,
        training=training,
        name='bn_pre'
    )

    conv1 = tf.layers.conv2d(
        inputs=pre_bn,
        filters=32,
        kernel_size=[3, 3],
        activation=tf.nn.relu,
        name='conv1'
    )
    _activation_summary(conv1)

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[3, 3],
        activation=tf.nn.relu,
        name='conv2'
    )
    _activation_summary(conv2)

    pool1 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2,
        name='pool1'
    )

    dropout1 = tf.layers.dropout(
        inputs=pool1,
        rate=0.25,
        training=training,
        name='dropout1'
    )

    dropout1_flat = tf.contrib.layers.flatten(dropout1)

    dense1 = tf.layers.dense(
        inputs=dropout1_flat,
        units=128,
        activation=tf.nn.relu,
        name='dense1'
    )

    dropout2 = tf.layers.dropout(
        inputs=dense1,
        rate=0.5,
        training=training,
        name='dropout5'
    )

    logits = tf.layers.dense(
        inputs=dropout2,
        units=settings.LABELS_SIZE,
        # activation=tf.nn.sigmoid
        name='logits'
    )
    _activation_summary(logits)
    return logits