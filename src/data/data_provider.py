import tensorflow as tf
import src.settings as settings
import matplotlib.pyplot as plt


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string)
        }
    )

    image = tf.decode_raw(features['image'], tf.float16)
    label = tf.decode_raw(features['label'], tf.int32)

    image = tf.reshape(image, [settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, settings.IMAGE_DEPTH])
    label = tf.reshape(label, [len(settings.LABELS[settings.NETWORK_ID])])

    return image, label


def generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples
        )
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size
        )
    return images, label_batch


def train_inputs(batch_size):
    filename_queue = tf.train.string_input_producer([settings.TRAIN_DATA_PATH])
    image, label = read_and_decode(filename_queue)

    # Augment the image set
    if settings.DATA_AUGMENTATION:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

    # Normalize the image
    if settings.NORMALIZE_IMAGE:
        image = tf.image.per_image_standardization(image)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(min_fraction_of_examples_in_queue * settings.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)

    print('Filling queue with {} images before starting to train.'.format(min_queue_examples))
    return generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle=True)


def inputs(eval_data, batch_size):
    if eval_data:
        filename_queue = tf.train.string_input_producer([settings.VALIDATION_DATA_PATH])
        num_examples_per_epoch = settings.NUM_EXAMPLES_PER_EPOCH_FOR_VALIDATION
    else:
        filename_queue = tf.train.string_input_producer([settings.TRAIN_DATA_PATH])
        num_examples_per_epoch = settings.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

    image, label = read_and_decode(filename_queue)

    # Normalize the image
    if settings.NORMALIZE_IMAGE:
        image = tf.image.per_image_standardization(image)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

    return generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle=False)


if __name__ == "__main__":
    images, label_batch = train_inputs(64)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        imgs, labs = sess.run([images, label_batch])
        img = imgs[25, :, :, 0]
        plt.imshow(img, cmap='gray')
        plt.show()

        coord.request_stop()
        coord.join(threads)
