import tensorflow as tf
import src.settings as settings


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

    image = tf.decode_raw(features['image'], tf.uint8)
    label = tf.decode_raw(features['label'], tf.int32)

    image = tf.reshape(image, [settings.RAW_IMAGE_HEIGHT, settings.RAW_IMAGE_WIDTH, settings.RAW_IMAGE_DEPTH])
    label = tf.reshape(label, [settings.LABELS_SIZE])

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

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(min_fraction_of_examples_in_queue * settings.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)

    print('Filling queue with {} images before starting to train.'.format(min_queue_examples))
    return generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle=True)


if __name__ == "__main__":
    images, label_batch = train_inputs(16)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        imgs, labs = sess.run([images, label_batch])
        print(imgs)
        coord.request_stop()
        coord.join(threads)
