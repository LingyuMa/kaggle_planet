import os
import tensorflow as tf

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 5000

current_dir = os.path.dirname(__file__)
train_data_path = os.path.join(current_dir, r'../../data/processed/train_set.txt')
validation_data_path = os.path.join(current_dir, r'../../data/processed/validation_set.txt')


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'shape': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string)
        }
    )

    image = tf.decode_raw(features['image'], tf.uint8)
    shape = tf.decode_raw(features['shape'], tf.int32)
    label = tf.decode_raw(features['label'], tf.int32)

    image = tf.reshape(image, shape)

    print(image)
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
    filename_queue = tf.train.string_input_producer([train_data_path])
    image, label = read_and_decode(filename_queue)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(min_fraction_of_examples_in_queue * NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)

    print('Filling queue with {} images before starting to train.'.format(min_queue_examples))
    return generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle=True)

if __name__ == "__main__":
    with tf.Session() as sess:
        images, label_batch = train_inputs(16)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coords=coord)

        imgs, labs = sess.run([images, label_batch])
        print(imgs)
        coord.request_stop()
        coord.join(threads)
