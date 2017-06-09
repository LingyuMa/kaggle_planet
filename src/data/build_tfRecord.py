import tensorflow as tf
from skimage import io
import numpy as np
import os
import glob
import pandas as pd
import random

LABELS = {'primary': 0,
          'clear': 1,
          'agriculture': 2,
          'road': 3,
          'water': 4,
          'partly_cloudy': 5,
          'cultivation': 6,
          'habitation': 7,
          'haze': 8,
          'cloudy': 9,
          'bare_ground': 10,
          'selective_logging': 11,
          'artisinal_mine': 12,
          'blooming': 13,
          'slash_burn': 14,
          'blow_down': 15,
          'conventional_mine': 16}

TRAIN_VALIDATION_RATIO = 0.9
SEED = 448


def get_image_binary(filename):
    image = io.imread(filename)
    image = np.asarray(image, np.uint8)
    shape = np.array(image.shape, np.int32)
    return shape.tobytes(), image.tobytes()


def write_to_tfrecord():
    current_dir = os.path.dirname(__file__)
    img_paths = os.path.join(current_dir, r'../../data/processed/train-tif-v2')
    label_path = os.path.join(current_dir, r'../../data/processed/train_v2.csv')

    train_data_path = os.path.join(current_dir, r'../../data/processed/train_set.txt')
    validation_data_path = os.path.join(current_dir, r'../../data/processed/validation_set.txt')

    writer_train = tf.python_io.TFRecordWriter(train_data_path)
    writer_validation = tf.python_io.TFRecordWriter(validation_data_path)

    img_list = glob.glob(os.path.join(img_paths, '*.tif'))
    random.seed(SEED)
    random.shuffle(img_list)
    df = pd.read_csv(label_path)
    df = df.set_index('image_name')

    total_size = len(img_list)

    for idx, img in enumerate(img_list):
        img_name = img.split('\\')[-1].split('.')[0]
        tags = df.at[img_name, 'tags'].split(' ')
        label = np.zeros(len(LABELS), np.int32)
        for tag in tags:
            label[LABELS[tag]] = 1
        shape, binary_image = get_image_binary(img)
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tobytes()])),
            'shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[shape])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_image]))
        }))
        if idx <= TRAIN_VALIDATION_RATIO * total_size:
            writer_train.write(example.SerializeToString())
        else:
            writer_validation.write(example.SerializeToString())
        print('write: {} label: {}'.format(img_name, label))
    writer_train.close()
    writer_validation.close()

if __name__ == "__main__":
    write_to_tfrecord()
