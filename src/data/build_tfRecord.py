import glob
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from skimage import io

import src.settings as settings


def get_image_binary(filename):
    return np.asarray(io.imread(filename), np.int32).tobytes()


def write_to_tfrecord():
    writer_train = tf.python_io.TFRecordWriter(settings.TRAIN_DATA_PATH)
    writer_validation = tf.python_io.TFRecordWriter(settings.VALIDATION_DATA_PATH)

    img_list = glob.glob(os.path.join(settings.IMG_PATH, '*.tif'))
    random.seed(settings.SEED)
    random.shuffle(img_list)
    df = pd.read_csv(settings.LABEL_PATH)
    df = df.set_index('image_name')

    total_size = len(img_list)

    for idx, img in enumerate(img_list):
        img_name = img.split('\\')[-1].split('.')[0]
        tags = df.at[img_name, 'tags'].split(' ')
        label = np.zeros(settings.LABELS_SIZE, np.int32)
        for tag in tags:
            label[settings.LABELS[tag]] = 1

        binary_image = get_image_binary(img)
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tobytes()])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_image]))
        }))
        if idx <= settings.TRAIN_VALIDATION_RATIO * total_size:
            writer_train.write(example.SerializeToString())
        else:
            writer_validation.write(example.SerializeToString())
        print('write: {} label: {}'.format(img_name, label))
    writer_train.close()
    writer_validation.close()

if __name__ == "__main__":
    write_to_tfrecord()
