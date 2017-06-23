import glob
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from skimage import io

import src.settings as settings


def get_image_binary(filename, offset):
    image = np.asarray(io.imread(filename), np.int32)
    if settings.ZERO_MEAN_IMAGE:
        print('apply offset')
        image -= offset
    return image.tobytes()


def get_mean():
    total_mean = np.empty([settings.RAW_IMAGE_HEIGHT, settings.RAW_IMAGE_WIDTH, settings.RAW_IMAGE_DEPTH]).\
        astype(np.int32)
    total_mean[:, :, 0] = np.loadtxt(os.path.join(settings.IMG_PATH, 'R_mean.txt')).astype(np.int32)
    total_mean[:, :, 1] = np.loadtxt(os.path.join(settings.IMG_PATH, 'G_mean.txt')).astype(np.int32)
    total_mean[:, :, 2] = np.loadtxt(os.path.join(settings.IMG_PATH, 'B_mean.txt')).astype(np.int32)
    total_mean[:, :, 3] = np.loadtxt(os.path.join(settings.IMG_PATH, 'I_mean.txt')).astype(np.int32)
    return total_mean


def write_to_tfrecord():
    writer_train = tf.python_io.TFRecordWriter(settings.TRAIN_DATA_PATH)
    writer_validation = tf.python_io.TFRecordWriter(settings.VALIDATION_DATA_PATH)

    img_list = glob.glob(os.path.join(settings.IMG_PATH, '*.tif'))
    random.seed(settings.SEED)
    random.shuffle(img_list)
    df = pd.read_csv(settings.LABEL_PATH)
    df = df.set_index('image_name')

    total_size = len(img_list)
    total_mean = get_mean()

    for idx, img in enumerate(img_list):
        img_name = img.split('\\')[-1].split('.')[0]
        tags = df.at[img_name, 'tags'].split(' ')
        label = np.zeros(settings.LABELS_SIZE, np.int32)
        for tag in tags:
            label[settings.LABELS[tag]] = 1

        binary_image = get_image_binary(img, total_mean)
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


def calc_mean():
    if os.path.exists(os.path.join(settings.IMG_PATH, 'R_mean.txt')) and \
       os.path.exists(os.path.join(settings.IMG_PATH, 'G_mean.txt')) and \
       os.path.exists(os.path.join(settings.IMG_PATH, 'B_mean.txt')) and \
       os.path.exists(os.path.join(settings.IMG_PATH, 'I_mean.txt')):
        return
    img_list = glob.glob(os.path.join(settings.IMG_PATH, '*.tif'))
    img_mean = np.zeros((settings.RAW_IMAGE_HEIGHT, settings.RAW_IMAGE_WIDTH, settings.RAW_IMAGE_DEPTH))
    for img_path in img_list:
        temp = np.asarray(io.imread(img_path), np.int32)
        img_mean += temp
    img_mean /= len(img_list)
    img_mean = np.round(img_mean).astype(np.int32)
    np.savetxt(os.path.join(settings.IMG_PATH, 'R_mean.txt'), img_mean[:, :, 0])
    np.savetxt(os.path.join(settings.IMG_PATH, 'G_mean.txt'), img_mean[:, :, 1])
    np.savetxt(os.path.join(settings.IMG_PATH, 'B_mean.txt'), img_mean[:, :, 2])
    np.savetxt(os.path.join(settings.IMG_PATH, 'I_mean.txt'), img_mean[:, :, 3])


if __name__ == "__main__":
    calc_mean()
    write_to_tfrecord()
