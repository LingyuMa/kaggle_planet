import glob
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from skimage import io, transform
from scipy import ndimage

import src.settings as settings


def get_image(filename, offset):
    image = np.asarray(io.imread(filename), np.int32)
    if settings.ZERO_MEAN_IMAGE:
        image -= offset
    return image


def get_mean():
    return np.load(os.path.join(settings.IMG_PATH, 'img_mean.npy')).astype(np.int32)


def write_one_record(tfwriter, image, label):
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tobytes()])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))
    }))
    tfwriter.write(example.SerializeToString())


def augment_images(image_list, augment_flag):
    original_image = image_list[0]
    if augment_flag.startswith('flip'):
        image_list.append(np.flipud(original_image))
        image_list.append(np.fliplr(original_image))
    if augment_flag.startswith('rotate'):
        n_times = int(np.floor(360 / int(augment_flag.split('_')[-1]) - 1))
        for i in range(n_times):
            angle = (i + 1) * float(augment_flag.split('_')[-1])
            image_list.append(ndimage.rotate(original_image, angle, reshape=False, mode='reflect'))
    return image_list


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

    labels_count = np.zeros(len(settings.LABELS[settings.NETWORK_ID]))
    total_count = 0

    for idx, img in enumerate(img_list):
        img_name = img.split(
            '\\')[-1].split('.')[0]
        tags = df.at[img_name, 'tags'].split(' ')
        label = np.zeros(len(settings.LABELS[settings.NETWORK_ID]), np.int32)
        augment_flag = None
        for tag in tags:
            if tag in settings.LABELS[settings.NETWORK_ID]:
                label[settings.LABELS[settings.NETWORK_ID][tag]] = 1
                if tag in settings.AUGMENT_FLAG:
                    augment_flag = settings.AUGMENT_FLAG[tag]

        images = [get_image(img, total_mean)]

        if idx <= settings.TRAIN_VALIDATION_RATIO * total_size and augment_flag is not None:
            images = augment_images(images, augment_flag)
        print('augmented images: {}'.format(len(images)))
        if idx <= settings.TRAIN_VALIDATION_RATIO * total_size:
            for i in range(len(images)):
                write_one_record(writer_train, transform.resize(images[i],
                                                                (settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH),
                                                                mode='constant').astype(np.float16), label)
                labels_count += label
                total_count += 1
        else:
            write_one_record(writer_validation, transform.resize(images[0],
                                                                 (settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH),
                                                                 mode='constant').astype(np.float16), label)
        print('write: {} label: {}'.format(img_name, label))
    writer_train.close()
    writer_validation.close()
    np.savetxt(os.path.join(settings.TRAIN_DATA_PATH, '../tfrecord_summary.txt'), np.hstack((labels_count, total_count)))


def calc_mean():
    if os.path.exists(os.path.join(settings.IMG_PATH, 'img_mean.npy')):
        return
    img_list = glob.glob(os.path.join(settings.IMG_PATH, '*.tif'))
    img_mean = np.zeros((settings.RAW_IMAGE_HEIGHT, settings.RAW_IMAGE_HEIGHT, settings.RAW_IMAGE_DEPTH))
    for img_path in img_list:
        temp = np.asarray(io.imread(img_path), np.int32)
        img_mean += temp
    img_mean /= len(img_list)
    img_mean = np.round(img_mean).astype(np.int32)
    np.save(os.path.join(settings.IMG_PATH, 'img_mean'), img_mean)


if __name__ == "__main__":
    calc_mean()
    write_to_tfrecord()


