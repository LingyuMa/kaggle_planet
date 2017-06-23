import tensorflow as tf
import time
from datetime import datetime
import numpy as np
import math
import os

import src.models.network as network
import src.data.data_provider as data

import src.settings as settings

from sklearn.metrics import fbeta_score


def f2_score(y_true, y_pred):
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(tf.round(y_pred), "int32") # implicit 0.5 threshold via tf.round
    y_correct = y_true * y_pred
    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_pred = tf.reduce_sum(y_pred, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)
    precision = sum_correct / sum_pred
    recall = sum_correct / sum_true
    f_score = 5 * precision * recall / (4 * precision + recall)
    f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
    return tf.reduce_mean(f_score)


def eval_once(saver, summary_writer, y_pred, y_labels, summary_op):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(settings.LOG_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model restored from {}'.format(ckpt.model_checkpoint_path))
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(settings.EVALUATION_NUM_EXAMPLES / settings.BATCH_SIZE))
            step = 0
            predictions = []
            labels = []
            while step < num_iter and not coord.should_stop():
                preds, labs = sess.run([y_pred, y_labels])
                predictions.append(np.array(preds).squeeze())
                labels.append(np.array(labs).squeeze())
                step += 1
            predictions = np.vstack(predictions)
            labels = np.vstack(labels)
            # Compute precision @ 1.
            np.savetxt(os.path.join(settings.EVALUATION_PATH, 'out_labels.txt'), labels)
            np.savetxt(os.path.join(settings.EVALUATION_PATH, 'out_predictions.txt'), predictions)
            precision = fbeta_score(labels.astype(np.float32),
                                    np.round((predictions > 0.5).astype(np.float32)),
                                    beta=2, average='samples')
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluation():
    with tf.Graph().as_default() as g:
        # Feed data
        images, labels = data.inputs(False, settings.BATCH_SIZE)

        # Inference model
        logits = network.inference(images, settings.NUM_RESIDUE_BLOCKS)

        # Calculate predictions
        y_pred = tf.sigmoid(logits)
        y_labels = labels

        # Restore model
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(settings.EVALUATION_PATH, g)

        while True:
            eval_once(saver, summary_writer, y_pred, y_labels, summary_op)
            if settings.RUN_ONCE:
                break
            time.sleep(settings.EVALUATION_INTERVAL_SECS)

if __name__ == "__main__":
    evaluation()
