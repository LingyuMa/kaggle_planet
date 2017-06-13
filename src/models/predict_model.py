import tensorflow as tf
import time
from datetime import datetime
import numpy as np
import math

import src.models.network as network
import src.data.data_provider as data

import src.settings as settings


def eval_once(saver, summary_writer, correct_prediction, summary_op):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(settings.LOG_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
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
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * settings.BATCH_SIZE
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = np.array(sess.run([correct_prediction])).squeeze()
                true_count += np.all(predictions, axis=1).sum()
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
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
        logits = network.inference(images, training=False)

        # Calculate predictions
        correct_prediction = tf.equal(tf.round(tf.sigmoid(logits)), tf.round(tf.cast(labels, tf.float32)))

        # Restore the moving average version of the learned variables for eval
        variable_averages = tf.train.ExponentialMovingAverage(
            settings.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(settings.EVALUATION_PATH, g)

        while True:
            eval_once(saver, summary_writer, correct_prediction, summary_op)
            if settings.RUN_ONCE:
                break
            time.sleep(settings.EVALUATION_INTERVAL_SECS)

if __name__ == "__main__":
    evaluation()
