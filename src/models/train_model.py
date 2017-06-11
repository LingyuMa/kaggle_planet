import tensorflow as tf
import time
from datetime import datetime

import src.models.network as network
import src.data.data_provider as data

import src.settings as settings


def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Force input on CPU
        with tf.device('/cpu:0'):
            images, labels = data.train_inputs(settings.BATCH_SIZE)

        # Build the graph
        logits = network.inference(images)

        # Calculate loss
        loss = network.loss(logits, labels)

        train_op = network.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % settings.LOG_FREQUENCY == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = settings.LOG_FREQUENCY * settings.BATCH_SIZE / duration
                    sec_per_batch = float(duration / settings.LOG_FREQUENCY)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=settings.LOG_PATH,
                hooks=[tf.train.StopAtStepHook(last_step=settings.MAX_STEPS),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                config=tf.ConfigProto(
                    log_device_placement=settings.LOG_DEVICE_PLACEMENT)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


if __name__ == "__main__":
    train()
