from contextlib import contextmanager

import tensorflow as tf


@contextmanager
def restored_session(model_path):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        yield sess
