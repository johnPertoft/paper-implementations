from contextlib import contextmanager

import tensorflow as tf


@contextmanager
def test_session_setup(sess):
    """
    Context manager for making tf.test.TestCase.test_session compatible with the models' train steps.
    Also runs initializer ops.
    :param sess: session
    :return: session
    """

    # The models' train_step expects a no_hook_run method, so we add this dynamically here on the test_session.
    def run_without_hooks(fetches, feed_dict=None, options=None, run_metadata=None):
        return sess.run(fetches, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
    sess.run_without_hooks = run_without_hooks

    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    yield sess
