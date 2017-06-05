import tensorflow as tf


class ExtendedMonitoredTrainingSession:
    """
    Simple wrapper class to provide a run method on which hooks will not run.
    """

    def __init__(self,
                 master='',
                 is_chief=True,
                 checkpoint_dir=None,
                 scaffold=None,
                 hooks=None,
                 chief_only_hooks=None,
                 save_checkpoint_secs=600,
                 save_summaries_steps=100,
                 save_summaries_secs=None):

        self._monitored_training_session = tf.train.MonitoredTrainingSession(master,
                                                                             is_chief,
                                                                             checkpoint_dir,
                                                                             scaffold,
                                                                             hooks,
                                                                             chief_only_hooks,
                                                                             save_checkpoint_secs,
                                                                             save_summaries_steps,
                                                                             save_summaries_secs)

    def __enter__(self):
        self._monitored_training_session.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._monitored_training_session.__exit__(exc_type, exc_val, exc_tb)

    def should_stop(self):
        return self._monitored_training_session.should_stop()

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        return self._monitored_training_session.run(fetches, feed_dict, options, run_metadata)

    def run_without_hooks(self, fetches, feed_dict=None, options=None, run_metadata=None):
        return self._monitored_training_session._sess._sess._sess._sess.run(fetches, feed_dict, options, run_metadata)
