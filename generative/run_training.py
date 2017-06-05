import tensorflow as tf

from tf_utils import ExtendedMonitoredTrainingSession


def run_training(model, log_dir, n_training_steps):
    training_hooks = [tf.train.StopAtStepHook(last_step=n_training_steps)]

    with ExtendedMonitoredTrainingSession(checkpoint_dir=log_dir,
                                          hooks=training_hooks,
                                          save_summaries_steps=10) as sess:
        while not sess.should_stop():
            # TODO: report progress.
            model.train_step(sess)

        return tf.train.latest_checkpoint(log_dir)
