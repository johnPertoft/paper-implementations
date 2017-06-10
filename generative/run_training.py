import tensorflow as tf
from tqdm import tqdm

from tf_utils import ExtendedMonitoredTrainingSession


def run_training(model, log_dir, n_training_steps):
    training_hooks = [tf.train.StopAtStepHook(last_step=n_training_steps)]

    with ExtendedMonitoredTrainingSession(checkpoint_dir=log_dir,
                                          hooks=training_hooks,
                                          save_summaries_steps=10) as sess:
        with tqdm(total=n_training_steps) as pbar:
            while not sess.should_stop():
                i = model.train_step(sess)
                pbar.update()

        return tf.train.latest_checkpoint(log_dir)
