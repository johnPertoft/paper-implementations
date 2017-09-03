import os

import tensorflow as tf

from data.mnist import maybe_download_and_prepare, create_input_fn
from generative.models.wgan_gp.wgan_gp2 import model_fn

# TODO: Most stuff in this file is common for all experiments, move it.

# TODO: import generator and critic from common module somewhere
# or define as functions in each experiment/run file
# should be flexible enough.

# TODO: create model_fn_wrapping function, decorator?


tf.logging.set_verbosity(tf.logging.INFO)

tf.flags.DEFINE_integer("save_summary_steps", 100, "Summary steps.")
tf.flags.DEFINE_integer("save_checkpoints_steps", 200, "Checkpoint steps.")
tf.flags.DEFINE_integer("eval_steps", None, "Number of eval steps.")
tf.flags.DEFINE_integer("eval_frequency", 1, "Eval frequency.")
FLAGS = tf.flags.FLAGS


mnist_tfrecords_path = maybe_download_and_prepare(os.path.join(os.path.expanduser("~"), "datasets"))


def create_model_fn(model_fn):
    def _model_fn(X_sampled, labels, mode, params, config):
        return model_fn(generator, critic, X_sampled, mode, params)

    return _model_fn

hparams = tf.contrib.training.HParams(latent_dim=25,
                                      wgan_lambda=10.0,
                                      n_critic_steps=5)

estimator = tf.estimator.Estimator(model_fn=create_model_fn(model_fn),
                                   params=hparams,
                                   config=create_default_run_config())

# TODO: Maybe better to define an experiment_fn to allow for hyper parameter tuning.
experiment = tf.contrib.learn.Experiment(estimator=estimator,
                                         train_input_fn=create_input_fn(mnist_tfrecords_path, batch_size=64),
                                         eval_input_fn=None,
                                         eval_steps=FLAGS.eval_steps,
                                         min_eval_frequency=FLAGS.eval_frequency)

# TODO: train hooks for stopping
train_hooks = []
experiment.extend_train_hooks(train_hooks)


def main(argv):

    tf.contrib.learn.learn_runner()


if __name__ == "__main__":
    tf.app.run()
