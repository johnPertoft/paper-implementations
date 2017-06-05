import os

import tensorflow as tf

from generative.models import VAE
from generative import run_training
from data.mnist import maybe_download_and_prepare, input_tensor


if __name__ == "__main__":
    # TODO: these parameters should be visible somewhere
    # Run parameters.
    batch_size = 64
    latent_dim = 25
    n_training_steps = 500000

    mnist_tfrecords_path = maybe_download_and_prepare(os.path.join(os.path.expanduser("~"), "datasets"))
    X_sampled = input_tensor(mnist_tfrecords_path, batch_size=batch_size, return_label=False)

    vae = VAE(X_sampled,
              latent_dim=latent_dim,
              global_step=tf.contrib.framework.get_or_create_global_step())

    # TODO: set unique log_dir for this run, clear it before?
    # TODO: Let this return generated stuff as well, then print them to a results file (markdown)
    saved_model_path = run_training(vae, "log_dir/vae", n_training_steps=n_training_steps)
