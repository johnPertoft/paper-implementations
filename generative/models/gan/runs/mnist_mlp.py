import os
from collections import OrderedDict

import tensorflow as tf

from generative import run_training
from generative.models import GAN
from data.mnist import maybe_download_and_prepare, input_tensor
from tf_utils import restored_session
from report import timestamp


if __name__ == "__main__":
    # Run parameters
    settings = OrderedDict([("Architecture", "mlp"),
                            ("Batch size", 64),
                            ("Latent dim", 25),
                            ("N training steps", 500000)])

    run_name = timestamp()  # TODO: Add run parameters to name

    mnist_tfrecords_path = maybe_download_and_prepare(os.path.join(os.path.expanduser("~"), "datasets"))
    X_sampled = input_tensor(mnist_tfrecords_path, batch_size=settings["Batch size"], return_label=False)

    gan = GAN(X_sampled,
              latent_dim=settings["Latent dim"],
              global_step=tf.contrib.framework.get_or_create_global_step())

    model_name = GAN.__name__
    log_dir = os.path.join("log_dir", model_name, run_name)
    results_dir = os.path.join("results", model_name, run_name)

    model_path = run_training(gan, log_dir, n_training_steps=settings["N training steps"])
    with restored_session(model_path) as sess:
        gan.generate_results(sess, results_dir, list(settings.items()))
