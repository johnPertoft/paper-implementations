import os
from collections import OrderedDict

import tensorflow as tf

from generative import run_training
from generative.models import WGAN_GP
from data.mnist import maybe_download_and_prepare, input_tensor
from tf_utils import restored_session
from report import timestamp


if __name__ == "__main__":
    # Run parameters
    settings = OrderedDict([("Batch size", 64),
                            ("Latent dim", 25),
                            ("lambda", 10.0),
                            ("N critic steps", 5),
                            ("N training steps", 500000)])

    mnist_tfrecords_path = maybe_download_and_prepare(os.path.join(os.path.expanduser("~"), "datasets"))
    X_sampled = input_tensor(mnist_tfrecords_path, batch_size=settings["Batch size"], return_label=False)

    wgan_gp = WGAN_GP.create("mnist",
                             generator_architecture="mlp",
                             critic_architecture="mlp",
                             X_sampled=X_sampled,
                             generator_final_activation_fn=tf.nn.sigmoid,
                             latent_dim=settings["Latent dim"],
                             wgan_lambda=settings["lambda"],
                             n_critic_steps=settings["N critic steps"])

    run_name = timestamp()  # TODO: Add run parameters to name
    model_name = WGAN_GP.__name__
    log_dir = os.path.join("log_dir", model_name, run_name)
    results_dir = os.path.join("results", model_name, run_name)

    model_path = run_training(wgan_gp, log_dir, n_training_steps=settings["N training steps"])
    with restored_session(model_path) as sess:
        wgan_gp.generate_results(sess, results_dir, list(settings.items()))
