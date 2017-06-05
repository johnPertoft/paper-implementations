from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

from generative.models.common import mlp


class VAE:
    def __init__(self, X_sampled, latent_dim, global_step):
        flat_data_dim = int(np.prod(X_sampled.get_shape().as_list()[1:]))
        X_sampled_flat = flatten(X_sampled)

        with tf.variable_scope("Encoder"):
            h = mlp(X_sampled_flat, layer_sizes=(128, 128),
                    intermediate_activation_fn=tf.nn.relu,
                    final_activation_fn=tf.nn.relu)

            # Computed parameters for encoding distribution.
            Z_mu = tf.layers.dense(h, latent_dim, activation=None)
            Z_log_var = tf.layers.dense(h, latent_dim, activation=None)

            # Sampling through reparametrization z = mu + sigma * epsilon
            # L = 1, i.e. one sampling of z is enough if the batch size is big enough according to VAE paper.
            epsilon = tf.random_normal(tf.shape(Z_log_var), mean=0.0, stddev=1.0)
            Z = Z_mu + tf.exp(0.5 * Z_log_var) * epsilon

        with tf.variable_scope("Decoder"):
            logits = mlp(Z, layer_sizes=(128, 128, flat_data_dim),
                         intermediate_activation_fn=tf.nn.relu,
                         final_activation_fn=None)

            # Computed parameters for multiple Bernoullis.
            bernoulli_probs = tf.nn.sigmoid(logits)

        with tf.name_scope("Training"):
            with tf.name_scope("Loss"):
                with tf.name_scope("KL_loss"):
                    # KL loss for gaussian encoding distribution, q(z|x).
                    # Negative since we want to maximize the lower bound estimator.
                    kl_loss = -0.5 * tf.reduce_sum(1.0 + Z_log_var - tf.square(Z_mu) - tf.exp(Z_log_var), axis=1)

                with tf.name_scope("Reconstruction_loss"):
                    # Expected negative log-likelihood as loss.
                    reconstruction_loss = \
                        tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                              labels=X_sampled_flat), axis=1)

                # TODO: Maybe try KL cost annealing.
                loss = tf.reduce_mean(kl_loss + reconstruction_loss)

                self.optimization_step = (tf.train.AdamOptimizer(learning_rate=1e-4)
                                          .minimize(loss, global_step=global_step))

        self.X_generated = bernoulli_probs

    def train_step(self, sess):
        sess.run(self.optimization_step)

    def generate_results(self, sess):
        pass
