import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

from generative.models.common import mlp
from report import create_default_report


class InfoGAN:
    def __init__(self, X_sampled, latent_dim, global_step):
        data_dim = X_sampled.get_shape().as_list()[1:]
        flat_data_dim = int(np.prod(data_dim))

        def generator(ZC):
            return mlp(ZC, layer_sizes=(128, 128, 128, flat_data_dim),
                       intermediate_activation_fn=tf.nn.relu,
                       final_activation_fn=tf.nn.sigmoid)

        def discriminator_q_shared(X):
            return mlp(X, layer_sizes=(128, 128, 128),
                       intermediate_activation_fn=tf.nn.relu,
                       final_activation_fn=tf.nn.relu)

        # TODO: see the cat generator project

        def discriminator(X):
            h = discriminator_q_shared(X)
            return tf.layers.dense(h, 1, activation_fn=tf.sigmoid)

        def q_network(X):
            h = discriminator_q_shared(X)
            # TODO: Use uniform distribution here

        with tf.variable_scope("Generator"):
            pass

        with tf.variable_scope("Discriminator"):
            pass

        with tf.variable_scope("Discriminator", reuse=True):
            pass

    def train_step(self, sess):
        pass

    def generate_results(self, sess, output_dir, param_settings):
        # TODO: this should have an extra experiment showing the disentanglement of latent space
        # TODO: make create_default_report be able to return the markdown builder

        images = sess.run(self.X_generated)
        md_builder = create_default_report(output_dir, param_settings, images, return_builder=True)
