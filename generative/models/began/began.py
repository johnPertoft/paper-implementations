import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

from generative.models.common import mlp
from report import create_default_report


class BEGAN:
    def __init__(self,
                 X_sampled,
                 latent_dim,
                 global_step,
                 autoencoder_hidden_dim=32,
                 k_lambda=0.001,
                 k_gamma=0.5):

        data_dim = X_sampled.get_shape().as_list()[1:]
        flat_data_dim = int(np.prod(data_dim))

        def generator(Z):
            return mlp(Z, layer_sizes=(128, 128, 128, flat_data_dim),
                       intermediate_activation_fn=tf.nn.relu,
                       final_activation_fn=tf.nn.sigmoid)

        def discriminator(X):
            # Note: Discriminator is an autoencoder in BEGAN paper.
            X_reconstructed = mlp(X, layer_sizes=(128, autoencoder_hidden_dim, 128, flat_data_dim),
                                  intermediate_activation_fn=tf.nn.relu,
                                  final_activation_fn=tf.nn.sigmoid)
            return tf.reduce_sum((X - X_reconstructed) ** 2, axis=1)

        N = tf.placeholder_with_default(tf.constant(64), shape=[])
        Z = tf.placeholder_with_default(tf.random_normal((N, latent_dim), mean=0.0, stddev=1.0),
                                        shape=(None, latent_dim))

        with tf.variable_scope("Generator"):
            X_fake = generator(Z)

        with tf.variable_scope("Discriminator"):
            D_real = discriminator(flatten(X_sampled))

        with tf.variable_scope("Discriminator", reuse=True):
            D_fake = discriminator(X_fake)

        with tf.name_scope("Training"):
            with tf.name_scope("Discriminator_loss"):
                k = tf.Variable(0.0)
                D_loss = tf.reduce_mean(D_real) - k * tf.reduce_mean(D_fake)

            with tf.name_scope("Generator_loss"):
                G_loss = tf.reduce_mean(D_fake)

            D_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")
            G_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator")

            self.global_step = global_step
            self.D_optimization_step = (tf.train.AdamOptimizer(learning_rate=1e-4)
                                        .minimize(D_loss, global_step=None, var_list=D_variables))
            self.G_optimization_step = (tf.train.AdamOptimizer(learning_rate=1e-4)
                                        .minimize(G_loss, global_step=global_step, var_list=G_variables))
            with tf.control_dependencies([self.G_optimization_step]):
                self.update_k = tf.assign_add(k, k_lambda * (k_gamma * tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)))

            self.summary = tf.summary.merge((tf.summary.scalar("D_loss", D_loss),
                                             tf.summary.scalar("G_loss", G_loss),
                                             tf.summary.scalar("k", k)))

        if X_fake.get_shape().ndims > 2:
            self.X_generated = X_fake
        else:
            self.X_generated = tf.reshape(X_fake, [-1] + X_sampled.get_shape().as_list()[1:])

    def train_step(self, sess):
        sess.run_without_hooks(self.D_optimization_step)
        sess.run((self.G_optimization_step, self.update_k, self.global_step))

    def generate_results(self, sess, output_dir, param_settings):
        images = sess.run(self.X_generated)
        create_default_report(output_dir, param_settings, images)
