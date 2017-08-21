import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

from generative.models.common import mlp
from report import create_default_report


class WGAN_GP:
    def __init__(self,
                 X_sampled,
                 latent_dim,
                 global_step,
                 generator_final_activation=tf.nn.sigmoid,
                 wgan_lambda=10.0,
                 n_critic_steps=5):

        flat_data_dim = int(np.prod(X_sampled.get_shape().as_list()[1:]))
        X_sampled_flat = flatten(X_sampled)

        def generator(Z):
            return mlp(Z, layer_sizes=(128, 128, 128, flat_data_dim),
                       intermediate_activation_fn=tf.nn.relu,
                       final_activation_fn=generator_final_activation)

        def critic(X):
            return mlp(X, layer_sizes=(128, 128, 128, 1),
                       intermediate_activation_fn=tf.nn.relu,
                       final_activation_fn=None)  # Note: No activation in final layer for Wasserstein GAN.

        N = tf.placeholder_with_default(tf.constant(64), shape=[])
        Z = tf.placeholder_with_default(tf.random_normal((N, latent_dim), mean=0.0, stddev=1.0),
                                        shape=(None, latent_dim))

        with tf.variable_scope("Generator"):
            X_fake = generator(Z)

        with tf.variable_scope("Critic"):
            C_real = critic(flatten(X_sampled))

        with tf.variable_scope("Critic", reuse=True):
            C_fake = critic(X_fake)

        with tf.variable_scope("Critic", reuse=True):
            epsilon = tf.random_uniform(shape=[tf.shape(X_fake)[0]], minval=0.0, maxval=1.0)
            epsilon = tf.expand_dims(epsilon, 1)  # TODO: Fix expand for higher dim data
            X_interpolated = epsilon * X_sampled_flat + (1.0 - epsilon) * X_fake
            C_X_interpolated_grads = tf.gradients(critic(X_interpolated), X_interpolated)[0]
            C_X_interpolated_grads_norm = tf.norm(C_X_interpolated_grads, ord=2, axis=1)
            gradient_penalty = wgan_lambda * tf.reduce_mean(tf.square((C_X_interpolated_grads_norm - 1.0)))

        with tf.name_scope("Training"):
            with tf.name_scope("Critic_loss"):
                # Note: Want to maximize mean output of C_real, minimize mean output of C_fake, and minimize
                # the gradient penalty.
                C_loss = -(tf.reduce_mean(C_real) - tf.reduce_mean(C_fake)) + gradient_penalty

            with tf.name_scope("Generator_loss"):
                # Note: Want to maximize mean output of C_fake.
                G_loss = -tf.reduce_mean(C_fake)

            C_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Critic")
            G_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator")

            self.C_optimization_step = (tf.train.AdamOptimizer(learning_rate=1e-4)
                                        .minimize(C_loss, global_step=None, var_list=C_variables))
            self.G_optimization_step = (tf.train.AdamOptimizer(learning_rate=1e-4)
                                        .minimize(G_loss, global_step=global_step, var_list=G_variables))

            self.global_step = global_step
            self.n_critic_steps = n_critic_steps

        if X_fake.get_shape().ndims > 2:
            self.X_generated = X_fake
        else:
            self.X_generated = tf.reshape(X_fake, [-1] + X_sampled.get_shape().as_list()[1:])

    def train_step(self, sess):
        for _ in range(self.n_critic_steps):
            sess.run_without_hooks(self.C_optimization_step)

        _, i = sess.run((self.G_optimization_step, self.global_step))
        return i

    def generate_results(self, sess, output_dir, param_settings):
        images = sess.run(self.X_generated)
        create_default_report(output_dir, param_settings, images)
