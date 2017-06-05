import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

from generative.models.common import mlp


class WGAN_GP:
    def __init__(self, X_sampled, latent_dim, global_step, wgan_lambda=10.0, n_critic_steps=5):
        flat_data_dim = int(np.prod(X_sampled.get_shape().as_list()[1:]))
        X_sampled_flat = flatten(X_sampled)

        def generator(Z):
            # TODO: Always want sigmoid at end here?
            return mlp(Z, layer_sizes=(128, 128, 128, flat_data_dim),
                       intermediate_activation_fn=tf.nn.relu,
                       final_activation_fn=tf.nn.sigmoid)

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
            # TODO: Maybe expand dims if epsilon needed
            X_interpolated = epsilon * X_sampled_flat + (1.0 - epsilon) * X_fake
            C_X_interpolated_grads = tf.gradients(critic(X_interpolated), X_interpolated)[0]
            C_X_interpolated_grads_norm = tf.norm(C_X_interpolated_grads, ord=2, axis=1)
            gradient_penalty = wgan_lambda * tf.reduce_mean(tf.square((C_X_interpolated_grads_norm - 1.0)))

        with tf.name_scope("Training"):
            with tf.name_scope("Critic_loss"):
                pass

            with tf.name_scope("Generator_loss"):
                pass

    def train_step(self, sess):
        pass

    def generate_results(self, sess):
        pass
