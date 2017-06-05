import numpy as np
import tensorflow as tf

from generative.models.common import mlp


class GAN:
    def __init__(self, X_sampled, latent_dim, global_step):
        flat_data_dim = int(np.prod(X_sampled.get_shape().as_list()[1:]))

        def generator(Z):
            return mlp(Z, layer_sizes=(128, 128, flat_data_dim),
                       intermediate_activation_fn=tf.nn.relu,
                       final_activation_fn=tf.nn.sigmoid)

        def discriminator(X):
            return mlp(X, layer_sizes=(128, 128, 1),
                       intermediate_activation_fn=tf.nn.relu,
                       final_activation_fn=tf.nn.sigmoid)

        N = tf.placeholder_with_default(tf.constant(64), shape=[])
        Z = tf.placeholder_with_default(tf.random_normal((N, latent_dim), mean=0.0, stddev=1.0),
                                        shape=(None, latent_dim))

        with tf.variable_scope("Generator"):
            X_fake = generator(Z)

        with tf.variable_scope("Discriminator"):
            D_real = discriminator(X_sampled)

        with tf.variable_scope("Discriminator", reuse=True):
            D_fake = discriminator(X_fake)

        with tf.name_scope("Training"):
            def log(x):
                return tf.log(tf.clip_by_value(x, 1e-6, 1.0))

            with tf.variable_scope("Discriminator_loss"):
                # Note: Want to maximize probability for real samples and minimize probability for fake samples.
                D_loss = -(tf.reduce_mean(log(D_real)) + tf.reduce_mean(log(1.0 - D_fake)))
                # TODO: Try some alternatives losses here maybe.

            with tf.variable_scope("Generator_loss"):
                G_loss = tf.reduce_mean(log(1.0 - D_fake))

            D_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")
            G_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator")

            self.D_optimization_step = (tf.train.AdamOptimizer(learning_rate=1e-4)
                                        .minimize(D_loss, global_step=None, var_list=D_variables))
            self.G_optimization_step = (tf.train.AdamOptimizer(learning_rate=1e-4)
                                        .minimize(G_loss, global_step=global_step, var_list=G_variables))

    def train_step(self, sess):
        # TODO: Or run them in same sess.run but with control dependencies.
        sess.run_without_hooks(self.D_optimization_step)
        sess.run(self.G_optimization_hook)

    def generate_results(self, sess):
        pass
