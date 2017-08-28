import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

from generative.models.common import mlp, BaseGenerativeModel
from report import create_default_report


class WGAN_GP(BaseGenerativeModel):
    def __init__(self,
                 generator,
                 critic,
                 X_sampled,
                 latent_dim,
                 wgan_lambda,
                 n_critic_steps):

        # TODO assertions about generator and critic being callable
        # and X_sampled shape fitting generator and critic?

        N = tf.placeholder_with_default(tf.constant(64), shape=[])
        Z = tf.placeholder_with_default(tf.random_normal((N, latent_dim), mean=0.0, stddev=1.0),
                                        shape=(None, latent_dim))

        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self.n_critic_steps = n_critic_steps

        with tf.variable_scope("Generator"):
            self.X_generated = X_fake = generator(Z)

        with tf.variable_scope("Critic"):
            C_real = critic(X_sampled)

        with tf.variable_scope("Critic", reuse=True):
            C_fake = critic(X_fake)

        with tf.variable_scope("Critic", reuse=True):
            epsilon_shape = tf.concat((tf.shape(X_fake)[:1], tf.ones_like(tf.shape(X_fake)[1:])), axis=0)
            epsilon = tf.random_uniform(shape=epsilon_shape, minval=0.0, maxval=1.0)
            X_interpolated = epsilon * X_sampled + (1.0 - epsilon) * X_fake
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
                                        .minimize(G_loss, global_step=self.global_step, var_list=G_variables))

    def train_step(self, sess):
        for _ in range(self.n_critic_steps):
            sess.run_without_hooks(self.C_optimization_step)

        _, i = sess.run((self.G_optimization_step, self.global_step))
        return i

    def generate_results(self, sess, output_dir, param_settings):
        images = sess.run(self.X_generated)
        create_default_report(output_dir, param_settings, images)

    @classmethod
    def create(cls,
               dataset,
               generator_architecture,
               critic_architecture,
               X_sampled,
               generator_final_activation_fn=tf.nn.sigmoid,
               latent_dim=25,
               wgan_lambda=10.0,
               n_critic_steps=5):

        generator = None
        critic = None

        flat_data_dim = int(np.prod(X_sampled.get_shape().as_list()[1:]))

        # TODO: Probably put all these in common somewhere? more fair comparison
        # but at the same time, some ideas like wgan allows for more complex architectures

        def mnist_mlp_generator(Z):
            X_flat = mlp(Z,
                         layer_sizes=(128, 128, 128, flat_data_dim),
                         intermediate_activation_fn=tf.nn.relu,
                         final_activation_fn=generator_final_activation_fn)
            return tf.reshape(X_flat, [-1] + X_sampled.get_shape().as_list()[1:])

        def mnist_mlp_critic(X):
            return mlp(flatten(X),
                       layer_sizes=(128, 128, 128, 1),
                       intermediate_activation_fn=tf.nn.relu,
                       final_activation_fn=None)  # Note: No activation in final layer for Wasserstein GAN.

        def mnist_conv_generator(Z):
            pass

        def mnist_conv_critic(X):
            pass

        def cifar10_mlp_generator(Z):
            pass

        def cifar10_mlp_critic(X):
            pass

        dataset = dataset.lower()
        if dataset == "mnist":
            if generator_architecture == "mlp":
                generator = mnist_mlp_generator

            if critic_architecture == "mlp":
                critic = mnist_mlp_critic

        if generator is None or critic is None:
            raise ValueError("Critic or discriminator could not be defined.")

        return cls(generator,
                   critic,
                   X_sampled,
                   latent_dim,
                   wgan_lambda,
                   n_critic_steps)
