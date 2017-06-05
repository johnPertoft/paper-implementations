import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from PIL import Image

from generative.models.common import mlp
from report import MarkdownDocumentBuilder


class WGAN:
    def __init__(self, X_sampled, latent_dim, global_step, weight_clip=0.01, n_critic_steps=5):
        data_dim = X_sampled.get_shape().as_list()[1:]
        flat_data_dim = int(np.prod(data_dim))

        # TODO: define generator to always output something of data_dim
        # TODO: critic should not take flattened stuff

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

        with tf.name_scope("Training"):
            with tf.name_scope("Critic_loss"):
                # Note: Want to maximize the mean output of C_real and minimize the mean output of C_fake.
                C_loss = -(tf.reduce_mean(C_real) - tf.reduce_mean(C_fake))

            with tf.name_scope("Generator_loss"):
                # Note: Want to maximize the mean output of C_fake.
                G_loss = -tf.reduce_mean(C_fake)

            C_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Critic")
            G_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator")

            # Note: Paper states that momentum based optimizers like Adam performs worse with WGAN training.
            self.C_optimization_step = (tf.train.RMSPropOptimizer(learning_rate=5e-5)
                                        .minimize(C_loss, global_step=None, var_list=C_variables))

            self.G_optimization_step = (tf.train.RMSPropOptimizer(learning_rate=5e-5)
                                        .minimize(G_loss, global_step=global_step, var_list=G_variables))

            self.global_step = global_step

            with tf.control_dependencies([self.C_optimization_step]):
                self.C_param_clip = tf.group(*[p.assign(tf.clip_by_value(p, -weight_clip, weight_clip))
                                               for p in C_variables])

            self.n_critic_steps = n_critic_steps

        if X_fake.get_shape().ndims > 2:
            self.X_generated = X_fake
        else:
            self.X_generated = tf.reshape(X_fake, [-1] + X_sampled.get_shape().as_list()[1:])

    def train_step(self, sess):
        for _ in range(self.n_critic_steps):
            sess.run_without_hooks((self.C_optimization_step, self.C_param_clip))

        _, i = sess.run((self.G_optimization_step, self.global_step))
        print(i)  # TODO: temp, do this with hooks instead.

    def generate_results(self, sess, output_dir, param_settings):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        img_dir = os.path.join(output_dir, "imgs")
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)

        images = sess.run(self.X_generated)
        images = (images * 255.0).astype(np.uint8)

        def save_img(img, path):
            Image.fromarray(img).save(path)
            return path

        img_paths = [save_img(img, os.path.join(img_dir, "img{}.png".format(i))) for i, img in enumerate(images)]
        relative_img_paths = [os.path.relpath(path, output_dir) for path in img_paths]

        md_builder = MarkdownDocumentBuilder()
        md_builder.add_header("Run Settings")
        md_builder.add_table(param_settings)
        md_builder.add_header("Generated Images")
        md_builder.add_images(relative_img_paths)
        md_builder.build(os.path.join(output_dir, "Results.md"))
