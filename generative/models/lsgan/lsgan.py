import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from PIL import Image

from generative.models.common import mlp, leaky_relu
from report import MarkdownDocumentBuilder


class LSGAN:
    def __init__(self, X_sampled, latent_dim, global_step):
        data_dim = X_sampled.get_shape().as_list()[1:]
        flat_data_dim = int(np.prod(data_dim))

        def generator(Z):
            return mlp(Z, layer_sizes=(128, 128, 128, flat_data_dim),
                       intermediate_activation_fn=tf.nn.relu,
                       final_activation_fn=tf.nn.sigmoid)

        def discriminator(X):
            return mlp(X, layer_sizes=(128, 128, 128, 1),
                       intermediate_activation_fn=leaky_relu,
                       final_activation_fn=None)  # Note: No activation in final layer for Least Squares GAN.

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
                # Note: Want to keep discriminator output for real samples close to 1 and minimize
                # discriminator output for fake samples.
                D_loss = 0.5 * (tf.reduce_mean((D_real - 1.0) ** 2.0) + tf.reduce_mean(D_fake ** 2.0))

            with tf.name_scope("Generator_loss"):
                # Note: Want to keep discriminator output for fake samples close to 1.
                G_loss = 0.5 * (tf.reduce_mean((D_fake - 1.0) ** 2.0))

            D_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")
            G_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator")

            self.D_optimization_step = (tf.train.AdamOptimizer(learning_rate=1e-4)
                                        .minimize(D_loss, global_step=None, var_list=D_variables))
            self.G_optimization_step = (tf.train.AdamOptimizer(learning_rate=1e-4)
                                        .minimize(G_loss, global_step=global_step, var_list=G_variables))

            self.global_step = global_step

        if X_fake.get_shape().ndims > 2:
            self.X_generated = X_fake
        else:
            self.X_generated = tf.reshape(X_fake, [-1] + X_sampled.get_shape().as_list()[1:])

    def train_step(self, sess):
        sess.run_without_hooks(self.D_optimization_step)
        _, i = sess.run((self.G_optimization_step, self.global_step))
        print(i)

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
