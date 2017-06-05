from functools import reduce

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

from generative.models.common import leaky_relu


# TODO: Possibly keep these defined in dcgan module instead to follow the "paper trail" more closely.
# TODO: Note: deconv is commonly used but not actually what happens, should be made clear.
# TODO: Options for weight inits


def _initial_dense_and_reshape(Z, shape,
                               activation_fn=None,
                               use_batchnorm=False, is_training=None):

    h = tf.reshape(tf.layers.dense(Z, int(np.prod(shape)), activation=None), (-1,) + shape)

    if use_batchnorm:
        h = tf.layers.batch_normalization(h, training=is_training)

    return activation_fn(h) if activation_fn is not None else h


# TODO: refactor out shared code between conv and deconv layer functions.

def _conv_layer(t, num_filters,
                kernel_size=(5, 5),
                strides=(2, 2),
                padding="same",
                use_bias=True,
                activation_fn=None,
                use_batchnorm=False, is_training=None):

    if use_batchnorm:
        assert is_training is not None, "If using batchnorm, must pass boolean tf.placeholder."

    h = tf.layers.conv2d(t, num_filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding=padding,
                         activation=None,
                         use_bias=use_bias)

    # TODO: dropout option?
    # TODO: Other normalization options as well, weight norm / layer norm.

    if use_batchnorm:
        h = tf.layers.batch_normalization(h, training=is_training)

    return activation_fn(h) if activation_fn is not None else h


def _deconv_layer(t, num_filters,
                  kernel_size=(5, 5),
                  strides=(2, 2),
                  padding="same",
                  use_bias=True,
                  activation_fn=None,
                  use_batchnorm=False, is_training=None):

    if use_batchnorm:
        assert is_training is not None, "If using batchnorm, must pass boolean tf.placeholder."

    h = tf.layers.conv2d_transpose(t, num_filters,
                                   kernel_size=kernel_size,
                                   strides=strides,
                                   padding=padding,
                                   activation=None,
                                   use_bias=use_bias)

    # TODO: dropout option?
    # TODO: Other normalization options as well, weight norm / layer norm.

    if use_batchnorm:
        h = tf.layers.batch_normalization(h, training=is_training)

    return activation_fn(h) if activation_fn is not None else h


def conv_net_mnist(X):
    # TODO: Batch norm params
    h1 = _conv_layer(X, 256, activation_fn=leaky_relu)
    h2 = _conv_layer(h1, 128, activation_fn=leaky_relu)
    h3 = _conv_layer(h2, 64, activation_fn=leaky_relu)
    out = tf.layers.dense(flatten(h3), 1, activation=tf.nn.sigmoid)  # TODO: Don't want sigmoid for wgan
    return out


def deconv_net_mnist(Z):
    # TODO: Batch norm params
    h1 = _initial_dense_and_reshape(Z, (7, 7, 256), activation_fn=tf.nn.relu)
    h2 = _deconv_layer(h1, 128, activation_fn=tf.nn.relu)
    h3 = _deconv_layer(h2, 1, activation_fn=tf.nn.tanh)
    return h3


def conv_net_cifar10(X):
    pass


def deconv_net_cifar10(Z):
    h1 = _initial_dense_and_reshape(Z, (4, 4, 512))
    h2 = _deconv_layer(h1, 256)
    h3 = _deconv_layer(h2, 128)
    h4 = _deconv_layer(h3, 3)
    return h4


def conv_net_celebA(X):
    pass


def deconv_net_celebA(Z):
    pass

# TODO: remove this?
def conv_net(inputs, filter_sizes,
             use_batchnorm, is_training=None,
             intermediate_activation_fn=leaky_relu,
             final_activation_fn=tf.nn.relu,
             use_bias=True):

    # Assume that the inputs is in nhwc format?

    if use_batchnorm:
        assert is_training is not None, "Must pass boolean tf.placeholder for batchnorm"

    # Helper function for creating the hidden convolutional layers
    def conv_layer(t, num_filters):
        c_linear = tf.layers.conv2d(t, num_filters,
                                    kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding="valid",
                                    activation=None,
                                    use_bias=use_bias)

        if use_batchnorm:
            c_linear = tf.layers.batch_normalization(c_linear, training=is_training)

        return intermediate_activation_fn(c_linear)

    # TODO: If we want to have more layers, we can use padding "same" so that the feature map doesnt
    # shrink, might have other consequences though.
    # This function should be about as easy to use as the corresponding mlp function
    # Could define layers as list of (num_filters, kernel_size, ...)

    # Note: with current fixed settings of kernel_size, padding strategy and strides
    # we can only support 3 hidden layers
    assert len(filter_sizes) == 3

    # Create the hidden convolutional layers
    last_c = reduce(lambda prev_layer, num_filters: conv_layer(prev_layer, num_filters), filter_sizes, inputs)

    # 1 fully connected layer at the end
    out = tf.layers.dense(flatten(last_c), 1,
                          use_bias=use_bias,
                          activation=final_activation_fn)

    return out