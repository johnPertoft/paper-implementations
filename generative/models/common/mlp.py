from functools import reduce

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, flatten


def mlp(inputs, layer_sizes,
        intermediate_activation_fn=tf.nn.relu,
        final_activation_fn=tf.nn.relu,
        use_bias=True,
        weight_initializer=None,
        return_layers=False,
        use_batchnorm=False, is_training=None,
        use_dropout=False):

    # TODO: add options for virtual batch normalization as well?

    if use_batchnorm or use_dropout:
        assert is_training is not None, "Must pass boolean tf.placeholder for batchnorm and/or dropout"

    inputs = flatten(inputs) if inputs.shape.ndims > 2 else inputs

    weight_init = weight_initializer if weight_initializer is not None else tf.contrib.layers.xavier_initializer

    def hidden_layer(t, num_hidden):
        linear = fully_connected(t, num_hidden,
                                 activation_fn=None,
                                 weights_initializer=weight_init(),
                                 biases_initializer=tf.zeros_initializer() if use_bias else None)

        out = tf.layers.batch_normalization(linear, training=is_training) if use_batchnorm else linear

        activated = intermediate_activation_fn(out)

        return tf.layers.dropout(activated, training=is_training) if use_dropout else activated

    # Create the hidden layers
    hidden_layer_sizes = layer_sizes[:-1]
    hidden_layers = reduce(lambda prev_layers, h_size: prev_layers + [hidden_layer(prev_layers[-1], h_size)],
                           hidden_layer_sizes, [inputs])

    # Create the output layer
    logits = fully_connected(hidden_layers[-1], layer_sizes[-1],
                             activation_fn=None,
                             weights_initializer=weight_init(),
                             biases_initializer=tf.zeros_initializer() if use_bias else None)

    out = final_activation_fn(logits) if final_activation_fn is not None else logits

    return out if not return_layers else (out, logits, hidden_layers[1:])
