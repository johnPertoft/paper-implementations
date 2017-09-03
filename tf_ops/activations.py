import tensorflow as tf


def leaky_relu(x, leak=0.2, name="leaky_relu"):
    assert 0.0 < leak < 1.0
    with tf.name_scope(name):
        return tf.maximum(leak * x, x)


def selu():
    pass
