import tensorflow as tf


def sequential_group(*inputs):
    """
    Create an op that groups multiple operations and runs them in the given order. Note that the operations
    must be given as functions creating the ops because dependencies can not be changed after operation creation.
    :param inputs: Functions creating the ops to group and run sequentially.
    :return: An operation that executes all the given ops sequentially.
    """

    with tf.name_scope("sequential_group"):
        if len(inputs) <= 1:
            return tf.group(*[op_fn() for op_fn in inputs])

        parent_op = inputs[0]()
        sequential_ops = [parent_op]
        for child_op_fn in inputs[1:]:
            with tf.control_dependencies([parent_op]):
                child_op = child_op_fn()
                sequential_ops.append(child_op)
                parent_op = child_op

        return tf.group(*sequential_ops)
