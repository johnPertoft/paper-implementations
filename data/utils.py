from multiprocessing import cpu_count

import tensorflow as tf


# TODO: Fix inconsistency here

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def uint8_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def tfrecords_input_tensor(tfrecords_path, from_single_record, batch_size):
    with tf.name_scope("Input"):
        reader = tf.TFRecordReader()
        queue = tf.train.string_input_producer([tfrecords_path], shuffle=True, capacity=2000)
        _, serialized_example = reader.read(queue)
        return tf.train.shuffle_batch(from_single_record(serialized_example),
                                      batch_size=batch_size,
                                      capacity=2000,
                                      num_threads=cpu_count(),
                                      enqueue_many=True,
                                      min_after_dequeue=1000,
                                      name="X_sampled")
