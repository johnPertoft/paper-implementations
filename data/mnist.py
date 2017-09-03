import os

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

from data.utils import int64_feature, uint8_feature, tfrecords_input_tensor


_IMG_SHAPE = (28, 28)


def maybe_download_and_prepare(data_dir):
    tfrecords_path = os.path.join(data_dir, "mnist.tfrecords")
    if os.path.exists(tfrecords_path):
        return tfrecords_path

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    print("Downloading mnist")
    data_sets = mnist.read_data_sets(data_dir,
                                     dtype=tf.uint8,
                                     reshape=True,
                                     validation_size=0)  # TODO: Set this by param instead

    print("Creating mnist.tfrecords")
    with tf.python_io.TFRecordWriter(tfrecords_path) as record_writer:
        for image, label in zip(data_sets.train.images, data_sets.train.labels):
            example = tf.train.Example(features=tf.train.Features(feature={
                "image": uint8_feature(image.tostring()),
                "label": int64_feature([int(label)])
            }))
            record_writer.write(example.SerializeToString())

    # Remove archive files
    for p in ["t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
              "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"]:
        os.remove(os.path.join(data_dir, p))

    return tfrecords_path


def input_tensor(tfrecords_path, batch_size, return_label, as_float=True):
    def from_records(serialized_examples):
        features = tf.parse_example(serialized_examples, features={"image": tf.FixedLenFeature([], tf.string),
                                                                   "label": tf.FixedLenFeature([], tf.int64)})
        images = tf.reshape(tf.decode_raw(features["image"], tf.uint8), (-1,) + _IMG_SHAPE)
        labels = tf.cast(features["label"], tf.int32)

        if as_float:
            images = tf.cast(images, tf.float32) / 255.0

        return [images, labels] if return_label else [images]

    return tfrecords_input_tensor(tfrecords_path, from_records, batch_size)


def create_input_fn(tfrecords_path, batch_size, as_float=True, shuffle=True, infinite=True):
    def parse_fn(serialized_example):
        parsed_example = tf.parse_single_example(serialized_example,
                                                 features={"image": tf.FixedLenFeature([], tf.string),
                                                           "label": tf.FixedLenFeature([], tf.int64)})
        images = tf.reshape(tf.decode_raw(parsed_example["image"], tf.uint8), (-1,) + _IMG_SHAPE)
        labels = tf.cast(parsed_example["label"], tf.int32)

        if as_float:
            images = tf.cast(images, tf.float32) / 255.0

        return images, labels

    def input_fn():
        dataset = tf.contrib.data.TFRecordDataset(tfrecords_path).map(parse_fn, num_threads=8)

        if shuffle:
            dataset = dataset.shuffle(batch_size * 10)

        if infinite:
            dataset.repeat(None)

        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    return input_fn
