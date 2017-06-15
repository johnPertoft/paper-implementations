import os
import shutil
import urllib.request
import tarfile
import pickle

import tensorflow as tf

from data.utils import int64_feature, uint8_feature, tfrecords_input_tensor


_DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_IMG_SHAPE = (3, 32, 32)


def _convert_cifar10_batch_file(path, record_writer):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="bytes")

    images = data[b"data"]
    labels = data[b"labels"]

    for image, label in zip(images, labels):
        example = tf.train.Example(features=tf.train.Features(feature={
            "image": uint8_feature(image.tostring()),
            "label": int64_feature([label])
        }))
        record_writer.write(example.SerializeToString())


def maybe_download_and_prepare(data_dir):
    tfrecords_path = os.path.join(data_dir, "cifar-10.tfrecords")
    if os.path.exists(tfrecords_path):
        return tfrecords_path

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    print("Downloading cifar-10")
    path, _ = urllib.request.urlretrieve(_DATA_URL, os.path.join(data_dir, "compressed.tar.gz"))

    print("Unpacking")
    tarfile.open(path, mode="r:gz").extractall(data_dir)
    os.remove(path)

    print("Creating cifar-10.tfrecords")
    with tf.python_io.TFRecordWriter(tfrecords_path) as record_writer:
        # TODO: option to add test batch too since we dont need test set for generative models

        for part in ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]:
            _convert_cifar10_batch_file(os.path.join(data_dir, "cifar-10-batches-py", part), record_writer)

    shutil.rmtree(os.path.join(data_dir, "cifar-10-batches-py"))

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
