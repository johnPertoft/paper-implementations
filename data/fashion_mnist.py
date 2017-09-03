import os
import urllib.request

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

from data.utils import int64_feature, uint8_feature, tfrecords_input_tensor


_IMG_SHAPE = (28, 28)
_DATA_URLs = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
              "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
              "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
              "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"]


def maybe_download_and_prepare(data_dir):

    print("Downloading fashion mnist.")
    #urllib.request.urlretrieve

    [urllib.request.urlretrieve(url, os.path.join(data_dir, "todo"))[0] for url in _DATA_URLs]


def input_tensor(tfrecords_path, batch_size, return_label, as_float=True):
    pass
