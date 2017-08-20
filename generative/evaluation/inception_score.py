import os
import urllib.request
import tarfile

# keep the inception model weights in res folder, download on first use? or just keep it there

# TODO maybe put this in improved gan since that paper introduced it?

_INCEPTION_MODEL_URL = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"


def _maybe_download(inception_model_dir):
    # TODO: check if model is already downloaded

    if not os.path.exists(inception_model_dir):
        os.mkdir(inception_model_dir)

    print("Downloading pretrained inception model.")
    path, _ = urllib.request.urlretrieve(_INCEPTION_MODEL_URL, os.path.join(inception_model_dir, "compressed.tar.gz"))

    print("Unpacking.")
    tarfile.open()

def inception_score(inception_model_dir):
    pass


