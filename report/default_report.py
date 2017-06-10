import os

import numpy as np
from PIL import Image

from report import MarkdownDocumentBuilder


def create_default_report(output_dir, param_settings, images):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_dir = os.path.join(output_dir, "imgs")
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

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
    return md_builder.build(os.path.join(output_dir, "Results.md"))
