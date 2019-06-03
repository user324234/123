import math
import os

import cntk as C
import numpy as np
import skimage.transform
from PIL import Image
from pathlib import Path


def load_image_as_hwc(path, output_shape, scale=None, rotation=None, shift=None, order=1, mode='edge', normalize=True):
    image = Image.open(path)

    if not len(image.getbands()) == 3:
        image = image.convert('RGB')

    source_width = image.size[0]
    source_height = image.size[1]
    target_width = output_shape[0]
    target_height = output_shape[1]

    if source_width == target_width and source_height == target_height and scale is None and rotation is None and shift is None:
        return np.asarray(image)

    source_ratio = source_width / source_height
    target_ratio = target_width / target_height

    if target_ratio < source_ratio:
        resize_scale = target_width / source_width
    else:
        resize_scale = target_height / source_height

    if scale:
        resize_scale *= scale

    image = image.resize(size=(int(source_width * resize_scale),
                               int(source_height * resize_scale)), resample=Image.LANCZOS)

    image_array = np.asarray(image, dtype=np.float64)  # HWC

    # centerize
    t = skimage.transform.AffineTransform(
        translation=(-image.size[0] * 0.5, -image.size[1] * 0.5))

    if rotation:
        radian = (rotation / 180.0) * math.pi
        t += skimage.transform.AffineTransform(rotation=radian)

    t += skimage.transform.AffineTransform(
        translation=(target_width * 0.5, target_height * 0.5))

    if shift:
        t += skimage.transform.AffineTransform(
            translation=(target_width * shift[0], target_height * shift[1]))

    warp_shape = (output_shape[1], output_shape[0])

    image_array = skimage.transform.warp(
        image_array, (t).inverse, output_shape=warp_shape, order=order, mode=mode)

    if normalize:
        image_array /= 255.0

    return image_array


def load_model_and_tags(project_path):
    if not os.path.exists(project_path):
        raise Exception('{project_path} is not exists.')

    tags_path = os.path.join(project_path, 'tags.txt')
    model_path = os.path.join(project_path, 'model.cntk')

    with open(tags_path, 'r') as tags_stream:
        tag_array = np.array([tag for tag in (tag.strip()
                                              for tag in tags_stream) if tag])

    model = C.load_model(model_path)

    return (model, tag_array)


def get_files_recursively(folder_path):
    allowed_patterns = [
        '*.[Pp][Nn][Gg]',
        '*.[Jj][Pp][Gg]',
        '*.[Jj][Pp][Ee][Gg]',
        '*.[Gg][Ii][Ff]'
    ]

    image_path_list = [
        str(path) for pattern in allowed_patterns for path in Path(folder_path).rglob(pattern)]

    return image_path_list
