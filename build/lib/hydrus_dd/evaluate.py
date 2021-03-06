import pathlib
from typing import Any, List, Optional, Sequence, Tuple, Union
import click
import six
import deepdanbooru
try:
    import tensorflow as tf
except ImportError:
    print('Tensorflow Import failed')
    tf = None


def load_tags(tags_path: Union[pathlib.Path, str, click.types.Path]):
    with open(tags_path, 'r') as stream:  # type: ignore
        tags = [tag for tag in (tag.strip() for tag in stream) if tag]
    return tags


def eval(
        image_path: Union[six.BytesIO, str, click.types.Path],
        threshold: float,
        return_score: bool = False,
        model: Optional[Any] = None, tags: Optional[List[str]] = None
) -> Sequence[Union[str, Tuple[str, Any], None]]:

    result_tags = []

    tag_sets = deepdanbooru.commands.evaluate_image(image_path, model, tags, threshold)
    for tag, score in tag_sets:
        if score >= threshold:
            if not return_score:
                result_tags.append(tag)
            else:
                result_tags.append((tag, score))  # type: ignore

    return result_tags
