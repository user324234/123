from __future__ import absolute_import

import os
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import click
from hydrus.utils import yield_chunks

from . import evaluate

try:
    import tensorflow as tf
except ImportError:
    print('Tensorflow Import failed')
    tf = None

try:
    import hydrus
except ImportError:
    hydrus = None
try:
    from flask import Flask, flash, request, redirect, json
    from werkzeug.utils import secure_filename
except ImportError:
    Flask = None  # type: ignore

DEFAULT_API_KEY = ""
TAG_FORMAT = '{tag}'
__version__ = '2.1.0'


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

def load_model_and_tags(model_path, tags_path, compile_):
    print("loading model...")
    if compile_ is None:
        model = tf.keras.models.load_model(model_path)
    else:
        model = tf.keras.models.load_model(model_path, compile=compile_)
    print("loading tags...")
    tags = evaluate.load_tags(tags_path)
    return model, tags

@click.version_option(prog_name='hydrus-dd', version=__version__)
@click.group()
def main():
    pass


@main.command('evaluate')
@click.argument('image_path', type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False))
@click.option('--threshold', default=0.5, help='Score threshold for result.')
@click.option('--cpu', default=False, is_flag=True, help="Use CPU")
@click.option(
    '--model_path', default=evaluate.MODEL_PATH, show_default=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Model path.")
@click.option(
    '--tags_path', default=evaluate.TAGS_PATH, show_default=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Tags file path.")
@click.option('--tag_format', default=TAG_FORMAT, show_default=True)
@click.option(
    '--compile/--no-compile', 'compile_', default=None,
    help='Compile/don\'t compile when loading model.')
def evaluate_sidecar(image_path: click.Path, threshold: float, cpu: bool, model_path: click.Path, tags_path: click.Path, tag_format: str, compile_: Optional[bool]):
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    model, tags = load_model_and_tags(model_path, tags_path, compile_)
    results = evaluate.eval(image_path, threshold, model=model, tags=tags)

    file = open(image_path + ".txt", "a")
    for tag in results:
        file.write(tag + "\n")
    file.close()


@main.command('evaluate-batch')
@click.argument('folder_path', type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True))
@click.option('--threshold', default=0.5, help='Score threshold for result.')
@click.option('--cpu', default=False, is_flag=True, help="Use CPU")
@click.option(
    '--model_path', default=evaluate.MODEL_PATH, show_default=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Model path.")
@click.option(
    '--tags_path', default=evaluate.TAGS_PATH, show_default=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Tags file path.")
@click.option('--tag_format', default=TAG_FORMAT, show_default=True)
@click.option(
    '--compile/--no-compile', 'compile_', default=None,
    help='Compile/don\'t compile when loading model.')
def evaluate_sidecar_batch(folder_path: click.Path, threshold: float, cpu: bool, model_path: click.Path, tags_path: click.Path, tag_format: str, compile_: Optional[bool]):
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    image_path_list = get_files_recursively(folder_path)
    model, tags = load_model_and_tags(model_path, tags_path, compile_)
    idx = 0

    with click.progressbar(length=len(image_path_list)) as image_path_progressbar:
        for image_path in image_path_list:
            image_path_progressbar.update(idx/len(image_path_list)*2)  # type: ignore
            idx += 1
            results = evaluate.eval(image_path, threshold, model=model, tags=tags)

            file = open(image_path + ".txt", "a")
            for tag in results:
                file.write(tag + "\n")
            file.close()


@main.command('evaluate-api-hash')
@click.option('--hash', '-h', 'hash_', multiple=True)
@click.option('--threshold', default=0.5, help='Score threshold for result.', show_default=True)
@click.option('--api_key', default=DEFAULT_API_KEY)
@click.option('--service', default="my tags", show_default=True)
@click.option(
    '--input', '-i', 'input_', nargs=1,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Input file with hashes to lookup, 1 hash per line.")
@click.option('--api_url', default=hydrus.DEFAULT_API_URL, show_default=True)
@click.option('--cpu', default=False, is_flag=True, help="Use CPU")
@click.option(
    '--model_path', default=evaluate.MODEL_PATH, show_default=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Model path.")
@click.option(
    '--tags_path', default=evaluate.TAGS_PATH, show_default=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Tags file path.")
@click.option('--tag_format', default=TAG_FORMAT, show_default=True)
@click.option(
    '--compile/--no-compile', 'compile_', default=None,
    help='Compile/don\'t compile when loading model.')
def evaluate_api_hash(
        threshold: float, service: str, api_key: str,
        hash_: List[str], api_url: str, input_: click.Path, cpu: bool,
        model_path: click.Path, tags_path: click.Path, tag_format: str, compile_: Optional[bool]):
    if hydrus is None:
        print("Hydrus API not found.\nPlease install hydrus-api python module.")
        return
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if input_:
        with open(input_, 'r') as f:  # type: ignore
            hash_ = [line.strip() for line in f]
    model, tags = load_model_and_tags(model_path, tags_path, compile_)
    with click.progressbar(hash_) as hash_progressbar:
        for hash_item in hash_progressbar:
            try:
                cl = hydrus.Client(api_key, api_url)
                print(f' tagging {hash_item}')
                image_path = BytesIO(cl.get_file(hash_=hash_item))
                hash_arg = [hash_item]
                if tag_format == TAG_FORMAT:
                    results = evaluate.eval(
                        image_path, threshold, model=model, tags=tags)
                    cl.add_tags(hash_arg, {service: results})
                else:
                    results = evaluate.eval(
                        image_path, threshold, return_score=True,  model=model, tags=tags)
                    if results:
                        service_tags = list(map(lambda x: tag_format.format(
                            tag=x[0], score=x[1], score10int=int(x[1]*11)  # type: ignore
                        ), results))
                        cl.add_tags(hash_arg, {service: service_tags})
            except Exception as e:
                print(e)
                print(f'tagging {hash_item} failed, skipping')


@main.command('evaluate-api-search')
@click.argument('search_tags', nargs=-1)
@click.option('--archive', default=False, is_flag=True)
@click.option('--inbox', default=False, is_flag=True)
@click.option('--threshold', default=0.5, help='Score threshold for result.', show_default=True)
@click.option('--api_key', default=DEFAULT_API_KEY)
@click.option('--service', default="my tags", show_default=True)
@click.option('--api_url', default=hydrus.DEFAULT_API_URL, show_default=True)
@click.option('--chunk_size', type=int, default=100, show_default=True)
@click.option('--cpu', default=False, is_flag=True, help="Use CPU")
@click.option(
    '--model_path', default=evaluate.MODEL_PATH, show_default=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Model path.")
@click.option(
    '--tags_path', default=evaluate.TAGS_PATH, show_default=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Tags file path.")
@click.option('--tag_format', default=TAG_FORMAT, show_default=True)
@click.option(
    '--compile/--no-compile', 'compile_', default=None,
    help='Compile/don\'t compile when loading model.')
def evaluate_api_search(
        archive: bool, inbox: bool, threshold: float, api_key: str, service: str, api_url: str,
        search_tags: str, chunk_size: int, cpu: bool,
        model_path: click.Path, tags_path: click.Path, tag_format: str, compile_: Optional[bool]):
    if hydrus is None:
        print("Hydrus API not found.\nPlease install hydrus-api python module.")
        return()
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    cl = hydrus.Client(api_key, api_url)
    clean_tags = cl.clean_tags(search_tags)
    fileIDs = list(cl.search_files(clean_tags, inbox, archive))
    idx = 0
    model, tags = load_model_and_tags(model_path, tags_path, compile_)
    with click.progressbar(length=len(fileIDs)) as file_id_progress_bar:
        for file_ids in yield_chunks(fileIDs, chunk_size):
            metadata = cl.file_metadata(file_ids=file_ids, only_identifiers=True)
            hashes = []
            for n, metadata in enumerate(metadata):
                hashes.append(metadata['hash'])
            for hash_ in hashes:
                file_id_progress_bar.update(idx/len(fileIDs)*2)  # type: ignore
                idx += 1
                try:
                    print(f'tagging {hash_}')
                    image_path = BytesIO(cl.get_file(hash_=hash_))
                    hash_arg = [hash_]
                    if tag_format == TAG_FORMAT:
                        results = evaluate.eval(image_path, threshold, model=model, tags=tags)
                        cl.add_tags(hash_arg, {service: results})
                    else:
                        results = evaluate.eval(
                            image_path, threshold, return_score=True, model=model, tags=tags)
                        if results:
                            service_tags = list(map(
                                lambda x: tag_format.format(
                                    tag=x[0], score=x[1], score10int=int(x[1]*11)),  # type: ignore
                                results))
                            cl.add_tags(hash_arg, {service: service_tags})
                except Exception as e:
                    print(e)
                    print(f'{hash_} does not appear to be an image, skipping')


@main.command('run-server')
@click.option('--threshold', default=0.5, help='Score threshold for result.', show_default=True)
@click.option('--host', default="0.0.0.0", show_default=True)
@click.option('--port', default="4443", show_default=True)
@click.option('--cpu', default=False, is_flag=True, help="Use CPU")
@click.option(
    '--model_path', default=evaluate.MODEL_PATH, show_default=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Model path.")
@click.option(
    '--tags_path', default=evaluate.TAGS_PATH, show_default=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Tags file path.")
@click.option('--tag_format', default=TAG_FORMAT, show_default=True)
@click.option(
    '--compile/--no-compile', 'compile_', default=None,
    help='Compile/don\'t compile when loading model.')
def run_server(threshold: float, host: str, port: int, cpu: bool, model_path: click.Path, tags_path: click.Path, tag_format: str, compile_: Optional[bool]):
    if Flask is None:
        print("flask not found.\nPlease install flask python module.")
        return()
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    app = Flask("hydrus-dd lookup server")
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
    model, tags = load_model_and_tags(model_path, tags_path, compile_)

    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route('/', methods=['GET', 'POST'])
    def upload_file():
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                _ = secure_filename(file.filename)
                image_path = BytesIO(file.read())
                results = evaluate.eval(image_path, threshold, model=model, tags=tags)
                deepdanbooru_response = json.dumps(results),
                response = app.response_class(
                    response=deepdanbooru_response,
                    status=200,
                    mimetype='application/json'
                )
                return response
        return '''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method=post enctype=multipart/form-data>
          <p><input type=file name=file>
             <input type=submit value=Upload>
        </form>
        '''

    app.run(host=host, port=port)


if __name__ == '__main__':
    main()
