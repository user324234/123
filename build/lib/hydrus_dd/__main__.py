from __future__ import absolute_import

import os
import threading
import traceback
from io import BytesIO
from pathlib import Path
from queue import Queue
from typing import Any, List, Optional, Tuple

import click
from hydrus_api.utils import yield_chunks
from PIL import Image
from tqdm import tqdm

from . import config, evaluate

try:
    import tensorflow as tf
except ImportError:
    print('Tensorflow import failed')
    tf = None

try:
    import hydrus_api
except ImportError:
    hydrus_api = None
try:
    from flask import Flask, flash, request, redirect, json
    from werkzeug.utils import secure_filename
except ImportError:
    Flask = None  # type: ignore

cfg = config.load_config()
TAG_FORMAT = '{tag}'
__version__ = '3.0.0'


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
@click.option('--threshold', type=float, default=cfg['general']['threshold'], help='Score threshold for result.')
@click.option('--cpu', default=cfg['general']['cpu'], is_flag=True, help="Use CPU")
@click.option(
    '--model_path', default=cfg['general']['model_path'], show_default=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Model path.")
@click.option(
    '--tags_path', default=cfg['general']['tags_path'], show_default=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Tags file path.")
@click.option('--tag_format', default=cfg['general']['tag_format'], show_default=True)
@click.option(
    '--compile/--no-compile', 'compile_', default=None,
    help='Compile/don\'t compile when loading model.')
def evaluate_sidecar(image_path: click.Path, threshold: float, cpu: bool, model_path: click.Path, tags_path: click.Path, tag_format: str, compile_: Optional[bool]):
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    model, tags = load_model_and_tags(model_path, tags_path, compile_)
    results = evaluate.eval(image_path, threshold, model=model, tags=tags)

    file = open(image_path + ".txt", "a")  # type: ignore
    for tag in results:
        file.write(tag + "\n")  # type: ignore
    file.close()


@main.command('evaluate-batch')
@click.argument('folder_path', type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True))
@click.option('--threshold', type=float, default=cfg['general']['threshold'], help='Score threshold for result.')
@click.option('--cpu', default=cfg['general']['cpu'], is_flag=True, help="Use CPU")
@click.option(
    '--model_path', default=cfg['general']['model_path'], show_default=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Model path.")
@click.option(
    '--tags_path', default=cfg['general']['tags_path'], show_default=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Tags file path.")
@click.option('--tag_format', default=cfg['general']['tag_format'], show_default=True)
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
            try:
                results = evaluate.eval(image_path, threshold, model=model, tags=tags)
                file = open(image_path + ".txt", "a")
                for tag in results:
                    file.write(tag + "\n")  # type: ignore
                file.close()
            except Exception as e:
                print(e)
                print(f'tagging {image_path} failed, skipping')


@main.command('evaluate-api-hash')
@click.option('--hash', '-h', 'hash_', multiple=True)
@click.option('--threshold', type=float, default=cfg['general']['threshold'], help='Score threshold for result.', show_default=True)
@click.option('--api_key', default=cfg['general']['api_key'])
@click.option('--tag_service', default=cfg['general']['tag_service'], show_default=True)
@click.option('--file_service', default=cfg['general']['file_service'], show_default=True)
@click.option(
    '--input', '-i', 'input_', nargs=1,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Input file with hashes to lookup, 1 hash per line.")
@click.option('--api_url', default=cfg['general']['api_url'], show_default=True)
@click.option('--cpu', default=cfg['general']['cpu'], is_flag=True, help="Use CPU")
@click.option(
    '--model_path', default=cfg['general']['model_path'], show_default=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Model path.")
@click.option(
    '--tags_path', default=cfg['general']['tags_path'], show_default=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Tags file path.")
@click.option('--tag_format', default=cfg['general']['tag_format'], show_default=True)
@click.option(
    '--compile/--no-compile', 'compile_', default=None,
    help='Compile/don\'t compile when loading model.')
def evaluate_api_hash(
        threshold: float, tag_service: str, file_service: str, api_key: str,
        hash_: List[str], api_url: str, input_: click.Path, cpu: bool,
        model_path: click.Path, tags_path: click.Path, tag_format: str, compile_: Optional[bool]):
    if hydrus_api is None:
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
                cl = hydrus_api.Client(api_key, api_url)
                print(f' tagging {hash_item}')
                image_path = BytesIO(cl.get_file(hash_=hash_item).content)
                hash_arg = [hash_item]
                if tag_format == TAG_FORMAT:
                    results = evaluate.eval(
                        image_path, threshold, model=model, tags=tags)
                    cl.add_tags(hash_arg, {tag_service: results})
                else:
                    results = evaluate.eval(
                        image_path, threshold, return_score=True,  model=model, tags=tags)
                    if results:
                        service_tags = list(map(lambda x: tag_format.format(
                            tag=x[0], score=x[1], score10int=int(x[1]*10)  # type: ignore
                        ), results))
                        cl.add_tags(hash_arg, {tag_service: service_tags})
            except Exception as e:
                print(e)
                print(f'tagging {hash_item} failed, skipping')


class Producer(threading.Thread):

    def __init__(self, hashes: List[str], client: Any):
        threading.Thread.__init__(self)
        self.queue = Queue(100)  # type: Queue[Tuple[str, Any]]
        self.hashes = hashes
        self.client = client
        self.finished = threading.Event()

    def run(self):
        cl = self.client
        for hash_ in self.hashes:
            tqdm.write(f"getting content {hash_}")
            try:
                image = BytesIO(cl.get_file(hash_=hash_).content)
                self.queue.put([hash_, image])
            except Exception as err:
                tqdm.write(f"Error when getting content for {hash_}: {err}")
        self.finished.set()


class ContentConsumer(threading.Thread):
    """Predict tags from image content."""

    queue: 'Queue[Tuple[str, List[Tuple[str, float]]]]'
    "Main queue for this class."
    producer_finished: threading.Event
    "Flag from producer, which produce hash and image content tuple."
    hash_content_queue: "Queue[Tuple[str, BytesIO]]"
    ":py:class:`Queue` consist of :py:class:`tuple` of hash and image content."
    model: Any
    "Model for deepdanbooru."
    tags: List[str]
    "Tags for deepdanbooru."
    threshold: float
    """Tags threshold."""
    finished: threading.Event
    """Flag when the run is finished."""

    def __init__(
            self,
            hash_content_queue: "Queue[Tuple[str, BytesIO]]",
            model: Any,
            tags: List[str],
            threshold: float,
            producer_finished: threading.Event
    ):
        threading.Thread.__init__(self)
        self.hash_content_queue = hash_content_queue
        self.model = model
        self.tags = tags
        self.threshold = threshold
        self.queue = Queue()  # type: Queue[Tuple[str, List[Tuple[str, float]]]]
        self.producer_finished = producer_finished
        self.finished = threading.Event()

    def run(self):
        while not self.producer_finished.is_set() or not self.hash_content_queue.empty():
            hash_, content = self.hash_content_queue.get()
            tqdm.write(f'estimating tags for {hash_}')
            results = None
            try:
                results = evaluate.eval(
                    content, self.threshold, return_score=True, model=self.model, tags=self.tags)
            except Exception:
                err_txt = traceback.format_exc().splitlines()[-1]
                im_format = None
                try:
                    im = Image.open(content)
                    im_format = im.format
                    if im_format == "WEBP":
                        o_im = BytesIO()
                        im.save(o_im, format='PNG')
                        results = evaluate.eval(
                            o_im, self.threshold, return_score=True, model=self.model, tags=self.tags)
                        tqdm.write(f'convert {hash_},format:{im_format}')
                except Exception:
                    err_txt = traceback.format_exc()
                if not results and err_txt:
                    tqdm.write(f'Error when estimating tags for {hash_},format:{im_format}\n{err_txt}')
            self.queue.put([hash_, results])
            self.hash_content_queue.task_done()
        self.finished.set()


class Consumer(threading.Thread):

    def __init__(
            self,
            queue: "Queue[Tuple[str, List[Tuple[str, float]]]]",
            client: Any,
            tag_format: str,
            tag_service: str,
            pbar: Any,
            content_consumer_finished: threading.Event
    ):
        threading.Thread.__init__(self)
        self.queue = queue
        self.client = client
        self.tag_format = tag_format
        self.tag_service = tag_service
        self.pbar = pbar
        self.content_consumer_finished = content_consumer_finished

    def run(self):
        cl = self.client
        tag_format = self.tag_format
        tag_service = self.tag_service
        while not self.content_consumer_finished.is_set() or not self.queue.empty():
            try:
                hash_, results = self.queue.get()
                tqdm.write(f'tagging {hash_}')
                hash_arg = [hash_]
                service_tags = list(map(
                    lambda x: tag_format.format(
                        tag=x[0], score=x[1], score10int=int(x[1]*10)),  # type: ignore
                    results))
                if service_tags:
                    cl.add_tags(hash_arg, {tag_service: service_tags})
            except Exception:
                err_txt = traceback.format_exc()
                tqdm.write(f'Error when tagging {hash_}\n{err_txt}')
            self.queue.task_done()
            self.pbar.update()


class ModelLoader(threading.Thread):

    def __init__(self, model_path: click.types.Path, compile_: Optional[bool]):
        threading.Thread.__init__(self)
        self.model_path = model_path
        self.compile = compile_
        self.model = None  # type: Optional[Any]

    def run(self):
        print("loading model...")
        compile_ = self.compile
        model_path = self.model_path
        if compile_ is None:
            model = tf.keras.models.load_model(model_path)
        else:
            model = tf.keras.models.load_model(model_path, compile=compile_)
        self.model = model


class HashLoader(threading.Thread):

    def __init__(
            self,
            api_key: str,
            api_url: str,
            search_tags: List[str],
            file_service: str,
            sort_type: int,
            sort_asc: bool,
            chunk_size: int
    ):
        threading.Thread.__init__(self)
        self.client = None
        self.api_key = api_key
        self.api_url = api_url
        self.search_tags = search_tags
        self.file_service = file_service
        self.sort_type = sort_type
        self.sort_asc = sort_asc
        self.chunk_size = chunk_size
        self.hashes = []  # type: List[str]

    def run(self):
        cl = hydrus_api.Client(self.api_key, self.api_url)
        self.client = cl
        fileIDs = list(cl.search_files(tags = self.search_tags, file_service_name = self.file_service, file_sort_type = self.sort_type, file_sort_asc = self.sort_asc))
        for file_ids in yield_chunks(fileIDs, self.chunk_size):
            metadata = cl.get_file_metadata(file_ids=file_ids, only_return_identifiers=True)
            self.hashes.extend(list(map(lambda x: x['hash'], metadata)))


@main.command('evaluate-api-search')
@click.argument('search_tags', nargs=-1)
@click.option('--threshold',  type=float, default=cfg['general']['threshold'], help='Score threshold for result.', show_default=True)
@click.option('--api_key', default=cfg['general']['api_key'])
@click.option('--tag_service', default=cfg['general']['tag_service'], show_default=True)
@click.option('--file_service', default=cfg['general']['file_service'], show_default=True)
@click.option('--sort_type', default=cfg['general']['sort_type'], show_default=True)
@click.option('--sort_asc/--sort_desc', default=cfg['general']['sort_asc'], is_flag=True, show_default=True)
@click.option('--api_url', default=cfg['general']['api_url'], show_default=True)
@click.option('--chunk_size', type=int, default=cfg['general']['chunk_size'], show_default=True)
@click.option('--cpu', default=cfg['general']['cpu'], is_flag=True, help="Use CPU")
@click.option(
    '--model_path', default=cfg['general']['model_path'], show_default=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Model path.")
@click.option(
    '--tags_path', default=cfg['general']['tags_path'], show_default=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Tags file path.")
@click.option('--tag_format', default=cfg['general']['tag_format'], show_default=True)
@click.option(
    '--compile/--no-compile', 'compile_', default=None,
    help='Compile/don\'t compile when loading model.')
@click.option(
    '--parallel/--no-parallel', 'parallel', default=True,
    help='Run parallel or not.')
def evaluate_api_search(
        threshold: float, api_key: str, tag_service: str, file_service: str, sort_type: int, sort_asc: bool, api_url: str,
        search_tags: List[str], chunk_size: int, cpu: bool,
        model_path: click.Path, tags_path: click.Path, tag_format: str, compile_: Optional[bool],
        parallel: bool = True
):
    if hydrus_api is None:
        print("Hydrus API not found.\nPlease install hydrus-api python module.")
        return()
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if parallel:
        hash_loader = HashLoader(api_key, api_url, search_tags, file_service, sort_type, sort_asc, chunk_size)
        model_loader = ModelLoader(model_path, compile_)
        hash_loader.start()
        model_loader.start()
        hash_loader.join()
        hashes = hash_loader.hashes
        cl = hash_loader.client
        if not hashes:
            print('No files found for this search.')
            return
        with tqdm(total=len(hashes)) as pbar:
            try:
                producer = Producer(sorted(hashes), cl)
                producer.start()
                model_loader.join()
                model = model_loader.model
                tags = evaluate.load_tags(tags_path)
                if not all([model, tags]):
                    return
                content_consumer = ContentConsumer(
                    producer.queue, model, tags, threshold, producer.finished)
                consumer = Consumer(
                    content_consumer.queue,
                    cl, tag_format, tag_service, pbar,
                    content_consumer.finished)
                content_consumer.start()
                consumer.start()
                producer.join()
                content_consumer.join()
                consumer.join()
            except KeyboardInterrupt:
                pass
    else:
        cl = hydrus_api.Client(api_key, api_url)
        fileIDs = list(cl.search_files(tags = search_tags, file_service_name = file_service, file_sort_type = sort_type, file_sort_asc = sort_asc))
        model, tags = load_model_and_tags(model_path, tags_path, compile_)
        hashes = []
        for file_ids in yield_chunks(fileIDs, chunk_size):
            metadata = cl.get_file_metadata(file_ids=file_ids, only_return_identifiers=True)
            for n, metadata in enumerate(metadata):
                hashes.append(metadata['hash'])
        if not hashes:
            print('No files found for this search.')
            return
        for hash_ in tqdm(hashes):
            try:
                print(f'tagging {hash_}')
                image_path = BytesIO(cl.get_file(hash_=hash_).content)
                hash_arg = [hash_]
                if tag_format == TAG_FORMAT:
                    service_tags = evaluate.eval(image_path, threshold, model=model, tags=tags)
                else:
                    results = evaluate.eval(
                        image_path, threshold, return_score=True, model=model, tags=tags)
                    service_tags = list(map(
                        lambda x: tag_format.format(
                            tag=x[0], score=x[1], score10int=int(x[1]*10)),  # type: ignore
                        results))
                if service_tags:
                    cl.add_tags(hash_arg, {tag_service: service_tags})
            except Exception as e:
                print(e)
                print(f'{hash_} does not appear to be an image, skipping')


@main.command('run-server')
@click.option('--threshold', type=float, default=cfg['general']['threshold'], help='Score threshold for result.', show_default=True)
@click.option('--host', default=cfg['server']['host'], show_default=True)
@click.option('--port', default=cfg['server']['port'], show_default=True)
@click.option('--cpu', default=cfg['general']['cpu'], is_flag=True, help="Use CPU")
@click.option(
    '--model_path', default=cfg['general']['model_path'], show_default=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Model path.")
@click.option(
    '--tags_path', default=cfg['general']['tags_path'], show_default=True,
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="Tags file path.")
@click.option('--tag_format', default=cfg['general']['tag_format'], show_default=True)
@click.option(
    '--compile/--no-compile', 'compile_', default=None,
    help='Compile/don\'t compile when loading model.')
def run_server(threshold: float, host: str, port: int, cpu: bool, model_path: click.Path, tags_path: click.Path, tag_format: str, compile_: Optional[bool]):
    if Flask is None:
        print("flask not found.\nPlease install flask python module.")
        return()
    config.load_config()
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
