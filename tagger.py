import click
import numpy as np
import threading
import core
import cntk
import os
import re
from io import BytesIO
try:
    import hydrus
    import hydrus.utils
except ImportError:
    hydrus_api = None
    

# Uncomment if you want to use CPU forcely
# cntk.try_set_default_device(cntk.device.cpu())


DEFAULT_API_KEY = ""

@click.version_option(prog_name='hydrus-dd', version='1.1.2')
@click.group()
def main():
    pass


@main.command('evaluate')
@click.argument('project_path', type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True))
@click.argument('image_path', type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False))
@click.option('--threshold', default=0.5, help='Score threshold for result.')
def evaluate_sidecar(project_path, image_path, threshold):
    model, tags = core.load_model_and_tags(project_path)

    image = core.load_image_as_hwc(
        image_path, (299, 299))  # resize to 299x299x3
    image = np.ascontiguousarray(np.transpose(
        image, (2, 0, 1)), dtype=np.float32)  # transpose HWC to CHW (3x299x299)

    results = model.eval(image).reshape(tags.shape[0])  # array of tag score
    file = open(image_path + ".txt", "a")
    alltags = []
    regex = re.compile('score:\w+')

    for i in range(len(tags)):
        if results[i] > threshold:
            alltags.append(tags[i])

    filtered = [x for x in alltags if not regex.match(x)]
    for tag in filtered:
        file.write(tag + "\n")
    file.close()


@main.command('evaluate-batch')
@click.argument('project_path', type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True))
@click.argument('folder_path', type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True))
@click.option('--threshold', default=0.5, help='Score threshold for result.')
def evaluate_sidecar_batch(project_path, folder_path, threshold):
    model, tags = core.load_model_and_tags(project_path)
    image_path_list = core.get_files_recursively(folder_path)

    for image_path in image_path_list:
        print(f'File={image_path}')

        image = core.load_image_as_hwc(
            image_path, (299, 299))  # resize to 299x299x3
        image = np.ascontiguousarray(np.transpose(
            image, (2, 0, 1)), dtype=np.float32)  # transpose HWC to CHW (3x299x299)

        results = model.eval(image).reshape(
            tags.shape[0])  # array of tag score
        filename = os.path.basename(image_path)
        path = os.path.join(folder_path, image_path)
        file = open(path + ".txt", "a")
        alltags = []
        regex = re.compile('score:\w+')

        for i in range(len(tags)):
            if results[i] > threshold:
                alltags.append(tags[i])

        filtered = [x for x in alltags if not regex.match(x)]
        for tag in filtered:
            file.write(tag + "\n")
    file.close()


@main.command('evaluate-api-hash')
@click.argument('project_path', type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True))
@click.option('--hash', '-h', multiple=True)
@click.option('--threshold', default=0.5, help='Score threshold for result.', show_default=True)
@click.option('--api_key', default=DEFAULT_API_KEY)
@click.option('--service', default="local tags", show_default=True)
@click.option('--input', '-i', nargs=1, type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False), help="Input file with hashes to lookup, 1 hash per line.")
@click.option('--api_url', default=hydrus.DEFAULT_API_URL, show_default=True)
def evaluate_api_hash(project_path, threshold, service, api_key, hash, api_url, input):
    if hydrus_api is None:
        print("Hydrus API not found.\nPlease install Hydrus API Python module.\nhttps://gitlab.com/cryzed/hydrus-api")
        exit()
    model, tags = core.load_model_and_tags(project_path)
    if input:
        with open(input, 'r') as f:
            hash = [line.strip() for line in f]
    for hash in hash:
        try:
            cl = hydrus.Client(api_key, api_url)
            print(f'tagging {hash}')
            image_path = BytesIO(cl.get_file(hash_=hash))

            image = core.load_image_as_hwc(
                image_path, (299, 299))  # resize to 299x299x3
            image = np.ascontiguousarray(np.transpose(
                image, (2, 0, 1)), dtype=np.float32)  # transpose HWC to CHW (3x299x299)

            results = model.eval(image).reshape(
                tags.shape[0])  # array of tag score
            alltags = []
            regex = re.compile('score:\w+')

            for i in range(len(tags)):
                if results[i] > threshold:
                    alltags.append(tags[i])

            filtered = [x for x in alltags if not regex.match(x)]
            hash = [hash]
            cl.add_tags(hash, {service: filtered})
        except:
            print(f'{hash} does not appear to be an image, skipping')


@main.command('evaluate-api-search')
@click.argument('project_path', type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True))
@click.argument('search_tags', nargs=-1)
@click.option('--archive', default=False, is_flag=True)
@click.option('--inbox', default=False, is_flag=True)
@click.option('--threshold', default=0.5, help='Score threshold for result.', show_default=True)
@click.option('--api_key', default=DEFAULT_API_KEY)
@click.option('--service', default="local tags", show_default=True)
@click.option('--api_url', default=hydrus.DEFAULT_API_URL, show_default=True)
@click.option('--chunk_size', type=int, default=100, show_default=True)
def evaluate_api_search(project_path, archive, inbox, threshold, api_key, service, api_url, search_tags, chunk_size):
    if hydrus_api is None:
        print("Hydrus API not found.\nPlease install Hydrus API Python module.\nhttps://gitlab.com/cryzed/hydrus-api")
        exit()
    model, tags = core.load_model_and_tags(project_path)
    cl = hydrus.Client(api_key, api_url)
    clean_tags = cl.clean_tags(search_tags)
    fileIDs = cl.search_files(clean_tags, inbox, archive)
    for file_ids in hydrus.utils.yield_chunks(list(fileIDs), chunk_size):
        metadata = cl.file_metadata(file_ids=file_ids, only_identifiers=True)
        hashes = []
        for n, metadata in enumerate(metadata):
            hashes.append(metadata['hash'])
        for hash in hashes:
            try:
                print(f'tagging {hash}')
                image_path = BytesIO(cl.get_file(hash_=hash))

                image = core.load_image_as_hwc(
                    image_path, (299, 299))  # resize to 299x299x3
                image = np.ascontiguousarray(np.transpose(
                    image, (2, 0, 1)), dtype=np.float32)  # transpose HWC to CHW (3x299x299)

                results = model.eval(image).reshape(
                    tags.shape[0])  # array of tag score
                alltags = []
                regex = re.compile('score:\w+')

                for i in range(len(tags)):
                    if results[i] > threshold:
                        alltags.append(tags[i])

                filtered = [x for x in alltags if not regex.match(x)]
                hash = [hash]
                cl.add_tags(hash, {service: filtered})
            except:
                print(f'{hash} does not appear to be an image, skipping')


def evaluate_post(image_path, threshold):
    model, tags = core.load_model_and_tags("danbooru-resnet_custom_v2-p4")

    image = core.load_image_as_hwc(
        image_path, (299, 299))  # resize to 299x299x3
    image = np.ascontiguousarray(np.transpose(
        image, (2, 0, 1)), dtype=np.float32)  # transpose HWC to CHW (3x299x299)

    results = model.eval(image).reshape(tags.shape[0])  # array of tag score
    alltags = []
    regex = re.compile('score:\w+')

    for i in range(len(tags)):
        if results[i] > threshold:
            alltags.append(tags[i])

    filtered = [x for x in alltags if not regex.match(x)]
    return filtered

if __name__ == '__main__':
    threading.stack_size(4 * 1024 * 1024)
    thread = threading.Thread(target=main)
    thread.start()
