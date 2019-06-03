import click
import numpy as np
import threading
import core
import cntk
import os
import re

# Uncomment if you want to use CPU forcely
# cntk.try_set_default_device(cntk.device.cpu())


@click.version_option(prog_name='DeepDanbooru-EvalOnly', version='1.0.0')
@click.group()
def main():
    pass


@main.command('evaluate')
@click.argument('project_path', type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True))
@click.argument('image_path', type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False))
@click.option('--threshold', default=0.5, help='Score threshold for result.')
def evaluate_project(project_path, image_path, threshold):
    model, tags = core.load_model_and_tags(project_path)

    image = core.load_image_as_hwc(
        image_path, (299, 299))  # resize to 299x299x3
    image = np.ascontiguousarray(np.transpose(
        image, (2, 0, 1)), dtype=np.float32)  # transpose HWC to CHW (3x299x299)

    results = model.eval(image).reshape(tags.shape[0])  # array of tag score
    filename = os.path.basename(image_path)
    file = open(filename + ".txt", "a")
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
def evaluate_project_batch(project_path, folder_path, threshold):
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