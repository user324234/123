#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re

import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()


version_path = os.path.join('hydrus_dd', '__main__.py')
with open(version_path, encoding='utf-8') as f:  # type: ignore
    version = re.search('__version__ = \'([^\']+)\'', f.read()).group(1)  # type: ignore


setuptools.setup(
    name="hydrus-dd",
    version=version,
    author="koto",
    #  author_email="author@example.com",
    description="DeepDanbooru neural network tagging for Hydrus",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitgud.io/koto/hydrus-dd",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        'Click>=7.0',
        'hydrus-api>=3.22.4',
        'numpy',
        'Pillow>=8.2.0',
        'scikit-image',
        'six',
        'appdirs',
        'tqdm',
        'deepdanbooru @ https://github.com/KichangKim/DeepDanbooru/tarball/master',
    ],
    extras_require={
        'server': ['flask'],
        'tensorflow': ['tensorflow>=2'],
        'tests': ['pytest', 'mypy', 'flake8', 'tox'],
    },
    entry_points={
        'console_scripts': [
            'hydrus-dd = hydrus_dd.__main__:main',
        ],
    }
)
