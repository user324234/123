import configparser, os  # NOQA
from appdirs import *  # NOQA

parser = configparser.ConfigParser()

default_config = """
[general]
api_url = http://127.0.0.1:45869
model_path = """+os.path.join("model", "model.h5")+"""
tags_path = """+os.path.join("model", "tags.txt")+"""
tag_format = {tag}
service = my tags
api_key = None
chunk_size = 100
threshold = 0.5
cpu = False
archive = False
inbox = False
[server]
host = 0.0.0.0
port = 4443
"""


def load_config():
    try:
        # Load default config
        parser.read_string(default_config)
        # Load user config from file to overwrite defaults
        parser.read_file(open(os.path.join(
            user_config_dir(), "hydrus-dd", "hydrus-dd.conf")))  # NOQA
        return parser
    except:  # NOQA
        parser.read_string(default_config)
        return parser
