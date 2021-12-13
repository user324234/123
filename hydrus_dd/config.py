import configparser, os  # NOQA
from appdirs import *  # NOQA

parser = configparser.ConfigParser()

default_config = """
[general]
api_url = http://127.0.0.1:45869
model_path = """+os.path.join("model", "model.h5")+"""
tags_path = """+os.path.join("model", "tags.txt")+"""
tag_format = {tag}
tag_service = my tags
file_service = my files
sort_type = 2
sort_asc = False
api_key = None
chunk_size = 100
threshold = 0.5
cpu = False
service =
archive =
inbox =
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
        # Check and warn about depreciated configuration
        if parser['general']['archive'] or parser['general']['inbox'] or parser['general']['service']:
            print("Your configuration file seems to be out of date for hydrus-dd >3.0.0. Please update your configuration file.")
        if parser['general']['archive'] or parser['general']['inbox']:
            print("WARNING: archive and inbox configuration options are depreciated. Use system predicates instead.")
        # Translate old config key,value to new to try and not cause a mess
        if parser['general']['service']:
            print("WARNING: service configuration option is depreciated. Please update your configuration file.")
            service_to_tag_service = """
            [general]
            tag_service = """+parser['general']['service']+"""
            """
            parser.read_string(service_to_tag_service)
        return parser
    except:  # NOQA
        parser.read_string(default_config)
        return parser
