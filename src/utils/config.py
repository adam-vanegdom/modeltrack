import os
from easydict import EasyDict

from utils.dirs import create_dirs


def get_config_from_json(config_dict):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided
    try:
        # EasyDict allows to access dict values as attributes (works recursively).
        config = EasyDict(config_dict)
        return config, config_dict
    except ValueError:
        print("INVALID JSON file format.. Please provide a good json file")
        exit(-1)


def process_config(exp_name, root, config):
    """
    Get the json file and return info as attributes, and
    create experiment in specified root directory
    :param json_file: the path of the config file
    :return: config object(namespace)
    """
    config, _ = get_config_from_json(config)
    print("Configuration of your experiment ..")

    config.exp_name = exp_name

    if root is None:
        root = os.path.join(os.getenv("HOME"), ".modeltrack")

    config.model_dir = os.path.join(root, config.exp_name, "model/")
    config.log_dir = os.path.join(root, config.exp_name, "logs/")
    config.out_dir = os.path.join(root, config.exp_name, "output/")
    create_dirs([config.model_dir, config.log_dir, config.out_dir])

    return config


def update_config(curr_config, new_config):
    config, _ = get_config_from_json(new_config)

    for item in config:
        curr_config[item] = config[item]
