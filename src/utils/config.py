import os
import glob
from easydict import EasyDict


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


def update_config(curr_config, new_config):
    config, _ = get_config_from_json(new_config)

    for item in config:
        curr_config[item] = config[item]


def process_config(exp_name, root, config):
    """
    Get the json file and return info as attributes, and
    create experiment in specified root directory
    :param json_file: the path of the config file
    :return: config object(namespace)
    """
    config, _ = get_config_from_json(config)

    if root is None:
        root = os.path.join(os.getenv("HOME"), ".modeltrack")

    config.model_dir, config.log_dir = create_unique_dir(
        root, exp_name, config.overwrite
    )

    exp_name = os.path.basename(os.path.normpath(config.model_dir))

    return config, exp_name


def create_unique_dir(path, name, overwrite):
    count = len(glob.glob(f"{path}/{name}*"))

    if overwrite and count > 0:
        dir_name = glob.glob(f"{path}/{name}*")[-1]
    else:
        dir_name = os.path.join(path, "{}-{}".format(name, str(count + 1).zfill(3)))

    log_name = os.path.join(dir_name, "logs/")

    try:
        for dir_ in [dir_name, log_name]:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return dir_name, log_name

    except Exception as err:
        print(err)
        exit(-1)
