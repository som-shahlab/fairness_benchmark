import os
import yaml


def maybe_makedir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def read_yaml(path):
    with open(path, "r") as f:
        data = yaml.load(f)
    return data
