# TODO this file is shared from SmartScreening project
import re
import os
from collections.abc import MutableMapping
from pathlib import Path
import yaml

def transform_string2env(text: str):
    if not isinstance(text, str):
        return text

    pattern = r"\$\{([^}]+)\}"
    list_env_vars = re.findall(pattern, text)

    if list_env_vars:
        for v in list_env_vars:
            return text.replace(f"${{{v}}}", os.getenv(v))
    else:
        return text


def parse_yaml(training_config_file: Path):
    """
    Parse the configuration YAML and flatten the output
    """
    with open(training_config_file) as f:
        config_file = flatten_dict(yaml.safe_load(f))

    for key, values in config_file.items():
        config_file[key] = transform_string2env(values)

    return config_file


def flatten_dict(d, parent_key="", sep="__"):
    """
    Flatten a dictionary of key-value pairs into a flat-level where keys are appended with __
    like in sklearn parameters.

    See here
    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)