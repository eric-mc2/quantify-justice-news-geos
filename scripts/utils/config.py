import yaml
import os
from functools import reduce
from pathlib import Path
from os.path import dirname

class Config:
    _COLAB_DATA_DIR = "gdrive/MyDrive/Work/quantify-news/data/"
    _LOCAL_PROJECT_DIR = dirname(dirname(dirname(Path(__file__).resolve())))
    _LOCAL_DATA_DIR = os.path.join(_LOCAL_PROJECT_DIR, "data/")
    _DATA_CONFIG_PATH = os.path.join(_LOCAL_PROJECT_DIR, "config/data_sources.yaml")
    _PARAM_CONFIG_PATH = os.path.join(_LOCAL_PROJECT_DIR, "config/params.yaml")

    def __init__(self):    
        with (open(self._DATA_CONFIG_PATH, "r") as f,
             open(self._PARAM_CONFIG_PATH, "r") as g):
            data_config = yaml.safe_load(f)
            params_config = yaml.safe_load(g)
            self.config = _deep_union(data_config, params_config)
        
        if "COLAB" in os.environ:
            self.data_dir = self._COLAB_DATA_DIR
        else:
            self.data_dir = self._LOCAL_DATA_DIR

    def get_data_path(self, data_key: str, makedirs: bool = True):
        basepath = self._get_value_reduce(data_key)
        data_path = os.path.join(self.data_dir, basepath)
        if makedirs:
            dir_path = os.path.dirname(data_path)
            os.makedirs(dir_path, exist_ok=True)
        return data_path
    
    def get_param(self, key: str):
        return self._get_value_reduce(key)

    def _get_value_reduce(self, path):
        keys = path.split('.')
        try:
            return reduce(lambda d, key: d[key], keys, self.config)
        except (TypeError, KeyError):
            raise KeyError(f"Key {path} not in config.")

def _deep_union(dict1, dict2):
    """Recursively merges two dictionaries."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_union(result[key], value)
        else:
            result[key] = value
    return result