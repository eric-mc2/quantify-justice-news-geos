import yaml
import os
from functools import reduce
from pathlib import Path
from os.path import dirname

class Config:
    _COLAB_DATA_DIR = "gdrive/MyDrive/Work/quantify-news/data/"
    _COLAB_SCRATCH_DIR = "/content/"
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
            self.scratch_dir = self._COLAB_SCRATCH_DIR
        else:
            self.data_dir = self._LOCAL_DATA_DIR
            # Note: Don't need actual separate scratch directory locally.
            #       This is just a colab thing because it versions gdrive files.
            self.scratch_dir = self._LOCAL_PROJECT_DIR

    def get_data_path(self, data_key: str, makedirs: bool = True):
        return self._get_path(data_key, self.data_dir, makedirs)
    
    def get_file_path(self, file_key: str, scratch: bool = False, makedirs: bool = True):
        dir_path = self.scratch_dir if scratch else self._LOCAL_PROJECT_DIR
        return self._get_path(file_key, dir_path, makedirs)
    
    def _get_path(self, key: str, basepath: str, makedirs: bool = True):
        childpath = self._get_value_reduce(key)
        full_path = os.path.join(basepath, childpath)
        if makedirs:
            dir_path = os.path.dirname(full_path)
            os.makedirs(dir_path, exist_ok=True)
        return full_path
    
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