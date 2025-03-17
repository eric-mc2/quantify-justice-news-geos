import spacy
from typing import Any
from scripts.utils.config import _deep_union

def load_spacy(model, **kwargs):
    try:
        nlp = spacy.load(model, **kwargs)
    except:
        spacy.cli.download(model)
        nlp = spacy.load(model, **kwargs)
    return nlp

def flatten_config(cfg: dict[str, Any], parent: str = ""):
    flat = {}
    for key, val in cfg.items():
        flat_key = parent + "." + key if parent else key
        if isinstance(val, dict):
            flat |= flatten_config(val, flat_key)
        else:
            flat[flat_key] = val
    return flat

def nest_config(flat: dict[str, Any]):
    cfg = {}
    for key, val in flat.items():
        nodes = key.split('.')
        leaf = val
        for node in reversed(nodes):
            leaf = {node: leaf}
        cfg = _deep_union(cfg, leaf)
    return cfg
