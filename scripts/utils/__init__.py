import spacy
from typing import Any

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