from scripts.utils.runners import cmd
import spacy
from typing import Any
import os
import json
from math import nan
from scripts.utils import flatten_config

def init_config(base_cfg, full_cfg):
    command = f"python -m spacy init fill-config {base_cfg} {full_cfg}"
    cmd(command)

def load_spacy(model, **kwargs):
    try:
        nlp = spacy.load(model, **kwargs)
    except:
        spacy.cli.download(model)
        nlp = spacy.load(model, **kwargs)
    return nlp

def train(train_path, dev_path, full_cfg, model_path, overrides: dict[str,Any] = {}):
    command = f"""python -m spacy train {full_cfg}
                --paths.train {train_path} --paths.dev {dev_path}
                --output {model_path}"""
    for key,val in overrides.items():
        command += f" --{key} {val}"
    cmd(command, "Training time: {:.1f}s")

def load_metrics(model_path):
    best_model_path = os.path.join(model_path, "model-best")
    metric_file = os.path.join(best_model_path, "meta.json")
    with open(metric_file) as fp:
        metrics = json.load(fp)
    return score_metrics(metrics)

def score_metrics(metrics):
    keys = ['cats_micro_p','cats_micro_r','cats_micro_f',
            'cats_macro_p','cats_macro_r','cats_macro_f',
            'cats_f_per_type','cats_auc_per_type','cats_score']
    out_metrics = {}
    for key in keys:
        if key in metrics['performance']:
            out_metrics |= flatten_config({key: metrics['performance'].get(key, nan)})
    return out_metrics
