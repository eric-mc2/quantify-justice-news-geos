from scripts.utils.runners import cmd
import spacy
from typing import Any
import os
import json
from math import nan
from scripts.utils import flatten_config
from scripts.utils.logging import setup_logger

logger = setup_logger(__name__)

def init_config(base_cfg, full_cfg, code=None):
    command = f"python -m spacy init fill-config {base_cfg} {full_cfg}"
    if code:
        command += f" --code {code}"
    logger.info(f"Running command {command}")
    cmd(command)

def load_spacy(model, **kwargs):
    try:
        nlp = spacy.load(model, **kwargs)
    except:
        spacy.cli.download(model)
        nlp = spacy.load(model, **kwargs)
    return nlp

def init_labels(full_cfg, out_path, code=None):
    # XXX: Couldn't get this to work! Throws strange error about
    # out path being a directory when it's unclear from docs if it should be
    # file or directory. 
    command = f"""python -m spacy init labels {full_cfg} {out_path}"""
    if code:
        command += f" --code {code}"
    cmd(command)

def train(train_path, dev_path, full_cfg, model_path, overrides: dict[str,Any] = {}):
    command = f"""python -m spacy train {full_cfg}
                --paths.train {train_path} --paths.dev {dev_path}
                --output {model_path}"""
    for key,val in overrides.items():
        command += f" --{key} {val}"
    cmd(command, "Training time: {:.1f}s")

def evaluate(model_path, test_path, out_metrics, out_data, overrides: dict[str,Any] = {}):
    command = f"""python -m spacy benchmark accuracy
                {model_path} {test_path} --output {out_metrics}"""
    for key,val in overrides.items():
        command += f" --{key} {val}"
    cmd(command, "Eval time: {:.1f}s")

    command = f"""python -m spacy apply
                {model_path} {test_path} {out_data} --force"""
    for key,val in overrides.items():
        command += f" --{key} {val}"
    cmd(command, "Eval time: {:.1f}s")

def assemble(full_cfg, model_path, overrides: dict[str,Any] = {}):
    command = f"""python -m spacy assemble {full_cfg} {model_path}"""
    for key,val in overrides.items():
        command += f" --{key} {val}"
    cmd(command)

def load_metrics(model_path, task='textcat'):
    best_model_path = os.path.join(model_path, "model-best")
    metric_file = os.path.join(best_model_path, "meta.json")
    with open(metric_file) as fp:
        metrics = json.load(fp)
    return score_metrics(metrics, task)

def score_metrics(metrics, task="textcat"):
    if task == "textcat":
        keys = ['cats_micro_p','cats_micro_r','cats_micro_f',
            'cats_macro_p','cats_macro_r','cats_macro_f',
            'cats_f_per_type','cats_auc_per_type','cats_score']
    elif task == "ner":
        keys = ['ents_f','ents_p','ents_r','ents_per_type','ner_loss']
    out_metrics = {}
    for key in keys:
        if key in metrics['performance']:
            out_metrics |= flatten_config({key: metrics['performance'].get(key, nan)})
    return out_metrics
