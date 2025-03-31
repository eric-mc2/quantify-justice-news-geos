"""Defines orchestration-independent data logic."""
import pandas as pd
import logging
import spacy
from spacy.tokens import DocBin

from scripts.utils import preprocessing as pre
from scripts.utils.labelstudio import extract as extract_ls
from scripts.utils.spacy import (init_config, train as train_spacy, load_metrics)
from scripts.utils.logging import setup_logger

logger = setup_logger(__name__)
logger.setLevel(logging.INFO)

def prototype_sample(in_path, k, seed):
    data = pd.read_parquet(in_path)
    proto = data.sample(k, random_state=seed)
    return proto

def pre_inference(in_path=None, in_df=None, mode="train"):
    select_cols = ['id','title'] + (["bodytext"] if mode == "inference" else [])
    if in_df is None:
        in_df = pd.read_parquet(in_path, columns=select_cols)
    else:
        in_df = in_df.filter(select_cols)
    logger.debug("Normalizing text.")
    return pre.normalize(in_df, "title")

def pre_annotate(in_path, out_path, k, seed):
    df = prototype_sample(in_path, k, seed)
    df = pre_inference(in_path=None, in_df=df)
    df.to_json(out_path, orient='records', index=False, force_ascii=True)
    return df

def annotate(in_path, out_path):
    data = extract_ls(in_path)
    data.to_json(out_path, lines=True, orient="records", index=False)
    return data

def split(in_path, train_path, dev_path, test_path):
    logger.debug("Splitting text.")
    article_data = pd.read_json(in_path, lines=True, orient="records")
    all_labels = article_data['label'].unique()
    train, dev, test = pre.split_train_dev_test(article_data)
    del article_data

    def _to_docbin(df, out_path):
        nlp = spacy.blank("en")
        doc_bin = DocBin()
        text = df['title']
        meta = df.drop(columns='title')
        data_tuples = ((t,m) for t,m in zip(text, meta.itertuples()))
        for doc, eg in nlp.pipe(data_tuples, as_tuples=True):
            for label in all_labels:
                doc.cats[label] = 1 if eg.label == label else 0
            doc_bin.add(doc)
        doc_bin.to_disk(out_path)

    logger.debug("Writing text.")
    _to_docbin(train, train_path)
    _to_docbin(dev, dev_path)
    _to_docbin(test, test_path)
    return train, dev, test

def train(base_cfg, full_cfg, train_path, dev_path, out_path, overrides={}):
    init_config(base_cfg, full_cfg)
    train_spacy(train_path, dev_path, full_cfg, out_path, overrides)
    metrics = load_metrics(out_path)
    return metrics

def inference(model_path, in_data_path, out_data_path, k, seed):
    df = prototype_sample(in_data_path, k, seed)
    df = pre_inference(in_path=None, in_df=df, mode="inference")
    
    nlp = spacy.load(model_path)
    labels = [doc.cats['CRIME'] > doc.cats['IRRELEVANT']
                for doc in nlp.pipe(df['title'])]
    relevant = pd.Series(labels, df.index)
    result = df[relevant]
    result.to_parquet(out_data_path)
    return result