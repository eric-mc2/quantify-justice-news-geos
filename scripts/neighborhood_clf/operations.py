import pandas as pd
from pathlib import Path
import os

import spacy
from spacy.tokens import DocBin, Doc
from spacy.cli import apply as spacy_infer
from thinc.api import Config

from scripts.utils import preprocessing as pre
from scripts.utils.logging import setup_logger
from scripts.utils.labelstudio import extract as extract_ls
from scripts.utils.spacy import (load_spacy, 
                                 init_config, 
                                 train as train_spacy,
                                 assemble)

logger = setup_logger(__name__)

def join_sentences(in_path, base_model, out_path):
    model = spacy.load(base_model)
    sentences = DocBin().from_disk(in_path).get_docs(model.vocab)
    sentences = sorted(sentences, key=lambda d: (d.user_data['id'], d.user_data['sentence_idx']))
    articles = DocBin(store_user_data=True)
    article = [sentences[0]]
    for sentence in sentences[1:]:
        if article[-1].user_data['id'] == sentence.user_data['id']:
            article.append(sentence)
        else:
            articles.add(Doc.from_docs(article))
            article = [sentence]
    articles.add(Doc.from_docs(article)) # process last sentence
    articles.to_disk(out_path)

def split(in_path, base_model, train_path, dev_path, test_path):    
    model = spacy.load(base_model)
    docs = list(DocBin().from_disk(in_path).get_docs(model.vocab))
    train, dev, test = pre.split_train_dev_test(docs)
    traindb, devdb, testdb = DocBin(), DocBin(), DocBin()
    for d in train:
        traindb.add(d)
    for d in dev:
        devdb.add(d)
    for d in test:
        testdb.add(d)
    traindb.to_disk(train_path)
    devdb.to_disk(dev_path)
    testdb.to_disk(test_path)
    return train, dev, test

def pre_annotate(in_path, base_model, out_path):
    model = spacy.load(base_model)
    docs = list(DocBin().from_disk(in_path).get_docs(model.vocab))
    # XXX Drops user info
    texts = [{'text': d.text} for d in docs]
    texts = pd.DataFrame.from_dict(texts)
    texts.to_json(out_path, orient='records', index=False, force_ascii=True)
    return texts
    
def annotate(in_path, out_path):
    data = extract_ls(in_path)
    groupby = data.columns.drop('label').to_list()
    data = data.groupby(groupby, as_index=False)['label'].agg(tuple)
    data = data.rename(columns={'label':'multilabel'})
    data.to_json(out_path, lines=True, orient="records", index=False)
    return data

def train(base_cfg, 
          full_cfg, 
          train_path, 
          dev_path, 
          out_path, 
          neighborhood_path,
          overrides = {}):
    cfg = Config().from_disk(base_cfg)
    cfg['initialize']['components']['nclf']['labels_path'] = neighborhood_path
    cfg['paths']['train'] = train_path
    cfg['paths']['dev'] = dev_path
    cfg['corpora']['train']['path'] = train_path
    cfg['corpora']['dev']['path'] = dev_path
    cfg.to_disk(base_cfg)

    code_path = os.path.join(os.path.dirname(__file__), "components.py")

    logger.debug("init config...")
    init_config(base_cfg, full_cfg, code_path)
    logger.debug("train...")
    assemble(full_cfg, out_path, overrides | {"code": code_path})

def inference(in_path, model_path, out_path):
    spacy_infer(Path(in_path), Path(out_path), model_path, None, 1, 1)
    docs = list(DocBin().from_disk(out_path).get_docs(spacy.load(model_path).vocab))
    return docs