import pandas as pd
from pathlib import Path
import os

import spacy
from spacy.tokens import DocBin, Doc
from spacy.cli import apply as spacy_infer
from thinc.api import Config
import srsly

from scripts.utils import preprocessing as pre
from scripts.utils.logging import setup_logger
from scripts.utils.labelstudio import extract as extract_ls
from scripts.utils.spacy import (load_spacy, 
                                 init_config, 
                                 load_metrics,
                                 train as train_spacy,
                                 assemble)

logger = setup_logger(__name__)


def _join_docs(docs):
    cats = {}
    for d in docs:
        for c,p in d.cats.items():
            cats[c] = max(cats.get(c, p), p)
    doc = Doc.from_docs(docs)#, attrs=['ents'])
    doc.cats = cats
    doc.user_data['id'] = docs[-1].user_data['id']
    return doc


# Commented out because other training data is per-span and hard-ish to aggregate to meaningful articles.
# Whereas the point of aggregating to articles was to do more sophisticated
# semantic relationship modeling, if needed.
# def join_sentences(in_path, base_model, out_path):
#     model = spacy.load(base_model)
#     sentences = DocBin().from_disk(in_path).get_docs(model.vocab)
#     sentences = sorted(sentences, key=lambda d: (d.user_data['id'], d.user_data['sentence_idx']))
#     articles = DocBin(store_user_data=True)
#     article = [sentences[0]]
#     for sentence in sentences[1:]:
#         if article[-1].user_data['id'] == sentence.user_data['id']:
#             article.append(sentence)
#         else:
#             articles.add(_join_docs(article))
#             article = [sentence]
#     # process last sentence
#     articles.add(_join_docs(article))
#     # output
#     articles.to_disk(out_path)

def split(in_path, train_path, dev_path, test_path):    
    docs = pd.read_parquet(in_path)
    train, dev, test = pre.split_train_dev_test(docs)
    traindb, devdb, testdb = DocBin(store_user_data=True), DocBin(store_user_data=True), DocBin(store_user_data=True)
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

def pre_inference(in_path, base_model) -> list[str]:
    model = spacy.load(base_model)
    docs = list(DocBin().from_disk(in_path).get_docs(model.vocab))
    labels = ['GPE','LOC','FAC']
    # XXX Drops user info
    ents = set([e.text for d in docs for e in d.ents if e.label_ in labels])
    return ents

def pre_annotate(in_path, base_model, out_path):
    ents = pre_inference(in_path, base_model)
    ents = [{'text': e} for e in ents]
    srsly.write_json(out_path, ents)
    return pd.DataFrame.from_records(ents)

def synthetic_data(intersections_path, out_path, k=100000):
    crosses = pd.read_parquet(intersections_path).rename(columns={'cross_name':'name'})
    crosses = crosses.sample(n=min(k, len(crosses)))
    model = spacy.blank("en")

    def _to_docs(df):
        docs = []
        inputs = zip(df['name'], df['community_name'])
        status = 0
        for i, (doc, label) in enumerate(model.pipe(inputs, as_tuples=True, batch_size=256)):
            new_status = (100 * i) // len(df)
            if new_status % 5 == 0 and new_status > status:
                logger.debug("Piped %d percent of rows", new_status)
                status = new_status
            doc.cats[label] = 1
            docs.append(doc)
        return docs
    
    def _to_docbin(docs, path):
        db = DocBin()
        for doc in docs:
            db.add(doc)
        db.to_disk(path)

    crosses = _to_docs(crosses)
    _to_docbin(crosses, out_path)
    return crosses


def geocodes(in_path, out_path):
    labels = (pd.read_parquet(in_path)
              .filter(['text','neighborhood','confidence'])
              .rename(columns={'neighborhood':'community_name'})
              .pipe(pre.normalize)
              .drop_duplicates(['text','community_name']))
    nlp = spacy.blank('en')
    docs = list(nlp.pipe(labels['text'], batch_size=64))
    db = DocBin()
    for doc, row in zip(docs, labels.itertuples()):
        doc.cats[row.community_name] = 1
        doc.user_data['confidence'] = row.confidence
        db.add(doc)
    db.to_disk(out_path)
    return labels


def train(full_cfg, train_path, dev_path, out_path, overrides={}):
    train_spacy(train_path, dev_path, full_cfg, out_path, overrides)
    metrics = load_metrics(out_path)
    return metrics
    
# def annotate(in_path, out_path):
#     data = extract_ls(in_path)
#     groupby = data.columns.drop('label').to_list()
#     data = data.groupby(groupby, as_index=False)['label'].agg(tuple)
#     data = data.rename(columns={'label':'multilabel'})
#     data.to_json(out_path, lines=True, orient="records", index=False)
#     return data

def init_model(base_cfg, 
          full_cfg, 
          out_path, 
          blocks_path,
          neighborhood_path,
          overrides = {}):
    cfg = Config().from_disk(base_cfg)
    cfg['initialize']['components']['nclf']['neighborhood_path'] = neighborhood_path
    cfg['initialize']['components']['nclf']['blocks_path'] = blocks_path
    # cfg['paths']['train'] = train_path
    # cfg['paths']['dev'] = dev_path
    # cfg['corpora']['train']['path'] = train_path
    # cfg['corpora']['dev']['path'] = dev_path
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