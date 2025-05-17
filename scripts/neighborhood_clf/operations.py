import pandas as pd
from pathlib import Path
import os

import spacy
from spacy.tokens import DocBin, Doc
from spacy.cli import apply as spacy_infer
from spacy.language import Language
import srsly

from scripts.utils import preprocessing as pre
from scripts.neighborhood_clf.components import make_nclf, cat_merger
from scripts.utils.logging import setup_logger
from scripts.utils.spacy import (init_config, 
                                 load_metrics,
                                 train as train_spacy,
                                 evaluate as eval_spacy)

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

def split(in_path, base_model, train_path, dev_path, test_path):    
    model = spacy.load(base_model)
    docs = list(DocBin().from_disk(in_path).get_docs(model.vocab))
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

def pre_inference(in_path, base_model) -> list[Doc]:
    model = spacy.load(base_model)
    docs = list(DocBin().from_disk(in_path).get_docs(model.vocab))
    labels = ['GPE','LOC','FAC']
    shapes = ['community', 'block']
    ents = set([e for d in docs 
                for e in d.ents 
                if e.label_ in labels
                and e._.gpe_shape not in shapes])
    return ents

def ner_labels(in_path, base_model, block_path, cross_path, out_path):
    model = spacy.load(base_model)
    docs = list(DocBin().from_disk(in_path).get_docs(model.vocab))
    blocks = pd.read_parquet(block_path).groupby('block_name')['community_name'].unique()
    crosses = pd.read_parquet(cross_path).groupby('cross_name')['community_name'].unique()
    labels = ['GPE','LOC','FAC']
    shapes = ['community', 'block', 'cross']
    ents = set([e for d in docs 
                for e in d.ents 
                if e.label_ in labels
                and e._.gpe_shape in shapes])
    out_data = []
    for e in ents:
        if e._.gpe_shape == "community":
            out_data.append({"text": e.text, "community_name": e.text, "gpe_shape": e._.gpe_shape})
        elif e._.gpe_shape == "block":
            if e.text not in blocks:
                continue
            for comm in blocks[e.text]:
                out_data.append({"text": e.text, "community_name": comm, "gpe_shape": e._.gpe_shape})
        elif e._.gpe_shape == "cross":
            if e.text not in crosses:
                continue
            for comm in crosses[e.text]:
                out_data.append({"text": e.text, "community_name": comm, "gpe_shape": e._.gpe_shape})
    out_data = pd.DataFrame.from_records(out_data)
    out_data.to_parquet(out_path)
    return out_data


def merge_synth_data(cross_path, geo_path, ner_path, base_model, k_train, out_path):
    crosses = pd.read_parquet(cross_path).rename(columns={"cross_name":"text"})
    geos = pd.read_parquet(geo_path)
    ners = pd.read_parquet(ner_path)
    cols = ['text','community_name']
    k = min(len(crosses), len(geos), len(ners), k_train)
    data = pd.concat([crosses.filter(cols).sample(k),
                      geos.filter(cols).sample(k),
                      ners.filter(cols + ['gpe_shape']).sample(k)])
    model = spacy.load(base_model)
    db = _to_docbin(data, model, "text", ["community_name"], ["gpe_shape"])
    db.to_disk(out_path)
    return data

def _to_docbin(df: pd.DataFrame, model, text_col="text", cats=[], user_data=[]):
    db = DocBin(store_user_data=True)
    ctx = df.drop(columns=text_col)
    data_iter = zip(df[text_col], ctx.itertuples())
    for doc, ctx in model.pipe(data_iter, as_tuples=True, batch_size=64):
        for cat in filter(lambda c: c in ctx._fields, cats):
            doc.cats[getattr(ctx, cat)] = 1
        for dat in filter(lambda d: d in ctx._fields, user_data):
            doc.user_data[dat] = getattr(ctx, dat)
        db.add(doc)
    return db
    

def pre_annotate(in_path, base_model, out_path):
    ents = pre_inference(in_path, base_model)
    ents = [{'text': e.text, "gpe_shape": e._.gpe_shape} for e in ents]
    srsly.write_json(out_path, ents)
    return pd.DataFrame.from_records(ents)


def geocodes(in_path, out_path):
    labels = (pd.read_parquet(in_path)
              .filter(['text','neighborhood','confidence'])
              .dropna()
              .rename(columns={'neighborhood':'community_name'})
              .pipe(pre.normalize)
              .drop_duplicates(['text','community_name'])
              .query('confidence > .9'))
    labels.to_parquet(out_path)
    # nlp = spacy.blank('en')
    # docs = list(nlp.pipe(labels['text'], batch_size=64))
    # db = DocBin()
    # for doc, row in zip(docs, labels.itertuples()):
    #     doc.cats[row.community_name] = 1
    #     doc.user_data['confidence'] = row.confidence
    #     db.add(doc)
    # db.to_disk(out_path)
    return labels


def train(base_cfg, full_cfg, train_path, dev_path, blocks_path, neighborhood_path, out_path, overrides={}):
    code_path = os.path.join(os.path.dirname(__file__), "components.py")
    overrides |= {"code": code_path,
                  "initialize.components.nclf.neighborhood_path": neighborhood_path,
                  "initialize.components.nclf.blocks_path": blocks_path}

    init_config(base_cfg, full_cfg, code_path)
    # assemble(full_cfg, out_path, overrides)                      
    train_spacy(train_path, dev_path, full_cfg, out_path, overrides)
    metrics = load_metrics(out_path)
    return metrics

def evaluate(model_path, in_path, out_metrics, out_data, overrides = {}):
    code_path = os.path.join(os.path.dirname(__file__), "components.py")
    overrides |= {"code": code_path}
                #   "initialize.components.nclf.neighborhood_path": neighborhood_path,
                #   "initialize.components.nclf.blocks_path": blocks_path}
    eval_spacy(model_path, in_path, out_metrics, out_data, overrides)
    
# def annotate(in_path, out_path):
#     data = extract_ls(in_path)
#     groupby = data.columns.drop('label').to_list()
#     data = data.groupby(groupby, as_index=False)['label'].agg(tuple)
#     data = data.rename(columns={'label':'multilabel'})
#     data.to_json(out_path, lines=True, orient="records", index=False)
#     return data


def inference(in_path, model_path, out_path):
    Language.component('cat_merger', func=cat_merger)
    Language.factory("nclf", func=make_nclf)
    spacy_infer(Path(in_path), Path(out_path), model_path, None, 64, 1)
    docs = list(DocBin().from_disk(out_path).get_docs(spacy.load(model_path).vocab))
    return docs