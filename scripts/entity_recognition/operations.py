import pandas as pd
import json 
from dataclasses import asdict
import os
from pathlib import Path

import spacy
from spacy.tokens import DocBin, Doc
from spacy.util import filter_spans
from thinc.api import Config
from spacy.language import Language
from spacy.cli import apply as spacy_infer

from scripts.utils.labelstudio import (extract as extract_ls,
                                       LSDoc, LSData, LSPrediction,
                                       LSResult, LSValue)
from scripts.utils import preprocessing as pre
from scripts.utils.spacy import (load_spacy, 
                                 init_config, 
                                 train as train_spacy)
from scripts.utils.logging import setup_logger
from scripts.entity_recognition.components import (block_matcher, intersection_matcher, street_vs_neighborhood)

logger = setup_logger(__name__)

def _to_docbin(df, out_path):
    nlp = spacy.blank("en")
    doc_bin = DocBin(store_user_data=True)
    text = df['sentence']
    meta = df.drop(columns='sentence')
    data_tuples = [(t,m) for t,m in zip(text, meta.itertuples(index=False))]
    for doc, eg in nlp.pipe(data_tuples, as_tuples=True):
        doc.user_data |= eg._asdict()
        doc_bin.add(doc)
    doc_bin.to_disk(out_path)

def split(in_path, train_path, dev_path, test_path):    
    logger.debug("Splitting text.")
    article_data = pd.read_parquet(in_path)
    train, dev, test = pre.split_train_dev_test(article_data, stratify=['id'])
    del article_data

    logger.debug("Writing text.")
    _to_docbin(train, train_path)
    _to_docbin(dev, dev_path)
    _to_docbin(test, test_path)
    return train, dev, test

def prelabel(in_path, out_path, model):
    nlp = load_spacy(model)
    docs = list(DocBin().from_disk(in_path).get_docs(nlp.vocab))
    docs = [d for d in docs if 'WHERE' in d.user_data and d.user_data['WHERE'] > .5]
    docs = nlp.pipe(docs)
    docs = [_convert(d, model) for d in docs]
    with open(out_path, "w") as f:
        json.dump(docs, f)
    return docs

def _convert(doc, model):
    return asdict(LSDoc(
        data=LSData(
            text=doc.text),
        predictions=[LSPrediction(
            model_version=model, 
            score=1, 
            result=[LSResult(id=str(i), 
                             from_name="label", 
                             to_name="text", 
                             type="labels",
                             value=LSValue(
                                 start=e.start_char,
                                 end=e.end_char,
                                 score=1,
                                 text=e.text,
                                 labels=[e.label_]
                             )) 
                    for i,e in enumerate(doc.ents)])]
    ))

def annotate(in_path, out_path):
    data = extract_ls(in_path, 'ner')
    logger.warning(f"Filtering out {data.isna().any(axis=1).sum()} null-ish rows.")
    data = data.dropna()
    data = data.groupby('text').apply(lambda x:
        x.drop(columns=['text','id']).to_dict(orient='records'))
    data = data.rename('meta').reset_index()
    _to_ents_docbin(data, out_path)
    return data

def _to_ents_docbin(df, out_path):
    nlp = spacy.blank("en")
    doc_bin = DocBin()
    data_tuples = [(t,m) for t,m in zip(df['text'], df['meta'])]
    for doc, ents in nlp.pipe(data_tuples, as_tuples=True):
        # char_span returns None if start/end are invalid token boundaries!
        spans = [doc.char_span(e['start'], e['end'], e['label']) for e in ents]
        for e,s in zip(ents, spans):
            if s is None:
                logger.warning("Invalid span boundaries: {}".format(e))
        spans = [s for s in spans if s is not None]
        # Still might have overlapping spans if dataset had duplicate sentences.
        spans = filter_spans(spans)
        doc.set_ents(spans)
        doc_bin.add(doc)
    doc_bin.to_disk(out_path)

def train(base_cfg, 
          full_cfg, 
          train_path, 
          dev_path, 
          out_path, 
          comm_area_path,
          neighborhood_path,
          street_path, 
          overrides = {}):
    cfg = Config().from_disk(base_cfg)
    cfg['paths']['train'] = train_path
    cfg['paths']['dev'] = dev_path
    cfg['corpora']['train']['path'] = train_path
    cfg['corpora']['dev']['path'] = dev_path
    cfg.to_disk(base_cfg)

    code_path = os.path.join(os.path.dirname(__file__), "components.py")
    more_overrides = {"code": code_path,
                      "initialize.components.community_matcher.comm_area_path": comm_area_path,
                      "initialize.components.neighborhood_name_matcher.neighborhood_path": neighborhood_path,
                      "initialize.components.neighborhood_matcher.neighborhood_path": neighborhood_path,
                      "initialize.components.street_matcher.street_name_path": street_path,
    }

    init_config(base_cfg, full_cfg, code_path)
    train_spacy(train_path, dev_path, full_cfg, out_path, overrides | more_overrides)
    # metrics = load_metrics(out_path)
    # return metrics

def inference(in_path, model_path, out_path, filter_=True):
    Language.component('block_matcher', func=block_matcher)
    Language.component('intersection_matcher', func=intersection_matcher)
    Language.component('street_vs_neighborhood', func=street_vs_neighborhood)
    spacy_infer(Path(in_path), Path(out_path), model_path, None, 1, 1)
    docs = list(DocBin().from_disk(out_path).get_docs(spacy.load(model_path).vocab))
    if filter_:
        crime_ents = ["FAC","GPE","LOC","PERSON"]
        docs = [d for d in docs if any([e.label_ in crime_ents for e in d.ents])]
    db = DocBin(store_user_data=True)
    for doc in docs:
        db.add(doc)
    db.to_disk(out_path)
    return docs
