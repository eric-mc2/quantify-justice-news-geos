import pandas as pd
import geopandas as gpd
import logging
import json 
from dataclasses import asdict

import spacy
from spacy.tokens import DocBin, Doc, Span
from spacy.util import filter_spans
from spacy.language import Language
from spacy.matcher.phrasematcher import PhraseMatcher
from spacy.matcher.matcher import Matcher
from thinc.api import Config

from scripts.geoms.operations import sides
from scripts.utils.labelstudio import (extract as extract_ls,
                                       LSDoc, LSData, LSPrediction,
                                       LSResult, LSValue)
from scripts.utils import preprocessing as pre
from scripts.utils.spacy import (load_spacy, 
                                 init_config, 
                                 load_metrics,
                                 init_labels,
                                 train as train_spacy)

# TODO: move basic config to top-level definitions
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
          overrides=None):
    cfg = Config().from_disk(base_cfg)
    cfg['components']['gpe_matcher']['comm_area_path'] = comm_area_path
    cfg['components']['gpe_matcher']['neighborhood_path'] = neighborhood_path
    cfg['components']['street_matcher']['street_name_path'] = street_path
    cfg.to_disk(base_cfg)
    init_config(base_cfg, full_cfg, __file__)
    label_path = cfg['initialize']['components']['ner']['labels']['path']
    init_labels(full_cfg, train_path, dev_path, label_path, __file__)
    # train_spacy(train_path, dev_path, full_cfg, out_path, overrides)
    # metrics = load_metrics(out_path)
    # return metrics

@Language.factory("gpe_matcher")
def create_gpe_matcher(nlp, name, comm_area_path=None, neighborhood_path=None):
    gpes = pd.concat([gpd.read_parquet(comm_area_path)['community_name'].rename('name'),
                        pd.read_csv(neighborhood_path)['name'],
                        pd.Series(sides)], ignore_index=True)
    gpes = gpes.str.split(",", expand=False).explode()
    gpes = gpes.str.title().drop_duplicates().sort_values()
    
    matcher = PhraseMatcher(nlp.vocab)
    patterns = list(nlp.tokenizer.pipe(gpes))
    matcher.add("GPE", patterns)

    def match_gpes(doc: Doc):
        matches = matcher(doc, as_spans=True)
        doc.ents = filter_spans(list(doc.ents) + matches)
        return doc
    
    return match_gpes
    
@Language.factory("street_matcher")
def register_street_matcher(nlp, name, street_name_path=None):
    street_names = pd.read_csv(street_name_path)
    street_names = street_names.filter(like='combined').melt()['value']
    street_names = street_names.str.title().drop_duplicates().sort_values()

    loc_matcher = PhraseMatcher(nlp.vocab)
    patterns = list(nlp.tokenizer.pipe(street_names))
    loc_matcher.add("FAC", patterns)

    def match_streets(doc: Doc):
        matches = loc_matcher(doc, as_spans=True)
        doc.ents = filter_spans(list(doc.ents) + matches)
        return doc
        
    return match_streets

@Language.factory("age_matcher")
def register_age_matcher(nlp, name):
    matcher = Matcher(nlp.vocab) # Matcher might not be the right thing here since it operates on tokens
    matcher.add("CARDINAL", [[{"TEXT": {"REGEX": r"\d+[ -]year[ -]old"}}]])
    
    def match_age(doc: Doc):
        matches = matcher(doc, as_spans=True)
        doc.ents = filter_spans(list(doc.ents) + matches)
        return doc

    return match_age

@Language.component("block_matcher")
def expand_street_blocks(doc: Doc):
    new_ents = []
    for idx, ent in enumerate(doc.ents):
        # Only check for title if it's a person and not the first token
        if ent.label_ == "FAC" and ent.start >= 3 and idx >= 1:
            prev_ent = list(doc.ents)[idx-1]
            prev_tokens = doc[ent.start - 3: ent.start]
            # Must match [CARDINAL] block of [FAC]
            if (prev_tokens[2].text == "of" and prev_tokens[1].text == "block"
                and prev_ent.label_ == "CARDINAL" and prev_tokens[0].text == prev_ent.text):
                new_ent = Span(doc, ent.start - 3, ent.end, label=ent.label)
                new_ents.append(new_ent)
    doc.ents = filter_spans(list(doc.ents) + new_ents)
    return doc

@Language.component("intersection_matcher")
def expand_intersections(doc: Doc):
    new_ents = []
    for idx, ent in enumerate(doc.ents):
        # Only check for title if it's a person and not the first token
        if ent.label_ == "FAC" and ent.start >= 2 and idx >= 1:
            prev_ent = list(doc.ents)[idx-1]
            prev_tokens = doc[ent.start - 2: ent.start]
            # Must match [STREET] and [STREET]
            if ((prev_tokens[1].text == "and" or prev_tokens[1].text == "&")
                and prev_ent.label_ == "FAC" and prev_tokens[0].text == prev_ent.text):
                new_ent = Span(doc, ent.start - 2, ent.end, label=ent.label)
                new_ents.append(new_ent)
    doc.ents = filter_spans(list(doc.ents) + new_ents)
    return doc