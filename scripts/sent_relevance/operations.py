import pandas as pd
from pathlib import Path
from tempfile import NamedTemporaryFile

import spacy 
from spacy.cli import apply
from spacy.tokens import DocBin

from scripts.utils import preprocessing as pre
from scripts.utils.spacy import (load_spacy, 
                                 init_config, 
                                 load_metrics,
                                 train as train_spacy)
from scripts.utils.labelstudio import extract as extract_ls
from scripts.art_relevance.operations import filter as art_filter
from scripts.utils.logging import setup_logger

logger = setup_logger(__name__)

def preprocess(in_path, base_model, out_path):
    logger.debug("Concatenating text.")
    article_data = pd.read_parquet(in_path)
    article_data['text'] = article_data['title'] + "\n.\n" + article_data['bodytext']
    article_data = article_data.drop(columns=['title','bodytext'])
    
    logger.debug("Normalizing text.")
    article_data = pre.normalize(article_data)

    logger.debug("Sentencizing text.")
    nlp = load_spacy(base_model)
    nlp.disable_pipes()
    nlp.add_pipe("sentencizer")
    docs = nlp.pipe(article_data['text'], batch_size=64)

    logger.debug("Re-data-framing.")
    article_data['docs'] = [[(i, s.text) for i,s in enumerate(d.sents)] for d in docs]
    article_data = (article_data
                .drop(columns=['text'])
                .explode('docs'))
    sentences = article_data['docs'].apply(pd.Series)
    sentences.columns = ['sentence_idx', 'sentence']
    article_data = pd.concat([article_data.drop(columns=['docs']), sentences], axis=1)
    
    logger.debug("Writing to disk.")
    article_data.to_json(out_path, orient='records', index=False, force_ascii=True)
    return article_data

def annotate(in_path, out_path):
    data = extract_ls(in_path)
    groupby = data.columns.drop('label').to_list()
    data = data.groupby(groupby, as_index=False)['label'].agg(tuple)
    data = data.rename(columns={'label':'multilabel'})
    data.to_json(out_path, lines=True, orient="records", index=False)
    return data

def _to_docbin(df, all_labels, out_path):
    nlp = spacy.blank("en")
    doc_bin = DocBin()
    text = df['sentence']
    meta = df.drop(columns='sentence')
    data_tuples = ((t,m) for t,m in zip(text, meta.itertuples()))
    for doc, eg in nlp.pipe(data_tuples, as_tuples=True):
        for label in all_labels:
            doc.cats[label] = 1 if label in eg.multilabel else 0
        doc_bin.add(doc)
    doc_bin.to_disk(out_path)

def split(in_path, train_path, dev_path, test_path):    
    logger.debug("Splitting text.")
    article_data = pd.read_json(in_path, lines=True, orient="records")
    all_labels = {lab for row in article_data['multilabel'] for lab in row}
    train, dev, test = pre.split_train_dev_test(article_data)
    del article_data

    logger.debug("Writing text.")
    _to_docbin(train, all_labels, train_path)
    _to_docbin(dev, all_labels, dev_path)
    _to_docbin(test, all_labels, test_path)

def train(base_cfg, full_cfg, train_path, dev_path, out_path, overrides=None):
    init_config(base_cfg, full_cfg)
    train_spacy(train_path, dev_path, full_cfg, out_path, overrides)
    metrics = load_metrics(out_path)
    return metrics

def filter(art_model, sent_model, seed, in_data_path, out_data_path):
    with (NamedTemporaryFile("wb", suffix=".parquet") as f1,
          NamedTemporaryFile("wb", suffix=".json") as f2):
        art_filter(art_model, in_data_path, f1.name, 600, seed)
        f1.seek(0)
        df_orig = preprocess(f1.name, sent_model, f2.name)

    nlp = spacy.load(sent_model)
    with (NamedTemporaryFile("wb", suffix=".spacy") as f1,
        NamedTemporaryFile("wb", suffix=".spacy") as f2):
        _to_docbin(df_orig, [], f1.name)
        f1.seek(0)
        apply(data_path=Path(f1.name), 
            output_file=Path(f2.name), 
            model=sent_model, 
            json_field="text", 
            batch_size=1,
            n_process=1)
        f2.seek(0)
        docs = list(DocBin().from_disk(f2.name).get_docs(nlp.vocab))
    relevant = [any(map(lambda x: x[0] != 'IRRELEVANT' and x[1]>.5, d.cats.items())) for d in docs]
    relevant = pd.Series(relevant, index=df_orig.index)
    cats = pd.Series([d.cats for d in docs], index=df_orig.index).apply(pd.Series)
    result = pd.concat([df_orig[relevant], cats[relevant]], axis=1)
    result.to_parquet(out_data_path)
    return result