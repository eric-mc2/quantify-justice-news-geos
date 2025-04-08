import pandas as pd

import spacy 
from spacy.tokens import DocBin

from scripts.utils import preprocessing as pre
from scripts.utils.spacy import (load_spacy, 
                                 init_config, 
                                 load_metrics,
                                 train as train_spacy)
from scripts.utils.labelstudio import extract as extract_ls
from scripts.utils.logging import setup_logger

logger = setup_logger(__name__)

def pre_inference(in_path):
    logger.debug("Concatenating text.")
    article_data = pd.read_parquet(in_path)
    article_data['text'] = article_data['title'] + "\n.\n" + article_data['bodytext']
    article_data = article_data.drop(columns=['title','bodytext'])
    
    logger.debug("Normalizing text.")
    article_data = pre.normalize(article_data)

    logger.debug("Sentencizing text.")
    nlp = spacy.blank('en')
    nlp.add_pipe("sentencizer")
    docs = nlp.pipe(article_data['text'], batch_size=64)

    logger.debug("Re-data-framing.")
    article_data['docs'] = [[(i, s.text) for i,s in enumerate(d.sents)] for d in docs]
    article_data = (article_data
                .drop(columns=['text'])
                .explode('docs'))
    sentences = article_data['docs'].apply(pd.Series)
    sentences.columns = ['sentence_idx', 'sentence']
    return pd.concat([article_data.drop(columns=['docs']), sentences], axis=1)
    
def pre_annotate(in_path, out_path):
    article_data = pre_inference(in_path)
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

def _to_docbin(df, all_labels, out_path, mode='train'):
    nlp = spacy.blank("en")
    doc_bin = DocBin(store_user_data=True)
    text = df['sentence']
    metadata = df.drop(columns='sentence')
    data_tuples = ((t,m) for t,m in zip(text, metadata.itertuples(index=False)))
    for doc, meta in nlp.pipe(data_tuples, as_tuples=True):
        metadict = meta._asdict()
        if mode == 'train':
            for label in all_labels:
                doc.cats[label] = 1 if label in meta.multilabel else 0
            del metadict['multilabel']
        else:
            for label in all_labels:
                doc.cats[label] = metadict[label]
                del metadict[label]
        doc.user_data |= metadict
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
    return train, dev, test

def train(base_cfg, full_cfg, train_path, dev_path, out_path, overrides={}):
    init_config(base_cfg, full_cfg)
    train_spacy(train_path, dev_path, full_cfg, out_path, overrides)
    metrics = load_metrics(out_path)
    return metrics

def inference(in_data_path, model_path, out_data_path):
    df = pre_inference(in_data_path)
    nlp = spacy.load(model_path)
    docs = list(nlp.pipe(df['sentence']))
    relevant = [any(map(lambda x: x[0] != 'IRRELEVANT' and x[1]>.5, d.cats.items())) for d in docs]
    relevant = pd.Series(relevant, index=df.index)
    cats = pd.Series([d.cats for d in docs], index=df.index).apply(pd.Series)
    result = pd.concat([df[relevant], cats[relevant]], axis=1)
    _to_docbin(result, list(cats.columns), out_data_path, mode='inference')
    return result