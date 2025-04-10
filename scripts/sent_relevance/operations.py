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
    logger.info("Concatenating text.")
    article_data = pd.read_parquet(in_path)
    article_data['text'] = article_data['title'] + "\n.\n" + article_data['bodytext']
    article_data = article_data.drop(columns=['title','bodytext'])
    logger.debug("article data has %d rows, %d unique ids, %d unique text",
                 len(article_data), article_data.id.nunique(), article_data.text.nunique())
    
    logger.info("Normalizing text.")
    article_data = pre.normalize(article_data)
    logger.debug("article data has %d rows, %d unique ids, %d unique text",
                 len(article_data), article_data.id.nunique(), article_data.text.nunique())

    logger.info("Sentencizing text.")
    nlp = spacy.blank('en')
    nlp.add_pipe("sentencizer")
    # Note: Make this a list now to immediately invoke pipe
    docs = list(nlp.pipe(article_data['text'], batch_size=64))
    logger.debug("Sentencized %d docs", len(docs))

    logger.info("Re-data-framing.")
    sentences = [[{'sentence_idx': i, 'sentence': s.text} for i,s in enumerate(d.sents)] for d in docs]
    sentences = pd.Series(sentences, article_data.index).explode().apply(pd.Series)
    logger.debug("Sentencized into %d sentences", len(sentences))
    sentence_data = article_data.drop(columns=['text']).join(sentences).reset_index(drop=True)
    logger.debug("Pre-inference data has %d rows, %d ids, max %d sentences",
                 len(sentence_data), sentence_data.id.nunique(), sentence_data.sentence_idx.max())
    return sentence_data
    
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

def inference(in_data_path, model_path, out_data_path, filter_=True):
    df = pre_inference(in_data_path)
    nlp = spacy.load(model_path)
    docs = list(nlp.pipe(df['sentence'], batch_size=64))
    logger.debug("Predicted on %d docs", len(docs))
    cats = pd.Series([d.cats for d in docs], index=df.index).apply(pd.Series)
    logger.debug("Predicted %d cats", len(cats))
    df = df.join(cats, lsuffix="_article", rsuffix="_sentence")
    logger.debug("Joined data has %d rows", len(df))
    relevant = (cats.drop(columns=['IRRELEVANT']) > .5).any(axis=1)
    logger.info("Pct relevant %.2f", relevant.mean())
    if filter_:
        df = df[relevant]
        docs = [d for d,r in zip(docs, relevant) if r]
    names = list(cats.columns) + ['CRIME','IRRELEVANT_sentence','IRRELEVANT_article']
    names = df.columns.intersection(names)
    _to_docbin(df, names, out_data_path, mode='inference')
    return df