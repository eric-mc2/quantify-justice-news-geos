import pandas as pd
import spacy 
import logging
from spacy.tokens import DocBin

from scripts.utils import preprocessing as pre
from scripts.utils.spacy import (load_spacy, 
                                 init_config, 
                                 load_metrics,
                                 train as train_spacy)
from scripts.utils.labelstudio import extract as extract_ls

# TODO: move basic config to top-level definitions
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def preprocess(dep_path, base_model, out_path):
    logger.debug("Concatenating text.")
    article_data = pd.read_parquet(dep_path)
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

def annotate(deps_path, out_path):
    data = extract_ls(deps_path)
    groupby = data.columns.drop('label').to_list()
    data = data.groupby(groupby, as_index=False)['label'].agg(tuple)
    data = data.rename(columns={'label':'multilabel'})
    data.to_json(out_path, lines=True, orient="records", index=False)
    return data

def split(dep_path, train_path, dev_path, test_path):    
    logger.debug("Splitting text.")
    article_data = pd.read_json(dep_path, lines=True, orient="records")
    all_labels = {lab for row in article_data['multilabel'] for lab in row}
    train, dev, test = pre.split_train_dev_test(article_data)
    del article_data

    def _to_docbin(df, out_path):
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
    
    logger.debug("Writing text.")
    _to_docbin(train, train_path)
    _to_docbin(dev, dev_path)
    _to_docbin(test, test_path)

def train(base_cfg, full_cfg, train_path, dev_path, out_path, overrides=None):
    init_config(base_cfg, full_cfg)
    train_spacy(train_path, dev_path, full_cfg, out_path, overrides)
    metrics = load_metrics(out_path)
    return metrics
