import pandas as pd
import logging
import json 
from dataclasses import dataclass, asdict

import spacy 
from spacy.tokens import DocBin

from scripts.utils import preprocessing as pre
from scripts.utils.spacy import load_spacy

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