"""Defines orchestration-independent data logic."""
from zipfile import ZipFile
import gzip
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import logging
import spacy
from spacy.tokens import DocBin
from tempfile import NamedTemporaryFile

from scripts.utils import preprocessing as pre
from scripts.utils.labelstudio import extract as extract_ls
from scripts.utils.spacy import (init_config, train as train_spacy, load_metrics)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def extract(in_path, out_path):
    with ZipFile(in_path, 'r') as zf:
        with zf.open("cjp_tables/newsarticles_article.csv.gz", "r") as zzf:
            with gzip.open(zzf) as zzzf:
                article_data_chunks = pd.read_csv(zzzf,
                        names=['id','feedname','url','orig_html','title','bodytext',
                                'relevant','created','last_modified','news_source_id', 'author'],
                            true_values=['t'], false_values=['f'],
                            iterator=True, chunksize=1000)
                writer = None
                for chunk in article_data_chunks:
                    chunk = chunk.filter(['id','title','bodytext','relevant'])
                    table = pa.Table.from_pandas(chunk)
                    if writer is None:
                        writer = pq.ParquetWriter(out_path, table.schema)
                    writer.write_table(table)
                writer.close()

def news_relevant(in_path, out_path):
    article_data = pd.read_parquet(in_path)
    filtered_articles = article_data[article_data.relevant].drop(columns='relevant')
    filtered_articles.to_parquet(out_path)

def prototype_sample(in_path, out_path=None, k=200, seed=31825):
    data = pd.read_parquet(in_path)
    proto = data.sample(k, random_state=seed)
    if out_path:
        proto.to_parquet(out_path)
    return proto
    
def preprocess(in_path, out_path):
    article_data = pd.read_parquet(in_path, columns=['id','title'])
    logger.debug("Normalizing text.")
    article_data = pre.normalize(article_data, "title")
    if out_path:
        article_data.to_json(out_path, orient='records', index=False, force_ascii=True)
    return article_data

def annotate(deps_path, out_path):
    data = extract_ls(deps_path)
    data.to_json(out_path, lines=True, orient="records", index=False)
    return data

def split(in_path, train_path, dev_path, test_path):
    logger.debug("Splitting text.")
    article_data = pd.read_json(in_path, lines=True, orient="records")
    all_labels = article_data['label'].unique()
    train, dev, test = pre.split_train_dev_test(article_data)
    del article_data

    def _to_docbin(df, out_path):
        nlp = spacy.blank("en")
        doc_bin = DocBin()
        text = df['title']
        meta = df.drop(columns='title')
        data_tuples = ((t,m) for t,m in zip(text, meta.itertuples()))
        for doc, eg in nlp.pipe(data_tuples, as_tuples=True):
            for label in all_labels:
                doc.cats[label] = 1 if eg.label == label else 0
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

def filter(model_path, in_data_path, out_data_path, k, seed):
    with NamedTemporaryFile("wb") as f:
        df_orig = prototype_sample(in_data_path, f.name, k, seed)
        f.seek(0)
        df_prep = preprocess(f.name, None)
    
    nlp = spacy.load(model_path)
    labels = [doc.cats['CRIME'] > doc.cats['IRRELEVANT']
                for doc in nlp.pipe(df_prep['title'])]
    relevant = pd.Series(labels, df_prep.index)
    result = df_orig[relevant]
    result.to_parquet(out_data_path)
    return result