"""Defines orchestration-independent data logic."""
from zipfile import ZipFile
import gzip
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from typing import Any
from scripts.utils import preprocessing as pre
from scripts.utils.runners import cmd
import logging
import spacy
from spacy.tokens import DocBin

def extract(dep_path, out_path):
    with ZipFile(dep_path, 'r') as zf:
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

def news_relevant(dep_path, out_path):
    article_data = pd.read_parquet(dep_path)
    filtered_articles = article_data[article_data.relevant].drop(columns='relevant')
    filtered_articles.to_parquet(out_path)

def prototype_sample(dep_path, out_path):
    data = pd.read_parquet(dep_path)
    proto = data.sample(200, random_state=31525)
    proto.to_parquet(out_path)
    return proto
    
def preprocess(dep_path, out_path):
    logger = logging.getLogger(__name__)

    article_data = pd.read_parquet(dep_path, columns=['id','title'])
    logger.debug("Normalizing text.")
    article_data = pre.normalize(article_data, "title")

    article_data.to_json(out_path, orient='records', index=False, force_ascii=True)
    return article_data

def annotate(deps_path, out_path):
    data = pd.read_json(deps_path)

    cols = ['data','annotations','file_upload','created_at','updated_at','total_annotations','cancelled_annotations']
    data = data.filter(cols)

    # Annotations is a list so unpack it.
    data = data.explode('annotations')

    # Recover original data
    data_input = data['data'].apply(pd.Series)

    # Parse out the actual label. Discard other label provenance metadata.
    data_annot = data['annotations'].apply(pd.Series)
    data_annot = data_annot['result'].explode().apply(pd.Series)
    data_annot = data_annot['value'].apply(pd.Series)
    data_annot = data_annot['choices'].explode()

    # Note: keeping as categorical for now because might want to add ACCIDENT label
    data_annot.fillna('IRRELEVANT',inplace=True)
    data_annot.rename('label', inplace=True)

    # Merge and write
    data = pd.concat([data_input, data_annot], axis=1)
    data.to_json(out_path, lines=True, orient="records", index=False)
    return data

def split(dep_path, train_path, dev_path, test_path):
    logger = logging.getLogger(__name__)

    logger.debug("Splitting text.")
    article_data = pd.read_json(dep_path, lines=True, orient="records")
    train, dev, test = pre.split_train_dev_test(article_data)
    del article_data

    def to_docbin(df, out_path):
        nlp = spacy.blank("en")
        doc_bin = DocBin()
        text = df['title']
        meta = df.drop(columns='title')
        data_tuples = ((t,m) for t,m in zip(text, meta.itertuples()))
        for doc, eg in nlp.pipe(data_tuples, as_tuples=True):
            doc.cats[eg.label] = 1
            doc_bin.add(doc)
        doc_bin.to_disk(out_path)
    
    logger.debug("Writing text.")
    to_docbin(train, train_path)
    to_docbin(dev, dev_path)
    to_docbin(test, test_path)

def init_config(base_cfg, full_cfg):
    command = f"python -m spacy init fill-config {base_cfg} {full_cfg}"
    cmd(command)

def train(train_path, dev_path, full_cfg, model_path, overrides: dict[str,Any] = {}):
    command = f"""python -m spacy train {full_cfg}
                --paths.train {train_path} --paths.dev {dev_path}
                --output {model_path}"""
    for key,val in overrides.items():
        command += f" --{key} {val}"
    cmd(command, "Training time: {:.1f}s")