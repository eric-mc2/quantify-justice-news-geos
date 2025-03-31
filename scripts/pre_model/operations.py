"""Defines orchestration-independent data logic."""
from zipfile import ZipFile
import gzip
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import logging

from scripts.utils.logging import setup_logger

logger = setup_logger(__name__)
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
    return filtered_articles