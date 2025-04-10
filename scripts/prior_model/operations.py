
"""Defines orchestration-independent data logic."""
import json
from zipfile import ZipFile
import gzip
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import logging
from spacy.util import filter_spans

from scripts.utils.logging import setup_logger
from scripts.utils import preprocessing as pre

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

def geocodes(in_path, out_path):
    with ZipFile(in_path, 'r') as zf:
        with zf.open("cjp_tables/newsarticles_trainedlocation.csv.gz", "r") as zzf:
            with gzip.open(zzf) as zzzf:
                geocodes = pd.read_csv(zzzf,
                        names=['id','text','latitude','longitude','coding_id',
                                'confidence','neighborhood','is_best'],
                        true_values=['t'],
                        false_values=['f'])
                geocodes.to_parquet(out_path)
    return geocodes

def user_coding(in_path, articles_path, out_path):
    # Raw locations
    with ZipFile(in_path, 'r') as zf:
        with zf.open("cjp_tables/newsarticles_usercoding.csv.gz", "r") as zzf:
            with gzip.open(zzf) as zzzf:
                loc_data = pd.read_csv(zzzf, 
                            names=['id','date','relevant','article_id','user_id','locations','sentiment'],
                            dtype={'locations':'str'},
                            true_values=['t'],
                            false_values=['f'])
                # Do NOT filter negatives yet. We want to compare
                # if new model flags any of these as positive!
                # mask = (loc_data['locations'] != '[]') & loc_data['relevant']
                mask = loc_data['relevant']
                loc_data = loc_data[mask]
    
    # Break out locations
    loc_data['location'] = loc_data['locations'].apply(json.loads)
    loc_data = loc_data.explode('location', ignore_index=True).drop(columns=['locations'])
    loc_data_locs = (loc_data.location.apply(pd.Series)
                    .filter(['start','end','text']))
    loc_data = pd.concat([loc_data, loc_data_locs], axis=1)

    # Quality check. Some texts are very very long. Eyeballing the top of the
    # distribution, SOME are obviously including extraneous text e.g.
    # "West Englewood neighborhood on the South Side, authorities said."
    # while MOST are compound location / prepositional phrases like:
    # "5200 block of West Kamerling Avenue in the city's Austin neighborhood"
    # TODO: it may / may not be worth training a rule-based matcher on these phrases
    # The error rate only seems to blow up in the way upper tail of the distribution,
    # the 99.5th percentile, which is 74 characters.
    cutoff = loc_data['text'].str.len().quantile(.995)
    logger.info("Dropping all locations >= %d characters", cutoff)
    loc_data = loc_data[loc_data['text'].str.len() < cutoff]

    loc_data = loc_data.filter(['id','article_id','user_id','start','end','text'])
    
    articles = pd.read_parquet(articles_path, columns=['id','title','bodytext'])
    articles['text'] = articles['title'] + "\n.\n" + articles['bodytext']
    loc_data = pre.fix_span_prelabels(loc_data, articles[['id','text']])
    loc_data.to_parquet(out_path)
    return loc_data
