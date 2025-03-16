from zipfile import ZipFile
import gzip
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import os
from scripts.utils.config import Config
from scripts.utils import preprocessing as pre
from scripts.utils import load_spacy
import spacy 

import dagster as dg
from dagster import get_dagster_logger


@dg.asset
def extract():
    config = Config()
    dep_path = config.get_data_path("raw.zip")
    out_path = config.get_data_path("raw.article_text")

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

@dg.asset(deps=[extract])
def news_relevant():
    config = Config()
    dep_path = config.get_data_path("raw.article_text")
    out_path = config.get_data_path("pre_relevance.article_text")
    article_data = pd.read_parquet(dep_path)
    filtered_articles = article_data[article_data.relevant].drop(columns='relevant')
    filtered_articles.to_parquet(out_path)

@dg.asset(deps=[news_relevant])
def prototype_sample():
    config = Config()
    dep_path = config.get_data_path("pre_relevance.article_text")
    out_path = config.get_data_path("relevance.article_text_prototype")
    data = pd.read_parquet(dep_path)
    proto = data.sample(200, random_state=31525)
    proto.to_parquet(out_path)
    
@dg.asset(deps=[prototype_sample])
def preprocess():
    config = Config()
    logger = get_dagster_logger()
    dep_path = config.get_data_path("relevance.article_text_prototype")
    out_path = config.get_data_path("relevance.article_text_preproc")
    
    logger.debug("Concatenating text.")
    article_data = pd.read_parquet(dep_path)
    article_data['text'] = article_data['title'] + "\n.\n" + article_data['bodytext']
    article_data = article_data.drop(columns=['title','bodytext'])
    logger.debug("Normalizing text.")
    article_data = pre.normalize(article_data)

    logger.debug("Sentencizing text.")
    nlp = load_spacy(config.get_param("relevance.base_model"))
    nlp.disable_pipes()
    nlp.add_pipe("sentencizer")

    docs = nlp.pipe(article_data['text'], batch_size=64)
    article_data['docs'] = [[(i, s.text) for i,s in enumerate(d.sents)] for d in docs]
    article_data = (article_data
                .drop(columns=['text'])
                .explode('docs'))
    sentences = pd.DataFrame.from_records(article_data['docs'], columns=['sentence_idx', 'sentence'])
    article_data = pd.concat([article_data.drop(columns=['docs']), sentences], axis=1)

    article_data.to_json(out_path, orient='records', index=False, force_ascii=True)

@dg.asset(deps=[preprocess])
def annotate():
    config = Config()
    out_path = config.get_data_path("relevance.article_text_labeled")
    assert os.path.exists(out_path)

@dg.multi_asset(
        deps=[annotate],
        outs={
            "relevance_article_text_train": dg.AssetOut(),
            "relevance_article_text_dev": dg.AssetOut(),
            "relevance_article_text_test": dg.AssetOut(),
        }
)
def split():
    config = Config()
    logger = get_dagster_logger()
    dep_path = config.get_data_path("relevance.article_text_preproc")
    train_path = config.get_data_path("relevance.article_text_train")
    dev_path = config.get_data_path("relevance.article_text_dev")
    test_path = config.get_data_path("relevance.article_text_test")

    logger.debug("Splitting text.")
    article_data = pd.read_json(dep_path, lines=True, orient="records")
    train, dev, test = pre.split_train_dev_test(article_data)
    del article_data

    logger.debug("Writing text.")
    train.filter(['id','text']).to_json(train_path, lines=True, orient='records', force_ascii=True)
    dev.filter(['id','text']).to_json(dev_path, lines=True, orient='records', force_ascii=True)
    test.filter(['id','text']).to_json(test_path, lines=True, orient='records', force_ascii=True)
    
    return ("relevance_article_text_train","relevance_article_text_dev","relevance_article_text_test")


defs = dg.Definitions(assets=[extract, 
                              news_relevant,
                              prototype_sample,
                              preprocess,
                              annotate,
                              split])

