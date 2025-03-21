"""Defines Dagster pipeline."""
import dagster as dg
import os
from functools import partial

from scripts.utils.config import Config
from scripts.art_relevance import operations as ops
from scripts.utils.dagster import dg_table_schema

config = Config()
PREFIX = "art_relevance"
dg_asset = partial(dg.asset, key_prefix=[PREFIX])

@dg_asset
def extract():
    dep_path = config.get_data_path("raw.zip")
    out_path = config.get_data_path("raw.article_text")
    ops.extract(dep_path, out_path)

@dg_asset(deps=[extract], description="Filter using external relevance model")
def pre_relevant():
    dep_path = config.get_data_path("raw.article_text")
    out_path = config.get_data_path("pre_relevance.article_text_filtered")
    ops.news_relevant(dep_path, out_path)

@dg_asset(deps=[pre_relevant], description="Pick k << N rows for rapid dev/exploration")
def prototype_sample():
    dep_path = config.get_data_path("pre_relevance.article_text_filtered")
    out_path = config.get_data_path("art_relevance.article_text_prototype")
    df = ops.prototype_sample(dep_path, out_path)
    return dg.MaterializeResult(metadata={
        "dagster/column_schema": dg_table_schema(df),
        "dagster/row_count": len(df),
    })
    
@dg_asset(deps=[prototype_sample], description="Normalize text for labeling")
def preprocess():
    dep_path = config.get_data_path("art_relevance.article_text_prototype")
    out_path = config.get_data_path("art_relevance.article_text_preproc")
    df = ops.preprocess(dep_path, out_path)
    return dg.MaterializeResult(metadata={
        "dagster/column_schema": dg_table_schema(df),
        "dagster/row_count": len(df),
    })

@dg_asset(deps=[preprocess], description="Manually label in Label Studio")
def annotate():
    # Creating the verbose labels is a manual process! 
    # Used Label Studio on preprocessed outs.
    deps_path = config.get_data_path("art_relevance.article_text_labeled_verbose")
    out_path = config.get_data_path("art_relevance.article_text_labeled")
    df = ops.annotate(deps_path, out_path)
    return dg.MaterializeResult(metadata={
        "dagster/column_schema": dg_table_schema(df),
        "dagster/row_count": len(df),
        "pct_positive": float((df['label'] == 'CRIME').mean()),
    })

split_train = dg.AssetSpec(dg.AssetKey([PREFIX,'split_train']), deps=[annotate], description="Training data")
split_dev = dg.AssetSpec(dg.AssetKey([PREFIX,'split_dev']), deps=[annotate], description="Dev data")
split_test = dg.AssetSpec(dg.AssetKey([PREFIX,'split_test']), deps=[annotate], description="Eval data")

@dg.multi_asset(
        specs=[split_train, split_dev, split_test],
)
def split():
    dep_path = config.get_data_path("art_relevance.article_text_labeled")
    train_path = config.get_data_path("art_relevance.article_text_train")
    dev_path = config.get_data_path("art_relevance.article_text_dev")
    test_path = config.get_data_path("art_relevance.article_text_test")
    ops.split(dep_path, train_path, dev_path, test_path)

@dg_asset(deps=[split_train, split_dev],
          description="Train article relevance classifier")
def train():
    train_path = config.get_data_path("art_relevance.article_text_train")
    dev_path = config.get_data_path("art_relevance.article_text_dev")
    base_cfg = config.get_param("art_relevance.base_cfg")
    full_cfg = config.get_param("art_relevance.full_cfg")
    out_path = config.get_param("art_relevance.trained_model")
    metrics = ops.train(base_cfg, full_cfg, train_path, dev_path, out_path)
    return dg.MaterializeResult(metadata=metrics)

@dg_asset(deps=[pre_relevant, train], 
          description="Pass original data through ml model")
def filter():
    in_data_path = config.get_data_path("pre_relevance.article_text_filtered")
    out_data_path = config.get_data_path("art_relevance.article_text_filtered")
    model_path = config.get_file_path("art_relevance.trained_model")
    seed = config.get_param("art_relevance.proto_seed")
    best_model_path = os.path.join(model_path, "model-best")
    df = ops.filter(best_model_path, in_data_path, out_data_path, 600, seed)
    return dg.MaterializeResult(metadata={
        "dagster/column_schema": dg_table_schema(df),
        "dagster/row_count": len(df),
    })

defs = dg.Definitions(assets=[extract, 
                              pre_relevant,
                              prototype_sample,
                              preprocess,
                              annotate,
                              split_train,
                              split_dev, 
                              split_test,
                              train,
                              filter])