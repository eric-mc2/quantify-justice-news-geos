from functools import partial
import dagster as dg
import os

from scripts.utils.config import Config
from scripts.sent_relevance import operations as ops
from scripts.utils.dagster import dg_table_schema

config = Config()
PREFIX = "sent_relevance"
dg_asset = partial(dg.asset, key_prefix=[PREFIX])

@dg_asset(deps=[dg.AssetDep(dg.AssetKey(['art_relevance','filter']))])
def preprocess():
    dep_path = config.get_data_path("art_relevance.article_text_filtered")
    out_path = config.get_data_path("sent_relevance.article_text_preproc")
    model_path = config.get_param("sent_relevance.base_model")
    df = ops.preprocess(dep_path, model_path, out_path)
    return dg.MaterializeResult(metadata={
        "dagster/column_schema": dg_table_schema(df),
        "dagster/row_count": len(df),
        "nunique_articles": df['id'].nunique(),
    })

@dg_asset(deps=[preprocess], description="Manually label in Label Studio")
def annotate():
    # Creating the verbose labels is a manual process! 
    # Used Label Studio on preprocessed outs.
    deps_path = config.get_data_path("sent_relevance.article_text_labeled_verbose")
    out_path = config.get_data_path("sent_relevance.article_text_labeled")
    df = ops.annotate(deps_path, out_path)
    all_labels = {lab for row in df['multilabel'] for lab in row}
    label_stats = {}
    for lab in all_labels:
        pct = df['multilabel'].apply(lambda xs: lab in xs).mean()
        label_stats[f"pct_{lab}"] = float(pct)
    return dg.MaterializeResult(metadata={
        "dagster/column_schema": dg_table_schema(df),
        "dagster/row_count": len(df)
        } | label_stats
    )

split_train = dg.AssetSpec(dg.AssetKey([PREFIX,'split_train']), deps=[annotate], description="Training data")
split_dev = dg.AssetSpec(dg.AssetKey([PREFIX,'split_dev']), deps=[annotate], description="Dev data")
split_test = dg.AssetSpec(dg.AssetKey([PREFIX,'split_test']),deps=[annotate], description="Eval data")

@dg.multi_asset(
        specs = [split_train, split_dev, split_test]
)
def split():
    dep_path = config.get_data_path("sent_relevance.article_text_labeled")
    train_path = config.get_data_path("sent_relevance.article_text_train")
    dev_path = config.get_data_path("sent_relevance.article_text_dev")
    test_path = config.get_data_path("sent_relevance.article_text_test")
    ops.split(dep_path, train_path, dev_path, test_path)

@dg_asset(deps=[split_train, split_dev],
          description="Train sentence relevance classifier")
def train():
    train_path = config.get_data_path("sent_relevance.article_text_train")
    dev_path = config.get_data_path("sent_relevance.article_text_dev")
    base_cfg = config.get_param("sent_relevance.base_cfg")
    full_cfg = config.get_param("sent_relevance.full_cfg")
    out_path = config.get_param("sent_relevance.trained_model")
    metrics = ops.train(base_cfg, full_cfg, train_path, dev_path, out_path)
    return dg.MaterializeResult(metadata=metrics)

@dg_asset(deps=[preprocess, train], 
          description="Pass original data through ml model")
def filter():
    in_data_path = config.get_data_path("pre_relevance.article_text_filtered")
    out_data_path = config.get_data_path("sent_relevance.article_text_filtered")
    seed = config.get_param("sent_relevance.proto_seed")
    art_model_path = config.get_file_path("art_relevance.trained_model")
    art_model_path = os.path.join(art_model_path, "model-best")
    sent_model_path = config.get_file_path("sent_relevance.trained_model")
    sent_model_path = os.path.join(sent_model_path, "model-best")
    df = ops.filter(art_model_path, sent_model_path, seed, in_data_path, out_data_path)
    
    return dg.MaterializeResult(metadata={
        "dagster/column_schema": dg_table_schema(df),
        "dagster/row_count": len(df),
    })

defs = dg.Definitions(assets=[preprocess,
                              annotate,
                              split_train,
                              split_dev,
                              split_test,
                              train,
                              filter])

