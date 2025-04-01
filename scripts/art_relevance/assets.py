"""Defines Dagster pipeline."""
import dagster as dg
import os
from functools import partial

from scripts.utils.config import Config
from scripts.pre_model import assets as pre_ops
from scripts.art_relevance import operations as ops
from scripts.utils.dagster import dg_standard_table

config = Config()
PREFIX = "art_relevance"
dg_asset = partial(dg.asset, key_prefix=[PREFIX], group_name=PREFIX)
dg_asset_out = partial(dg.AssetOut, key_prefix=[PREFIX], group_name=PREFIX)

@dg_asset(deps=[pre_ops.pre_relevant], description="Normalize text for labeling")
def pre_annotate():
    in_path = config.get_data_path("pre_relevance.article_text_filtered")
    out_path = config.get_data_path("art_relevance.article_text_pre_annotate")
    k = config.get_param("art_relevance.k")
    seed = config.get_param("art_relevance.proto_seed")
    df = ops.pre_annotate(in_path, out_path, k, seed)
    return dg_standard_table(df)

@dg_asset(deps=[pre_annotate], description="Manually label in Label Studio")
def annotate():
    # Creating the verbose labels is a manual process! 
    # Used Label Studio on preprocessed outs.
    in_path = config.get_data_path("art_relevance.article_text_labeled_verbose")
    out_path = config.get_data_path("art_relevance.article_text_labeled")
    df = ops.annotate(in_path, out_path)
    return dg_standard_table(df, meta = {
        "pct_positive": float((df['label'] == 'CRIME').mean())
    })

split_train_key = dg.AssetKey([PREFIX, "split_train"])
split_dev_key = dg.AssetKey([PREFIX, "split_dev"])
split_test_key = dg.AssetKey([PREFIX, "split_test"])

@dg.multi_asset(
    outs= {"split_train": dg_asset_out(description="Training data"),
            "split_dev": dg_asset_out(description="Dev data"),
            "split_test": dg_asset_out(description="Eval data")
    },
    deps = [annotate],
    name = dg.AssetKey([PREFIX, "split"]).to_python_identifier()
)
def split():
    in_path = config.get_data_path("art_relevance.article_text_labeled")
    train_path = config.get_data_path("art_relevance.article_text_train")
    dev_path = config.get_data_path("art_relevance.article_text_dev")
    test_path = config.get_data_path("art_relevance.article_text_test")
    train, dev, test = ops.split(in_path, train_path, dev_path, test_path)
    yield dg_standard_table(train, asset_key=split_train_key)
    yield dg_standard_table(dev, asset_key=split_dev_key)
    yield dg_standard_table(test, asset_key=split_test_key)


@dg_asset(deps=[split_train_key, split_dev_key, split_test_key],
          description="Train article relevance classifier")
def train():
    train_path = config.get_data_path("art_relevance.article_text_train")
    dev_path = config.get_data_path("art_relevance.article_text_dev")
    base_cfg = config.get_param("art_relevance.base_cfg")
    full_cfg = config.get_param("art_relevance.full_cfg")
    out_path = config.get_param("art_relevance.trained_model")
    metrics = ops.train(base_cfg, full_cfg, train_path, dev_path, out_path)
    return dg.MaterializeResult(metadata=metrics)

@dg_asset(deps=[pre_ops.pre_relevant, train], 
          description="Pass original data through ml model")
def inference():
    in_data_path = config.get_data_path("pre_relevance.article_text_filtered")
    out_data_path = config.get_data_path("art_relevance.article_text_filtered")
    model_path = config.get_file_path("art_relevance.trained_model")
    seed = config.get_param("entity_recognition.proto_seed")
    k = config.get_param("entity_recognition.k")
    best_model_path = os.path.join(model_path, "model-best")
    df = ops.inference(best_model_path, in_data_path, out_data_path, k, seed)
    return dg_standard_table(df)

defs = dg.Definitions(assets=[pre_annotate,
                              annotate,
                              split,
                              train,
                              inference])