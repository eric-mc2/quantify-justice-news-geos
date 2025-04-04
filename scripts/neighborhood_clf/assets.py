"""Defines Dagster pipeline."""
import dagster as dg
import os
from functools import partial

from scripts.utils.config import Config
from scripts.neighborhood_clf import operations as ops
from scripts.utils.dagster import dg_standard_table, dg_standard_doc

config = Config()
PREFIX = "neighborhood_clf"
dg_asset = partial(dg.asset, key_prefix=[PREFIX], group_name=PREFIX)
dg_asset_out = partial(dg.AssetOut, key_prefix=[PREFIX], group_name=PREFIX)

prev = dg.AssetKey(["entity_recognition", "inference"])

@dg_asset(deps = [prev], description="Combine to articles again.")
def join_sentences():
    in_path = config.get_data_path("entity_recognition.inference")
    model = config.get_param("neighborhood_clf.base_model")
    out_path = config.get_data_path("neighborhood_clf.articles")
    ops.join_sentences(in_path, model, out_path)

split_train_key = dg.AssetKey([PREFIX, "split_train"])
split_dev_key = dg.AssetKey([PREFIX, "split_dev"])
split_test_key = dg.AssetKey([PREFIX, "split_test"])

@dg.multi_asset(
    outs= {"split_train": dg_asset_out(description="Training data"),
            "split_dev": dg_asset_out(description="Dev data"),
            "split_test": dg_asset_out(description="Eval data")
    },
    deps = [join_sentences],
    name = dg.AssetKey([PREFIX, "split"]).to_python_identifier()
)
def split():
    in_path = config.get_data_path("neighborhood_clf.articles")
    train_path = config.get_data_path("neighborhood_clf.train")
    dev_path = config.get_data_path("neighborhood_clf.dev")
    test_path = config.get_data_path("neighborhood_clf.test")
    train, dev, test = ops.split(in_path, train_path, dev_path, test_path)
    yield dg_standard_doc(train, asset_key=split_train_key)
    yield dg_standard_doc(dev, asset_key=split_dev_key)
    yield dg_standard_doc(test, asset_key=split_test_key)

@dg_asset(deps=[split_train_key], description="Normalize text for labeling")
def pre_annotate_train():
    in_path = config.get_data_path("neighborhood_clf.train")
    out_path = config.get_data_path("neighborhood_clf.pre_annotate_train")
    df = ops.pre_annotate(in_path, out_path)
    return dg_standard_table(df)

@dg_asset(deps=[split_dev_key], description="Normalize text for labeling")
def pre_annotate_dev():
    in_path = config.get_data_path("neighborhood_clf.dev")
    out_path = config.get_data_path("neighborhood_clf.pre_annotate_dev")
    df = ops.pre_annotate(in_path, out_path)
    return dg_standard_table(df)

@dg_asset(deps=[pre_annotate_train], description="Manually label in Label Studio")
def annotate_train():
    # Creating the verbose labels is a manual process! 
    # Used Label Studio on preprocessed outs.
    in_path = config.get_data_path("neighborhood_clf.labeled_verbose_train")
    out_path = config.get_data_path("neighborhood_clf.labeled_train")
    df = ops.annotate(in_path, out_path)
    return dg_standard_table(df)

@dg_asset(deps=[pre_annotate_dev], description="Manually label in Label Studio")
def annotate_dev():
    # Creating the verbose labels is a manual process! 
    # Used Label Studio on preprocessed outs.
    in_path = config.get_data_path("neighborhood_clf.labeled_verbose_dev")
    out_path = config.get_data_path("neighborhood_clf.labeled_dev")
    df = ops.annotate(in_path, out_path)
    return dg_standard_table(df)

@dg_asset(deps=[annotate_train, annotate_dev],
          description="Train article relevance classifier")
def train():
    train_path = config.get_data_path("neighborhood_clf.labeled_train")
    dev_path = config.get_data_path("neighborhood_clf.labeled_dev")
    base_cfg = config.get_param("neighborhood_clf.base_cfg")
    full_cfg = config.get_param("neighborhood_clf.full_cfg")
    out_path = config.get_param("neighborhood_clf.trained_model")
    metrics = ops.train(base_cfg, full_cfg, train_path, dev_path, out_path)
    return dg.MaterializeResult(metadata=metrics)

@dg_asset(deps=[prev, train],
          description="Pass original data through ml model")
def inference():
    in_data_path = config.get_data_path("entity_recognition.inference")
    out_data_path = config.get_data_path("neighborhood_clf.inference")
    model_path = config.get_file_path("neighborhood_clf.trained_model")
    best_model_path = os.path.join(model_path, "model-best")
    df = ops.inference(best_model_path, in_data_path, out_data_path)
    return dg_standard_table(df)

defs = dg.Definitions(assets=[join_sentences,
                              split,
                              pre_annotate_dev,
                              pre_annotate_train,
                              annotate_train,
                              annotate_dev,
                              train,
                              inference])