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

# @dg_asset(deps = [prev], description="Combine to articles again.")
# def join_sentences():
#     in_path = config.get_data_path("entity_recognition.inference")
#     model = config.get_param("neighborhood_clf.base_model")
#     out_path = config.get_data_path("neighborhood_clf.articles")
#     ops.join_sentences(in_path, model, out_path)


@dg_asset(deps = [dg.AssetKey(["geoms","intersection_labels"])],
          description="Create synthetic labels from known intersection shapefile")
def synthetic_data():
    intersections_path = config.get_data_path("geoms.intersection_labels")
    out_path = config.get_data_path("neighborhood_clf.crosses")
    df = ops.synthetic_data(intersections_path, out_path)
    return dg_standard_doc(df)


@dg_asset(deps=dg.AssetKey(["prior_model","geocodes"]))
def geocodes():
    in_path = config.get_data_path("prior_model.geocodes")
    out_path = config.get_data_path("neighborhood_clf.geocodes")
    df = ops.geocodes(in_path, out_path)
    return dg_standard_table(df)


@dg_asset(deps=[dg.AssetKey(["entity_recognition", "inference"]),
                dg.AssetKey(["geoms", "block_labels"]),
                dg.AssetKey(["geoms", "intersection_labels"])])
def ner_labels():
    in_path = config.get_data_path("entity_recognition.inference")
    block_labels = config.get_data_path("geoms.block_labels")
    cross_labels = config.get_data_path("geoms.intersection_labels")
    model = config.get_param("neighborhood_clf.base_model")
    out_path = config.get_data_path("neighborhood_clf.ner_labels")
    df = ops.ner_labels(in_path, model, block_labels, cross_labels, out_path)
    return dg_standard_table(df)

@dg_asset(deps = [dg.AssetKey(["geoms","intersection_labels"]), 
                  geocodes, 
                  ner_labels],
          description="Combine data for training")
def merge_synth_data():
    cross_path = config.get_data_path("geoms.intersection_labels")
    geo_path = config.get_data_path("neighborhood_clf.geocodes")
    ner_path = config.get_data_path("neighborhood_clf.ner_labels")
    model = config.get_param("neighborhood_clf.base_model")
    out_path = config.get_data_path("neighborhood_clf.synth_data")
    df = ops.merge_synth_data(cross_path, geo_path, ner_path, model, out_path)
    return dg_standard_table(df)

# @dg_asset(deps=[dg.AssetKey(["entity_recognition", "inference"])], 
#           description="Normalize text for labeling")
# def pre_annotate():
#     in_path = config.get_data_path("entity_recognition.inference")
#     model = config.get_param("neighborhood_clf.base_model")
#     out_path = config.get_data_path("neighborhood_clf.pre_annotate")
#     df = ops.pre_annotate(in_path, model, out_path)
#     return dg_standard_table(df)


# @dg_asset(deps=[pre_annotate], description="Manually label in Label Studio")
# def annotate():
#     # Creating the verbose labels is a manual process! 
#     # Used Label Studio on preprocessed outs.
#     in_path = config.get_data_path("neighborhood_clf.labeled_verbose_train")
#     out_path = config.get_data_path("neighborhood_clf.labeled_train")
#     df = ops.annotate(in_path, out_path)
#     return dg_standard_table(df)

split_train_key = dg.AssetKey([PREFIX, "split_train"])
split_dev_key = dg.AssetKey([PREFIX, "split_dev"])
split_test_key = dg.AssetKey([PREFIX, "split_test"])

@dg.multi_asset(
    outs= {"split_train": dg_asset_out(description="Training data"),
            "split_dev": dg_asset_out(description="Dev data"),
            "split_test": dg_asset_out(description="Eval data")
    },
    deps = [merge_synth_data],
    name = dg.AssetKey([PREFIX, "split"]).to_python_identifier()
)
def split():
    in_path = config.get_data_path("neighborhood_clf.synth_data")
    model = config.get_param("neighborhood_clf.base_model")
    train_path = config.get_data_path("neighborhood_clf.train")
    dev_path = config.get_data_path("neighborhood_clf.dev")
    test_path = config.get_data_path("neighborhood_clf.test")
    train, dev, test = ops.split(in_path, model, train_path, dev_path, test_path)
    yield dg_standard_doc(train, asset_key=split_train_key)
    yield dg_standard_doc(dev, asset_key=split_dev_key)
    yield dg_standard_doc(test, asset_key=split_test_key)


@dg_asset(deps=[dg.AssetKey(["geoms", "block_labels"]),
                dg.AssetKey(["geoms","neighborhood_labels"])],
          description="Train article relevance classifier")
def init_model():
    blocks_path = config.get_data_path("geoms.block_labels")
    neighborhood_path = config.get_data_path("geoms.neighborhood_labels")
    base_cfg = config.get_file_path("neighborhood_clf.base_cfg")
    full_cfg = config.get_file_path("neighborhood_clf.full_cfg")
    out_path = config.get_file_path("neighborhood_clf.trained_model")
    out_path = os.path.join(out_path, "model-best")
    ops.init_model(base_cfg, full_cfg, out_path, blocks_path, neighborhood_path)


@dg_asset(deps=[split_train_key, split_dev_key, init_model])
def train():
    train_path = config.get_data_path("neighborhood_clf.train")
    dev_path = config.get_data_path("neighborhood_clf.dev")
    full_cfg = config.get_file_path("neighborhood_clf.full_cfg")
    block_labels = config.get_data_path("geoms.block_labels")
    comm_labels = config.get_data_path("geoms.neighborhood_labels")
    out_path = config.get_file_path("neighborhood_clf.trained_model")
    metrics = ops.train(full_cfg, train_path, dev_path, block_labels, comm_labels, out_path)
    return dg.MaterializeResult(metadata=metrics)

# @dg_asset(deps=[prev, init_model],
#           description="Pass original data through ml model")
@dg_asset(deps=[split_train_key, init_model],
          description="Pass original data through ml model")
def inference():
    in_data_path = config.get_data_path("entity_recognition.inference")
    out_data_path = config.get_data_path("neighborhood_clf.inference")
    model_path = config.get_file_path("neighborhood_clf.trained_model")
    best_model_path = os.path.join(model_path, "model-best")
    docs = ops.inference(best_model_path, in_data_path, out_data_path)
    return dg_standard_doc(docs)

defs = dg.Definitions(assets=[
    synthetic_data,
    geocodes,
    ner_labels,
    merge_synth_data,
    # pre_annotate,
    split,
#   annotate_train,
#   annotate_dev,
    init_model,
    # train_synthetic,
    train,
    inference])