"""Defines Dagster pipeline."""
from scripts.utils.config import Config
import dagster as dg
from scripts.art_relevance import operations as ops
from scripts.utils.runners import dg_table_schema
import os
import json

config = Config()

@dg.asset
def extract():
    dep_path = config.get_data_path("raw.zip")
    out_path = config.get_data_path("raw.article_text")
    ops.extract(dep_path, out_path)

@dg.asset(deps=[extract], description="Filter using external relevance model")
def pre_relevant():
    dep_path = config.get_data_path("raw.article_text")
    out_path = config.get_data_path("pre_relevance.article_text")
    ops.news_relevant(dep_path, out_path)

@dg.asset(deps=[pre_relevant], description="Pick k << N rows for rapid dev/exploration")
def prototype_sample():
    dep_path = config.get_data_path("pre_relevance.article_text")
    out_path = config.get_data_path("art_relevance.article_text_prototype")
    df = ops.prototype_sample(dep_path, out_path)
    return dg.MaterializeResult(metadata={
        "dagster/column_schema": dg_table_schema(df),
        "dagster/row_count": len(df),
    })
    
@dg.asset(deps=[prototype_sample], description="Normalize text for labeling")
def preprocess():
    dep_path = config.get_data_path("art_relevance.article_text_prototype")
    out_path = config.get_data_path("art_relevance.article_text_preproc")
    df = ops.preprocess(dep_path, out_path)
    return dg.MaterializeResult(metadata={
        "dagster/column_schema": dg_table_schema(df),
        "dagster/row_count": len(df),
    })

@dg.asset(deps=[preprocess], description="Manually label in Label Studio")
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

@dg.multi_asset(
        deps=[annotate],
        outs={
            "art_relevance_article_text_train": dg.AssetOut(description="Training data"),
            "art_relevance_article_text_dev": dg.AssetOut(description="Testing data"),
            "art_relevance_article_text_test": dg.AssetOut(description="Final performance estimate"),
        },
)
def split():
    dep_path = config.get_data_path("art_relevance.article_text_labeled")
    train_path = config.get_data_path("art_relevance.article_text_train")
    dev_path = config.get_data_path("art_relevance.article_text_dev")
    test_path = config.get_data_path("art_relevance.article_text_test")
    ops.split(dep_path, train_path, dev_path, test_path)
    return ("art_relevance_article_text_train","art_relevance_article_text_dev","art_relevance_article_text_test")

@dg.asset(deps=["art_relevance_article_text_train","art_relevance_article_text_dev"],
          description="Train article relevance classifier")
def train():
    train_path = config.get_data_path("art_relevance.article_text_train")
    dev_path = config.get_data_path("art_relevance.article_text_dev")
    base_cfg = config.get_param("art_relevance.base_cfg")
    full_cfg = config.get_param("art_relevance.full_cfg")
    out_path = config.get_param("art_relevance.trained_model")
    ops.init_config(base_cfg, full_cfg)
    ops.train(train_path, dev_path, full_cfg, out_path)
    
    best_model_path = os.path.join(out_path, "model-best")
    metric_file = os.path.join(best_model_path, "meta.json")
    with open(metric_file) as fp:
        metrics = json.load(fp)['performance']
    track_metrics = dict(
        precision = metrics['cats_f_per_type']['CRIME']['p'],
        recall = metrics['cats_f_per_type']['CRIME']['r'],
        f1 = metrics['cats_f_per_type']['CRIME']['f'],
    )
    return dg.MaterializeResult(metadata=track_metrics)

@dg.asset(deps=[pre_relevant, train], description="Pass original data through ml model")
def filter():
    in_data_path = config.get_data_path("pre_relevance.article_text")
    out_data_path = config.get_data_path("sent_relevance.article_text_prototype")
    model_path = config.get_param("art_relevance.trained_model")
    best_model_path = os.path.join(model_path, "model-best")
    df = ops.filter(best_model_path, in_data_path, out_data_path)
    return dg.MaterializeResult(metadata={
        "dagster/column_schema": dg_table_schema(df),
        "dagster/row_count": len(df),
    })

defs = dg.Definitions(assets=[extract, 
                              pre_relevant,
                              prototype_sample,
                              preprocess,
                              annotate,
                              split,
                              train,
                              filter])