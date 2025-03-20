from functools import partial
import dagster as dg

from scripts.utils.config import Config
from scripts.sent_relevance import operations as ops
from scripts.utils.dagster import dg_table_schema

config = Config()
dg_asset = partial(dg.asset, key_prefix=__name__.replace(".","_"))

@dg_asset(deps=["scripts_art_relevance_filter"])
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

@dg.multi_asset(
        deps=[annotate],
        outs={
            "sent_relevance_article_text_train": dg.AssetOut(description="Training data"),
            "sent_relevance_article_text_dev": dg.AssetOut(description="Testing data"),
            "sent_relevance_article_text_test": dg.AssetOut(description="Final performance estimate"),
        },
)
def split():
    dep_path = config.get_data_path("sent_relevance.article_text_labeled")
    train_path = config.get_data_path("sent_relevance.article_text_train")
    dev_path = config.get_data_path("sent_relevance.article_text_dev")
    test_path = config.get_data_path("sent_relevance.article_text_test")
    ops.split(dep_path, train_path, dev_path, test_path)
    return ("sent_relevance_article_text_train",
            "sent_relevance_article_text_dev",
            "sent_relevance_article_text_test")

@dg_asset(deps=["sent_relevance_article_text_train","sent_relevance_article_text_dev"],
          description="Train sentence relevance classifier")
def train():
    train_path = config.get_data_path("sent_relevance.article_text_train")
    dev_path = config.get_data_path("sent_relevance.article_text_dev")
    base_cfg = config.get_param("sent_relevance.base_cfg")
    full_cfg = config.get_param("sent_relevance.full_cfg")
    out_path = config.get_param("sent_relevance.trained_model")
    metrics = ops.train(base_cfg, full_cfg, train_path, dev_path, out_path)
    return dg.MaterializeResult(metadata=metrics)

defs = dg.Definitions(assets=[preprocess,
                              annotate,
                              split,
                              train])

