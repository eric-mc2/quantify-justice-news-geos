from functools import partial
import dagster as dg
import os

from scripts.utils.config import Config
from scripts.entity_recognition import operations as ops
from scripts.utils.dagster import dg_table_schema

config = Config()
PREFIX = "entity_recognition"
dg_asset = partial(dg.asset, key_prefix=[PREFIX])

prev = dg.AssetDep(dg.AssetKey(['sent_relevance','filter']))

train_key = dg.AssetKey([PREFIX,'split_train'])
dev_key = dg.AssetKey([PREFIX,'split_dev'])
test_key = dg.AssetKey([PREFIX,'split_test'])
split_train = dg.AssetSpec(train_key, deps=[prev], description="Training data")
split_dev = dg.AssetSpec(dev_key, deps=[prev], description="Dev data")
split_test = dg.AssetSpec(test_key, deps=[prev], description="Eval data")

@dg.multi_asset(specs = [split_train, split_dev, split_test])
def split():
    dep_path = config.get_data_path("sent_relevance.article_text_filtered")
    train_path = config.get_data_path("entity_recognition.article_text_train")
    dev_path = config.get_data_path("entity_recognition.article_text_dev")
    test_path = config.get_data_path("entity_recognition.article_text_test")
    train, dev, test = ops.split(dep_path, train_path, dev_path, test_path)
    yield dg.MaterializeResult(asset_key=train_key, metadata={
        "dagster/column_schema": dg_table_schema(train),
        "dagster/row_count": len(train),
        "nunique_articles": train['id'].nunique(),
    })
    yield dg.MaterializeResult(asset_key=dev_key, metadata={
        "dagster/column_schema": dg_table_schema(dev),
        "dagster/row_count": len(dev),
        "nunique_articles": dev['id'].nunique(),
    })
    yield dg.MaterializeResult(asset_key=test_key, metadata={
        "dagster/column_schema": dg_table_schema(test),
        "dagster/row_count": len(test),
        "nunique_articles": test['id'].nunique(),
    })

defs = dg.Definitions(assets=[split])