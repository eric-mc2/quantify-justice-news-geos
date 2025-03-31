"""Defines Dagster pipeline."""
import dagster as dg
from functools import partial

from scripts.utils.config import Config
from scripts.pre_model import operations as ops
from scripts.utils.dagster import dg_standard_table

config = Config()
PREFIX = "pre_model"
dg_asset = partial(dg.asset, key_prefix=[PREFIX], group_name=PREFIX)

@dg_asset
def extract():
    in_path = config.get_data_path("raw.zip")
    out_path = config.get_data_path("raw.article_text")
    ops.extract(in_path, out_path)

@dg_asset(deps=[extract], description="Filter using external relevance model")
def pre_relevant():
    in_path = config.get_data_path("raw.article_text")
    out_path = config.get_data_path("pre_relevance.article_text_filtered")
    df = ops.news_relevant(in_path, out_path)
    return dg_standard_table(df)

defs = dg.Definitions(assets=[extract, 
                              pre_relevant])