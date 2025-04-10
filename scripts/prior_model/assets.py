"""Defines Dagster pipeline."""
import dagster as dg
from functools import partial

from scripts.utils.config import Config
from scripts.prior_model import operations as ops
from scripts.utils.dagster import dg_standard_table

config = Config()
PREFIX = "prior_model"
dg_asset = partial(dg.asset, key_prefix=[PREFIX], group_name=PREFIX)

@dg_asset
def extract():
    in_path = config.get_data_path("raw.zip")
    out_path = config.get_data_path("raw.article_text")
    ops.extract(in_path, out_path)

@dg_asset(deps=[extract], description="Filter using external relevance model")
def pre_relevant():
    in_path = config.get_data_path("raw.article_text")
    out_path = config.get_data_path("prior_model.article_text_filtered")
    df = ops.news_relevant(in_path, out_path)
    return dg_standard_table(df)

@dg_asset
def geocodes():
    in_path = config.get_data_path("raw.zip")
    out_path = config.get_data_path("prior_model.geocodes")
    df = ops.geocodes(in_path, out_path)
    return dg_standard_table(df)

@dg_asset(description="Aligned user-coded location entities")
def user_coding():
    in_path = config.get_data_path("raw.zip")
    articles_path = config.get_data_path("prior_model.article_text_filtered")
    out_path = config.get_data_path("prior_model.user_coding")
    df = ops.user_coding(in_path, articles_path, out_path)
    return dg_standard_table(df)


defs = dg.Definitions(assets=[extract, 
                              pre_relevant,
                              geocodes,
                              user_coding])