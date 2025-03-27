from functools import partial
import dagster as dg

from scripts.utils.config import Config
from scripts.entity_recognition import operations as ops
from scripts.utils.dagster import dg_table_schema

config = Config()
PREFIX = "entity_recognition"
dg_asset = partial(dg.asset, key_prefix=[PREFIX], group_name=PREFIX)
dg_asset_out = partial(dg.AssetOut, key_prefix=[PREFIX], group_name=PREFIX)

prev = dg.AssetDep(dg.AssetKey(['sent_relevance','filter']))

split_train_key = dg.AssetKey([PREFIX,'split_train'])
split_dev_key = dg.AssetKey([PREFIX,'split_dev'])
split_test_key = dg.AssetKey([PREFIX,'split_test'])
split_train = dg.AssetSpec(split_train_key, deps=[prev], group_name=PREFIX, description="Training data")
split_dev = dg.AssetSpec(split_dev_key, deps=[prev], group_name=PREFIX, description="Dev data")
split_test = dg.AssetSpec(split_test_key, deps=[prev], group_name=PREFIX, description="Eval data")

@dg.multi_asset(specs = [split_train, split_dev, split_test])
def split():
    in_path = config.get_data_path("sent_relevance.article_text_filtered")
    train_path = config.get_data_path("entity_recognition.article_text_train")
    dev_path = config.get_data_path("entity_recognition.article_text_dev")
    test_path = config.get_data_path("entity_recognition.article_text_test")
    train, dev, test = ops.split(in_path, train_path, dev_path, test_path)
    yield dg.MaterializeResult(asset_key=split_train_key, metadata={
        "dagster/column_schema": dg_table_schema(train),
        "dagster/row_count": len(train),
        "nunique_articles": train['id'].nunique(),
    })
    yield dg.MaterializeResult(asset_key=split_dev_key, metadata={
        "dagster/column_schema": dg_table_schema(dev),
        "dagster/row_count": len(dev),
        "nunique_articles": dev['id'].nunique(),
    })
    yield dg.MaterializeResult(asset_key=split_test_key, metadata={
        "dagster/column_schema": dg_table_schema(test),
        "dagster/row_count": len(test),
        "nunique_articles": test['id'].nunique(),
    })

prelabels_train_key = dg.AssetKey([PREFIX,'prelabels_train'])
prelabels_dev_key = dg.AssetKey([PREFIX,'prelabels_dev'])
prelabels_train = dg.AssetSpec(prelabels_train_key, deps=[split_train], group_name=PREFIX, description="Training data for LabelStudio")
prelabels_dev = dg.AssetSpec(prelabels_dev_key, deps=[split_dev], group_name=PREFIX, description="Dev data for LabelStudio")

@dg.multi_asset(specs = [prelabels_train, prelabels_dev])
def prelabel():
    in_path = config.get_data_path("entity_recognition.article_text_train")
    out_path = config.get_data_path("entity_recognition.prelabel_train")
    base_model = config.get_param("entity_recognition.base_model")
    docs_train = ops.prelabel(in_path, out_path, base_model)
    in_path = config.get_data_path("entity_recognition.article_text_dev")
    out_path = config.get_data_path("entity_recognition.prelabel_dev")
    docs_dev = ops.prelabel(in_path, out_path, base_model)
    yield dg.MaterializeResult(asset_key=prelabels_train_key, metadata={
        "dagster/row_count": len(docs_train)
    })
    yield dg.MaterializeResult(asset_key=prelabels_dev_key, metadata={
        "dagster/row_count": len(docs_dev)
    })

labels_train_key = dg.AssetKey([PREFIX,'labels_train'])
labels_dev_key = dg.AssetKey([PREFIX,'labels_dev'])

@dg.multi_asset(outs = {"labels_train": dg_asset_out(description="Training data from LabelStudio"), 
                        "labels_dev": dg_asset_out(description="Dev data from LabelStudio")},
                deps=[prelabels_train, prelabels_dev])
def annotate():
    # Creating the verbose labels is a manual process! 
    # Used Label Studio on preprocessed outs.
    in_path_dev = config.get_data_path("entity_recognition.labels_dev_verbose")
    out_path_dev = config.get_data_path("entity_recognition.labels_dev")
    df_dev = ops.annotate(in_path_dev, out_path_dev)
    in_path_train = config.get_data_path("entity_recognition.labels_train_verbose")
    out_path_train = config.get_data_path("entity_recognition.labels_train")
    df_train = ops.annotate(in_path_train, out_path_train)
    yield dg.MaterializeResult(asset_key=labels_train_key, metadata={
        "dagster/column_schema": dg_table_schema(df_train),
        "dagster/row_count": len(df_train)
        }
    )
    yield dg.MaterializeResult(asset_key=labels_dev_key, metadata={
        "dagster/column_schema": dg_table_schema(df_dev),
        "dagster/row_count": len(df_dev)
        }
    )

@dg_asset(deps=[labels_train_key, labels_dev_key],
          description="Train sentence relevance classifier")
def train():
    train_path = config.get_data_path("entity_recognition.labels_train")
    dev_path = config.get_data_path("entity_recognition.labels_dev")
    base_cfg = config.get_file_path("entity_recognition.base_cfg")
    full_cfg = config.get_file_path("entity_recognition.full_cfg")
    out_path = config.get_file_path("entity_recognition.trained_model")
    comm_area_path = config.get_data_path("geoms.comm_areas")
    neighborhood_path = config.get_data_path("geoms.neighborhoods")
    street_name_path = config.get_data_path("geoms.street_names")

    metrics = ops.train(base_cfg, 
                        full_cfg, 
                        train_path, 
                        dev_path, 
                        out_path,
                        comm_area_path,
                        neighborhood_path,
                        street_name_path)
    return dg.MaterializeResult(metadata=metrics)

# TODO: Continue implementing language components:
# read https://spacy.io/usage/processing-pipelines#component-data-initialization
# and read https://spacy.io/usage/training#config
# Not sure if i want to make it an assset or just as part of train op.



# NOTE: A good way to make this half-idempotent is to have a "sink" version
# of each re-run. So you run the pipe again and get new seeds and new rows
# and whatnot and then after that you put it in an "accumulator" func that
# just takes the new stuff and APPENDS to the sink asset. And then the 
# rest of the pipeline uses the sink asset.

defs = dg.Definitions(assets=[split_train, split_dev, split_test, 
                              prelabels_train, prelabels_dev,
                              annotate,
                              train])