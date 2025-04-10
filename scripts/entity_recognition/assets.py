import os
from functools import partial
import dagster as dg

from scripts.utils.config import Config
from scripts.entity_recognition import operations as ops
from scripts.utils.dagster import dg_standard_table, dg_standard_doc

config = Config()
PREFIX = "entity_recognition"
dg_asset = partial(dg.asset, key_prefix=[PREFIX], group_name=PREFIX)
dg_asset_out = partial(dg.AssetOut, key_prefix=[PREFIX], group_name=PREFIX)

split_train_key = dg.AssetKey([PREFIX,'split_train'])
split_dev_key = dg.AssetKey([PREFIX,'split_dev'])
split_test_key = dg.AssetKey([PREFIX,'split_test'])

@dg.multi_asset(outs= {
    "split_train": dg_asset_out(description="Training data"),
    "split_dev": dg_asset_out(description="Dev data"),
    "split_test": dg_asset_out(description="Eval data")
    },
    deps = [dg.AssetDep(dg.AssetKey(['sent_relevance','inference']))],
    name = dg.AssetKey([PREFIX, "split"]).to_python_identifier()
)
def split():
    in_path = config.get_data_path("sent_relevance.article_text_filtered")
    train_path = config.get_data_path("entity_recognition.article_text_train")
    dev_path = config.get_data_path("entity_recognition.article_text_dev")
    test_path = config.get_data_path("entity_recognition.article_text_test")
    train, dev, test = ops.split(in_path, train_path, dev_path, test_path)
    yield dg_standard_table(train, asset_key=split_train_key, meta={
        "nunique_articles": train['id'].nunique(),
    })
    yield dg_standard_table(dev, asset_key=split_dev_key, meta={
        "nunique_articles": dev['id'].nunique(),
    })
    yield dg_standard_table(test, asset_key=split_test_key, meta={
        "nunique_articles": test['id'].nunique(),
    })

@dg_asset(deps=[split_train_key], description="Training data for label studio")
def prelabel_train():
    in_path = config.get_data_path("entity_recognition.article_text_train")
    out_path = config.get_data_path("entity_recognition.prelabel_train")
    base_model = config.get_param("entity_recognition.base_model")
    docs_train = ops.prelabel(in_path, out_path, base_model)
    yield dg.MaterializeResult(metadata={
        "dagster/row_count": len(docs_train)
    })

@dg_asset(deps=[split_dev_key], description="Training data for label studio")
def prelabel_dev():
    in_path = config.get_data_path("entity_recognition.article_text_dev")
    out_path = config.get_data_path("entity_recognition.prelabel_dev")
    base_model = config.get_param("entity_recognition.base_model")
    docs_dev = ops.prelabel(in_path, out_path, base_model)
    yield dg.MaterializeResult(metadata={
        "dagster/row_count": len(docs_dev)
    })

@dg_asset(description="Dev data from LabelStudio", deps=[prelabel_dev])
def annotate_dev():
    # Creating the verbose labels is a manual process! 
    # Used Label Studio on preprocessed outs.
    in_path_dev = config.get_data_path("entity_recognition.labels_dev_verbose")
    out_path_dev = config.get_data_path("entity_recognition.labels_dev")
    df_dev = ops.annotate(in_path_dev, out_path_dev)
    dg_standard_table(df_dev)

@dg_asset(description="Training data from LabelStudio", deps=[prelabel_train])
def annotate_train():
    in_path_train = config.get_data_path("entity_recognition.labels_train_verbose")
    out_path_train = config.get_data_path("entity_recognition.labels_train")
    df_train = ops.annotate(in_path_train, out_path_train)
    dg_standard_table(df_train)

@dg_asset(deps=[annotate_dev, annotate_train,
                dg.AssetKey(["geoms",'street_names']), 
                dg.AssetKey(["geoms","comm_areas"]),
                dg.AssetKey(["geoms", "neighborhoods"])],
          description="Train NER")
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

@dg_asset(deps=[dg.AssetKey(["sent_relevance","inference"]), train],
                description="Run NER")
def inference():
    in_path = config.get_data_path("sent_relevance.article_text_filtered")
    out_path = config.get_data_path("entity_recognition.inference")
    model_path = config.get_file_path("entity_recognition.trained_model")
    model_path = os.path.join(model_path, "model-best")
    docs = ops.inference(in_path, model_path, out_path, filter_=True)
    return dg_standard_doc(docs)


# NOTE: A good way to make this half-idempotent is to have a "sink" version
# of each re-run. So you run the pipe again and get new seeds and new rows
# and whatnot and then after that you put it in an "accumulator" func that
# just takes the new stuff and APPENDS to the sink asset. And then the 
# rest of the pipeline uses the sink asset.

defs = dg.Definitions(assets=[split,
                              prelabel_train, 
                              prelabel_dev,
                              annotate_train,
                              annotate_dev,
                              train,
                              inference])