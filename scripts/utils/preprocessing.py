import pandas as pd
from textacy import preprocessing as tp
from functools import singledispatch
from spacy.tokens import Doc, DocBin
from typing import TypeAlias

Text: TypeAlias = pd.DataFrame | list[Doc]

def split_train_dev_test(data: Text,
                         train_path: str = None, 
                         dev_path: str = None, 
                         test_path: str = None,
                         train_frac = .8,
                         test_frac = .5,
                         stratify: list[str] = None) -> tuple[Text,Text,Text]:
    train, dev_test = _split(data, train_frac, stratify)
    dev, test = _split(dev_test, test_frac, stratify)
    if train_path:
        train.to_parquet(train_path)
    if dev_path:
        dev.to_parquet(dev_path)
    if test_path:
        test.to_parquet(test_path)
    return train, dev, test

@singledispatch
def _split(data: Text, frac: float, stratify: list[str] = None) -> tuple[Text,Text]:
    pass

@_split.register(list)
def _(data: list[Doc], frac: float, stratify: list[str] = None) -> tuple[list[Doc],list[Doc]]:
    left_idx = pd.Series(list(range(len(data)))).sample(frac=frac, random_state=3925)
    left, right = [], []
    for i, doc in enumerate(data):
        if i in left_idx:
            left.append(doc)
        else:
            right.append(doc)
    return (left, right)

@_split.register(pd.DataFrame)
def _(data: pd.DataFrame, frac: float, stratify: list[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if stratify:
        df_items = data[stratify].drop_duplicates().sample(frac=frac, random_state=3925)
        splits = df_items.merge(data, how='outer', indicator=True)
        left = splits[splits._merge == 'both'].drop(columns='_merge')
        right = splits[splits._merge == 'right_only'].drop(columns='_merge')
    else:
        left = data.sample(frac=frac, random_state=3925)
        right = data.loc[data.index.difference(left.index)]
    return (left, right)

def normalize(df: pd.DataFrame, text_col: str = "text"):
    preproc = tp.make_pipeline(
        tp.normalize.unicode,
        tp.replace.urls,
        tp.replace.user_handles,
        tp.replace.emails,
        tp.normalize.whitespace,
    )
    df = df.dropna(subset=text_col)
    df = df.assign(**{text_col: df[text_col].str.replace('\n', ' ').apply(preproc)})
    return df