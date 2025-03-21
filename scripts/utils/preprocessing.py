import pandas as pd
from textacy import preprocessing as tp

def split_train_dev_test(data: pd.DataFrame, 
                         train_path: str = None, 
                         dev_path: str = None, 
                         test_path: str = None,
                         stratify: list[str] = None):
    train, dev_test = _split(data, .8, stratify)
    dev, test = _split(dev_test, .5, stratify)
    if train_path:
        train.to_parquet(train_path)
    if dev_path:
        dev.to_parquet(dev_path)
    if test_path:
        test.to_parquet(test_path)
    return train, dev, test

def _split(df: pd.DataFrame, frac: float, stratify: list[str] = None):
    if stratify:
        df_items = df[stratify].drop_duplicates().sample(frac=frac, random_state=3925)
        splits = df_items.merge(df, how='outer', indicator=True)
        left = splits[splits._merge == 'both'].drop(columns='_merge')
        right = splits[splits._merge == 'right_only'].drop(columns='_merge')
    else:
        left = df.sample(frac=frac, random_state=3925)
        right = df.loc[df.index.difference(left.index)]
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